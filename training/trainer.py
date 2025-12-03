# src/training/trainer.py
"""
Research-grade Trainer for IDS models.

Improvements / fixes over earlier version:
 - Accepts ExperimentLogger or stdlib logging.Logger (and other logger-like objects)
 - Casts to stdlib logger when passing to callbacks to satisfy static type checkers
 - Safe and typed handling of logger.progress_bar(...) which may return tqdm or raw iterable
 - Guards use of tqdm-specific methods like set_postfix_str
 - Normalizes checkpoint path returned by ModelCheckpoint before passing to EarlyStopping
 - Supports resume, atomic last checkpoint saves, AMP, gradient accumulation
 - Clear type hints and defensive programming for robust research usage
"""
from __future__ import annotations

import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Iterator, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score  # For F1-macro during validation

import logging as _logging

# Local callback imports
from ..training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, atomic_save_state  # type: ignore
# Local logging utilities (ExperimentLogger provides progress_bar etc.)
from ..utils.logging_utils import ExperimentLogger, MetricLogger  # type: ignore

# Optional system utils
try:
    from ..utils.system_utils import SystemMonitor, TimingContext  # type: ignore
except Exception:
    SystemMonitor = None
    TimingContext = None


DEFAULT_TRAINING_CONFIG = {
    "optimizer": {"name": "adam", "learning_rate": 1e-3, "weight_decay": 0.0},
    "loss": {"name": "crossentropy"},
    "callbacks": {
        "early_stopping": {"enabled": False, "patience": 10, "min_delta": 1e-4, "monitor": "val_loss", "mode": "min"},
        "model_checkpoint": {
            "enabled": True,
            "monitor": "val_loss",
            "mode": "min",
            "save_best_only": True,
            "save_weights_only": True,
            "filename": "best_model_epoch{epoch}_metric{metric}.pth",
            "save_last": True
        },
        "lr_scheduler": {"enabled": False}
    },
    "training": {
        "epochs": 20,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": None,
        "use_amp": False
    }
}


def _ensure_stdlib_logger(logger_like: Optional[Union[ExperimentLogger, _logging.Logger]]) -> _logging.Logger:
    """
    Return a stdlib logging.Logger that can be passed to callbacks that expect it.
    If an ExperimentLogger (or other wrapper) is provided, attempt to extract the underlying
    stdlib logger via attribute `logger`. Fallback: cast provided object to Logger (best-effort).
    """
    if logger_like is None:
        return _logging.getLogger(__name__)
    # If it's already a stdlib logger
    if isinstance(logger_like, _logging.Logger):
        return logger_like
    # Common pattern: wrapper exposes .logger (the stdlib logger)
    std = getattr(logger_like, "logger", None)
    if isinstance(std, _logging.Logger):
        return std
    # Last resort: cast (satisfy type-checker; runtime may still work if object implements .info/.debug)
    return cast(_logging.Logger, logger_like)  # type: ignore[return-value]


class Trainer:
    """
    Research-grade Trainer.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader, val_loader : DataLoader
        Data loaders for train & validation.
    experiment_dir : Union[str, Path]
        Directory to save checkpoints, logs (Trainer will create subdirs).
    config : Optional[dict]
        Training configuration (defaults merged with DEFAULT_TRAINING_CONFIG).
    device : Union[str, torch.device]
        Device string or torch.device.
    logger : Optional[ExperimentLogger | logging.Logger]
        Logger helper. Trainer will use its .info/.debug methods; if ExperimentLogger is provided
        its progress_bar(...) will be used for iterables.
    metric_logger : Optional[MetricLogger]
        Metric logger used to persist epoch-level metrics.
    resume_from : Optional[Union[str, Path]]
        Checkpoint path to resume from.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        logger: Optional[Union[ExperimentLogger, _logging.Logger]] = None,
        metric_logger: Optional[MetricLogger] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ):
        # Merge configs (simple shallow merge)
        self.config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}
        # Top-level training config may be nested
        self.training_cfg = self.config.get("training", DEFAULT_TRAINING_CONFIG["training"])
        self.callbacks_cfg = self.config.get("callbacks", DEFAULT_TRAINING_CONFIG["callbacks"])

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device handling
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.model.to(self.device)

        # Logger (accept ExperimentLogger or stdlib logger)
        # Keep the wrapper (if provided) for progress bars; but obtain stdlib logger for callbacks
        self.logger_wrapper: Optional[ExperimentLogger] = logger if isinstance(logger, ExperimentLogger) else None
        self.logger_std: _logging.Logger = _ensure_stdlib_logger(logger)

        # If no wrapper passed, create a minimal ExperimentLogger for progress_bar convenience
        if self.logger_wrapper is None:
            try:
                # Create a simple ExperimentLogger that writes to experiment_dir/logs
                self.logger_wrapper = ExperimentLogger(name=self.model.__class__.__name__, log_dir=self.experiment_dir / "logs")
            except Exception:
                # Fallback: None (we'll only use stdlib logger then)
                self.logger_wrapper = None

        # Metric logger (used for epoch-level metric persistence)
        self.metric_logger = metric_logger or MetricLogger(log_dir=self.experiment_dir / "logs")

        # System monitor / timing context (optional)
        self.system_monitor = SystemMonitor(logger=self.logger_std) if SystemMonitor is not None else None
        self.timing_context = TimingContext() if TimingContext is not None else None

        # Optimizer
        opt_cfg = self.config.get("optimizer", DEFAULT_TRAINING_CONFIG["optimizer"])
        opt_name = opt_cfg.get("name", "adam").lower()
        lr = float(opt_cfg.get("learning_rate", 1e-3))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = float(opt_cfg.get("momentum", 0.9))
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        self.logger_std.info(f"Optimizer: {opt_name} lr={lr} wd={weight_decay}")

        # Criterion with optional class weights for imbalanced data
        loss_cfg = self.config.get("loss", DEFAULT_TRAINING_CONFIG["loss"])
        loss_name = loss_cfg.get("name", "crossentropy").lower()
        class_weights = loss_cfg.get("class_weights", None)
        
        if loss_name in ("crossentropy", "cross_entropy", "cross_entropy_loss"):
            if class_weights is not None:
                if isinstance(class_weights, (list, tuple)):
                    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                elif class_weights == "auto":
                    # Will be computed from data later
                    weight_tensor = None
                    self._compute_class_weights = True
                else:
                    weight_tensor = None
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif loss_name in ("mse", "mse_loss", "mean_squared_error"):
            self.criterion = nn.MSELoss()
        elif loss_name in ("focal", "focal_loss", "focalloss"):
            # Focal Loss for handling class imbalance
            try:
                from .focal_loss import FocalLoss
                alpha = loss_cfg.get("alpha", 0.25)
                gamma = loss_cfg.get("gamma", 2.0)
                num_classes = self.config.get("num_classes", 2)
                self.criterion = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
                self.logger_std.info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
            except ImportError:
                self.logger_std.warning("Focal Loss not available, falling back to CrossEntropyLoss")
                self.criterion = nn.CrossEntropyLoss()
        elif loss_name in ("weighted_focal", "weighted_focal_loss"):
            # Weighted Focal Loss with automatic class weight computation
            try:
                from .focal_loss import WeightedFocalLoss
                class_counts = loss_cfg.get("class_counts", None)
                gamma = loss_cfg.get("gamma", 2.0)
                self.criterion = WeightedFocalLoss(class_counts=class_counts, gamma=gamma)
                self.logger_std.info(f"Using Weighted Focal Loss with gamma={gamma}")
            except ImportError:
                self.logger_std.warning("Weighted Focal Loss not available, falling back to CrossEntropyLoss")
                self.criterion = nn.CrossEntropyLoss()
        elif loss_name in ("class_balanced", "class_balanced_loss"):
            # Class-Balanced Loss based on effective number of samples
            try:
                from .focal_loss import ClassBalancedLoss
                class_counts = loss_cfg.get("class_counts", None)
                gamma = loss_cfg.get("gamma", 2.0)
                beta = loss_cfg.get("beta", 0.9999)
                if class_counts is None:
                    raise ValueError("class_counts required for class_balanced loss")
                self.criterion = ClassBalancedLoss(class_counts=class_counts, gamma=gamma, beta=beta)
                self.logger_std.info(f"Using Class-Balanced Loss with gamma={gamma}, beta={beta}")
            except ImportError:
                self.logger_std.warning("Class-Balanced Loss not available, falling back to CrossEntropyLoss")
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")
        self.logger_std.info(f"Criterion: {loss_name}")

        # Callbacks - create using stdlib logger passed in (callbacks expect logging.Logger)
        cb_cfg = self.callbacks_cfg or DEFAULT_TRAINING_CONFIG["callbacks"]

        # EarlyStopping
        self.early_stopping: Optional[EarlyStopping] = None
        es_cfg = cb_cfg.get("early_stopping", {})
        if es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=int(es_cfg.get("patience", 10)),
                min_delta=float(es_cfg.get("min_delta", 1e-4)),
                monitor=es_cfg.get("monitor", "val_loss"),
                mode=es_cfg.get("mode", "min"),
                logger=self.logger_std,
            )
            self.logger_std.info("EarlyStopping enabled")

        # ModelCheckpoint
        self.checkpoint: Optional[ModelCheckpoint] = None
        mc_cfg = cb_cfg.get("model_checkpoint", {})
        if mc_cfg.get("enabled", True):
            filename = mc_cfg.get("filename", "best_model_epoch{epoch}_metric{metric}.pth")
            self.checkpoint = ModelCheckpoint(
                save_dir=self.checkpoint_dir,
                monitor=mc_cfg.get("monitor", "val_loss"),
                mode=mc_cfg.get("mode", "min"),
                save_best_only=mc_cfg.get("save_best_only", True),
                save_weights_only=mc_cfg.get("save_weights_only", True),
                filename=filename,
                logger=self.logger_std,
            )
            self.save_last = bool(mc_cfg.get("save_last", True))
            self.logger_std.info(f"ModelCheckpoint enabled. save_last={self.save_last}")
        else:
            self.save_last = False

        # LR scheduler wrapper (optional)
        self.lr_scheduler_wrapper: Optional[LearningRateScheduler] = None
        lr_cfg = cb_cfg.get("lr_scheduler", {})
        if lr_cfg.get("enabled", False):
            name = lr_cfg.get("name", "ReduceLROnPlateau")
            pytorch_cls = getattr(torch.optim.lr_scheduler, name, None)
            if pytorch_cls is None:
                raise ValueError(f"Scheduler {name} not found in torch.optim.lr_scheduler")
            # Build kwargs for constructor
            kwargs = {"optimizer": self.optimizer}
            for key in ("mode", "factor", "patience", "step_size", "T_max"):
                if key in lr_cfg:
                    kwargs[key] = lr_cfg[key]
            # Instantiate in a best-effort manner (filter signature)
            try:
                pytorch_scheduler = pytorch_cls(**{k: v for k, v in kwargs.items() if k in pytorch_cls.__init__.__code__.co_varnames})
            except Exception:
                pytorch_scheduler = pytorch_cls(self.optimizer)
            self.lr_scheduler_wrapper = LearningRateScheduler(scheduler=pytorch_scheduler, monitor=lr_cfg.get("monitor"), logger=self.logger_std)
            self.logger_std.info(f"LR scheduler {name} wrapped")

        # Training hyperparams
        self.epochs = int(self.training_cfg.get("epochs", 20))
        self.grad_accum_steps = int(self.training_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = self.training_cfg.get("max_grad_norm", None)
        self.use_amp = bool(self.training_cfg.get("use_amp", False))
        self.global_step = 0
        self.start_epoch = 1

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        if self.use_amp and self.scaler is None:
            self.logger_std.warning("use_amp=True but CUDA not available; AMP disabled.")

        # Resume
        self.resume_from = resume_from
        if self.resume_from is not None:
            self._resume_from_checkpoint(self.resume_from)

        # History
        self.history: Dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

        # Friendly info
        self.logger_std.info(f"Trainer initialized on device {self.device}")

    # -------------------------
    # Checkpoint / resume
    # -------------------------
    def _resume_from_checkpoint(self, ckpt_path: Union[str, Path]) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        self.logger_std.info(f"Resuming training from checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_state = ckpt["model_state_dict"]
            self.model.load_state_dict(model_state)
            if "optimizer_state_dict" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    self.logger_std.warning(f"Failed to load optimizer state: {e}")
            epoch = ckpt.get("epoch")
            if epoch is not None:
                self.start_epoch = int(epoch) + 1
            if "scheduler_state_dict" in ckpt and self.lr_scheduler_wrapper is not None:
                try:
                    self.lr_scheduler_wrapper.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    self.logger_std.debug("Failed to restore scheduler state (incompatible state dict).")
            if "scaler_state_dict" in ckpt and self.scaler is not None:
                try:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    self.logger_std.debug("Failed to restore AMP scaler state.")
        else:
            # assume raw state_dict
            self.model.load_state_dict(ckpt)
        self.logger_std.info(f"Resumed; starting epoch = {self.start_epoch}")

    # -------------------------
    # Batch handling and forward
    # -------------------------
    def _forward_and_loss(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Robust batch parsing. Accepts (inputs, targets) tuples or dicts containing 'x'/'y' or 'inputs'/'labels'.
        Returns (logits, loss).
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            if "x" in batch and "y" in batch:
                inputs, targets = batch["x"], batch["y"]
            elif "inputs" in batch and "labels" in batch:
                inputs, targets = batch["inputs"], batch["labels"]
            else:
                keys = list(batch.keys())
                if len(keys) >= 2:
                    inputs, targets = batch[keys[0]], batch[keys[1]]
                else:
                    raise ValueError("Cannot parse batch dict for inputs/targets")
        else:
            raise ValueError("Unsupported batch type returned by DataLoader")

        # Move to device (try-except to handle numpy arrays)
        if torch.is_tensor(inputs):
            inputs = inputs.to(self.device)
        else:
            inputs = torch.tensor(inputs, device=self.device)

        if torch.is_tensor(targets):
            targets = targets.to(self.device)
        else:
            targets = torch.tensor(targets, device=self.device)

        # Forward + loss (support AMP)
        if self.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
        return logits, loss

    # -------------------------
    # Single epoch train/validate
    # -------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch with gradient accumulation support.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        # Prepare iterable via wrapper.progress_bar if present (narrow its type for static checkers)
        iter_loader: Iterable[Any] = self.train_loader
        try:
            pbar_fn = getattr(self.logger_wrapper, "progress_bar", None)
            if callable(pbar_fn):
                iter_loader = cast(Iterable[Any], pbar_fn(self.train_loader, desc=f"Epoch {epoch} [Train]", total=len(self.train_loader)))
        except Exception:
            iter_loader = self.train_loader  # fallback

        # Zero grads before epoch
        self.optimizer.zero_grad()

        iterator: Iterator[Any] = iter(iter_loader)
        for local_step, batch in enumerate(iterator, start=1):
            batch_count += 1
            self.global_step += 1

            # forward + loss
            logits, loss = self._forward_and_loss(batch)
            loss_to_backprop = loss / float(self.grad_accum_steps)

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            # step when accumulation boundary reached
            if (local_step % self.grad_accum_steps) == 0:
                if self.use_amp and self.scaler is not None:
                    # unscale before clipping
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception:
                        pass
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.max_grad_norm))

                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # metrics (best-effort)
            total_loss += float(loss.item())
            batch_count_for_acc = 0
            try:
                pred_labels = logits.argmax(dim=1)
                # extract targets
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    t = batch[1]
                elif isinstance(batch, dict) and "y" in batch:
                    t = batch["y"]
                elif isinstance(batch, dict) and "labels" in batch:
                    t = batch["labels"]
                else:
                    t = None
                if t is not None:
                    t = t.to(self.device) if torch.is_tensor(t) else torch.tensor(t, device=self.device)
                    correct += int((pred_labels == t).sum().item())
                    total += int(t.size(0))
                    batch_count_for_acc = int(t.size(0))
            except Exception:
                # ignore metric for this batch (best-effort)
                pass

            # update progress bar postfix if available
            try:
                # logger_wrapper's progress_bar may return tqdm-like object or the raw loader.
                pbar_obj = iter_loader  # the object returned; may or may not be tqdm instance
                if hasattr(pbar_obj, "set_postfix_str"):
                    cast(Any, pbar_obj).set_postfix_str(f"loss: {loss.item():.4f}")
            except Exception:
                pass

        avg_loss = total_loss / (batch_count if batch_count > 0 else 1)
        accuracy = (correct / total) if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Run validation epoch (no grads).
        Now also computes F1-macro for imbalanced data monitoring.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        all_preds = []
        all_targets = []

        iter_loader: Iterable[Any] = self.val_loader
        try:
            pbar_fn = getattr(self.logger_wrapper, "progress_bar", None)
            if callable(pbar_fn):
                iter_loader = cast(Iterable[Any], pbar_fn(self.val_loader, desc=f"Epoch {epoch} [Val]", total=len(self.val_loader)))
        except Exception:
            iter_loader = self.val_loader

        with torch.no_grad():
            iterator: Iterator[Any] = iter(iter_loader)
            for batch in iterator:
                batch_count += 1
                logits, loss = self._forward_and_loss(batch)
                total_loss += float(loss.item())

                # compute accuracy and collect predictions for F1
                try:
                    pred_labels = logits.argmax(dim=1)
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        t = batch[1]
                    elif isinstance(batch, dict) and "y" in batch:
                        t = batch["y"]
                    elif isinstance(batch, dict) and "labels" in batch:
                        t = batch["labels"]
                    else:
                        t = None
                    if t is not None:
                        t = t.to(self.device) if torch.is_tensor(t) else torch.tensor(t, device=self.device)
                        correct += int((pred_labels == t).sum().item())
                        total += int(t.size(0))
                        # Collect for F1 calculation
                        all_preds.extend(pred_labels.cpu().numpy().tolist())
                        all_targets.extend(t.cpu().numpy().tolist())
                except Exception:
                    pass

        avg_loss = total_loss / (batch_count if batch_count > 0 else 1)
        accuracy = (correct / total) if total > 0 else 0.0
        
        # Compute F1-macro for better monitoring of imbalanced data performance
        val_f1_macro = 0.0
        if len(all_preds) > 0 and len(all_targets) > 0:
            try:
                val_f1_macro = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))
            except Exception:
                val_f1_macro = 0.0
        
        return {"loss": avg_loss, "accuracy": accuracy, "f1_macro": val_f1_macro}

    # -------------------------
    # Training loop
    # -------------------------
    def fit(self, epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Run training for `epochs` epochs (overrides config if provided).
        Returns the history dict.
        """
        epochs = int(epochs) if epochs is not None else int(self.epochs)
        self.logger_std.info(f"Training {self.model.__class__.__name__} for {epochs} epochs on {self.device}")

        start_time = time.time()
        try:
            for epoch in range(self.start_epoch, epochs + 1):
                epoch_start_time = time.time()

                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)

                # LR scheduler step (wrapper handles Plateau vs epoch schedulers)
                if self.lr_scheduler_wrapper is not None:
                    try:
                        # Prefer passing the metric for ReduceLROnPlateau
                        self.lr_scheduler_wrapper(metrics={"val_loss": val_metrics["loss"]}, epoch=epoch)
                    except Exception:
                        # fallback to epoch stepping
                        self.lr_scheduler_wrapper(metrics=None, epoch=epoch)

                # history & metric logging
                lr = float(self.optimizer.param_groups[0]["lr"])
                val_f1 = val_metrics.get("f1_macro", 0.0)
                self.history["train_loss"].append(train_metrics["loss"])
                self.history["train_acc"].append(train_metrics["accuracy"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                self.history["lr"].append(lr)

                metrics_to_log = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_f1_macro": val_f1,
                    "lr": lr,
                    "epoch_time_s": time.time() - epoch_start_time
                }
                # MetricLogger handles epoch/step columns
                try:
                    self.metric_logger.log_metrics(metrics_to_log, step=self.global_step, epoch=epoch)
                except Exception:
                    self.logger_std.debug("MetricLogger.log_metrics failed (non-fatal).")

                # Console logging - now includes F1-macro
                self.logger_std.info(
                    f"Epoch {epoch}/{epochs} - train_loss: {train_metrics['loss']:.4f}, val_loss: {val_metrics['loss']:.4f}, "
                    f"train_acc: {train_metrics['accuracy']:.4f}, val_acc: {val_metrics['accuracy']:.4f}, val_f1: {val_f1:.4f}, lr: {lr:.6g}"
                )

                # Checkpointing - model checkpoint may return Path or other form; normalize
                if self.checkpoint:
                    saved_path = self.checkpoint(
                        model=self.model,
                        metrics={"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"], "val_f1_macro": val_f1},
                        epoch=epoch,
                        optimizer=self.optimizer
                    )
                    # Normalize saved_path to a Path if possible
                    sp: Optional[Path] = None
                    try:
                        if isinstance(saved_path, Path):
                            sp = saved_path
                        elif isinstance(saved_path, (list, tuple)) and len(saved_path) > 0:
                            sp = saved_path[0] if isinstance(saved_path[0], Path) else Path(str(saved_path[0]))
                        elif saved_path is not None:
                            sp = Path(str(saved_path))
                    except Exception:
                        sp = None

                    if self.early_stopping and sp is not None:
                        try:
                            self.early_stopping.set_best_checkpoint(sp)
                        except Exception as e:
                            self.logger_std.debug(f"Could not set best checkpoint on EarlyStopping: {e}")

                # Save last checkpoint (atomic) if requested
                if getattr(self, "save_last", False):
                    last_path = self.checkpoint_dir / f"last_epoch_{epoch}.pth"
                    try:
                        payload = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "metrics": {"train": train_metrics, "val": val_metrics}
                        }
                        atomic_save_state(payload, last_path)
                        self.logger_std.debug(f"Saved last checkpoint to {last_path}")
                    except Exception as e:
                        self.logger_std.warning(f"Failed to save last checkpoint: {e}")

                # Early stopping check - now includes F1 metric
                if self.early_stopping:
                    stop = self.early_stopping(metrics={"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"], "val_f1_macro": val_f1}, epoch=epoch)
                    if stop:
                        self.logger_std.info(f"EarlyStopping triggered at epoch {epoch}")
                        break

        except KeyboardInterrupt:
            self.logger_std.warning("Training interrupted by user (KeyboardInterrupt). Saving final model...")
        finally:
            # finalize
            total_time = time.time() - start_time
            self.logger_std.info(f"Training finished in {total_time/60:.2f} minutes")
            # Save metrics JSON/CSV via MetricLogger
            try:
                self.metric_logger.save_metrics()
                self.logger_std.info("Saved metric logs.")
            except Exception:
                self.logger_std.debug("Failed to save metric logs (non-fatal).")

            # Optionally restore best checkpoint into model if EarlyStopping requested
            if self.early_stopping and getattr(self.early_stopping, "restore_best", False) and getattr(self.early_stopping, "best_checkpoint_path", None):
                try:
                    self.early_stopping.restore_best_checkpoint(self.model)
                    self.logger_std.info("Restored best checkpoint after training.")
                except Exception as e:
                    self.logger_std.warning(f"Failed to restore best checkpoint: {e}")

            # Save final model (weights-only)
            final_path = self.checkpoint_dir / "final_model.pth"
            try:
                atomic_save_state({"model_state_dict": self.model.state_dict()}, final_path)
                self.logger_std.info(f"Saved final model to {final_path}")
            except Exception as e:
                self.logger_std.warning(f"Failed to save final model: {e}")

        return self.history

    # -------------------------
    # Utilities
    # -------------------------
    def evaluate_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        """
        Convenience evaluation on an arbitrary loader (returns loss & accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        with torch.no_grad():
            for batch in loader:
                logits, loss = self._forward_and_loss(batch)
                total_loss += float(loss.item())
                pred = logits.argmax(dim=1)

                # robust target extraction
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    t = batch[1]
                elif isinstance(batch, dict) and "y" in batch:
                    t = batch["y"]
                elif isinstance(batch, dict) and "labels" in batch:
                    t = batch["labels"]
                else:
                    t = None

                if t is not None:
                    t = t.to(self.device) if torch.is_tensor(t) else torch.tensor(t, device=self.device)
                    correct += int((pred == t).sum().item())
                    total += int(t.size(0))
                batch_count += 1
        avg_loss = total_loss / (batch_count if batch_count > 0 else 1)
        acc = (correct / total) if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": acc}

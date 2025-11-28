"""
Research-grade training callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler.

Improvements & design notes (Objective Logic):
- Deterministic comparator for 'min'/'max' modes
- Atomic checkpoint saving (temp file + rename) to avoid corrupted saves
- Optionally save full checkpoint (model + optimizer + epoch + metrics) or weights-only
- EarlyStopping stores/restores state and can optionally restore best checkpoint
- Callbacks expose .state_dict() / .load_state_dict() for reproducibility
- LearningRateScheduler wrapper supports both metric-driven schedulers (ReduceLROnPlateau)
  and epoch-based schedulers; logs LR after stepping
- Detailed logging (no dependency on ExperimentLogger — any logger works)
- Type hints and docstrings for clarity
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import logging

# Module logger (can be replaced by ExperimentLogger if desired)
_logger = logging.getLogger(__name__)


# ---------------------------
# EarlyStopping
# ---------------------------
class EarlyStopping:
    """
    Early stopping callback with optional checkpoint restoration support.

    Usage pattern:
        earlystop = EarlyStopping(patience=10, monitor="val_loss", mode="min")
        for epoch in range(...):
            ...
            stop = earlystop(metrics, epoch)
            if earlystop.is_improvement:
                ckpt_path = checkpoint(model, metrics, epoch, optimizer)
                earlystop.set_best_checkpoint(ckpt_path)
            if stop:
                if earlystop.restore_best and earlystop.best_checkpoint_path:
                    earlystop.restore_best_checkpoint(model)
                break

    Methods:
    - __call__(metrics, epoch): update state and return boolean (should stop)
    - is_improvement property: whether the last call was an improvement
    - set_best_checkpoint(path): record path to best checkpoint
    - restore_best_checkpoint(model, map_location): load best checkpoint into model
    - state_dict/load_state_dict: for reproducibility
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor: str = "val_loss",
        mode: str = "min",
        restore_best: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor)
        self.mode = mode
        self.logger = logger or _logger
        self.restore_best = bool(restore_best)

        # internal state
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_epoch: Optional[int] = None
        self.best_checkpoint_path: Optional[Path] = None

        # comparator: returns True if current is better than best
        if mode == "min":
            self._is_better = lambda current, best: current < best - self.min_delta
            self._init_best = float("inf")
        else:
            self._is_better = lambda current, best: current > best + self.min_delta
            self._init_best = float("-inf")

        # transient flag for external use (e.g., to decide checkpoint saving)
        self.is_improvement: bool = False

    def __call__(self, metrics: Dict[str, float], epoch: Optional[int] = None) -> bool:
        """
        Update internal early stopping state using the provided metrics.

        Returns True if training should stop (i.e., early stop triggered).
        Also sets .is_improvement to True when current metric improves best_score.
        """
        if self.monitor not in metrics:
            self.logger.warning(f"EarlyStopping: monitor '{self.monitor}' not present in metrics.")
            self.is_improvement = False
            return False

        current = float(metrics[self.monitor])

        # initial best
        if self.best_score is None:
            self.best_score = current
            self.counter = 0
            self.best_epoch = int(epoch) if epoch is not None else None
            self.is_improvement = True
            self.logger.debug(f"EarlyStopping: initial best {self.monitor} = {self.best_score:.6f}")
            return False

        if self._is_better(current, self.best_score):
            self.logger.debug(f"EarlyStopping: improvement {self.best_score:.6f} -> {current:.6f}")
            self.best_score = current
            self.counter = 0
            self.best_epoch = int(epoch) if epoch is not None else self.best_epoch
            self.is_improvement = True
            return False
        else:
            self.counter += 1
            self.is_improvement = False
            self.logger.debug(f"EarlyStopping: no improvement. counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"EarlyStopping: triggered after {self.counter} epochs without improvement.")
        return self.early_stop

    def set_best_checkpoint(self, path: Path) -> None:
        """Record path to the best checkpoint (so the early-stopper can restore later)."""
        self.best_checkpoint_path = Path(path)
        self.logger.debug(f"EarlyStopping: best checkpoint set to {self.best_checkpoint_path}")

    def restore_best_checkpoint(self, model: torch.nn.Module, map_location: Optional[str] = None) -> None:
        """
        Load the best checkpoint into model.state_dict().

        Behaviors:
        - Accepts both "weights-only" state_dict and full checkpoint dict with keys
          'model_state_dict' and optionally 'optimizer_state_dict', 'epoch', 'metrics'.
        - Uses provided map_location or model.device if present; falls back to CPU.
        """
        if not self.best_checkpoint_path:
            raise RuntimeError("No best checkpoint path recorded. Call set_best_checkpoint(path) first.")
        path = str(self.best_checkpoint_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Best checkpoint not found: {path}")

        self.logger.info(f"EarlyStopping: restoring best checkpoint from {path}")

        # Determine effective map_location
        if map_location is not None:
            effective_map_location = map_location
        elif hasattr(model, "device"):
            dev = getattr(model, "device")
            effective_map_location = dev if isinstance(dev, torch.device) else torch.device("cpu")
        else:
            effective_map_location = torch.device("cpu")

        ckpt = torch.load(path, map_location=effective_map_location)

        # Support full checkpoint dictionary or raw state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and any(k in model.state_dict() for k in ckpt.keys()):
            # older style: top-level keys are model params
            state = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        else:
            state = ckpt  # assume raw state_dict

        model.load_state_dict(state)
        model.to(getattr(model, "device", torch.device("cpu")))
        self.logger.info("EarlyStopping: model restored to best checkpoint.")

    # reproducible saving/loading of callback state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "mode": self.mode,
            "restore_best": self.restore_best,
            "best_score": self.best_score,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "best_epoch": self.best_epoch,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.monitor = str(state.get("monitor", self.monitor))
        self.mode = str(state.get("mode", self.mode))
        self.restore_best = bool(state.get("restore_best", self.restore_best))
        self.best_score = state.get("best_score", self.best_score)
        self.counter = int(state.get("counter", self.counter))
        self.early_stop = bool(state.get("early_stop", self.early_stop))
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        bpath = state.get("best_checkpoint_path", None)
        self.best_checkpoint_path = Path(bpath) if bpath else None


# ---------------------------
# ModelCheckpoint
# ---------------------------
class ModelCheckpoint:
    """
    Save model checkpoints in a robust, atomic way.

    Features:
    - Atomic save (save to tmp file then rename)
    - Option to save only weights or full checkpoint (weights + optimizer + epoch + metrics)
    - Optionally include optimizer state_dict if provided to __call__
    - Optionally save the last checkpoint at every call (save_last=True)
    - Returns path to saved best checkpoint or None if not saved
    """

    def __init__(
        self,
        save_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = True,
        filename: str = "best_model.pth",
        save_last: bool = True,
        last_filename: str = "last_model.pth",
        logger: Optional[logging.Logger] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.save_best_only = bool(save_best_only)
        self.save_weights_only = bool(save_weights_only)
        self.filename = str(filename)
        self.save_last = bool(save_last)
        self.last_filename = str(last_filename)
        self.logger = logger or _logger

        self.best_score: Optional[float] = None
        if self.mode == "min":
            self._is_better = lambda cur, best: cur < best
            self._init_best = float("inf")
        else:
            self._is_better = lambda cur, best: cur > best
            self._init_best = float("-inf")

    def _atomic_save(self, payload: Any, dest: Path) -> None:
        dest = Path(dest)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.save_dir))
        os.close(tmp_fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, str(dest))
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def __call__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Possibly save best and/or last checkpoint based on metrics.

        Returns:
            Tuple[best_path, last_path] - Paths to saved checkpoints (or None if not saved)
        """
        raw_metric = metrics.get(self.monitor, float("nan"))
        try:
            current = float(raw_metric)
        except (TypeError, ValueError):
            current = float("nan")
        save_best = False

        if not self.save_best_only:
            save_best = True
        elif self.best_score is None:
            save_best = True
        elif self._is_better(current, self.best_score):
            save_best = True

        best_path: Optional[Path] = None
        last_path: Optional[Path] = None

        # Prepare payload
        if self.save_weights_only:
            payload = model.state_dict()
        else:
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }
            if optimizer is not None:
                payload["optimizer_state_dict"] = optimizer.state_dict()

        # Save best checkpoint if appropriate
        if save_best:
            # Handle NaN by replacing with 0.0 for filename formatting
            metric_for_filename = current if not (current != current) else 0.0  # NaN check
            try:
                fname = self.filename.format(epoch=epoch if epoch is not None else "NA", metric=metric_for_filename)
            except (ValueError, KeyError):
                # Fallback if format string doesn't work
                fname = f"best_model_epoch{epoch}.pth"
            dest = self.save_dir / fname
            try:
                self._atomic_save(payload, dest)
                best_path = dest
                self.logger.info(f"ModelCheckpoint: saved best checkpoint to {dest}")
                self.best_score = current
            except Exception as e:
                self.logger.error(f"ModelCheckpoint: failed to save best checkpoint to {dest}: {e}")

        # Save last checkpoint if enabled
        if self.save_last:
            dest_last = self.save_dir / self.last_filename
            try:
                self._atomic_save(payload, dest_last)
                last_path = dest_last
                self.logger.debug(f"ModelCheckpoint: saved last checkpoint to {dest_last}")
            except Exception as e:
                self.logger.error(f"ModelCheckpoint: failed to save last checkpoint to {dest_last}: {e}")

        return best_path, last_path

    def state_dict(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "save_best_only": self.save_best_only,
            "save_weights_only": self.save_weights_only,
            "filename": self.filename,
            "save_last": self.save_last,
            "last_filename": self.last_filename,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.monitor = state.get("monitor", self.monitor)
        self.mode = state.get("mode", self.mode)
        self.save_best_only = state.get("save_best_only", self.save_best_only)
        self.save_weights_only = state.get("save_weights_only", self.save_weights_only)
        self.filename = state.get("filename", self.filename)
        self.save_last = state.get("save_last", self.save_last)
        self.last_filename = state.get("last_filename", self.last_filename)
        self.best_score = state.get("best_score", self.best_score)

# ---------------------------
# LearningRateScheduler wrapper
# ---------------------------
class LearningRateScheduler:
    """
    Lightweight wrapper for PyTorch LR schedulers to integrate into training loops.

    - If scheduler is ReduceLROnPlateau, call with metrics (and configure monitor param).
    - Otherwise, call step() per epoch or per batch as appropriate externally.
    - Exposes last_lr for quick access.
    """

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, monitor: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.scheduler = scheduler
        self.monitor = monitor  # e.g., 'val_loss' for ReduceLROnPlateau
        self.logger = logger or _logger
        self.last_lr: Optional[float] = None

    def __call__(self, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None) -> None:
        """
        Step the scheduler. If monitor is set, a metrics dict must be provided.
        For ReduceLROnPlateau, pass the metric value; for other schedulers use epoch or just step().

        Notes:
        - ReduceLROnPlateau.step() accepts a float metric (not epoch). Static type checkers may complain;
          we call it directly and silence type warnings where necessary.
        """
        is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        if is_plateau:
            if self.monitor is None:
                self.logger.warning("LearningRateScheduler: ReduceLROnPlateau used but monitor not set; skipping step.")
                return
            if metrics is None or self.monitor not in metrics:
                self.logger.warning(f"LearningRateScheduler: monitor '{self.monitor}' required but not in metrics.")
                return
            val = float(metrics[self.monitor])
            # For ReduceLROnPlateau, the PyTorch API expects the monitored metric (float).
            # Static type checkers think this is 'epoch' -> ignore the call-arg typing.
            self.scheduler.step(val)  # type: ignore[call-arg]
            self.logger.debug(f"LearningRateScheduler: ReduceLROnPlateau stepped with {self.monitor}={val:.6f}")
        else:
            # For other schedulers, prefer stepping by epoch (int), else step() without args.
            try:
                if epoch is not None:
                    self.scheduler.step(int(epoch))
                else:
                    self.scheduler.step()
                self.logger.debug("LearningRateScheduler: stepped scheduler (non-Plateau).")
            except TypeError:
                # Fallback: call without epoch
                self.scheduler.step()
                self.logger.debug("LearningRateScheduler: stepped scheduler (fallback).")

        # Expose current LR for monitoring
        try:
            # prefer scheduler.get_last_lr() if available
            last_lrs = getattr(self.scheduler, "get_last_lr", None)
            if callable(last_lrs):
                lrs = self.scheduler.get_last_lr()
                self.last_lr = float(lrs[0]) if lrs else None
            else:
                # fallback to optimizer structure
                self.last_lr = float(self.scheduler.optimizer.param_groups[0]["lr"])
            self.logger.debug(f"LearningRateScheduler: current LR = {self.last_lr:.8e}")
        except Exception:
            self.last_lr = None

    def state_dict(self) -> Dict[str, Any]:
        # Many torch schedulers are not fully serializable without optimizer,
        # so we return minimal metadata. Full checkpointing should include optimizer + scheduler state.
        return {
            "scheduler_class": self.scheduler.__class__.__name__,
            "monitor": self.monitor,
            "last_lr": self.last_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.monitor = state.get("monitor", self.monitor)
        self.last_lr = state.get("last_lr", getattr(self, "last_lr", None))


# ---------------------------
# Convenience helpers
# ---------------------------
def atomic_save_state(obj: Any, path: Path) -> None:
    """
    Helper to atomically save a python object using torch.save.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, str(path))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

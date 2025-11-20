"""
Research-grade training callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler.

Design goals / improvements (OL):
- Deterministic comparator for 'min'/'max' modes
- Atomic checkpoint saving (temp file + rename) to avoid corrupted saves
- Optionally save full checkpoint (model + optimizer + epoch + metrics) or weights-only
- EarlyStopping stores/restores state + can optionally restore best checkpoint
- Callbacks expose .state_dict() / .load_state_dict() for experiment reproducibility
- LearningRateScheduler wrapper supports both metric-driven schedulers (ReduceLROnPlateau)
  and epoch-based schedulers; logs LR after stepping
- Detailed logging (no dependency on ExperimentLogger — any logger works)
- Type hints and docstrings for clarity
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import logging

# Module logger (can be replaced by ExperimentLogger if desired)
_logger = logging.getLogger(__name__)


# ---------------------------
# EarlyStopping
# ---------------------------
class EarlyStopping:
    """
    Early stopping callback with checkpoint restoration support.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    monitor : str
        Metric name to monitor, e.g., 'val_loss' or 'val_acc'.
    mode : {'min', 'max'}
        Whether lower or higher values are better.
    restore_best : bool
        If True and a checkpoint_path is set (see `set_checkpoint_path`), `restore_best_checkpoint`
        can load best weights into the model after stop.
    logger : Optional[logging.Logger]
        Logger instance. If None, module logger is used.
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

        # initialize
        self.best_score = None

    def __call__(self, metrics: Dict[str, float], epoch: Optional[int] = None) -> bool:
        """
        Call once per epoch with current metrics.

        Returns True if training should be stopped (i.e., early_stop == True).
        """
        if self.monitor not in metrics:
            self.logger.warning(f"EarlyStopping: monitor '{self.monitor}' not present in metrics.")
            return False

        current = float(metrics[self.monitor])

        if self.best_score is None:
            self.best_score = current
            self.counter = 0
            if epoch is not None:
                self.best_epoch = int(epoch)
            self.logger.debug(f"EarlyStopping: initial best {self.monitor} = {self.best_score:.6f}")
            return False

        if self._is_better(current, self.best_score):
            self.logger.debug(f"EarlyStopping: improvement {self.best_score:.6f} -> {current:.6f}")
            self.best_score = current
            self.counter = 0
            if epoch is not None:
                self.best_epoch = int(epoch)
            return False
        else:
            self.counter += 1
            self.logger.debug(f"EarlyStopping: no improvement. counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"EarlyStopping: triggered after {self.counter} epochs without improvement.")
        return self.early_stop

    def set_best_checkpoint(self, path: Path) -> None:
        """
        Record path to the best checkpoint (so the early-stopper can restore later).
        """
        self.best_checkpoint_path = Path(path)

    def restore_best_checkpoint(self, model: torch.nn.Module, map_location: Optional[str] = None) -> None:
        """
        Load the best checkpoint into model.state_dict().

        If the checkpoint is a full checkpoint (dict with 'model_state_dict'), it loads that;
        otherwise it attempts to load the file as a raw state_dict.
        """
        if not self.best_checkpoint_path:
            raise RuntimeError("No best checkpoint path recorded. Call set_best_checkpoint(path) first.")

        path = str(self.best_checkpoint_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Best checkpoint not found: {path}")

        self.logger.info(f"EarlyStopping: restoring best checkpoint from {path}")
        # map_location can be: None, a device string, or torch.device; avoid passing arbitrary attributes
        if map_location is not None:
            effective_map_location = map_location
        elif hasattr(model, "device"):
            dev = getattr(model, "device")
            # torch.device is acceptable; fall back to CPU if it's something else
            effective_map_location = dev if isinstance(dev, torch.device) else torch.device("cpu")
        else:
            effective_map_location = None

        ckpt = torch.load(path, map_location=effective_map_location)
        # Support both raw state_dict and full checkpoint dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # newer style: saved top-level dict with 'model_state_dict' key
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and any(k in model.state_dict() for k in ckpt.keys()):
            # older style: saved top-level dict but contains model_state_dict keys directly
            state = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        else:
            # raw state_dict
            state = ckpt
        model.load_state_dict(state)
        # Ensure model is on the correct device after loading
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

    Usage:
        ckpt = ModelCheckpoint(save_dir="checkpoints", save_weights_only=True, monitor="val_loss", mode="min")
        ckpt(model, metrics=metrics, epoch=epoch, optimizer=optimizer)  # called each epoch

    Features:
    - Atomic save (save to tmp file then rename)
    - Option to save only weights or full checkpoint (weights + optimizer + epoch + metrics)
    - Optionally include optimizer state_dict if provided to __call__
    - Returns path to saved file (or None if not saved)
    """

    def __init__(
        self,
        save_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = True,
        filename: str = "best_model.pth",
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
        self.logger = logger or _logger

        self.best_score: Optional[float] = None
        # comparator
        if self.mode == "min":
            self._is_better = lambda cur, best: cur < best
            self._init_best = float("inf")
        else:
            self._is_better = lambda cur, best: cur > best
            self._init_best = float("-inf")

    def _atomic_save(self, payload: Any, dest: Path) -> None:
        """
        Atomically save payload to dest using a temp file and os.replace.
        """
        dest = Path(dest)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.save_dir))
        os.close(tmp_fd)
        try:
            torch.save(payload, tmp_path)
            # os.replace is atomic on POSIX
            os.replace(tmp_path, str(dest))
        except Exception:
            # cleanup on error
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
    ) -> Optional[Path]:
        """
        Possibly save checkpoint based on metrics.

        Returns:
            Path to saved checkpoint or None.
        """
        if self.monitor not in metrics:
            self.logger.warning(f"ModelCheckpoint: monitor '{self.monitor}' not found in metrics.")
            return None

        current = float(metrics[self.monitor])
        save_this_time = False

        if not self.save_best_only:
            save_this_time = True
        elif self.best_score is None:
            save_this_time = True
        elif self._is_better(current, self.best_score):
            save_this_time = True

        saved_path: Optional[Path] = None
        if save_this_time:
            # Build destination filename (allow epoch interpolation)
            fname = self.filename.format(epoch=epoch if epoch is not None else "NA", metric=f"{current:.6f}")
            dest = self.save_dir / fname

            # Build payload
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

            # Atomic save
            try:
                self._atomic_save(payload, dest)
                saved_path = dest
                self.logger.info(f"ModelCheckpoint: saved checkpoint to {dest}")
            except Exception as e:
                self.logger.error(f"ModelCheckpoint: failed to save checkpoint to {dest}: {e}")
                saved_path = None

            # update best_score if using save_best_only semantics
            if self.best_score is None or self._is_better(current, self.best_score):
                self.logger.debug(f"ModelCheckpoint: updating best score {self.best_score} -> {current}")
                self.best_score = current

        return saved_path

    def state_dict(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "save_best_only": self.save_best_only,
            "save_weights_only": self.save_weights_only,
            "filename": self.filename,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.monitor = state.get("monitor", self.monitor)
        self.mode = state.get("mode", self.mode)
        self.save_best_only = state.get("save_best_only", self.save_best_only)
        self.save_weights_only = state.get("save_weights_only", self.save_weights_only)
        self.filename = state.get("filename", self.filename)
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

    def __call__(self, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None) -> None:
        """
        Step the scheduler. If monitor is set, a metrics dict must be provided.
        For ReduceLROnPlateau, pass the metric value; for other schedulers use epoch or just step().

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
            # PyTorch API expects the monitored metric for ReduceLROnPlateau,
            # type checker thinks this is 'epoch', so we ignore type here
            self.scheduler.step(metrics=val)  # type: ignore[call-arg]
            self.logger.debug(f"LearningRateScheduler: ReduceLROnPlateau stepped with {self.monitor}={val:.6f}")
        else:
            # For other schedulers, pass epoch if supported, else just step()
            # Some schedulers accept epoch (CosineAnnealingWarmRestarts), others don't.
            try:
                if epoch is not None:
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step()
                self.logger.debug("LearningRateScheduler: stepped scheduler (non-Plateau).")
            except TypeError:
                # Fallback: call without epoch
                self.scheduler.step()
                self.logger.debug("LearningRateScheduler: stepped scheduler (fallback).")

        # Expose current LR for monitoring
        try:
            self.last_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            self.logger.debug(f"LearningRateScheduler: current LR = {self.last_lr:.8e}")
        except Exception:
            self.last_lr = None


    def state_dict(self) -> Dict[str, Any]:
        # Many torch schedulers are not serializable via state_dict without the optimizer,
        # so we only return minimal metadata here. Full checkpointing should include optimizer + scheduler state.
        return {
            "scheduler_class": self.scheduler.__class__.__name__,
            "monitor": self.monitor,
            "last_lr": getattr(self, "last_lr", None),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # Nothing to restore without the actual scheduler object/optimizer; this is a placeholder.
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
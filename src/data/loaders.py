"""
Research-grade PyTorch data loaders for IDS datasets.

Features:
- Safe memmap loading in multi-worker settings (memmaps reopened per worker)
- Deterministic worker seeding for reproducibility
- Automatic DistributedSampler support
- Optional transforms and dict-based output (return_dict)
- Default collate for variable-length sequences (compatible with dict- or tuple-samples)
- Conservative, portable defaults (pin_memory=False, persistent_workers=False)
- Clear, actionable errors for common edge-cases (empty dataset, non-numeric features/labels)
- Detailed logging
"""
from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, TypedDict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, get_worker_info

# Module logger
logger = logging.getLogger(__name__)

# Type aliases
PathOrStr = Union[Path, str]
TransformFn = Optional[Callable[[Any], Any]]


class DataLoaderKwargs(TypedDict, total=False):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    collate_fn: Optional[Callable]
    worker_init_fn: Callable
    prefetch_factor: int


# ---------------------------
# Worker initialization
# ---------------------------
from typing import Any
from torch.utils.data import get_worker_info
# ... other imports remain ...

def _default_worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """
    Worker initialization:
      - deterministic numpy / torch seeding
      - re-open memmap files in the worker process if dataset supports X_path/y_path

    This avoids memmap descriptor sharing issues that can occur when forking loaders
    with num_workers > 0 on some platforms.
    NOTE: Uses getattr/setattr to avoid static type-checker (Pylance) warnings about
    unknown Dataset attributes.
    """
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + worker_id)

    worker_info = get_worker_info()
    if worker_info is None:
        # Single-process data loading.
        logger.debug(f"Worker {worker_id}: single-process mode, seeded {seed + worker_id}")
        return

    dataset: Any = worker_info.dataset  # type: ignore[assignment]
    try:
        # Use getattr to safely probe attributes (avoids Pylance errors).
        mode = getattr(dataset, "mode", None)
        x_path = getattr(dataset, "X_path", None)
        y_path = getattr(dataset, "y_path", None)

        # Only attempt memmap reopen if dataset indicates memmap mode and paths exist
        if mode == "memmap" and x_path is not None and y_path is not None:
            # Convert PathLike to string if necessary
            try:
                x_path_str = str(x_path)
                y_path_str = str(y_path)
            except Exception:
                x_path_str = x_path
                y_path_str = y_path

            # Re-open memmaps inside worker process
            X_mm = np.load(x_path_str, mmap_mode="r")
            y_mm = np.load(y_path_str, mmap_mode="r")

            # Use setattr (avoids static attribute access warnings)
            setattr(dataset, "X", X_mm)
            setattr(dataset, "y", y_mm)

            logger.debug(
                f"Worker {worker_id}: reopened memmaps for dataset {getattr(dataset, 'data_path', 'unknown')} "
                f"(X_path={x_path_str})"
            )
        else:
            logger.debug(f"Worker {worker_id}: dataset not memmap or missing paths (mode={mode}, X_path={x_path}, y_path={y_path})")
    except Exception as e:
        # Never let worker init silently crash; log the issue so training doesn't hang mysteriously.
        logger.warning(f"Worker {worker_id}: failed to re-open memmaps (non-fatal): {e}")


# ---------------------------
# Collate function
# ---------------------------
def default_collate_variable_length(batch: List[Any]) -> Dict[str, torch.Tensor]:
    """
    Collate function that supports both:
      - dataset returning tuples (x, y)
      - dataset returning dicts {"x": x, "y": y}

    Pads variable-length feature tensors to the maximum length in the batch (zero padding).
    Expects features to be torch.Tensor (IDSDataset returns torch tensors).
    """
    if len(batch) == 0:
        raise ValueError("default_collate_variable_length: received empty batch")

    # Normalize batch items to (x, y)
    first = batch[0]
    if isinstance(first, dict):
        xs = [b["x"] for b in batch]
        ys = [b["y"] for b in batch]
    else:
        xs, ys = zip(*batch)

    # Ensure xs are tensors
    if not all(isinstance(x, torch.Tensor) for x in xs):
        # convert numpy arrays to tensors if necessary
        xs = [torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in xs]

    # Determine if padding is needed (assume seq length is dim 0)
    seq_lens = [x.shape[0] if x.ndim >= 1 else 1 for x in xs]
    max_len = max(seq_lens)

    # If all equal lengths, just stack
    if all(l == max_len for l in seq_lens):
        x_batch = torch.stack([x.to(dtype=torch.float32) for x in xs])
    else:
        # Pad each tensor along dim 0 to max_len
        padded = []
        for x in xs:
            if x.ndim == 0:
                # scalar feature -> treat as length-1 vector
                x = x.unsqueeze(0)
            if x.shape[0] < max_len:
                pad_shape = (max_len - x.shape[0],) + tuple(x.shape[1:])
                pad_tensor = x.new_zeros(pad_shape).to(dtype=x.dtype)
                x_padded = torch.cat([x, pad_tensor], dim=0)
            else:
                x_padded = x
            padded.append(x_padded.to(dtype=torch.float32))
        x_batch = torch.stack(padded)

    # Stack labels (ys may be tensors of varying shapes)
    if not all(isinstance(y, torch.Tensor) for y in ys):
        ys = [torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y for y in ys]

    # Convert integer-like label tensors to long, floats to float32
    # If ys are scalar tensors, stacking produces shape (batch,)
    sample_y = ys[0]
    if sample_y.ndim == 0:
        # scalar labels
        if torch.is_floating_point(sample_y):
            y_batch = torch.stack([y.to(dtype=torch.float32) for y in ys])
        else:
            y_batch = torch.stack([y.to(dtype=torch.long) for y in ys])
    else:
        # vector/multi-label: cast appropriately
        if torch.is_floating_point(sample_y):
            y_batch = torch.stack([y.to(dtype=torch.float32) for y in ys])
        else:
            y_batch = torch.stack([y.to(dtype=torch.long) for y in ys])

    return {"x": x_batch, "y": y_batch}


# ---------------------------
# Dataset class
# ---------------------------
class IDSDataset(Dataset):
    """
    PyTorch dataset reading preprocessed numpy arrays (X.npy, y.npy).

    Parameters
    ----------
    data_path : Path | str
        Directory containing X.npy and y.npy.
    mode : {"memmap", "memory"}
        'memmap' uses np.load(..., mmap_mode="r") (safe for large datasets).
        'memory' loads into RAM (faster but requires memory).
    transform, target_transform : optional callables applied to X and y respectively.
    return_dict : if True, __getitem__ returns {"x": x_tensor, "y": y_tensor}, else (x, y).
    """

    def __init__(
        self,
        data_path: PathOrStr,
        mode: str = "memmap",
        transform: TransformFn = None,
        target_transform: TransformFn = None,
        return_dict: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.return_dict = bool(return_dict)

        self.X_path = self.data_path / "X.npy"
        self.y_path = self.data_path / "y.npy"

        if not self.X_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"Required files not found in {self.data_path}: X.npy and y.npy")

        # Load arrays (memmap by default)
        if self.mode == "memmap":
            self.X = np.load(str(self.X_path), mmap_mode="r")
            self.y = np.load(str(self.y_path), mmap_mode="r")
        elif self.mode == "memory":
            self.X = np.load(str(self.X_path))
            self.y = np.load(str(self.y_path))
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'memmap' or 'memory'.")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y sample count mismatch: {self.X.shape[0]} vs {self.y.shape[0]}")

        self.n_samples = int(self.X.shape[0])
        if self.n_samples == 0:
            raise ValueError(f"Dataset at {self.data_path} is empty (0 samples).")

        logger.info(f"IDSDataset loaded: {self.data_path} mode={self.mode} n_samples={self.n_samples}")

    def __len__(self) -> int:
        return self.n_samples

    @staticmethod
    def _to_tensor_x(x: Any) -> torch.Tensor:
        """
        Convert features to torch.FloatTensor.
        Raises if non-numeric/object/string dtype is encountered.
        """
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float32)

        arr = np.asarray(x)
        if arr.dtype == object or arr.dtype.kind in {"U", "S"}:
            raise TypeError(
                "_to_tensor_x: found non-numeric feature dtype (object/string). "
                "Ensure FeatureEngineer outputs numeric features."
            )
        try:
            return torch.from_numpy(arr).to(dtype=torch.float32)
        except Exception as e:
            raise TypeError(f"_to_tensor_x: cannot convert input to tensor: {e}")

    @staticmethod
    def _to_tensor_y(y: Any) -> torch.Tensor:
        """
        Convert label to an appropriate tensor:
          - integer scalar -> torch.long scalar
          - float scalar -> torch.float32 scalar
          - numpy array -> torch tensor (long if integer dtype, else float32)
        """
        if isinstance(y, torch.Tensor):
            # preserve scalar vs vector shape but ensure dtype is appropriate
            if y.ndim == 0:
                if torch.is_floating_point(y):
                    return y.to(dtype=torch.float32)
                return y.to(dtype=torch.long)
            else:
                if torch.is_floating_point(y):
                    return y.to(dtype=torch.float32)
                return y.to(dtype=torch.long)

        if isinstance(y, numbers.Integral):
            return torch.tensor(int(y), dtype=torch.long)
        if isinstance(y, numbers.Real):
            return torch.tensor(float(y), dtype=torch.float32)

        arr = np.asarray(y)
        if arr.dtype == object or arr.dtype.kind in {"U", "S"}:
            # string labels should be encoded before saving to numpy arrays
            raise TypeError(
                "_to_tensor_y: found string/object dtype for labels. "
                "Run label encoding (map labels to integers) before using the dataset."
            )

        if np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr).to(dtype=torch.long)
        return torch.from_numpy(arr).to(dtype=torch.float32)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        # support negative indices
        if idx < 0:
            idx = self.n_samples + idx
        if not (0 <= idx < self.n_samples):
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_samples} samples")

        X_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.transform:
            X_sample = self.transform(X_sample)
        if self.target_transform:
            y_sample = self.target_transform(y_sample)

        X_tensor = self._to_tensor_x(X_sample)
        y_tensor = self._to_tensor_y(y_sample)

        if self.return_dict:
            return {"x": X_tensor, "y": y_tensor}
        return X_tensor, y_tensor


# ---------------------------
# DataLoader factory
# ---------------------------
def create_data_loaders(
    train_path: PathOrStr,
    val_path: PathOrStr,
    test_path: PathOrStr,
    batch_size: int = 64,
    num_workers: int = 4,
    mode: str = "memmap",
    transform: TransformFn = None,
    target_transform: TransformFn = None,
    return_dict: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    sampler_train: Optional[Sampler] = None,
    sampler_val: Optional[Sampler] = None,
    sampler_test: Optional[Sampler] = None,
    distributed: bool = False,
    seed: int = 42,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with research-grade defaults.

    Notes:
      - If using memmap with num_workers > 0, worker_init_fn reopens memmaps per worker.
      - If return_dict=True and no collate_fn is provided, a default collate function that pads variable-length
        sequences is used.
    """
    # Choose sensible collate default when return_dict=True
    if collate_fn is None and return_dict:
        collate_fn = default_collate_variable_length

    # Datasets
    ds_kwargs = {"mode": mode, "transform": transform, "target_transform": target_transform, "return_dict": return_dict}
    train_ds = IDSDataset(train_path, **ds_kwargs)
    val_ds = IDSDataset(val_path, **ds_kwargs)
    test_ds = IDSDataset(test_path, **ds_kwargs)

    # Early sanity checks
    if train_ds.n_samples == 0:
        raise ValueError(f"Train dataset at {train_path} is empty.")
    if val_ds.n_samples == 0:
        logger.warning(f"Validation dataset at {val_path} is empty.")
    if test_ds.n_samples == 0:
        logger.warning(f"Test dataset at {test_path} is empty.")

    # Distributed sampler handling
    if distributed:
        sampler_train = DistributedSampler(train_ds, shuffle=True, seed=seed)
        sampler_val = DistributedSampler(val_ds, shuffle=False, seed=seed)
        sampler_test = DistributedSampler(test_ds, shuffle=False, seed=seed)
        logger.info("create_data_loaders: using DistributedSampler for datasets (distributed=True)")

    # Worker init: prefer provided function; else use our memmap-safe default
    chosen_worker_init = (lambda wid: _default_worker_init_fn(wid, seed)) if worker_init_fn is None else worker_init_fn

    # Build DataLoader kwargs
    num_workers = max(0, int(num_workers))
    dl_kwargs: DataLoaderKwargs = {
        "batch_size": int(batch_size),
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) and num_workers > 0,
        "drop_last": bool(drop_last),
        "collate_fn": collate_fn,
        "worker_init_fn": chosen_worker_init,
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    # Create loaders
    train_loader = DataLoader(train_ds, shuffle=(sampler_train is None), sampler=sampler_train, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=sampler_val, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, sampler=sampler_test, **dl_kwargs)

    logger.info(
        f"DataLoaders created: train_batches={len(train_loader) if hasattr(train_loader, '__len__') else 'unknown'} "
        f"val_batches={len(val_loader) if hasattr(val_loader, '__len__') else 'unknown'} "
        f"test_batches={len(test_loader) if hasattr(test_loader, '__len__') else 'unknown'}"
    )
    return train_loader, val_loader, test_loader

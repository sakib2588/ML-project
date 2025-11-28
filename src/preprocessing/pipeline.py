""""
Research-grade PreprocessingPipeline

Improvements:
- Deterministic seeding
- Chronological Split -> FeatureEngineer per split -> Label mapping -> Windowing
- Atomic artifact writes + metadata (includes git SHA if available)
- Defensive handling of FeatureEngineer return signatures
- Safe scaler transform wrt NA handling (prevents shape-mismatch)
- Optional streaming path placeholder for >RAM datasets
- Diagnostics in preprocessing_report.json
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import logging
from datetime import datetime

# persistence
try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    joblib = None
    HAS_JOBLIB = False

from sklearn.model_selection import train_test_split
# Local imports (adjust these relative paths to match your actual project structure)
from .feature_engineering import FeatureEngineer
from .windowing import FlowWindower
from .scalers import FeatureScaler
from ..data.validators import DataValidator

# ---------- helpers ----------

def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON to a tmp file then rename for atomicity."""
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, default=str, indent=2)
    tmp.replace(path)

def _atomic_save_numpy(path: Path, arr: np.ndarray) -> None:
    """Save numpy array to tmp file then rename for atomicity."""
    # Ensure tmp file ends in .npy so np.save doesn't append it again
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(str(tmp), arr)
    tmp.replace(path)

def _git_sha() -> Optional[str]:
    """Attempt to get current git SHA for reproducibility."""
    try:
        root = Path.cwd()
        git_dir = root / ".git"
        if git_dir.exists():
            import subprocess
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root)).decode().strip()
            return sha
    except Exception:
        return None
    return None

def _ensure_joblib():
    if not HAS_JOBLIB:
        raise RuntimeError("joblib is required; install it with `pip install joblib`.")

@dataclass
class PipelineMetadata:
    created_at: str
    dataset: str
    mode: str
    seed: int
    git_sha: Optional[str]

# ---------- main pipeline ----------

class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, seed: int = 42):
        self.config = config
        self.preprocess_config = config.get("preprocess", {})
        self.data_config = config.get("data", {})
        self.logger = logger or logging.getLogger(__name__)
        self.seed = int(seed)

        # Validator
        strict_missing = self.preprocess_config.get("missing_values", {}).get("mode", "strict") == "strict"
        self.validator = DataValidator(strict_mode=strict_missing, logger=self.logger)

        # Scaler (FeatureScaler signature assumed; pass only minimal args)
        scaler_cfg = self.preprocess_config.get("transformations", {}).get("normalization", {})
        self.scaler = FeatureScaler(method=scaler_cfg.get("method", "standard"), logger=self.logger)

        # Windower
        window_cfg = self.preprocess_config.get("windowing", {})
        self.windower = FlowWindower(
            window_length=int(window_cfg.get("window_length", 8)),
            stride=int(window_cfg.get("stride", 1)),
            mode=window_cfg.get("mode", "sliding"),
            padding_strategy=window_cfg.get("padding_strategy", "zero"),
            label_strategy=window_cfg.get("label_strategy", "any_malicious"),
            logger=self.logger
        )

        # Sampling
        self.sampling_config = self.data_config.get("sampling", {})

    # ---------------- core API ----------------
    def process_dataset(self, dataset_name: str, output_dir: Union[str, Path], mode: str = "quick", overwrite: bool = False) -> Dict[str, Any]:
        """
        High-level processing entrypoint.
        """
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)
        if not overwrite and any((outp / p).exists() for p in ("train", "val", "test")):
            raise FileExistsError(f"Output dir {outp} already contains splits. Use overwrite=True to replace.")

        # seed all randomness
        np.random.seed(self.seed)
        import random
        random.seed(self.seed)

        if self.logger:
            self.logger.info(f"Pipeline start: dataset={dataset_name} mode={mode} seed={self.seed}")

        # metadata for reproducibility
        metadata = PipelineMetadata(created_at=datetime.utcnow().isoformat(), dataset=dataset_name, mode=mode, seed=self.seed, git_sha=_git_sha())
        _atomic_write_json(outp / "preprocessing_metadata.json", asdict(metadata))

        # 1) load raw data (DEV: in-memory; FULL: placeholder streaming)
        df = self._load_raw_data(dataset_name, mode)

        # 2) chronological sort if time column provided (no separate sampling step now - done in _load_raw_data)
        ds_cfg = self.data_config.get("datasets", {}).get(dataset_name, {})
        time_col = ds_cfg.get("time_column")
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.sort_values(time_col).reset_index(drop=True)
            if self.logger:
                self.logger.info(f"Sorted by time column: {time_col}")
        else:
            if self.logger:
                self.logger.info("No time column provided — using file/row order as chronological proxy")

        # 3) chronological hard-split
        train_ratio = float(self.data_config.get("split_ratios", {}).get("train", 0.7))
        val_ratio = float(self.data_config.get("split_ratios", {}).get("val", 0.15))
        n = len(df)
        n_raw_rows_cached = n  # Cache before deleting df
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Type cast to help Pylance understand these are DataFrames, not Series
        df_train = pd.DataFrame(df.iloc[:n_train]).reset_index(drop=True)
        df_val = pd.DataFrame(df.iloc[n_train:n_train + n_val]).reset_index(drop=True)
        df_test = pd.DataFrame(df.iloc[n_train + n_val:]).reset_index(drop=True)
        if self.logger:
            self.logger.info(f"Split sizes (chronological) train={len(df_train)} val={len(df_val)} test={len(df_test)}")
        
        # Clear original df to save memory
        del df
        import gc
        gc.collect()

        # 4) feature-engineer per split (defensive unpack)
        fe_train_df, fe_train_meta = self._apply_feature_engineer(df_train, dataset_name)
        import gc
        gc.collect()
        
        fe_val_df, fe_val_meta = self._apply_feature_engineer(df_val, dataset_name)
        gc.collect()
        
        fe_test_df, fe_test_meta = self._apply_feature_engineer(df_test, dataset_name)
        gc.collect()

        # 5) validate that label column exists in train FE output
        label_col = ds_cfg.get("label_column", "Label")
        # Allow lowercase canonical variant if original casing absent
        if label_col not in fe_train_df.columns and label_col.lower() in fe_train_df.columns:
            label_col = label_col.lower()
        if label_col not in fe_train_df.columns:
            raise RuntimeError(f"Label column '{label_col}' missing in feature-engineered train dataframe (available: {list(fe_train_df.columns)[:10]} ...)")

        # 7) label mapping (train-only), deterministic
        y_train_flow = fe_train_df[label_col].to_numpy()
        unique_train = list(pd.unique(y_train_flow))
        try:
            unique_train_sorted = sorted(unique_train)
        except Exception:
            unique_train_sorted = unique_train
        label_map = {lab: int(i) for i, lab in enumerate(unique_train_sorted)}
        inv_label_map = {v: k for k, v in label_map.items()}
        if self.logger:
            self.logger.info(f"Label map trained with {len(label_map)} classes (train-only)")

        def _map(arr):
            return np.array([label_map.get(x, -1) for x in arr], dtype=np.int64)

        # 8) map labels, log unseen
        y_train_mapped = _map(y_train_flow)
        y_val_mapped = _map(fe_val_df[label_col].to_numpy()) if label_col in fe_val_df.columns else np.array([], dtype=np.int64)
        y_test_mapped = _map(fe_test_df[label_col].to_numpy()) if label_col in fe_test_df.columns else np.array([], dtype=np.int64)

        # FIX 2: Explicitly construct the check list to satisfy Pylance
        val_orig = fe_val_df[label_col].to_numpy() if label_col in fe_val_df.columns else np.array([])
        test_orig = fe_test_df[label_col].to_numpy() if label_col in fe_test_df.columns else np.array([])

        check_splits: List[Tuple[str, np.ndarray, np.ndarray]] = [
            ("VAL", val_orig, y_val_mapped),
            ("TEST", test_orig, y_test_mapped)
        ]

        for split_name, orig, mapped in check_splits:
            if len(orig) > 0:
                # check for -1 in mapped
                unseen_mask = (mapped == -1)
                if np.any(unseen_mask):
                    unseen_vals = np.unique(np.array(orig)[unseen_mask])
                    if self.logger:
                        self.logger.warning(f"{split_name} contains unseen labels -> mapped to -1: {unseen_vals}")

        # 9) window per split
        # Drop label col if present, else use full DF
        # 9) window per split
        # Drop label col if present, else use full DF
        def _drop_label(df_in: pd.DataFrame) -> np.ndarray:
            if label_col in df_in.columns:
                return df_in.drop(columns=[label_col]).to_numpy(dtype=np.float32)
            return df_in.to_numpy(dtype=np.float32)

        X_train_flow = _drop_label(fe_train_df)
        X_val_flow = _drop_label(fe_val_df)
        X_test_flow = _drop_label(fe_test_df)

        # FIX: Unpack 4 values (windows, labels, diagnostics, indices) and ignore the last 2
        X_train_w, y_train_w, _, _ = self.windower.create_windows(X_train_flow, y_train_mapped)
        X_val_w, y_val_w, _, _ = self.windower.create_windows(X_val_flow, y_val_mapped)
        X_test_w, y_test_w, _, _ = self.windower.create_windows(X_test_flow, y_test_mapped)

        if self.logger:
            self.logger.info(f"Windowed shapes -> train {X_train_w.shape}, val {X_val_w.shape}, test {X_test_w.shape}")

        # 10) scaler: fit on train windows; *safe transform* (no row dropping)
        self._fit_scaler(X_train_w)
        X_train_scaled = self._safe_transform(X_train_w)
        X_val_scaled = self._safe_transform(X_val_w)
        X_test_scaled = self._safe_transform(X_test_w)

        # 11) save splits & artifacts (atomic)
        for split_name, X_arr, y_arr in (("train", X_train_scaled, y_train_w), ("val", X_val_scaled, y_val_w), ("test", X_test_scaled, y_test_w)):
            split_dir = outp / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            _atomic_save_numpy(split_dir / "X.npy", X_arr)
            _atomic_save_numpy(split_dir / "y.npy", y_arr)
            if self.logger:
                self.logger.info(f"Saved split {split_name} -> {split_dir}")

        # save scaler and label_map using joblib (atomic)
        _ensure_joblib()
        # FIX 1: Assert joblib is not None for Pylance
        assert joblib is not None 

        scaler_path = outp / "scaler.joblib"
        tmp_scaler = outp / ("scaler.joblib.tmp")
        if self.scaler is None:
             raise RuntimeError("Scaler is None, cannot save.")
        
        joblib.dump(self.scaler, str(tmp_scaler))
        tmp_scaler.replace(scaler_path)

        label_map_path = outp / "label_map.joblib"
        joblib.dump({"label_map": label_map, "inv_label_map": inv_label_map}, str(label_map_path))

        # 12) diagnostics and report
        # Convert label_map keys/values to JSON-serializable types (Python native int/str, not numpy.int64)
        label_map_serializable = {str(k): int(v) for k, v in label_map.items()}
        
        report = {
            "dataset": dataset_name,
            "mode": mode,
            "metadata_path": str(outp / "preprocessing_metadata.json"),
            "n_raw_rows": n_raw_rows_cached,
            "n_train_rows": len(df_train),
            "n_val_rows": len(df_val),
            "n_test_rows": len(df_test),
            "n_train_windows": int(X_train_scaled.shape[0]),
            "n_val_windows": int(X_val_scaled.shape[0]),
            "n_test_windows": int(X_test_scaled.shape[0]),
            "label_map": label_map_serializable,
            "git_sha": metadata.git_sha,
            "scaler_method": getattr(self.scaler, "method", "unknown"),
            "diagnostics": self._compute_diagnostics(fe_train_df, fe_val_df, fe_test_df)
        }
        _atomic_write_json(outp / "preprocessing_report.json", report)
        if self.logger:
            self.logger.info(f"Preprocessing complete -> {outp}")
        return report

    # --------------- internal helpers ---------------

    def _apply_feature_engineer(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        fe = FeatureEngineer(config=self.preprocess_config, dataset_name=dataset_name, logger=self.logger)
        res = fe.transform(df)
        # Handle return signature polymorphism: can be DF or (DF, meta)
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], pd.DataFrame):
            return res[0], res[1]
        elif isinstance(res, pd.DataFrame):
            return res, {}
        else:
            raise RuntimeError("FeatureEngineer.transform() returned unexpected type. Expected DataFrame or (DataFrame, meta).")

    def _load_raw_data(self, dataset_name: str, mode: str) -> pd.DataFrame:
        ds_cfg = self.data_config.get("datasets", {}).get(dataset_name, {})
        raw_dir = Path(ds_cfg.get("raw_data_dir", f"data/{dataset_name}"))
        if not raw_dir.exists():
            raise FileNotFoundError(raw_dir)
        csvs = sorted(raw_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSVs in {raw_dir}")

        # Get sampling config
        sample_size = self.sampling_config.get(f"{mode}_mode", {}).get("samples_per_dataset")
        
        # Standard loading (optimized for machines with sufficient RAM like your Arch Linux with 11.8GB)
        dfs = []
        for p in csvs:
            try:
                if self.logger:
                    self.logger.info(f"Loading {p.name}")
                
                # Load file directly - pandas is efficient with modern RAM
                df = pd.read_csv(p, low_memory=False)
                
                if self.logger:
                    self.logger.info(f"  Loaded {p.name}: {len(df)} rows")
                
                dfs.append(df)
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed reading {p}: {e}")
                import traceback
                traceback.print_exc()
        
        if not dfs:
            raise RuntimeError(f"No CSVs successfully loaded from {raw_dir}")
        
        if self.logger:
            self.logger.info(f"Concatenating {len(dfs)} files...")
        
        combined = pd.concat(dfs, ignore_index=True)
        # Normalize column names (strip leading/trailing whitespace) to avoid mapping failures
        # CIC-IDS2017 raw CSVs often contain leading spaces after commas (e.g. " Flow Duration")
        # which caused FeatureEngineer to think the source column was missing.
        combined.columns = [c.strip() for c in combined.columns]
        if self.logger:
            self.logger.info(f"Combined dataset: {len(combined)} rows (normalized {len(dfs)} files)")
        
        # Apply sampling AFTER loading all data (preserves chronological order better)
        if sample_size and len(combined) > sample_size:
            if self.logger:
                self.logger.info(f"Sampling {sample_size} rows from {len(combined)} total (preserving chronological order)")
            combined = combined.head(int(sample_size))
        
        return combined

    def _fit_scaler(self, X_train_w: np.ndarray) -> None:
        # scaler.fit supports 3D windowed data in your FeatureScaler
        if self.scaler is not None:
            self.scaler.fit(X_train_w)
        if self.logger:
            self.logger.info("Scaler fitted on train windows")

    def _safe_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Warning: if scaler or config uses 'drop' for NA, transform would break shape.
        Strategy: If scaler.transform would drop rows, we refuse that in transform stage.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler is None, cannot transform.")
            
        try:
            out = self.scaler.transform(X)
        except ValueError as e:
            # likely caused by dropped rows during transform (shape mismatch)
            msg = (
                "Scaler.transform failed — possible na_strategy='drop' causing shape mismatch.\n"
                "Recommendation: set na_strategy to 'mean' or 'none' for transform-time (only allow 'drop' during fit),\n"
                "or rerun fit/transform with a strategy that preserves sample count."
            )
            if self.logger:
                self.logger.error(msg)
            raise
        # shape check
        if out.shape[0] != X.shape[0]:
            raise RuntimeError("Scaler.transform changed number of rows. This would break windowed data shapes.")
        return out

    def _compute_diagnostics(self, fe_train: pd.DataFrame, fe_val: pd.DataFrame, fe_test: pd.DataFrame) -> Dict[str, Any]:
        def basic(df: pd.DataFrame) -> Dict[str, Any]:
            return {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "nan_rate": float(df.isna().sum().sum()) / max(1, len(df) * len(df.columns)),
                "zero_variance_cols": int((df.nunique(dropna=False) <= 1).sum())
            }
        return {"train": basic(fe_train), "val": basic(fe_val), "test": basic(fe_test)}
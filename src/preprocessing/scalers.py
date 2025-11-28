"""
Research-grade feature scaler utilities (type-safe, Pylance-friendly).

Features:
- feature_range coerced to floats for MinMaxScaler
- Internal scaler annotated with Any, guarded by _ensure_scaler()
- Handles 2D and 3D arrays (windowed)
- NA strategies: drop/mean/constant/none
- Fit modes: global or per-window
- Save/load via joblib with metadata
"""

from __future__ import annotations
from typing import Any
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple, Literal, Type, cast

# persistence
try:
    import joblib
    HAS_JOBLIB: bool = True
except Exception:
    joblib: Optional[Any] = None
    HAS_JOBLIB = False

# sklearn scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
try:
    from sklearn.preprocessing import PowerTransformer as _PowerTransformer
    PowerTransformer: Optional[Type[Any]] = _PowerTransformer
    HAS_POWER_TRANSFORMER: bool = True
except Exception:
    PowerTransformer: Optional[Type[Any]] = None
    HAS_POWER_TRANSFORMER = False

# Type aliases
ArrayLike = Union[np.ndarray]
ScalerMethod = Literal["standard", "minmax", "robust", "power"]
NaStrategy = Literal["drop", "mean", "constant", "none"]
FitMode = Literal["global", "per_window"]


class FeatureScaler:
    """
    Research-grade feature scaler wrapper.

    Parameters
    ----------
    method : 'standard' | 'minmax' | 'robust' | 'power'
    na_strategy : how to handle NaNs before fitting ('drop','mean','constant','none')
    na_fill_value : used when na_strategy == 'constant'
    fit_mode : 'global' or 'per_window'
    feature_range : for MinMaxScaler (tuple of two numbers)
    copy : passed to sklearn scalers
    logger : optional logger
    sk_params : additional sklearn params
    """

    def __init__(
        self,
        method: ScalerMethod = "standard",
        na_strategy: NaStrategy = "mean",
        na_fill_value: float = 0.0,
        fit_mode: FitMode = "global",
        feature_range: Tuple[float, float] = (0.0, 1.0),
        copy: bool = True,
        logger: Optional[logging.Logger] = None,
        **sk_params: Any,
    ) -> None:
        self.method = method
        self.na_strategy = na_strategy
        self.na_fill_value = na_fill_value
        self.fit_mode = fit_mode
        self.feature_range = feature_range
        self.copy = copy
        self.logger = logger or logging.getLogger(__name__)
        self._sk_params = sk_params

        # Internal scaler, runtime-guarded
        self._scaler: Any = None
        self.fitted = False
        self.original_shape: Optional[Tuple[int, ...]] = None

        # Instantiate underlying sklearn scaler
        if method == "standard":
            self._scaler = StandardScaler(copy=copy, **sk_params)
        elif method == "minmax":
            fr = (float(feature_range[0]), float(feature_range[1]))
            self._scaler = MinMaxScaler(feature_range=cast(Any, fr), copy=copy, **sk_params)

        elif method == "robust":
            self._scaler = RobustScaler(copy=copy, **sk_params)
        elif method == "power":
            if not HAS_POWER_TRANSFORMER or PowerTransformer is None:
                raise ImportError("PowerTransformer not available. Install a recent scikit-learn.")
            self._scaler = PowerTransformer(copy=copy, **sk_params)
        else:
            raise ValueError(f"Unknown method: {method}")

    # -------------------------
    # Internal utilities
    # -------------------------
    def _ensure_scaler(self) -> None:
        if self._scaler is None:
            raise RuntimeError("Underlying sklearn scaler not initialized. Check constructor arguments.")

    def _ensure_array(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(X) if not isinstance(X, np.ndarray) else X

    def _validate_ndim(self, X: np.ndarray) -> None:
        if X.ndim not in (2, 3):
            raise ValueError(f"X must be 2D or 3D array. Received shape: {X.shape}")

    def _impute_with_mean_or_constant(self, X2: np.ndarray) -> np.ndarray:
        if X2.size == 0:
            return X2
        col_mean = np.nanmean(X2, axis=0)
        col_mean_fixed = np.where(np.isnan(col_mean), self.na_fill_value, col_mean)
        inds = np.where(np.isnan(X2))
        if inds[0].size > 0:
            X2 = X2.copy()
            X2[inds] = np.take(col_mean_fixed, inds[1])
        return X2

    def _prepare_for_fit(self, X: np.ndarray) -> np.ndarray:
        X = self._ensure_array(X)
        self._validate_ndim(X)
        if X.ndim == 3 and self.fit_mode == "global":
            n, L, f = X.shape
            X2 = X.reshape(-1, f)
        elif X.ndim == 3 and self.fit_mode == "per_window":
            return X
        else:
            X2 = X

        # Handle NA strategies
        if self.na_strategy == "none":
            return X2
        elif self.na_strategy == "drop":
            mask = ~np.isnan(X2).any(axis=1)
            X2 = X2[mask]
            return X2
        elif self.na_strategy == "mean" or self.na_strategy == "constant":
            return self._impute_with_mean_or_constant(X2)
        else:
            return X2

    def _prepare_for_transform(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        X = self._ensure_array(X)
        self._validate_ndim(X)
        self.original_shape = X.shape

        if X.ndim == 3 and self.fit_mode == "global":
            n, L, f = X.shape
            X2 = X.reshape(-1, f)
            out_shape = (n, L, f)
            if self.na_strategy == "drop":
                self.logger.warning(
                    "na_strategy='drop' configured. Falling back to mean-imputation to preserve shape."
                )
                X2 = self._impute_with_mean_or_constant(X2)
            elif self.na_strategy in ("mean", "constant"):
                X2 = self._impute_with_mean_or_constant(X2)
            return X2, out_shape
        elif X.ndim == 3 and self.fit_mode == "per_window":
            return X, X.shape
        else:
            out_shape = X.shape
            X2 = X
            if self.na_strategy == "drop":
                self.logger.warning(
                    "na_strategy='drop' configured. Falling back to mean-imputation to preserve shape."
                )
                X2 = self._impute_with_mean_or_constant(X2)
            elif self.na_strategy in ("mean", "constant"):
                X2 = self._impute_with_mean_or_constant(X2)
            return X2, out_shape

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X: ArrayLike, y=None) -> "FeatureScaler":
        self._ensure_scaler()
        X = self._ensure_array(X)
        self._validate_ndim(X)

        if X.ndim == 3:
            n, L, f = X.shape
            X_flat = X.reshape(-1, f)
            X2 = self._prepare_for_fit(X_flat)
        else:
            X2 = self._prepare_for_fit(X)

        if X2.shape[0] == 0:
            raise ValueError("No data left after NA handling to fit scaler")

        self._scaler.fit(X2)
        self.fitted = True
        self.logger.info(f"Fitted {self.method} scaler on data of shape {X2.shape}")
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._ensure_scaler()
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() before transform().")

        X = self._ensure_array(X)
        X_prep, out_shape = self._prepare_for_transform(X)

        if X.ndim == 3 and self.fit_mode == "per_window":
            n, L, f = X.shape
            out = np.empty_like(X, dtype=float)
            for i in range(n):
                chunk = X[i]
                chunk_prep = self._prepare_for_fit(chunk)
                if chunk_prep.size == 0:
                    out[i] = chunk
                    continue
                out_flat = self._scaler.transform(chunk_prep)
                out[i] = out_flat.reshape((L, f))
            return out

        X_scaled_flat = self._scaler.transform(X_prep)
        if X.ndim == 3:
            n, L, f = out_shape
            X_scaled = X_scaled_flat.reshape(n, L, f)
        else:
            X_scaled = X_scaled_flat
        return X_scaled

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_scaled: ArrayLike) -> np.ndarray:
        self._ensure_scaler()
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() before inverse_transform().")

        Xs = self._ensure_array(X_scaled)
        if Xs.ndim == 3 and self.fit_mode == "global":
            n, L, f = Xs.shape
            X_flat = Xs.reshape(-1, f)
            X_inv = self._scaler.inverse_transform(X_flat)
            return X_inv.reshape(n, L, f)
        elif Xs.ndim == 3 and self.fit_mode == "per_window":
            n, L, f = Xs.shape
            out = np.empty_like(Xs, dtype=float)
            for i in range(n):
                out[i] = self._scaler.inverse_transform(Xs[i])
            return out
        else:
            return self._scaler.inverse_transform(Xs)

    def get_scaler_params(self) -> Dict[str, Any]:
        if not self.fitted:
            return {}
        params: Dict[str, Any] = {"method": self.method}
        if hasattr(self._scaler, "n_features_in_"):
            params["n_features_in_"] = int(getattr(self._scaler, "n_features_in_"))
        if self.method == "standard":
            params.update({"mean_": getattr(self._scaler, "mean_", None), "scale_": getattr(self._scaler, "scale_", None)})
        elif self.method == "minmax":
            params.update({"data_min_": getattr(self._scaler, "data_min_", None), "data_max_": getattr(self._scaler, "data_max_", None)})
        elif self.method == "robust":
            params.update({"center_": getattr(self._scaler, "center_", None), "scale_": getattr(self._scaler, "scale_", None)})
        elif self.method == "power":
            params.update({"lambdas_": getattr(self._scaler, "lambdas_", None)})
        return params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "na_strategy": self.na_strategy,
            "na_fill_value": self.na_fill_value,
            "fit_mode": self.fit_mode,
            "feature_range": self.feature_range,
            "copy": self.copy,
            "fitted": bool(self.fitted),
            "sk_params": self._sk_params,
            "scaler_params": self.get_scaler_params(),
        }

    # -------------------------
    # Persistence
    # -------------------------
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not HAS_JOBLIB or joblib is None:
            raise RuntimeError("joblib is required to save FeatureScaler objects reliably")
        payload = {"meta": {"wrapper": self.to_dict()}, "scaler": self._scaler}
        joblib.dump(payload, path)
        self.logger.info(f"FeatureScaler saved to {path}")

    @staticmethod
    def load(path: Union[str, Path], logger: Optional[logging.Logger] = None) -> "FeatureScaler":
        path = Path(path)
        if not HAS_JOBLIB or joblib is None:
            raise RuntimeError("joblib is required to load FeatureScaler objects reliably")
        obj = joblib.load(path)
        if isinstance(obj, dict) and "scaler" in obj:
            wrapper_info = obj.get("meta", {}).get("wrapper", {}) if obj.get("meta") else {}
            method = wrapper_info.get("method", "standard")
            fs = FeatureScaler(method=method, logger=logger)
            fs._scaler = obj["scaler"]
            fs.fitted = True
            if logger:
                logger.info(f"FeatureScaler loaded from {path}")
            return fs
        if isinstance(obj, FeatureScaler):
            if logger is not None:
                obj.logger = logger
            return obj
        raise RuntimeError("Unexpected object loaded from path; expected saved FeatureScaler structure")

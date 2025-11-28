"""
Research-grade windowing utilities for IDS flow sequences.

Features:
- Sliding and grouped windows
- Vectorized sliding-window where possible (fast)
- Several padding strategies with configurable pad values
- Multiple labeling strategies (any_malicious, majority, first, last)
- Optional DataFrame-grouping helper (window by session/flow id)
- Optionally return original row indices for each window (useful for debugging)
- Reports diagnostics about created windows
"""

from __future__ import annotations
from typing import Optional, Callable
from typing import Tuple, Literal, Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Try to import sliding_window_view for vectorized windows (NumPy >= 1.20)
from typing import cast, Callable, Optional
try:
    from numpy.lib.stride_tricks import sliding_window_view
    sliding_window_view = cast(Optional[Callable], sliding_window_view)
    HAS_SLIDING_WINDOW: bool = True
except Exception:
    sliding_window_view = None
    HAS_SLIDING_WINDOW = False



WindowMode = Literal["sliding", "grouped"]
PaddingStrategy = Literal["zero", "repeat", "none"]
LabelStrategy = Literal["any_malicious", "majority", "first", "last"]


class FlowWindower:
    """
    Create fixed-length windows from sequential flow features and labels.

    Core methods:
      - create_windows(features, labels) -> (windows, window_labels, diagnostics)
      - from_dataframe(df, feature_columns, label_column, groupby=None, ...) -> windows across grouped flows

    Important options:
      - window_length: int
      - stride: int (effective stride for sliding; ignored for grouped mode)
      - mode: 'sliding' or 'grouped'
      - padding_strategy: 'zero', 'repeat', or 'none' (drop incomplete windows)
      - pad_feature_value: numeric pad value for features when padding_strategy == 'zero'
      - pad_label: label value used when padding labels (e.g., -1 or 0)
      - label_strategy: how to compute the window label
      - return_indices: include original row indices for each window
    """

    def __init__(
        self,
        window_length: int = 8,
        stride: int = 1,
        mode: WindowMode = "sliding",
        padding_strategy: PaddingStrategy = "zero",
        label_strategy: LabelStrategy = "any_malicious",
        pad_feature_value: float = 0.0,
        pad_label: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        if window_length <= 0:
            raise ValueError("window_length must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.window_length = int(window_length)
        self.stride = int(stride)
        self.mode = mode
        self.padding_strategy = padding_strategy
        self.label_strategy = label_strategy
        self.pad_feature_value = pad_feature_value
        self.pad_label = pad_label
        self.logger = logger or logging.getLogger(__name__)

        if self.mode == "grouped":
            # grouped windows are non-overlapping
            self.stride = self.window_length

        if padding_strategy not in ("zero", "repeat", "none"):
            raise ValueError("padding_strategy must be 'zero','repeat',or 'none'")

        if label_strategy not in ("any_malicious", "majority", "first", "last"):
            raise ValueError("unsupported label_strategy")

    # -------------------------
    # Public API
    # -------------------------
    def create_windows(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        return_indices: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        """
        Create windows from numpy arrays.

        Args:
            features: shape (n_flows, n_features)
            labels: shape (n_flows,) or (n_flows, 1)
            return_indices: if True returns indices ndarray shape (n_windows, window_length)

        Returns:
            windows: np.ndarray shape (n_windows, window_length, n_features)
            window_labels: np.ndarray shape (n_windows,)
            diagnostics: dict with keys like n_windows_created, n_padded_windows
            indices (optional): np.ndarray shape (n_windows, window_length) of source row indices
        """
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a numpy array")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a numpy array")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same first dimension length")

        n_flows, n_features = features.shape
        if n_flows == 0:
            return (
                np.empty((0, self.window_length, n_features), dtype=features.dtype),
                np.empty((0,), dtype=labels.dtype),
                {"n_windows_created": 0, "n_padded_windows": 0},
                (np.empty((0, self.window_length), dtype=int) if return_indices else None),
            )

        if self.mode == "sliding":
            windows, window_labels, diag, indices = self._sliding_windows(features, labels, return_indices=return_indices)
        else:
            windows, window_labels, diag, indices = self._grouped_windows(features, labels, return_indices=return_indices)

        return windows, window_labels, diag, indices

    def from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str,
        *,
        groupby: Optional[Union[str, List[str]]] = None,
        concat_groups: bool = False,
        **create_windows_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        """
        Window flows stored in a pandas DataFrame.

        Args:
            df: DataFrame with flow rows in chronological order (per group).
            feature_columns: list of columns used as features (order preserved).
            label_column: name of the label column.
            groupby: if provided, window per group key (e.g., session_id). Can be column name or list of columns.
            concat_groups: if True, concatenate windows from all groups into one numpy array result (default True).
            create_windows_kwargs: forwarded to create_windows (e.g., return_indices=True)

        Returns:
            same tuple as create_windows: windows, window_labels, diagnostics, indices
        """
        if groupby is None:
            # Simple path: entire DF as one sequence
            features = df[feature_columns].to_numpy()
            labels = df[label_column].to_numpy().reshape(-1)
            return self.create_windows(features, labels, **create_windows_kwargs)

        # Grouped path: process each group independently and concatenate results
        grouped = df.groupby(groupby, sort=False)
        windows_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        indices_list: List[np.ndarray] = []
        total_diag = {"n_windows_created": 0, "n_padded_windows": 0, "groups_processed": 0}

        for _, group in grouped:
            total_diag["groups_processed"] += 1
            feats = group[feature_columns].to_numpy()
            labs = group[label_column].to_numpy().reshape(-1)
            w, wl, diag, idx = self.create_windows(feats, labs, **create_windows_kwargs)
            if w.size:
                windows_list.append(w)
                labels_list.append(wl)
                if idx is not None:
                    indices_list.append(idx)
            # accumulate diagnostics
            total_diag["n_windows_created"] += diag.get("n_windows_created", 0)
            total_diag["n_padded_windows"] += diag.get("n_padded_windows", 0)

        if not windows_list:
            return (np.empty((0, self.window_length, len(feature_columns))),
                    np.empty((0,)),
                    total_diag,
                    (np.empty((0, self.window_length), dtype=int) if create_windows_kwargs.get("return_indices") else None))

        # Concatenate windows from groups
        windows = np.concatenate(windows_list, axis=0)
        window_labels = np.concatenate(labels_list, axis=0)
        indices = np.concatenate(indices_list, axis=0) if indices_list and create_windows_kwargs.get("return_indices") else None

        return windows, window_labels, total_diag, indices

    # -------------------------
    # Internal implementations
    # -------------------------
    
    def _sliding_windows(self, features: np.ndarray, labels: np.ndarray, *, return_indices: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        n_flows, n_features = features.shape
        L = self.window_length
        s = self.stride

        windows_list: List[np.ndarray] = []
        label_list: List[int] = []
        indices_list: List[np.ndarray] = []
        n_padded = 0

        # Fast path: use sliding_window_view if available and stride == 1 and n_flows >= L
        if HAS_SLIDING_WINDOW and s == 1 and n_flows >= L and sliding_window_view is not None:
            # sliding_window_view with 2D window_shape returns dimension order slightly different; safer to use 1D window on axis 0:
            all_windows = sliding_window_view(features, window_shape=L, axis=0)  # shape (n_flows-L+1, L, n_features)
            # select every stride (but stride ==1 here)
            selected = all_windows[::s]
            for i in range(selected.shape[0]):
                windows_list.append(selected[i])
                win_idx = np.arange(i * s, i * s + L)
                indices_list.append(win_idx)
                label_list.append(self._label_window(labels[win_idx]))
        else:
            # Generic path (supports arbitrary stride)
            n_full_windows = max(0, (n_flows - L) // s + 1) if n_flows >= L else 0
            for i in range(n_full_windows):
                start = i * s
                end = start + L
                win_feats = features[start:end]
                win_labels = labels[start:end]
                windows_list.append(win_feats)
                indices_list.append(np.arange(start, end))
                label_list.append(self._label_window(win_labels))

        # Handle tail remainder windows (last window that may be partial)
        last_window_start = n_flows - L
        tail_added = False
        if last_window_start < 0:
            # sequence shorter than window_length
            if self.padding_strategy != "none":
                padded = self._pad_sequence(features, is_label=False)
                padded_labels = self._pad_sequence(labels, is_label=True)
                windows_list.append(padded)
                indices_list.append(np.arange(0, n_flows).tolist() + [-1] * (L - n_flows))
                label_list.append(self._label_window(padded_labels))
                n_padded += 1
                tail_added = True
        else:
            # There may be a leftover tail that was not covered by full windows depending on stride
            last_start = ( (n_flows - L) // s ) * s if n_flows >= L else 0
            end_of_last = last_start + L
            if end_of_last < n_flows and self.padding_strategy != "none":
                # start the final window at max(0, n_flows - L)
                start = max(0, n_flows - L)
                remain_feats = features[start:]
                remain_labels = labels[start:]
                if remain_feats.shape[0] < L:
                    padded = self._pad_sequence(remain_feats, is_label=False)
                    padded_labels = self._pad_sequence(remain_labels, is_label=True)
                    windows_list.append(padded)
                    indices_row = np.concatenate([np.arange(start, n_flows), np.full(L - (n_flows - start), -1, dtype=int)])
                    indices_list.append(indices_row)
                    label_list.append(self._label_window(padded_labels))
                    n_padded += 1
                    tail_added = True

        # Convert lists to arrays
        if windows_list:
            windows = np.stack(windows_list, axis=0)
            window_labels = np.array(label_list, dtype=int)
            indices_arr = np.stack([np.array(idx, dtype=int) for idx in indices_list], axis=0) if return_indices else None
            diagnostics = {"n_windows_created": windows.shape[0], "n_padded_windows": n_padded}
            return windows, window_labels, diagnostics, indices_arr
        else:
            # No windows created (e.g., short sequence & padding_strategy == 'none')
            diagnostics = {"n_windows_created": 0, "n_padded_windows": 0}
            empty_windows = np.empty((0, L, n_features), dtype=features.dtype)
            empty_labels = np.empty((0,), dtype=labels.dtype)
            empty_indices = np.empty((0, L), dtype=int) if return_indices else None
            return empty_windows, empty_labels, diagnostics, empty_indices

    def _grouped_windows(self, features: np.ndarray, labels: np.ndarray, *, return_indices: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        n_flows, n_features = features.shape
        L = self.window_length
        groups = n_flows // L
        n_remaining = n_flows % L

        windows_list = []
        labels_list = []
        indices_list = []
        n_padded = 0

        for g in range(groups):
            start = g * L
            end = start + L
            windows_list.append(features[start:end])
            labels_list.append(self._label_window(labels[start:end]))
            indices_list.append(np.arange(start, end))

        if n_remaining > 0:
            if self.padding_strategy == "none":
                # drop remainder
                pass
            else:
                remaining_feats = features[-n_remaining:]
                remaining_labels = labels[-n_remaining:]
                padded_feats = self._pad_sequence(remaining_feats, is_label=False)
                padded_labels = self._pad_sequence(remaining_labels, is_label=True)
                windows_list.append(padded_feats)
                labels_list.append(self._label_window(padded_labels))
                indices_row = np.concatenate([np.arange(n_flows - n_remaining, n_flows), np.full(L - n_remaining, -1, dtype=int)])
                indices_list.append(indices_row)
                n_padded += 1

        if windows_list:
            windows = np.stack(windows_list, axis=0)
            window_labels = np.array(labels_list, dtype=int)
            indices_arr = np.stack(indices_list, axis=0) if return_indices else None
            diagnostics = {"n_windows_created": windows.shape[0], "n_padded_windows": n_padded}
            return windows, window_labels, diagnostics, indices_arr
        else:
            diagnostics = {"n_windows_created": 0, "n_padded_windows": 0}
            empty_windows = np.empty((0, L, n_features), dtype=features.dtype)
            empty_labels = np.empty((0,), dtype=labels.dtype)
            empty_indices = np.empty((0, L), dtype=int) if return_indices else None
            return empty_windows, empty_labels, diagnostics, empty_indices

    # -------------------------
    # helpers
    # -------------------------
    def _pad_sequence(self, seq: np.ndarray, *, is_label: bool) -> np.ndarray:
        """
        Pad a 1D label sequence or 2D feature sequence to window_length.
        For features: seq shape (m, n_features) -> padded shape (L, n_features)
        For labels: seq shape (m,) -> padded shape (L,)
        """
        m = seq.shape[0]
        L = self.window_length
        if m >= L:
            return seq[:L]

        pad_len = L - m
        if is_label:
            if self.padding_strategy == "zero":
                pad_val = self.pad_label
                padded = np.concatenate([seq, np.full(pad_len, pad_val, dtype=seq.dtype)], axis=0)
            elif self.padding_strategy == "repeat":
                if m == 0:
                    padded = np.full(L, self.pad_label, dtype=seq.dtype)
                else:
                    padded = np.concatenate([seq, np.tile(seq[-1], pad_len)], axis=0)
            else:
                raise ValueError("padding_strategy 'none' should have been caught earlier")
            return padded
        else:
            # features (2D)
            n_features = seq.shape[1] if seq.ndim == 2 and seq.shape[0] > 0 else (seq.shape[1] if seq.ndim == 2 else 0)
            if self.padding_strategy == "zero":
                pad_row = np.full((pad_len, seq.shape[1]), self.pad_feature_value, dtype=seq.dtype)
                padded = np.vstack([seq, pad_row])
            elif self.padding_strategy == "repeat":
                if seq.shape[0] == 0:
                    pad_row = np.full((L, n_features), self.pad_feature_value, dtype=seq.dtype)
                    padded = pad_row
                else:
                    last_row = seq[-1:]
                    tiled = np.repeat(last_row, pad_len, axis=0)
                    padded = np.vstack([seq, tiled])
            else:
                raise ValueError("padding_strategy 'none' should have been caught earlier")
            return padded

    def _label_window(self, window_labels: np.ndarray) -> int:
        """
        Compute the label for a window according to strategy.
        Assumes labels are integers where 0 is benign by convention (adjust if needed).
        """
        if window_labels.size == 0:
            return int(self.pad_label)

        if self.label_strategy == "any_malicious":
            return int(np.any(window_labels != 0))
        if self.label_strategy == "majority":
            unique, counts = np.unique(window_labels, return_counts=True)
            max_count = counts.max()
            candidates = unique[counts == max_count]
            return int(np.min(candidates))  # tie-breaker: smallest class id
        if self.label_strategy == "first":
            return int(window_labels[0])
        if self.label_strategy == "last":
            return int(window_labels[-1])

        # fallback (shouldn't happen)
        return int(window_labels[0])


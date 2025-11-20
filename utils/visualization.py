"""
Research-grade visualization utilities for IDS project.

Improvements / fixes:
- Import Figure from matplotlib.figure to satisfy static type checkers.
- Use tuple for tight_layout rect parameter (not list).
- Use typed helper function for DataFrame.apply to keep Pylance happy.
- Convert pandas Series -> numpy arrays explicitly before plotting.
- Robust input validation, helpful errors, and returning Figure objects for downstream usage.
- headless-friendly by default (show=False); can be overridden.
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

logger = logging.getLogger(__name__)

# sensible defaults
DEFAULT_DPI = 300
DEFAULT_STYLE = "whitegrid"


def _safe_mkdir(path: Optional[Union[str, Path]]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: Figure, save_path: Optional[Union[str, Path]], dpi: int = DEFAULT_DPI, close: bool = True) -> None:
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    logger.info(f"Saved figure to {p}")
    if close:
        plt.close(fig)


def _format_number_for_label(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    v = float(value)
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.0f}k"
    # generic formatting
    return f"{v:.3f}"


def _pad_series(a: List[float], length: int) -> np.ndarray:
    out = np.full(length, np.nan, dtype=float)
    if a is None:
        return out
    arr = np.asarray(a, dtype=float)
    out[: min(len(arr), length)] = arr[: min(len(arr), length)]
    return out


def plot_training_curves(
    metrics_history: Dict[str, Dict[str, List[float]]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    dpi: int = DEFAULT_DPI,
    show: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    palette: Optional[Union[str, List[str]]] = "muted",
) -> Figure:
    """
    Plot training/validation curves.

    metrics_history: {'loss': {'train':[...], 'val':[...]} , 'accuracy': {...}, ...}
    Returns: matplotlib.figure.Figure
    """
    if not metrics_history:
        raise ValueError("metrics_history cannot be empty")

    metric_names = list(metrics_history.keys())
    n = len(metric_names)
    figsize = figsize or (6 * n, 4)

    with sns.axes_style(DEFAULT_STYLE), sns.color_palette(palette):
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, metric in zip(axes, metric_names):
            hist = metrics_history.get(metric, {})
            train = hist.get("train", []) or []
            val = hist.get("val", []) or []

            max_len = max(len(train), len(val), 1)
            x = np.arange(1, max_len + 1)

            y_train = _pad_series(train, max_len)
            y_val = _pad_series(val, max_len)

            ax.plot(x, y_train, marker="o", markersize=3, label="train")
            if np.any(~np.isnan(y_val)):
                ax.plot(x, y_val, marker="s", markersize=3, label="val")

            ax.set_xlabel("Epoch")
            ax.set_title(metric.replace("_", " ").title())
            if metric.lower().endswith("loss"):
                ax.set_ylabel("Loss")
            elif metric.lower().endswith("acc") or metric.lower().startswith("accuracy"):
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylabel(metric.replace("_", " ").title())

            ax.grid(alpha=0.25)
            ax.legend()
            # nice margins
            try:
                vals = np.concatenate([y_train[~np.isnan(y_train)], y_val[~np.isnan(y_val)]])
                if vals.size > 0:
                    ymin, ymax = float(vals.min()), float(vals.max())
                    margin = (ymax - ymin) * 0.05 if ymax != ymin else 0.1 * (abs(ymax) + 1)
                    ax.set_ylim(ymin - margin, ymax + margin)
            except Exception:
                pass

        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))  # <-- tuple, not list

        _save_figure(fig, save_path, dpi=dpi, close=not show)
        if show:
            plt.show()
        return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[float]]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> Figure:
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2:
        raise ValueError("cm must be 2D array-like (rows=true labels, cols=predictions)")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)
        display = cm_normalized
        fmt = ".2%"
        cbar_label = "Proportion"
    else:
        display = cm
        fmt = "d"
        cbar_label = "Count"

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.6), max(5, n_classes * 0.6)))
        sns.heatmap(
            display,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": cbar_label},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

        _save_figure(fig, save_path, dpi=dpi, close=not show)
        if show:
            plt.show()
        return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Model Comparison",
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> Figure:
    if not results:
        raise ValueError("results must be non-empty")

    df = pd.DataFrame(results).T  # models x metrics

    default_metrics = ["accuracy", "f1_macro", "parameters", "inference_time_mean_ms"]
    metrics_to_try = metrics_to_plot or default_metrics

    # If inference_time column exists but contains dicts, attempt to extract mean_ms
    if "inference_time" in df.columns:
        def _extract_mean(v: Any) -> float:
            if isinstance(v, dict):
                return float(v.get("mean_ms", np.nan))
            if pd.api.types.is_scalar(v) and not isinstance(v, (str, bytes)):
                try:
                    return float(v)
                except Exception:
                    return float(np.nan)
            return float(np.nan)
        df["inference_time_mean_ms"] = df["inference_time"].apply(_extract_mean)

    # choose available metrics
    available_metrics = [m for m in metrics_to_try if m in df.columns]
    if not available_metrics:
        available_metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        if not available_metrics:
            raise ValueError("No numeric metrics available to plot.")

    num_metrics = len(available_metrics)
    figsize = (5 * num_metrics, 5)
    with sns.axes_style(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        models = df.index.tolist()
        palette = sns.color_palette("husl", len(models))

        for ax, metric in zip(axes, available_metrics):
            # convert to numpy array to satisfy type checkers
            values = df[metric].astype(float).to_numpy()
            bars = ax.bar(models, values, color=palette, edgecolor="k", linewidth=0.5)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.2)

            for bar, val in zip(bars, values):
                ax.annotate(
                    _format_number_for_label(float(val)) if not np.isnan(val) else "NaN",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        _save_figure(fig, save_path, dpi=dpi, close=not show)
        if show:
            plt.show()
        return fig


def plot_parameter_efficiency(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Parameter Efficiency: Accuracy vs Model Size",
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> Figure:
    if not results:
        raise ValueError("results must be non-empty")

    df = pd.DataFrame(results).T
    if "accuracy" not in df.columns or "parameters" not in df.columns:
        raise ValueError("results must contain 'accuracy' and 'parameters' for each model")

    x = df["parameters"].astype(float).to_numpy()
    y = df["accuracy"].astype(float).to_numpy()
    models = df.index.tolist()

    with sns.axes_style(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(x, y, s=150, alpha=0.85, edgecolors="k")
        for i, model in enumerate(models):
            ax.annotate(model, (x[i], y[i]), xytext=(6, 6), textcoords="offset points", fontsize=9)

        ax.set_xscale("log")
        ax.set_xlabel("Number of Parameters (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        _save_figure(fig, save_path, dpi=dpi, close=not show)
        if show:
            plt.show()
        return fig


def plot_cross_dataset_performance(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Cross-Dataset Performance",
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> Figure:
    if not results:
        raise ValueError("results required")

    df = pd.DataFrame(results).T
    if df.empty:
        raise ValueError("Empty results DataFrame")

    datasets = list(df.columns)
    models = list(df.index)
    n_models = len(models)
    n_dsets = len(datasets)

    x = np.arange(n_models)
    total_width = 0.8
    bar_width = total_width / max(1, n_dsets)
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_dsets)

    with sns.axes_style(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("tab10", n_colors=n_dsets)
        for i, dset in enumerate(datasets):
            vals = df[dset].astype(float).to_numpy()
            ax.bar(x + offsets[i], vals, width=bar_width, label=dset, color=palette[i], edgecolor="k", alpha=0.9)
            for j, val in enumerate(vals):
                if not np.isnan(val):
                    ax.text(x[j] + offsets[i], val + 0.01 * max(1.0, abs(val)), f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        _save_figure(fig, save_path, dpi=dpi, close=not show)
        if show:
            plt.show()
        return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    save_path: Optional[Union[str, Path]] = None,
    top_k: Optional[int] = 10,
    title: str = "Top Feature Importance",
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> Figure:
    if not feature_names or not importances or len(feature_names) != len(importances):
        raise ValueError("feature_names and importances must be non-empty and of the same length")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    if top_k:
        df = df.head(top_k)

    with sns.axes_style(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
        sns.barplot(x="importance", y="feature", data=df, ax=ax, palette="viridis")
        ax.set_xlabel("Importance")
        ax.set_title(title)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        _save_figure(fig, save_path, dpi=dpi, close=not show)
        if show:
            plt.show()
        return fig


def export_plot_data(data: Any, path: Union[str, Path], fmt: str = "json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    elif fmt == "csv":
        try:
            pd.DataFrame(data).to_csv(path, index=True)
        except Exception:
            with open(path, "w") as f:
                f.write(str(data))
    else:
        raise ValueError("Unsupported export fmt. Use 'json' or 'csv'.")


__all__ = [
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_parameter_efficiency",
    "plot_cross_dataset_performance",
    "plot_feature_importance",
    "export_plot_data",
]

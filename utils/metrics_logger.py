#!/usr/bin/env python3
"""
Per-Class Metrics Logger for Training

Tracks precision, recall, F1 for all 10 attack classes every epoch.
Critical for honest rare-class reporting.

Usage:
    from src.utils.metrics_logger import MetricsLogger
    
    logger = MetricsLogger(output_dir='experiments/run1', class_names=CLASS_NAMES)
    
    # During training
    logger.log_epoch(epoch, y_true, y_pred, phase='train')
    logger.log_epoch(epoch, y_val_true, y_val_pred, phase='val')
    
    # After training
    logger.save()
    logger.plot_curves()
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

# Optional imports with proper handling
PANDAS_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import pandas as _pd
    PANDAS_AVAILABLE = True
except ImportError:
    _pd = None  # type: ignore

try:
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    _plt = None  # type: ignore
    _Rectangle = None  # type: ignore


# CIC-IDS-2017 attack classes
CLASS_NAMES = [
    'BENIGN',
    'Bot',
    'DDoS',
    'DoS GoldenEye',
    'DoS Hulk',
    'DoS Slowhttptest',
    'DoS slowloris',
    'FTP-Patator',
    'PortScan',
    'SSH-Patator'
]

# Classes requiring special attention
CRITICAL_CLASSES = ['DDoS', 'PortScan']  # Must maintain >98% recall
RARE_CLASSES = ['Bot', 'SSH-Patator', 'DoS GoldenEye']  # Report honestly


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 for each class.
    
    Args:
        y_true: Ground truth labels (integers)
        y_pred: Predicted labels (integers)
        class_names: List of class names
        
    Returns:
        Dict mapping class_name -> {precision, recall, f1, support}
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    labels = list(range(len(class_names)))
    result = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Unpack - result is tuple of arrays
    p_arr = np.asarray(result[0])
    r_arr = np.asarray(result[1])
    f1_arr = np.asarray(result[2])
    support_arr = np.asarray(result[3])
    
    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            'precision': float(p_arr[i]),
            'recall': float(r_arr[i]),
            'f1': float(f1_arr[i]),
            'support': int(support_arr[i])
        }
    
    return metrics


class MetricsLogger:
    """
    Log per-class metrics every epoch for later analysis.
    
    Tracks all 10 attack classes to ensure honest rare-class reporting.
    """
    
    def __init__(self, output_dir: Union[str, Path], 
                 class_names: Optional[List[str]] = None):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save logs
            class_names: List of class names (default: CIC-IDS-2017 classes)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names or CLASS_NAMES
        self.history: List[Dict] = []
        
        # Track best metrics
        self.best_macro_f1 = 0.0
        self.best_epoch = 0
        self.best_rare_recall = {}  # Track best for each rare class
        
    def log_epoch(self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray,
                  phase: str = 'val', loss: Optional[float] = None) -> Dict:
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Epoch number
            y_true: Ground truth labels
            y_pred: Predicted labels
            phase: 'train' or 'val'
            loss: Optional loss value
            
        Returns:
            Dictionary of computed metrics
        """
        per_class = compute_per_class_metrics(y_true, y_pred, self.class_names)
        
        # Compute aggregates
        macro_f1 = np.mean([m['f1'] for m in per_class.values()])
        weighted_f1 = np.average(
            [m['f1'] for m in per_class.values()],
            weights=[m['support'] for m in per_class.values()]
        )
        
        # Build row
        row = {
            'epoch': epoch,
            'phase': phase,
            'loss': loss,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
        }
        
        # Add per-class metrics
        for name, metrics in per_class.items():
            row[f'{name}_precision'] = metrics['precision']
            row[f'{name}_recall'] = metrics['recall']
            row[f'{name}_f1'] = metrics['f1']
            row[f'{name}_support'] = metrics['support']
        
        self.history.append(row)
        
        # Update best tracking (validation only)
        if phase == 'val' and macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.best_epoch = epoch
        
        # Track rare class bests
        for cls in RARE_CLASSES:
            recall = per_class.get(cls, {}).get('recall', 0)
            if cls not in self.best_rare_recall or recall > self.best_rare_recall[cls]:
                self.best_rare_recall[cls] = recall
        
        return row
    
    def get_latest(self, phase: str = 'val') -> Optional[Dict]:
        """Get most recent metrics for a phase."""
        for row in reversed(self.history):
            if row['phase'] == phase:
                return row
        return None
    
    def get_epoch(self, epoch: int, phase: str = 'val') -> Optional[Dict]:
        """Get metrics for a specific epoch."""
        for row in self.history:
            if row['epoch'] == epoch and row['phase'] == phase:
                return row
        return None
    
    def save(self, filename: str = 'per_class_metrics.csv'):
        """Save metrics history to CSV."""
        output_path = self.output_dir / filename
        
        if PANDAS_AVAILABLE and _pd is not None:
            df = _pd.DataFrame(self.history)
            df.to_csv(output_path, index=False)
        else:
            # Fallback: save as JSON
            output_path = output_path.with_suffix('.json')
            with open(output_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"💾 Saved metrics to {output_path}")
        
        # Also save summary
        summary = self.get_summary()
        summary_path = self.output_dir / 'metrics_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved summary to {summary_path}")
    
    def get_summary(self) -> Dict:
        """Get summary of training."""
        summary = {
            'best_epoch': self.best_epoch,
            'best_macro_f1': self.best_macro_f1,
            'best_rare_class_recall': self.best_rare_recall,
            'total_epochs': max(r['epoch'] for r in self.history) if self.history else 0,
        }
        
        # Get final metrics
        final_val = self.get_latest('val')
        if final_val:
            summary['final_metrics'] = {
                'macro_f1': final_val['macro_f1'],
                'critical_classes': {
                    cls: final_val.get(f'{cls}_recall', 0) 
                    for cls in CRITICAL_CLASSES
                },
                'rare_classes': {
                    cls: final_val.get(f'{cls}_recall', 0) 
                    for cls in RARE_CLASSES
                }
            }
        
        return summary
    
    def print_epoch_summary(self, epoch: int, phase: str = 'val'):
        """Print formatted summary for an epoch."""
        row = self.get_epoch(epoch, phase)
        if not row:
            print(f"No data for epoch {epoch} ({phase})")
            return
        
        print(f"\n{'─' * 50}")
        print(f"Epoch {epoch} ({phase})")
        print(f"{'─' * 50}")
        
        print(f"Macro F1: {row['macro_f1']:.4f}")
        if row.get('loss') is not None:
            print(f"Loss: {row['loss']:.4f}")
        
        print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 62)
        
        for cls in self.class_names:
            p = row.get(f'{cls}_precision', 0)
            r = row.get(f'{cls}_recall', 0)
            f1 = row.get(f'{cls}_f1', 0)
            s = row.get(f'{cls}_support', 0)
            
            # Highlight rare/critical classes
            marker = ""
            if cls in RARE_CLASSES:
                marker = " ⚠️" if r < 0.6 else " ✅"
            elif cls in CRITICAL_CLASSES:
                marker = " ❌" if r < 0.98 else " ✅"
            
            print(f"{cls:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {s:>10}{marker}")
    
    def plot_curves(self, save: bool = True):
        """Plot per-class recall curves over epochs."""
        if not MATPLOTLIB_AVAILABLE or _plt is None:
            print("⚠️  matplotlib not available, skipping plots")
            return
        
        if not self.history:
            print("No data to plot")
            return
        
        # Extract validation data
        val_data = [r for r in self.history if r['phase'] == 'val']
        if not val_data:
            print("No validation data to plot")
            return
        
        epochs = [r['epoch'] for r in val_data]
        
        # Plot 1: All classes recall
        fig, axes = _plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: Critical + Rare classes
        ax1 = axes[0]
        for cls in CRITICAL_CLASSES + RARE_CLASSES:
            recalls = [r.get(f'{cls}_recall', 0) for r in val_data]
            style = '-' if cls in CRITICAL_CLASSES else '--'
            ax1.plot(epochs, recalls, style, label=cls, linewidth=2)
        
        ax1.axhline(y=0.98, color='red', linestyle=':', alpha=0.5, label='Critical threshold')
        ax1.axhline(y=0.60, color='orange', linestyle=':', alpha=0.5, label='Rare target')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Recall')
        ax1.set_title('Critical & Rare Class Recall')
        ax1.legend(loc='lower right')
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3)
        
        # Right: Macro F1
        ax2 = axes[1]
        macro_f1s = [r['macro_f1'] for r in val_data]
        ax2.plot(epochs, macro_f1s, 'b-', linewidth=2, label='Macro F1')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Macro F1 Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        _plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'training_curves.png'
            _plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved training curves to {output_path}")
        
        _plt.close()
        
        # Plot 2: Per-class heatmap (final epoch)
        self._plot_confusion_heatmap(val_data[-1])
    
    def _plot_confusion_heatmap(self, row: Dict[str, Any]):
        """Plot per-class metrics as heatmap."""
        if not MATPLOTLIB_AVAILABLE or _plt is None or _Rectangle is None:
            return
        
        metrics = ['precision', 'recall', 'f1']
        data = np.zeros((len(self.class_names), len(metrics)))
        
        for i, cls in enumerate(self.class_names):
            for j, metric in enumerate(metrics):
                data[i, j] = row.get(f'{cls}_{metric}', 0)
        
        fig, ax = _plt.subplots(figsize=(8, 10))
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(['Precision', 'Recall', 'F1'])
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        
        # Add text annotations
        for i in range(len(self.class_names)):
            for j in range(len(metrics)):
                val = data[i, j]
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
        
        # Highlight rare classes
        for i, cls in enumerate(self.class_names):
            if cls in RARE_CLASSES:
                ax.add_patch(_Rectangle((-.5, i-.5), 3, 1, 
                                        fill=False, edgecolor='orange', linewidth=2))
        
        _plt.colorbar(im, ax=ax, label='Score')
        ax.set_title(f'Per-Class Metrics (Epoch {row["epoch"]})')
        
        output_path = self.output_dir / 'per_class_heatmap.png'
        _plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved heatmap to {output_path}")
        _plt.close()


def check_acceptance_criteria(metrics: Dict) -> Dict[str, bool]:
    """
    Check if metrics meet acceptance criteria.
    
    Returns dict of criterion -> passed (bool)
    """
    criteria = {}
    
    # DDoS recall >= 98%
    ddos_recall = metrics.get('DDoS_recall', 0)
    criteria['DDoS_recall >= 98%'] = ddos_recall >= 0.98
    
    # PortScan recall >= 98%
    portscan_recall = metrics.get('PortScan_recall', 0)
    criteria['PortScan_recall >= 98%'] = portscan_recall >= 0.98
    
    # Macro F1 (check if within target)
    macro_f1 = metrics.get('macro_f1', 0)
    criteria['Macro_F1 >= 91%'] = macro_f1 >= 0.91
    
    # Rare class targets (softer)
    for cls in RARE_CLASSES:
        recall = metrics.get(f'{cls}_recall', 0)
        criteria[f'{cls}_recall >= 50%'] = recall >= 0.50
    
    return criteria


if __name__ == '__main__':
    # Demo usage
    print("MetricsLogger Demo")
    print("=" * 50)
    
    # Create logger
    logger = MetricsLogger(output_dir='demo_logs', class_names=CLASS_NAMES)
    
    # Simulate some epochs
    np.random.seed(42)
    n_classes = len(CLASS_NAMES)
    
    for epoch in range(10):
        # Fake predictions (improving over time)
        n_samples = 1000
        y_true = np.random.randint(0, n_classes, n_samples)
        
        # Predictions get better each epoch
        noise = np.random.random(n_samples) > (0.3 + epoch * 0.05)
        y_pred = np.where(noise, y_true, np.random.randint(0, n_classes, n_samples))
        
        logger.log_epoch(epoch, y_true, y_pred, phase='val', loss=1.0 - epoch * 0.08)
    
    # Print summary
    logger.print_epoch_summary(9, 'val')
    
    # Check criteria
    final = logger.get_latest('val')
    if final:
        criteria = check_acceptance_criteria(final)
        print("\n📋 Acceptance Criteria:")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")
    
    # Save
    logger.save()
    logger.plot_curves()
    
    print("\n✅ Demo complete!")

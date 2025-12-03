"""
Phase 2 Compression Pipeline - Common Utilities
================================================

Shared utilities for reproducibility, logging, metrics, and data loading.
"""

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    roc_auc_score, precision_recall_fscore_support
)

# ============ REPRODUCIBILITY ============
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()[:8]
    except:
        return "unknown"

def get_env_info() -> Dict[str, Any]:
    """Get environment information for logging."""
    import platform
    
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'platform': platform.system(),
        'hostname': platform.node(),
    }
    
    if torch.cuda.is_available():
        env_info['cuda_version'] = torch.version.cuda  # type: ignore[attr-defined]
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return env_info

# ============ METRICS ============
@dataclass
class Metrics:
    """Comprehensive metrics container."""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    detection_rate: float = 0.0  # Recall for attack class
    false_alarm_rate: float = 0.0  # FPR for benign class
    precision: float = 0.0
    auc: float = 0.0
    
    # Per-class metrics
    f1_per_class: Optional[Dict[str, float]] = None
    recall_per_class: Optional[Dict[str, float]] = None
    precision_per_class: Optional[Dict[str, float]] = None
    
    # Per-attack-type metrics (NEW: critical for Phase 2)
    per_attack_detection_rate: Optional[Dict[str, float]] = None
    per_attack_precision: Optional[Dict[str, float]] = None
    per_attack_f1: Optional[Dict[str, float]] = None
    per_attack_support: Optional[Dict[str, int]] = None
    
    # Critical class metrics
    ddos_recall: float = 0.0
    portscan_recall: float = 0.0
    bot_recall: float = 0.0
    ssh_patator_recall: float = 0.0
    ftp_patator_recall: float = 0.0
    
    # Confusion matrix
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Handle None values
        for k, v in d.items():
            if v is None:
                d[k] = {}
        return d
    
    def passes_critical_recall(self, threshold: float = 98.0) -> bool:
        """Check if critical class recalls pass threshold."""
        return (self.ddos_recall >= threshold and 
                self.portscan_recall >= threshold)
    
    def passes_far(self, threshold: float = 1.5) -> bool:
        """Check if FAR is acceptable."""
        return self.false_alarm_rate <= threshold
    
    def get_attack_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of per-attack metrics."""
        summary = {}
        if self.per_attack_detection_rate:
            for attack, dr in self.per_attack_detection_rate.items():
                summary[attack] = {
                    'detection_rate': dr,
                    'precision': self.per_attack_precision.get(attack, 0.0) if self.per_attack_precision else 0.0,
                    'f1': self.per_attack_f1.get(attack, 0.0) if self.per_attack_f1 else 0.0,
                    'support': self.per_attack_support.get(attack, 0) if self.per_attack_support else 0
                }
        return summary
    
    def check_degradation(self, baseline: 'Metrics', threshold: float = 5.0) -> Dict[str, float]:
        """Check which attacks degraded more than threshold % from baseline."""
        degraded = {}
        if self.per_attack_detection_rate and baseline.per_attack_detection_rate:
            for attack, dr in self.per_attack_detection_rate.items():
                baseline_dr = baseline.per_attack_detection_rate.get(attack, 0.0)
                drop = baseline_dr - dr
                if drop > threshold:
                    degraded[attack] = drop
        return degraded

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    attack_types: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Metrics:
    """Compute comprehensive metrics including per-attack-type tracking."""
    
    metrics = Metrics()
    
    # Basic metrics
    metrics.accuracy = float(accuracy_score(y_true, y_pred) * 100)
    metrics.f1_macro = float(f1_score(y_true, y_pred, average='macro') * 100)
    metrics.f1_weighted = float(f1_score(y_true, y_pred, average='weighted') * 100)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics.tn, metrics.fp, metrics.fn, metrics.tp = cm.ravel()
        
        # Detection rate (recall for attacks)
        if (metrics.tp + metrics.fn) > 0:
            metrics.detection_rate = metrics.tp / (metrics.tp + metrics.fn) * 100
        
        # False alarm rate
        if (metrics.fp + metrics.tn) > 0:
            metrics.false_alarm_rate = metrics.fp / (metrics.fp + metrics.tn) * 100
        
        # Precision
        if (metrics.tp + metrics.fp) > 0:
            metrics.precision = metrics.tp / (metrics.tp + metrics.fp) * 100
    
    # AUC
    if y_prob is not None:
        try:
            metrics.auc = float(roc_auc_score(y_true, y_prob) * 100)
        except:
            metrics.auc = 0.0
    
    # Per-class F1
    if class_names is not None:
        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(len(class_names)))
        )
        # f1_arr and recall_arr are ndarray when average=None
        f1_arr = np.atleast_1d(f1_arr)
        recall_arr = np.atleast_1d(recall_arr)
        precision_arr = np.atleast_1d(precision_arr)
        metrics.f1_per_class = {name: float(f1_arr[i]*100) for i, name in enumerate(class_names)}
        metrics.recall_per_class = {name: float(recall_arr[i]*100) for i, name in enumerate(class_names)}
        metrics.precision_per_class = {name: float(precision_arr[i]*100) for i, name in enumerate(class_names)}
    
    # ============ PER-ATTACK-TYPE METRICS (CRITICAL FOR PHASE 2) ============
    if attack_types is not None:
        unique_attacks = np.unique(attack_types)
        
        # Initialize per-attack dictionaries
        metrics.per_attack_detection_rate = {}
        metrics.per_attack_precision = {}
        metrics.per_attack_f1 = {}
        metrics.per_attack_support = {}
        
        for attack in unique_attacks:
            if attack == 'BENIGN':
                continue  # Skip benign class
            
            mask = attack_types == attack
            support = int(mask.sum())
            metrics.per_attack_support[attack] = support
            
            if support == 0:
                metrics.per_attack_detection_rate[attack] = 0.0
                metrics.per_attack_precision[attack] = 0.0
                metrics.per_attack_f1[attack] = 0.0
                continue
            
            # Detection rate (recall): of all attack samples, how many were predicted as attack (1)
            true_positives = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
            total_attack_samples = (y_true[mask] == 1).sum()
            dr = (true_positives / total_attack_samples * 100) if total_attack_samples > 0 else 0.0
            metrics.per_attack_detection_rate[attack] = float(dr)
            
            # Precision: of samples predicted as attack, how many were this attack type
            predicted_positive = (y_pred[mask] == 1).sum()
            prec = (true_positives / predicted_positive * 100) if predicted_positive > 0 else 0.0
            metrics.per_attack_precision[attack] = float(prec)
            
            # F1
            if dr + prec > 0:
                f1 = 2 * dr * prec / (dr + prec)
            else:
                f1 = 0.0
            metrics.per_attack_f1[attack] = float(f1)
            
            # Update critical attack recalls
            attack_lower = attack.lower().replace('-', '_').replace(' ', '_')
            if 'ddos' in attack_lower:
                metrics.ddos_recall = float(dr)
            elif 'portscan' in attack_lower:
                metrics.portscan_recall = float(dr)
            elif 'bot' == attack_lower:
                metrics.bot_recall = float(dr)
            elif 'ssh' in attack_lower and 'patator' in attack_lower:
                metrics.ssh_patator_recall = float(dr)
            elif 'ftp' in attack_lower and 'patator' in attack_lower:
                metrics.ftp_patator_recall = float(dr)
    
    return metrics


def compute_per_attack_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: np.ndarray,
    stage_name: str = "unknown"
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-attack-type metrics for tracking degradation across compression stages.
    
    Returns:
        Dict mapping attack type -> {detection_rate, precision, f1, support}
    """
    results = {}
    unique_attacks = np.unique(attack_types)
    
    for attack in unique_attacks:
        if attack == 'BENIGN':
            continue
        
        mask = attack_types == attack
        support = int(mask.sum())
        
        if support == 0:
            results[attack] = {
                'detection_rate': 0.0,
                'precision': 0.0,
                'f1': 0.0,
                'support': 0,
                'stage': stage_name
            }
            continue
        
        # Metrics for this attack type
        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]
        
        tp = ((y_pred_subset == 1) & (y_true_subset == 1)).sum()
        fn = ((y_pred_subset == 0) & (y_true_subset == 1)).sum()
        fp = ((y_pred_subset == 1) & (y_true_subset == 0)).sum()
        
        dr = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * dr * prec / (dr + prec)) if (dr + prec) > 0 else 0.0
        
        results[attack] = {
            'detection_rate': float(dr),
            'precision': float(prec),
            'f1': float(f1),
            'support': support,
            'stage': stage_name
        }
    
    return results


def track_compression_degradation(
    stage_metrics: Dict[str, Dict[str, Dict[str, float]]],
    critical_attacks: List[str] = ['DDoS', 'PortScan', 'Bot', 'SSH-Patator'],
    max_drop: float = 5.0
) -> Dict[str, List[str]]:
    """
    Track per-attack degradation across compression stages.
    
    Args:
        stage_metrics: Dict[stage_name -> per_attack_metrics]
        critical_attacks: List of attack types to monitor closely
        max_drop: Maximum allowed detection rate drop (%)
    
    Returns:
        Dict with 'warnings' and 'critical' lists
    """
    alerts = {'warnings': [], 'critical': []}
    stages = list(stage_metrics.keys())
    
    if len(stages) < 2:
        return alerts
    
    baseline_stage = stages[0]
    baseline = stage_metrics[baseline_stage]
    
    for stage in stages[1:]:
        current = stage_metrics[stage]
        
        for attack in critical_attacks:
            if attack not in baseline or attack not in current:
                continue
            
            baseline_dr = baseline[attack]['detection_rate']
            current_dr = current[attack]['detection_rate']
            drop = baseline_dr - current_dr
            
            if drop > max_drop * 2:  # More than 2x threshold
                alerts['critical'].append(
                    f"CRITICAL: {attack} DR dropped {drop:.1f}% ({baseline_dr:.1f}% -> {current_dr:.1f}%) at {stage}"
                )
            elif drop > max_drop:
                alerts['warnings'].append(
                    f"WARNING: {attack} DR dropped {drop:.1f}% ({baseline_dr:.1f}% -> {current_dr:.1f}%) at {stage}"
                )
    
    return alerts

# ============ MODEL UTILITIES ============
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model: nn.Module) -> int:
    """Count all parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Metrics,
    path: Path,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
    }
    
    if config is not None:
        checkpoint['config'] = config
    if extra is not None:
        checkpoint.update(extra)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

# ============ DATA LOADING ============
def load_data(
    data_dir: Path,
    load_attack_types: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load preprocessed data."""
    
    X_train = np.load(data_dir / 'train/X.npy')
    y_train = np.load(data_dir / 'train/y.npy')
    X_val = np.load(data_dir / 'val/X.npy')
    y_val = np.load(data_dir / 'val/y.npy')
    X_test = np.load(data_dir / 'test/X.npy')
    y_test = np.load(data_dir / 'test/y.npy')
    
    attack_types_test: Optional[np.ndarray] = None
    if load_attack_types:
        attack_types_path = data_dir / 'test/attack_types.npy'
        if attack_types_path.exists():
            attack_types_test = np.load(attack_types_path, allow_pickle=True)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, attack_types_test

def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def get_balanced_sampler(y: np.ndarray) -> "torch.utils.data.WeightedRandomSampler":  # type: ignore[name-defined]
    """Create balanced sampler for imbalanced data."""
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    sample_weights = weights[y]
    return torch.utils.data.WeightedRandomSampler(  # type: ignore[attr-defined]
        weights=sample_weights.tolist(),
        num_samples=len(y),
        replacement=True
    )

def get_representative_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 1000
) -> np.ndarray:
    """Get balanced representative dataset for calibration."""
    # Balance across classes
    unique_classes = np.unique(y)
    samples_per_class = n_samples // len(unique_classes)
    
    indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(
            cls_indices, 
            min(samples_per_class, len(cls_indices)),
            replace=False
        )
        indices.extend(selected)
    
    np.random.shuffle(indices)
    return X[indices[:n_samples]]

# ============ FLOPS COMPUTATION ============
def compute_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Compute FLOPs using thop or manual counting."""
    try:
        from thop import profile  # type: ignore[import-not-found]
        dummy_input = torch.randn(1, *input_shape)
        result = profile(model, inputs=(dummy_input,), verbose=False)
        flops = result[0]
        return int(flops)
    except ImportError:
        # Manual estimation for Conv1d
        total_flops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                # FLOPs = 2 * kernel_size * in_channels * out_channels * output_length
                # Approximate output_length as input_length
                out_len = input_shape[0]  # window_size
                flops = 2 * module.kernel_size[0] * module.in_channels * module.out_channels * out_len
                total_flops += flops
            elif isinstance(module, nn.Linear):
                flops = 2 * module.in_features * module.out_features
                total_flops += flops
        return total_flops

# ============ LOGGING ============
class ExperimentLogger:
    """Logger for experiment tracking."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"
        
        # Initialize metrics CSV
        self._init_metrics_csv()
    
    def _init_metrics_csv(self):
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("seed,stage,epoch,accuracy,f1_macro,detection_rate,far,auc,ddos_recall,portscan_recall,timestamp\n")
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        print(log_line)
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def log_metrics(self, seed: int, stage: str, epoch: int, metrics: Metrics):
        timestamp = datetime.now().isoformat()
        with open(self.metrics_file, 'a') as f:
            f.write(f"{seed},{stage},{epoch},{metrics.accuracy:.2f},{metrics.f1_macro:.2f},"
                    f"{metrics.detection_rate:.2f},{metrics.false_alarm_rate:.2f},"
                    f"{metrics.auc:.2f},{metrics.ddos_recall:.2f},{metrics.portscan_recall:.2f},"
                    f"{timestamp}\n")
    
    def log_config(self, config: Dict):
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

# ============ MEMORY MANAGEMENT ============
def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def memory_status() -> Dict[str, float]:
    """Get memory status."""
    import psutil
    
    status = {
        'cpu_percent': psutil.virtual_memory().percent,
        'cpu_available_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    if torch.cuda.is_available():
        status['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
        status['gpu_cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
    
    return status

# ============ STATISTICAL UTILITIES ============
def compute_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and confidence interval."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    n = len(values)
    
    # z-score for confidence level
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    
    ci = float(z * std / np.sqrt(n))
    return mean, std, ci

def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples."""
    from scipy.stats import wilcoxon
    result = wilcoxon(a, b)
    return float(result.statistic), float(result.pvalue)  # type: ignore[union-attr]

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    pooled_std = float(np.sqrt((np.std(a)**2 + np.std(b)**2) / 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)

# ============ HARDWARE DETECTION ============
def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def is_pi() -> bool:
    """Check if running on Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except:
        return False


# ============ FOCAL LOSS ============
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============ KD LOSS ============
class KDLoss(nn.Module):
    """Knowledge Distillation loss combining soft and hard targets."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Soft loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Hard loss (cross entropy)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


# ============ METRICS TRACKER ============
class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or []
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
        }
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics: Dict[str, float] = {
            'accuracy': float(accuracy_score(y_true, y_pred) * 100),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100),
        }
        
        # Per-class metrics
        precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Binary metrics (assuming class 0 is benign)
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['detection_rate'] = float((tp / (tp + fn)) * 100) if (tp + fn) > 0 else 0.0
            metrics['false_alarm_rate'] = float((fp / (fp + tn)) * 100) if (fp + tn) > 0 else 0.0
        else:
            # Multi-class: DR = macro recall, FAR from benign class
            cm = confusion_matrix(y_true, y_pred)
            if isinstance(recall_arr, np.ndarray):
                metrics['detection_rate'] = float(np.mean(recall_arr[1:]) * 100)  # Exclude benign
            else:
                metrics['detection_rate'] = 0.0
            metrics['false_alarm_rate'] = float((cm[0].sum() - cm[0, 0]) / cm[0].sum() * 100) if cm[0].sum() > 0 else 0.0
        
        # Per-class recall for critical classes
        if isinstance(recall_arr, np.ndarray):
            for i, r in enumerate(recall_arr):
                if self.class_names and i < len(self.class_names):
                    name = self.class_names[i].lower().replace(' ', '_')
                    metrics[f'{name}_recall'] = float(r * 100)
                else:
                    metrics[f'class_{i}_recall'] = float(r * 100)
        
        # AUC if probabilities provided
        if y_prob is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr') * 100)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def update(self, **kwargs: float) -> None:
        """Update history with new values."""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best(self, metric: str = 'val_f1') -> Tuple[int, float]:
        """Get best epoch and value for a metric."""
        if metric not in self.history or not self.history[metric]:
            return 0, 0.0
        values = self.history[metric]
        best_idx = int(np.argmax(values))
        return best_idx, values[best_idx]


# ============ DATA LOADER FACTORY ============
class DataLoaderFactory:
    """Factory for creating data loaders from processed data."""
    
    def __init__(self, data_dir: Path, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self._load_data()
        
    def _load_data(self):
        """Load train/val/test data."""
        import json
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Load data
        self.X_train = np.load(self.data_dir / 'train' / 'X.npy')
        self.y_train = np.load(self.data_dir / 'train' / 'y.npy')
        self.X_val = np.load(self.data_dir / 'val' / 'X.npy')
        self.y_val = np.load(self.data_dir / 'val' / 'y.npy')
        self.X_test = np.load(self.data_dir / 'test' / 'X.npy')
        self.y_test = np.load(self.data_dir / 'test' / 'y.npy')
        
        self.n_features = self.X_train.shape[1]
        self.n_classes = len(np.unique(self.y_train))
        
    def get_train_loader(self, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
        """Get training data loader."""
        dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def get_val_loader(self, batch_size: int = 256) -> DataLoader:
        """Get validation data loader."""
        dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.LongTensor(self.y_val)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def get_test_loader(self, batch_size: int = 256) -> DataLoader:
        """Get test data loader."""
        dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.LongTensor(self.y_test)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def get_info(self) -> Dict[str, Any]:
        """Get data info."""
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
        }


# Alias for compatibility
set_reproducibility = set_seed

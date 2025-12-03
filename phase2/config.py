"""
Phase 2 Compression Pipeline - Configuration
=============================================

All hyperparameters, paths, and settings for reproducible compression.
Based on the 10/10 theoretical plan.

Targets (hard pass/fail):
- Critical recall (DDoS/PortScan): > 98%
- Overall F1-Macro drop vs baseline: ≤ 1% (production), ≤ 2% (research)
- False Alarm Rate (FAR): ≤ 1% preferred; ≤ 1.5% maximum
- Latency (batch=1, Pi 4): p50 ≤ 10 ms, p95 ≤ 40 ms
- Model size: final TFLite ≤ 0.05 MB (50 KB) ideal
- Energy per inference: target ≤ 15 mJ; ideal ≤ 10 mJ
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

# ============ PATHS ============
ROOT_DIR = Path(__file__).parent.parent
PHASE2_DIR = ROOT_DIR / "phase2"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data/processed/cic_ids_2017_v2"

# Phase 1 baseline path (teacher model)
PHASE1_CV_DIR = ROOT_DIR / "experiments/multitask_cv"
PHASE1_BEST_MODEL = PHASE1_CV_DIR / "fold1" / "best_model.pt"  # Best fold as teacher

# Create artifact subdirs
for stage in range(6):
    (ARTIFACTS_DIR / f"stage{stage}").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "teacher").mkdir(parents=True, exist_ok=True)

# ============ SEEDS FOR REPRODUCIBILITY ============
SEEDS = [0, 7, 21, 42, 101, 202, 303]  # N=7 for statistical rigor
QUICK_SEEDS = [0, 7, 42]  # N=3 for quick experiments
N_SEEDS = len(SEEDS)

# ============ STUDENT PARAMETER TARGETS ============
# Multiple sizes to test compression vs accuracy tradeoff
STUDENT_PARAM_TARGETS = [5000, 50000, 200000]  # 5K, 50K, 200K
BASELINE_PARAM_TARGET = 14000  # Matches Phase 1 teacher (13,902 params)

# ============ HARDWARE TARGETS ============
@dataclass
class HardwareTargets:
    """Target metrics for Pi 4 deployment."""
    # Latency targets (ms)
    latency_p50_max: float = 10.0
    latency_p50_ideal: float = 7.0
    latency_p95_max: float = 40.0
    
    # Energy targets (mJ per inference)
    energy_max: float = 15.0
    energy_ideal: float = 10.0
    
    # Model size targets (bytes)
    model_size_ideal: int = 50 * 1024      # 50 KB
    model_size_max: int = 2 * 1024 * 1024  # 2 MB (OTA limit)
    
    # Memory (bytes)
    memory_peak_max: int = 64 * 1024 * 1024  # 64 MB

HARDWARE_TARGETS = HardwareTargets()

# ============ ACCURACY TARGETS ============
@dataclass
class AccuracyTargets:
    """Target metrics for accuracy preservation."""
    # HARD CONSTRAINTS - must be met
    critical_recall_min: float = 98.0  # DDoS/PortScan recall
    f1_drop_max: float = 2.0           # ≤ 2% absolute drop acceptable
    far_max: float = 1.5               # False Alarm Rate max
    
    # Soft targets (preferred)
    f1_drop_preferred: float = 1.0     # Production target
    far_preferred: float = 1.0
    
    # Per-stage acceptable drops
    kd_improvement_min: float = 0.3      # KD should improve by ≥0.3%
    prune_immediate_drop_max: float = 3.0  # Post-prune before KD-FT
    kd_ft_recovery_max: float = 1.0       # After KD-FT vs Stage 1
    qat_drop_max: float = 1.0             # QAT vs Stage 3
    
    # Phase 1 baseline metrics (from 5-fold CV)
    baseline_binary_acc: float = 87.4    # ±0.7%
    baseline_benign_recall: float = 96.3 # ±1.0%
    baseline_portscan_recall: float = 99.9  # ±0.0%
    baseline_ddos_recall: float = 97.2   # ±0.5%

ACCURACY_TARGETS = AccuracyTargets()

# ============ CRITICAL CLASSES ============
CRITICAL_CLASSES = ['DDoS', 'PortScan']  # Must maintain >98% recall

# ============ MODEL CONFIGURATIONS ============
@dataclass
class StudentConfig:
    """DS-1D-CNN Student model configuration."""
    # Architecture
    stem_channels: int = 64
    stage1_channels: int = 128
    stage2_channels: int = 128
    stage3_channels: int = 256
    classifier_hidden: int = 128
    dropout: float = 0.3
    
    # Input shape
    window_size: int = 15
    n_features: int = 65  # Will be set from data
    n_classes: int = 2
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

@dataclass  
class TeacherConfig:
    """Teacher model configuration (3-5x student params)."""
    # Architecture (larger than student)
    stem_channels: int = 128
    stage1_channels: int = 256
    stage2_channels: int = 256
    stage3_channels: int = 512
    classifier_hidden: int = 256
    dropout: float = 0.2
    
    # Additional capacity
    use_multi_head_attention: bool = True
    n_attention_heads: int = 4
    
    # Input shape (same as student)
    window_size: int = 15
    n_features: int = 65
    n_classes: int = 2
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

STUDENT_CONFIG = StudentConfig()
TEACHER_CONFIG = TeacherConfig()

# ============ TRAINING CONFIGURATIONS ============
@dataclass
class BaselineTrainingConfig:
    """Stage 0: Baseline student training."""
    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # or "plateau"
    early_stop_patience: int = 10
    focal_loss_gamma: float = 2.0
    
    # Quick run for validation
    quick_epochs: int = 25

@dataclass
class KDTrainingConfig:
    """Stage 1 & 3: Knowledge Distillation configuration."""
    temperature: float = 4.0  # Tune: [2, 4, 8]
    alpha: float = 0.5        # Tune: [0.3, 0.5, 0.7]
    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    
    # KD fine-tune after prune (Stage 3)
    ft_epochs: int = 50
    ft_lr: float = 1e-4
    
    # Temperature/alpha sweep for tuning
    temperature_sweep: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    alpha_sweep: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])

@dataclass
class PruningConfig:
    """Stage 2: Structured pruning configuration."""
    # Sensitivity analysis
    sensitivity_prune_pcts: List[int] = field(
        default_factory=lambda: [10, 20, 30, 40, 50, 60, 70]
    )
    sensitivity_finetune_epochs: int = 3
    
    # Candidate schedules (uniform)
    uniform_schedules: List[int] = field(
        default_factory=lambda: [30, 50, 70]
    )
    
    # Non-uniform schedule template (layer: prune_pct)
    # Will be filled based on sensitivity analysis
    nonuniform_schedules: List[Dict[str, int]] = field(default_factory=list)
    
    # Pruning method
    method: str = "l1"  # L1-norm based filter pruning

@dataclass
class QATConfig:
    """Stage 4: Quantization-Aware Training configuration."""
    epochs: int = 30
    lr: float = 1e-5
    batch_size: int = 128
    
    # Calibration
    calibration_samples: int = 5000  # Balanced across classes
    
    # Quantization settings
    quantize_type: str = "int8"  # or "float16" fallback
    per_channel: bool = True

@dataclass
class ConversionConfig:
    """Stage 5: TFLite conversion configuration."""
    # Optimization flags
    optimize: bool = True
    use_xnnpack: bool = True
    
    # Representative dataset
    representative_samples: int = 1000
    
    # Fallback options
    allow_float16_fallback: bool = True

BASELINE_CONFIG = BaselineTrainingConfig()
KD_CONFIG = KDTrainingConfig()
PRUNING_CONFIG = PruningConfig()
QAT_CONFIG = QATConfig()
CONVERSION_CONFIG = ConversionConfig()

# ============ BENCHMARKING CONFIGURATION ============
@dataclass
class BenchmarkConfig:
    """Pi 4 benchmarking configuration."""
    # Inference runs
    warmup_runs: int = 200
    benchmark_runs: int = 1000
    batch_size: int = 1
    
    # Energy measurement (INA219)
    ina219_sample_rate_hz: int = 100
    idle_measurement_seconds: int = 60
    
    # Memory sampling
    memory_sample_interval: float = 0.1  # seconds
    
    # Thermal limits
    max_cpu_temp_c: int = 75
    
    # Percentiles to report
    latency_percentiles: List[int] = field(
        default_factory=lambda: [50, 75, 90, 95, 99]
    )

BENCHMARK_CONFIG = BenchmarkConfig()

# ============ STATISTICAL ANALYSIS ============
@dataclass
class StatisticalConfig:
    """Statistical analysis configuration."""
    confidence_level: float = 0.95
    ci_multiplier: float = 1.96  # For 95% CI
    
    # Statistical tests
    use_wilcoxon: bool = True  # Paired comparisons
    significance_threshold: float = 0.05
    
    # Effect size thresholds
    cohens_d_small: float = 0.2
    cohens_d_medium: float = 0.5
    cohens_d_large: float = 0.8

STATISTICAL_CONFIG = StatisticalConfig()

# ============ HELPER FUNCTIONS ============
def save_config(config_dict: dict, path: Path):
    """Save configuration to JSON."""
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

def load_config(path: Path) -> dict:
    """Load configuration from JSON."""
    with open(path, 'r') as f:
        return json.load(f)

def get_all_configs() -> dict:
    """Get all configurations as dictionary."""
    return {
        'seeds': SEEDS,
        'hardware_targets': HARDWARE_TARGETS.__dict__,
        'accuracy_targets': ACCURACY_TARGETS.__dict__,
        'critical_classes': CRITICAL_CLASSES,
        'student_config': STUDENT_CONFIG.to_dict(),
        'teacher_config': TEACHER_CONFIG.to_dict(),
        'baseline_config': BASELINE_CONFIG.__dict__,
        'kd_config': KD_CONFIG.__dict__,
        'pruning_config': PRUNING_CONFIG.__dict__,
        'qat_config': QAT_CONFIG.__dict__,
        'conversion_config': CONVERSION_CONFIG.__dict__,
        'benchmark_config': BENCHMARK_CONFIG.__dict__,
        'statistical_config': STATISTICAL_CONFIG.__dict__,
    }

if __name__ == "__main__":
    # Save all configs for documentation
    all_configs = get_all_configs()
    save_config(all_configs, PHASE2_DIR / "all_configs.json")
    print(f"Saved all configurations to {PHASE2_DIR / 'all_configs.json'}")
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 2 COMPRESSION PIPELINE CONFIGURATION")
    print("="*70)
    print(f"Seeds: {SEEDS} (N={N_SEEDS})")
    print(f"Data: {DATA_DIR}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"\nTarget Metrics:")
    print(f"  - Critical recall (DDoS/PortScan): > {ACCURACY_TARGETS.critical_recall_min}%")
    print(f"  - F1-Macro drop: ≤ {ACCURACY_TARGETS.f1_drop_max}% (hard), ≤ {ACCURACY_TARGETS.f1_drop_preferred}% (preferred)")
    print(f"  - FAR: ≤ {ACCURACY_TARGETS.far_max}%")
    print(f"  - Latency p50: ≤ {HARDWARE_TARGETS.latency_p50_max} ms")
    print(f"  - Model size: ≤ {HARDWARE_TARGETS.model_size_ideal/1024:.0f} KB ideal")
    print(f"\nStudent sizes to test: {STUDENT_PARAM_TARGETS}")

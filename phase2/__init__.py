"""
Phase 2 - DS-CNN Compression Pipeline
======================================

Complete compression pipeline for deploying DS-1D-CNN on edge devices.

Stages:
    0. Baseline Training (train_baseline.py)
    1. Knowledge Distillation (train_kd.py)
    2. Structured Pruning (prune_model.py)
    3. KD Fine-tuning (fine_tune_kd.py)
    4. Quantization-Aware Training (qat_train.py)
    5. TFLite Conversion (convert_to_tflite.py)
    6. Pi Benchmarking (pi_bench.py)

Usage:
    # Run full pipeline
    ./run_pipeline.sh
    
    # Run specific stage
    python phase2/train/train_baseline.py --seed 42
"""

from .config import SEEDS, ARTIFACTS_DIR, ACCURACY_TARGETS, HARDWARE_TARGETS
from .utils import set_seed, set_reproducibility, MetricsTracker, FocalLoss, KDLoss, DataLoaderFactory
from .models import DSCNNStudent, DSCNNTeacher

__version__ = '0.1.0'
__author__ = 'Phase 2 Compression'

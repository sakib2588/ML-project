"""
Training module for IDS models.

Contains:
- Trainer: Main training loop with callbacks
- Evaluator: Model evaluation utilities  
- Callbacks: EarlyStopping, ModelCheckpoint, LRScheduler
- FocalLoss: Loss functions for handling class imbalance
- IDSMetrics: IDS-specific evaluation metrics
"""

from .trainer import Trainer
from .evaluator import ModelEvaluator
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Loss functions
try:
    from .focal_loss import (
        FocalLoss,
        WeightedFocalLoss,
        ClassBalancedLoss,
        compute_class_weights,
        get_loss_function,
    )
except ImportError:
    pass

# IDS-specific metrics
try:
    from .ids_metrics import (
        IDSMetrics,
        IDSEvaluator,
        compare_models,
    )
except ImportError:
    pass

__all__ = [
    "Trainer",
    "ModelEvaluator",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "FocalLoss",
    "WeightedFocalLoss", 
    "ClassBalancedLoss",
    "compute_class_weights",
    "get_loss_function",
    "IDSMetrics",
    "IDSEvaluator",
    "compare_models",
]
"""
Research-Grade Model Evaluator for IDS Models

Features:
 - Standard classification metrics: accuracy, F1, precision, recall
 - Confusion matrix and per-class metrics
 - Optional efficiency profiling (FLOPs, params)
 - Optional progress bars via logger
 - Supports cross-dataset evaluation
 - Supports custom metric hooks
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import logging


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cpu',
        logger: Optional[logging.Logger] = None,
        metric_hooks: Optional[Dict[str, Callable]] = None
    ):
        """
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for evaluation
            device: Device to run evaluation on ('cpu' or 'cuda')
            logger: Optional logger (e.g., ExperimentLogger)
            metric_hooks: Optional dict of custom metric functions
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.metric_hooks = metric_hooks or {}

    def evaluate(
        self,
        calculate_efficiency: bool = True,
        input_shape_for_profiling: Optional[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        """
        Perform evaluation on the test set.

        Args:
            calculate_efficiency: If True, compute model efficiency metrics (FLOPs, params)
            input_shape_for_profiling: Input shape tuple (required if calculate_efficiency is True)

        Returns:
            Dictionary with evaluation results
        """
        if calculate_efficiency and input_shape_for_profiling is None:
            raise ValueError("input_shape_for_profiling is required if calculate_efficiency is True.")

        self.model.eval()
        all_preds = []
        all_targets = []

        if self.logger:
            self.logger.info("Starting model evaluation...")

        # Evaluation loop with optional progress bar
        loader_iter = self.test_loader
        if self.logger and hasattr(self.logger, 'progress_bar'):
            loader_iter = getattr(self.logger, 'progress_bar')(self.test_loader, desc="Evaluating")

        with torch.no_grad():
            for data, target in loader_iter:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # --- Standard Metrics ---
        results: Dict[str, Any] = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(all_targets, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_targets, all_preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
            'per_class_metrics': classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
        }

        # --- Custom Metrics ---
        for name, func in self.metric_hooks.items():
            try:
                results[name] = func(all_targets, all_preds)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Custom metric '{name}' failed: {e}")

        # --- Optional Efficiency Metrics ---
        if calculate_efficiency and input_shape_for_profiling is not None:
            try:
                from ..utils.metrics import profile_model
                eff_results = profile_model(self.model, input_shape_for_profiling, self.device)
                results.update(eff_results)
            except ImportError:
                if self.logger:
                    self.logger.warning("profile_model not available. Skipping efficiency metrics.")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Efficiency profiling failed: {e}")

        if self.logger:
            self.logger.info(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}, F1 Macro: {results['f1_macro']:.4f}")

        return results

    def evaluate_cross_dataset(
        self,
        model: torch.nn.Module,
        cross_test_loader: torch.utils.data.DataLoader,
        input_shape_for_profiling: Optional[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a different dataset (e.g., for cross-dataset testing).

        Args:
            model: Model to evaluate (weights should already be loaded)
            cross_test_loader: DataLoader for the cross dataset
            input_shape_for_profiling: Input shape for efficiency metrics (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        # Backup original state
        original_model = self.model
        original_loader = self.test_loader

        # Set new model and loader
        self.model = model.to(self.device)
        self.test_loader = cross_test_loader

        # Evaluate
        results = self.evaluate(
            calculate_efficiency=(input_shape_for_profiling is not None),
            input_shape_for_profiling=input_shape_for_profiling
        )

        # Restore original state
        self.model = original_model
        self.test_loader = original_loader

        return results

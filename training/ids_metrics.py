"""
IDS-specific evaluation metrics and reporting.

Key metrics for Intrusion Detection Systems:
- Detection Rate (DR) / Attack Recall: TP / (TP + FN) - CRITICAL
- False Alarm Rate (FAR): FP / (FP + TN) - Should be low
- Precision: TP / (TP + FP)
- F1-Score: Harmonic mean of precision and recall

Publication standards for IDS:
- Attack Recall > 80% (ideally > 90%)
- False Alarm Rate < 10% (ideally < 5%)
- F1-Macro > 85%
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class IDSMetrics:
    """Container for IDS evaluation metrics."""
    
    # Overall metrics
    accuracy: float
    f1_macro: float
    f1_weighted: float
    
    # Benign class (typically class 0)
    benign_precision: float
    benign_recall: float
    benign_f1: float
    
    # Attack class (typically class 1) - MOST IMPORTANT
    attack_precision: float
    attack_recall: float  # = Detection Rate
    attack_f1: float
    
    # IDS-specific metrics
    detection_rate: float  # Same as attack_recall
    false_alarm_rate: float
    
    # Confusion matrix
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    
    # Optional
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'benign_precision': self.benign_precision,
            'benign_recall': self.benign_recall,
            'benign_f1': self.benign_f1,
            'attack_precision': self.attack_precision,
            'attack_recall': self.attack_recall,
            'attack_f1': self.attack_f1,
            'detection_rate': self.detection_rate,
            'false_alarm_rate': self.false_alarm_rate,
            'true_negative': self.true_negative,
            'false_positive': self.false_positive,
            'false_negative': self.false_negative,
            'true_positive': self.true_positive,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
        }
    
    def is_publishable(self) -> Tuple[bool, list]:
        """
        Check if results meet publication standards.
        
        Returns:
            (is_publishable, list of issues)
        """
        issues = []
        
        if self.attack_recall < 0.80:
            issues.append(f"Attack recall too low: {self.attack_recall*100:.1f}% < 80%")
        
        if self.false_alarm_rate > 0.10:
            issues.append(f"False alarm rate too high: {self.false_alarm_rate*100:.1f}% > 10%")
        
        if self.f1_macro < 0.80:
            issues.append(f"F1-Macro too low: {self.f1_macro*100:.1f}% < 80%")
        
        if self.attack_precision < 0.60:
            issues.append(f"Attack precision low: {self.attack_precision*100:.1f}% < 60%")
        
        return len(issues) == 0, issues


class IDSEvaluator:
    """
    Comprehensive evaluator for Intrusion Detection Systems.
    """
    
    def __init__(self, positive_label: int = 1):
        """
        Args:
            positive_label: Label for attack class (typically 1)
        """
        self.positive_label = positive_label
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> IDSMetrics:
        """
        Compute comprehensive IDS metrics.
        
        Args:
            y_true: Ground truth labels (0=benign, 1=attack)
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (batch, num_classes)
        
        Returns:
            IDSMetrics dataclass
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            confusion_matrix, roc_auc_score, average_precision_score
        )
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        # Benign (class 0)
        benign_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        benign_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        benign_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        # Attack (class 1) - CRITICAL
        attack_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        attack_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        attack_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            tn = fp = fn = tp = 0
            if cm.shape[0] >= 1:
                tn = cm[0, 0] if cm.shape[1] >= 1 else 0
                fp = cm[0, 1] if cm.shape[1] >= 2 else 0
            if cm.shape[0] >= 2:
                fn = cm[1, 0] if cm.shape[1] >= 1 else 0
                tp = cm[1, 1] if cm.shape[1] >= 2 else 0
        
        # IDS-specific metrics
        detection_rate = attack_recall  # Same as true positive rate for attacks
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # AUC metrics if probabilities available
        auc_roc = None
        auc_pr = None
        if y_pred_proba is not None:
            try:
                # Get probability of attack class
                if y_pred_proba.ndim == 2:
                    attack_proba = y_pred_proba[:, 1]
                else:
                    attack_proba = y_pred_proba
                
                auc_roc = roc_auc_score(y_true, attack_proba)
                auc_pr = average_precision_score(y_true, attack_proba)
            except Exception:
                pass
        
        return IDSMetrics(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            benign_precision=benign_precision,
            benign_recall=benign_recall,
            benign_f1=benign_f1,
            attack_precision=attack_precision,
            attack_recall=attack_recall,
            attack_f1=attack_f1,
            detection_rate=detection_rate,
            false_alarm_rate=false_alarm_rate,
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn),
            true_positive=int(tp),
            auc_roc=auc_roc,
            auc_pr=auc_pr,
        )
    
    def print_report(self, metrics: IDSMetrics, model_name: str = "Model") -> None:
        """Print detailed evaluation report."""
        
        print("\n" + "=" * 70)
        print(f"IDS EVALUATION REPORT: {model_name}")
        print("=" * 70)
        
        print("\n📊 Overall Metrics:")
        print(f"  Accuracy:      {metrics.accuracy * 100:6.2f}%")
        print(f"  F1-Macro:      {metrics.f1_macro * 100:6.2f}%")
        print(f"  F1-Weighted:   {metrics.f1_weighted * 100:6.2f}%")
        if metrics.auc_roc is not None:
            print(f"  AUC-ROC:       {metrics.auc_roc:6.4f}")
        if metrics.auc_pr is not None:
            print(f"  AUC-PR:        {metrics.auc_pr:6.4f}")
        
        print("\n✅ Benign Traffic (Class 0):")
        print(f"  Precision:     {metrics.benign_precision * 100:6.2f}%")
        print(f"  Recall:        {metrics.benign_recall * 100:6.2f}%")
        print(f"  F1-Score:      {metrics.benign_f1 * 100:6.2f}%")
        
        print("\n🚨 Attack Detection (Class 1) - CRITICAL:")
        # Status indicators
        recall_ok = metrics.attack_recall >= 0.80
        precision_ok = metrics.attack_precision >= 0.60
        
        recall_symbol = "✅" if recall_ok else "❌"
        precision_symbol = "✅" if precision_ok else "⚠️"
        
        print(f"  Precision:     {metrics.attack_precision * 100:6.2f}% {precision_symbol}")
        print(f"  Recall:        {metrics.attack_recall * 100:6.2f}% {recall_symbol} {'← GOOD!' if recall_ok else '← NEEDS IMPROVEMENT'}")
        print(f"  F1-Score:      {metrics.attack_f1 * 100:6.2f}%")
        
        print("\n📈 IDS-Specific Metrics:")
        dr_ok = metrics.detection_rate >= 0.80
        far_ok = metrics.false_alarm_rate <= 0.10
        
        dr_symbol = "✅" if dr_ok else "❌"
        far_symbol = "✅" if far_ok else "⚠️"
        
        print(f"  Detection Rate:     {metrics.detection_rate * 100:6.2f}% {dr_symbol} (target: >80%)")
        print(f"  False Alarm Rate:   {metrics.false_alarm_rate * 100:6.2f}% {far_symbol} (target: <10%)")
        
        print("\n🔢 Confusion Matrix:")
        print(f"  True Negatives:  {metrics.true_negative:>10,}  (Benign correctly classified)")
        print(f"  False Positives: {metrics.false_positive:>10,}  (False alarms)")
        print(f"  False Negatives: {metrics.false_negative:>10,}  (Missed attacks) ← MINIMIZE THIS!")
        print(f"  True Positives:  {metrics.true_positive:>10,}  (Attacks detected)")
        
        # Publishability assessment
        is_pub, issues = metrics.is_publishable()
        
        print("\n📝 Publication Assessment:")
        if is_pub:
            print("  ✅ Results meet publication standards!")
            print("  ✅ Attack detection >80%, False alarms <10%")
        else:
            print("  ⚠️ Issues preventing publication:")
            for issue in issues:
                print(f"     ❌ {issue}")
        
        print("=" * 70 + "\n")
    
    def save_report(self, metrics: IDSMetrics, save_path: Path, model_name: str = "Model") -> None:
        """Save evaluation report to JSON."""
        report = {
            'model_name': model_name,
            'metrics': metrics.to_dict(),
            'publishability': {
                'is_publishable': metrics.is_publishable()[0],
                'issues': metrics.is_publishable()[1]
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)


def compare_models(results: Dict[str, IDSMetrics]) -> None:
    """
    Print comparison table for multiple models.
    
    Args:
        results: Dict mapping model_name -> IDSMetrics
    """
    print("\n" + "=" * 90)
    print("MODEL COMPARISON - IDS METRICS")
    print("=" * 90)
    
    # Header
    print(f"{'Model':<15} {'Accuracy':>10} {'F1-Macro':>10} {'Attack Recall':>14} {'FAR':>8} {'Status':>10}")
    print("-" * 90)
    
    for name, metrics in results.items():
        is_pub, _ = metrics.is_publishable()
        status = "✅ GOOD" if is_pub else "❌ FIX"
        
        print(f"{name:<15} {metrics.accuracy*100:>9.2f}% {metrics.f1_macro*100:>9.2f}% "
              f"{metrics.attack_recall*100:>13.2f}% {metrics.false_alarm_rate*100:>7.2f}% {status:>10}")
    
    print("=" * 90)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].attack_recall + x[1].f1_macro)
    print(f"\n🏆 Best Model: {best_model[0]} (Attack Recall: {best_model[1].attack_recall*100:.2f}%)")
    print()


# Quick test
if __name__ == '__main__':
    print("Testing IDS Evaluator...")
    
    # Simulated predictions (imbalanced)
    np.random.seed(42)
    n_samples = 1000
    
    # Ground truth: 80% benign, 20% attack
    y_true = np.array([0] * 800 + [1] * 200)
    
    # Simulated good model predictions
    y_pred_good = y_true.copy()
    # Add some errors
    y_pred_good[:50] = 1  # 50 false positives
    y_pred_good[820:850] = 0  # 30 missed attacks
    
    # Simulated bad model predictions
    y_pred_bad = y_true.copy()
    y_pred_bad[:100] = 1  # 100 false positives
    y_pred_bad[800:950] = 0  # 150 missed attacks (75% of attacks missed!)
    
    evaluator = IDSEvaluator()
    
    # Evaluate good model
    print("\n--- Good Model ---")
    metrics_good = evaluator.evaluate(y_true, y_pred_good)
    evaluator.print_report(metrics_good, "GoodModel")
    
    # Evaluate bad model
    print("\n--- Bad Model ---")
    metrics_bad = evaluator.evaluate(y_true, y_pred_bad)
    evaluator.print_report(metrics_bad, "BadModel")
    
    # Compare
    compare_models({
        "GoodModel": metrics_good,
        "BadModel": metrics_bad
    })
    
    print("✅ IDS Evaluator test complete!")

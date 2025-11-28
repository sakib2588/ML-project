#!/usr/bin/env python
"""
IDS Experiment Diagnostic Tool

This script diagnoses issues with the current experiment setup and
provides actionable recommendations.

Run: python diagnose_experiment.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


def check_feature_count(data_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Check if sufficient features are being used."""
    train_X_path = data_dir / "train" / "X.npy"
    
    if not train_X_path.exists():
        return False, "Training data not found", {}
    
    X = np.load(train_X_path)
    n_samples, window_len, n_features = X.shape
    
    info = {
        "shape": X.shape,
        "n_samples": n_samples,
        "window_length": window_len,
        "n_features": n_features
    }
    
    if n_features < 50:
        return False, f"Only {n_features} features used. CIC-IDS2017 has 78 features - losing information!", info
    elif n_features < 70:
        return True, f"Using {n_features}/78 features. Consider using more.", info
    else:
        return True, f"Using {n_features} features - good!", info


def check_class_balance(data_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Check class distribution and imbalance ratio."""
    train_y_path = data_dir / "train" / "y.npy"
    
    if not train_y_path.exists():
        return False, "Training labels not found", {}
    
    y = np.load(train_y_path)
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))
    
    imbalance_ratio = max(counts) / min(counts)
    
    info = {
        "class_distribution": class_dist,
        "imbalance_ratio": float(imbalance_ratio),
        "total_samples": len(y)
    }
    
    if imbalance_ratio > 10:
        return False, f"Severe class imbalance ({imbalance_ratio:.1f}:1). Must use SMOTE or Focal Loss!", info
    elif imbalance_ratio > 5:
        return False, f"Significant imbalance ({imbalance_ratio:.1f}:1). Use Focal Loss or class weights.", info
    elif imbalance_ratio > 2:
        return True, f"Moderate imbalance ({imbalance_ratio:.1f}:1). Consider Focal Loss.", info
    else:
        return True, f"Class balance is acceptable ({imbalance_ratio:.1f}:1)", info


def check_previous_results(experiment_dir: Path) -> Dict[str, Any]:
    """Analyze previous experiment results."""
    results = {}
    
    results_file = experiment_dir / "all_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        for model_name, metrics in all_results.items():
            accuracy = metrics.get('accuracy', 0)
            f1_macro = metrics.get('f1_macro', 0)
            
            # Try to get per-class metrics
            per_class = metrics.get('per_class_metrics', {})
            attack_recall = None
            if per_class and '1' in per_class:
                attack_recall = per_class['1'].get('recall', None)
            
            results[model_name] = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "attack_recall": attack_recall,
                "issues": []
            }
            
            # Diagnose issues
            if accuracy < 0.75:
                results[model_name]["issues"].append(f"Accuracy too low ({accuracy*100:.1f}%)")
            if f1_macro < 0.70:
                results[model_name]["issues"].append(f"F1-Macro too low ({f1_macro*100:.1f}%)")
            if attack_recall is not None and attack_recall < 0.50:
                results[model_name]["issues"].append(f"Attack recall critical ({attack_recall*100:.1f}%)")
    
    return results


def generate_recommendations(issues: List[str]) -> List[str]:
    """Generate actionable recommendations based on identified issues."""
    recommendations = []
    
    if any("feature" in issue.lower() for issue in issues):
        recommendations.append(
            "1. USE MORE FEATURES:\n"
            "   - Edit configs/preprocess_config.yaml to add more CIC-IDS2017 features\n"
            "   - Or use configs/preprocess_config_enhanced.yaml which has 60+ features\n"
            "   - Re-run preprocessing: python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full"
        )
    
    if any("imbalance" in issue.lower() for issue in issues):
        recommendations.append(
            "2. HANDLE CLASS IMBALANCE:\n"
            "   - Use Focal Loss: Set loss.name='focal' in main_phase1.py config\n"
            "   - Increase class weights: loss.class_weights = [1.0, 8.0]\n"
            "   - Consider SMOTE in preprocessing (requires imblearn)"
        )
    
    if any("recall" in issue.lower() or "attack" in issue.lower() for issue in issues):
        recommendations.append(
            "3. IMPROVE ATTACK DETECTION:\n"
            "   - Use Focal Loss with alpha=0.75, gamma=2.0\n"
            "   - Monitor val_f1_macro instead of val_loss for early stopping\n"
            "   - Increase epochs to 50+ with early_stopping patience=12"
        )
    
    if any("accuracy" in issue.lower() and "low" in issue.lower() for issue in issues):
        recommendations.append(
            "4. IMPROVE MODEL CAPACITY:\n"
            "   - For MLP: Increase hidden_sizes to (256, 128, 64)\n"
            "   - For DS-CNN: Increase conv_channels to (64, 128, 128, 256)\n"
            "   - For LSTM: Use bidirectional=True, hidden_size=128"
        )
    
    return recommendations


def main():
    print("=" * 70)
    print("IDS EXPERIMENT DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # Paths
    data_dir = Path("data/processed/cic_ids_2017")
    experiment_dir = Path("experiments/Phase1_Baseline_CIC_IDS_2017")
    
    all_issues = []
    all_info = {}
    
    # 1. Check feature count
    print("\n📊 FEATURE ANALYSIS")
    print("-" * 40)
    ok, msg, info = check_feature_count(data_dir)
    status = "✅" if ok else "❌"
    print(f"{status} {msg}")
    all_info["features"] = info
    if not ok:
        all_issues.append(msg)
    
    if info:
        print(f"   Shape: {info['shape']}")
        print(f"   Samples: {info['n_samples']:,}")
        print(f"   Window length: {info['window_length']}")
        print(f"   Features: {info['n_features']}")
    
    # 2. Check class balance
    print("\n📊 CLASS BALANCE ANALYSIS")
    print("-" * 40)
    ok, msg, info = check_class_balance(data_dir)
    status = "✅" if ok else "❌"
    print(f"{status} {msg}")
    all_info["class_balance"] = info
    if not ok:
        all_issues.append(msg)
    
    if info:
        print(f"   Distribution: {info['class_distribution']}")
        print(f"   Imbalance ratio: {info['imbalance_ratio']:.2f}:1")
    
    # 3. Check previous results
    print("\n📊 PREVIOUS EXPERIMENT RESULTS")
    print("-" * 40)
    prev_results = check_previous_results(experiment_dir)
    
    if prev_results:
        for model, data in prev_results.items():
            print(f"\n   {model}:")
            print(f"      Accuracy: {data['accuracy']*100:.2f}%")
            print(f"      F1-Macro: {data['f1_macro']*100:.2f}%")
            if data['attack_recall'] is not None:
                print(f"      Attack Recall: {data['attack_recall']*100:.2f}%")
            if data['issues']:
                for issue in data['issues']:
                    print(f"      ❌ {issue}")
                    all_issues.append(f"{model}: {issue}")
    else:
        print("   No previous results found")
    
    # 4. Generate recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if all_issues:
        print("\n🚨 ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        
        recommendations = generate_recommendations(all_issues)
        print("\n💡 RECOMMENDED ACTIONS:")
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\n✅ No critical issues found!")
        print("   Your experiment setup looks good.")
    
    # 5. Quick fix commands
    print("\n" + "=" * 70)
    print("QUICK FIX COMMANDS")
    print("=" * 70)
    
    print("""
To re-run with Focal Loss (fastest fix):
    python main_phase1.py --quick

To re-preprocess with enhanced features (recommended):
    1. Edit configs/phase1_config.yaml to use preprocess_config_enhanced.yaml
    2. python -m src.data.cli preprocess --dataset cic_ids_2017 --mode full
    3. python main_phase1.py

To test Focal Loss implementation:
    python -c "from src.training.focal_loss import FocalLoss; print('✅ Focal Loss available')"

To test IDS metrics:
    python -c "from src.training.ids_metrics import IDSEvaluator; print('✅ IDS Metrics available')"
""")
    
    print("=" * 70)
    print("Diagnostic complete!")
    print("=" * 70 + "\n")
    
    return len(all_issues)


if __name__ == "__main__":
    n_issues = main()
    sys.exit(0 if n_issues == 0 else 1)

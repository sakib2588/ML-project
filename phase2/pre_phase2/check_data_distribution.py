#!/usr/bin/env python3
"""
Pre-Phase-2 Data Distribution Analysis

CRITICAL: Run this BEFORE starting Phase 2 compression!

This script:
1. Checks training data class distribution
2. Identifies underrepresented attack classes (Bot, SSH-Patator, etc.)
3. Reports class imbalance statistics
4. Recommends augmentation strategy

Usage:
    python phase2/pre_phase2/check_data_distribution.py --data-dir data/processed/cic_ids_2017
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Minimum samples threshold for each class
MIN_SAMPLES_THRESHOLD = 100
CRITICAL_ATTACK_TYPES = ['DDoS', 'PortScan', 'Bot', 'SSH-Patator', 'FTP-Patator']


def load_attack_types(data_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load attack type labels from train/val/test splits."""
    # Try multiple filename patterns
    train_types = None
    val_types = None
    test_types = None
    
    for name in ['attack_types.npy', 'y_attack_types.npy', 'attack_type.npy']:
        train_path = data_dir / 'train' / name
        if train_path.exists() and train_types is None:
            train_types = np.load(train_path, allow_pickle=True)
        
        val_path = data_dir / 'val' / name
        if val_path.exists() and val_types is None:
            val_types = np.load(val_path, allow_pickle=True)
        
        test_path = data_dir / 'test' / name
        if test_path.exists() and test_types is None:
            test_types = np.load(test_path, allow_pickle=True)
    
    return train_types, val_types, test_types


def load_binary_labels(data_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load binary labels from train/val/test splits."""
    # Try multiple filename patterns
    train_labels = None
    val_labels = None
    test_labels = None
    
    for name in ['y.npy', 'y_train.npy', 'labels.npy']:
        train_path = data_dir / 'train' / name
        if train_path.exists() and train_labels is None:
            train_labels = np.load(train_path)
        
        val_path = data_dir / 'val' / name
        if val_path.exists() and val_labels is None:
            val_labels = np.load(val_path)
        
        test_path = data_dir / 'test' / name
        if test_path.exists() and test_labels is None:
            test_labels = np.load(test_path)
    
    return train_labels, val_labels, test_labels


def analyze_distribution(attack_types: np.ndarray, split_name: str) -> Dict:
    """Analyze class distribution for a given split."""
    counter = Counter(attack_types)
    total = len(attack_types)
    
    distribution = {}
    for attack_type, count in counter.most_common():
        distribution[attack_type] = {
            'count': count,
            'percentage': count / total * 100,
            'status': 'OK' if count >= MIN_SAMPLES_THRESHOLD else 'CRITICAL'
        }
    
    # Check for missing critical attacks
    missing = []
    for critical in CRITICAL_ATTACK_TYPES:
        if critical not in distribution:
            missing.append(critical)
            distribution[critical] = {
                'count': 0,
                'percentage': 0.0,
                'status': 'MISSING'
            }
    
    return {
        'split': split_name,
        'total_samples': total,
        'n_classes': len(counter),
        'distribution': distribution,
        'missing_critical': missing,
        'underrepresented': [k for k, v in distribution.items() 
                            if v['count'] < MIN_SAMPLES_THRESHOLD and v['count'] > 0]
    }


def compute_class_weights(attack_types: np.ndarray) -> Dict[str, float]:
    """Compute balanced class weights for loss function."""
    counter = Counter(attack_types)
    total = len(attack_types)
    n_classes = len(counter)
    
    weights = {}
    for attack_type, count in counter.items():
        # Balanced weight formula: n_samples / (n_classes * n_samples_for_class)
        weights[attack_type] = total / (n_classes * count)
    
    # Normalize so min weight is 1.0
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    return weights


def compute_imbalance_ratio(attack_types: np.ndarray) -> float:
    """Compute imbalance ratio (max_count / min_count)."""
    counter = Counter(attack_types)
    counts = list(counter.values())
    return max(counts) / min(counts) if min(counts) > 0 else float('inf')


def print_report(analysis: Dict, class_weights: Dict[str, float]) -> None:
    """Print formatted analysis report."""
    print("\n" + "=" * 80)
    print(f"📊 DATA DISTRIBUTION ANALYSIS: {analysis['split'].upper()}")
    print("=" * 80)
    
    print(f"\nTotal Samples: {analysis['total_samples']:,}")
    print(f"Number of Classes: {analysis['n_classes']}")
    
    print("\n📈 Class Distribution:")
    print("-" * 60)
    print(f"{'Attack Type':<25} {'Count':>10} {'Percentage':>12} {'Status':<10}")
    print("-" * 60)
    
    # Sort by count descending
    sorted_dist = sorted(analysis['distribution'].items(), 
                         key=lambda x: x[1]['count'], reverse=True)
    
    for attack_type, stats in sorted_dist:
        status_icon = "✅" if stats['status'] == 'OK' else "❌" if stats['status'] == 'MISSING' else "⚠️"
        print(f"{attack_type:<25} {stats['count']:>10,} {stats['percentage']:>11.2f}% {status_icon} {stats['status']}")
    
    print("-" * 60)
    
    # Critical alerts
    if analysis['missing_critical']:
        print(f"\n🚨 CRITICAL: Missing attack types: {analysis['missing_critical']}")
    
    if analysis['underrepresented']:
        print(f"\n⚠️  WARNING: Underrepresented classes (<{MIN_SAMPLES_THRESHOLD} samples):")
        for attack in analysis['underrepresented']:
            count = analysis['distribution'][attack]['count']
            print(f"   - {attack}: {count} samples")
    
    # Class weights
    print("\n⚖️  Recommended Class Weights (for weighted loss):")
    print("-" * 40)
    for attack_type, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"   {attack_type:<25}: {weight:.2f}")


def generate_recommendations(train_analysis: Dict, val_analysis: Dict, test_analysis: Dict) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # Check for missing critical attacks
    all_missing = set(train_analysis['missing_critical']) | \
                  set(val_analysis['missing_critical']) | \
                  set(test_analysis['missing_critical'])
    
    if 'Bot' in all_missing or 'SSH-Patator' in all_missing:
        recommendations.append(
            "🚨 CRITICAL: Bot/SSH-Patator samples are MISSING from training data!\n"
            "   → Your Phase 1 model shows 0% detection for these classes.\n"
            "   → FIX: Check if these attacks exist in raw CSV files.\n"
            "   → If present but filtered: Modify preprocessing to include them.\n"
            "   → If absent: Consider synthetic generation or different dataset."
        )
    
    # Check for underrepresented classes
    underrep = set(train_analysis['underrepresented'])
    if underrep:
        recommendations.append(
            f"⚠️  WARNING: Classes with <{MIN_SAMPLES_THRESHOLD} samples: {underrep}\n"
            "   → These classes may have poor detection rates.\n"
            "   → FIX 1: Use SMOTE augmentation (run augment_rare_classes.py)\n"
            "   → FIX 2: Use class-weighted loss function\n"
            "   → FIX 3: Use Focal Loss to focus on hard examples"
        )
    
    # Check imbalance ratio
    train_counts = [v['count'] for v in train_analysis['distribution'].values() if v['count'] > 0]
    if train_counts:
        imbalance = max(train_counts) / min(train_counts)
        if imbalance > 100:
            recommendations.append(
                f"⚠️  HIGH IMBALANCE: Ratio = {imbalance:.0f}:1\n"
                "   → Majority class dominates training.\n"
                "   → FIX: Use balanced sampling or class weights."
            )
    
    # Check train/test distribution mismatch
    train_attacks = set(train_analysis['distribution'].keys())
    test_attacks = set(test_analysis['distribution'].keys())
    train_only = train_attacks - test_attacks
    test_only = test_attacks - train_attacks
    
    if train_only:
        recommendations.append(
            f"⚠️  Distribution Mismatch: In train but not test: {train_only}\n"
            "   → Model may overfit to these classes."
        )
    if test_only:
        recommendations.append(
            f"🚨 CRITICAL: In test but not train: {test_only}\n"
            "   → Model will fail on these unseen classes!"
        )
    
    return recommendations


def save_analysis_json(output_path: Path, analyses: Dict, recommendations: List[str]) -> None:
    """Save analysis results to JSON file."""
    output = {
        'timestamp': str(np.datetime64('now')),
        'analyses': analyses,
        'recommendations': recommendations,
        'thresholds': {
            'min_samples': MIN_SAMPLES_THRESHOLD,
            'critical_attacks': CRITICAL_ATTACK_TYPES
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Check data distribution before Phase 2")
    parser.add_argument('--data-dir', type=str, default='data/processed/cic_ids_2017_v2',
                        help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='phase2/pre_phase2/data_analysis.json',
                        help='Output JSON file for analysis results')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 PRE-PHASE-2 DATA DISTRIBUTION CHECK")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    
    # Check if attack type files exist
    train_types, val_types, test_types = load_attack_types(data_dir)
    
    if train_types is None:
        print("\n⚠️  WARNING: y_attack_types.npy not found!")
        print("   Checking binary labels instead...")
        
        train_labels, val_labels, test_labels = load_binary_labels(data_dir)
        
        if train_labels is not None:
            print(f"\n   Binary distribution (train):")
            print(f"   - Benign: {(train_labels == 0).sum():,}")
            print(f"   - Attack: {(train_labels == 1).sum():,}")
            print(f"   - Ratio: {(train_labels == 0).sum() / (train_labels == 1).sum():.2f}:1")
        
        print("\n❌ CRITICAL: Cannot perform per-class analysis without attack type labels!")
        print("   → You need to save y_attack_types.npy during preprocessing.")
        print("   → See preprocessing/pipeline.py for modifications needed.")
        sys.exit(1)
    
    # Analyze each split
    train_analysis = analyze_distribution(train_types, 'train')
    val_analysis = analyze_distribution(val_types, 'val') if val_types is not None else {'distribution': {}, 'missing_critical': [], 'underrepresented': []}
    test_analysis = analyze_distribution(test_types, 'test') if test_types is not None else {'distribution': {}, 'missing_critical': [], 'underrepresented': []}
    
    # Compute class weights
    train_weights = compute_class_weights(train_types)
    
    # Print reports
    print_report(train_analysis, train_weights)
    if val_types is not None:
        print_report(val_analysis, compute_class_weights(val_types))
    if test_types is not None:
        print_report(test_analysis, compute_class_weights(test_types))
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = generate_recommendations(train_analysis, val_analysis, test_analysis)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n✅ All checks passed! Data distribution looks healthy.")
    
    # Save analysis
    save_analysis_json(output_path, {
        'train': train_analysis,
        'val': val_analysis if val_types is not None else None,
        'test': test_analysis if test_types is not None else None,
        'class_weights': train_weights
    }, recommendations)
    
    # Final verdict
    print("\n" + "=" * 80)
    if train_analysis['missing_critical'] or train_analysis['underrepresented']:
        print("❌ VERDICT: Fix data issues BEFORE proceeding to Phase 2!")
        print("   Run: python phase2/pre_phase2/augment_rare_classes.py")
        return 1
    else:
        print("✅ VERDICT: Data distribution OK. Proceed to Phase 2!")
        return 0


if __name__ == '__main__':
    sys.exit(main())

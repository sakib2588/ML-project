#!/usr/bin/env python3
"""
Ultra-Lite Augmentation (Jitter + Mixup Only)

For severe hardware constraints:
- No SMOTE (memory-safe)
- Jitter: ~2× samples per class
- Mixup: ~3× samples per class  
- Total: ~6× samples per rare class with ~100MB RAM

Usage:
    python scripts/augment_ultra_lite.py \
        --data-dir data/processed/cic_ids_2017_v2 \
        --rare-classes Bot SSH-Patator \
        --augment-factor 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

ArrayLike = NDArray[Any]


def load_data_lazy(data_dir: Path) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Load X as mmap (don't load into RAM), load y and attack_types."""
    train_dir = data_dir / 'train'
    
    X = np.load(train_dir / 'X.npy', mmap_mode='r')  # Memory mapped, not loaded
    y = np.load(train_dir / 'y.npy')
    attack_types = np.load(train_dir / 'attack_types.npy', allow_pickle=True)
    
    return X, y, attack_types


def jitter_augment(X: ArrayLike, noise_std: float = 0.01, seed: int = 42) -> ArrayLike:
    """Add jitter - creates ~2x samples from input."""
    np.random.seed(seed)
    feature_std = np.std(X, axis=(0, 1), keepdims=True)
    feature_std = np.maximum(feature_std, 1e-8)
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise * feature_std


def mixup_augment(X: ArrayLike, alpha: float = 0.2, seed: int = 42) -> ArrayLike:
    """Mixup - interpolates between samples, creates ~1x new samples."""
    np.random.seed(seed)
    n = len(X)
    if n < 2:
        return X.copy()
    
    lam = np.random.beta(alpha, alpha, size=n).reshape(-1, 1, 1)
    indices = np.random.permutation(n)
    return lam * X + (1 - lam) * X[indices]


def augment_simple(
    data_dir: Path,
    output_dir: Path,
    rare_classes: List[str],
    augment_factor: int = 3,
    seed: int = 42
) -> None:
    """
    Ultra-simple augmentation: jitter + mixup only.
    
    Memory usage: ~100-200MB (only one class at a time in RAM)
    Time: ~2 minutes
    Augmentation: ~3-6x per rare class
    """
    np.random.seed(seed)
    
    print("\n" + "=" * 70)
    print("ULTRA-LITE AUGMENTATION (Jitter + Mixup)")
    print("=" * 70)
    print(f"Memory efficient: processes one class at a time")
    print(f"Augment factor: {augment_factor}×")
    
    train_dir = data_dir / 'train'
    output_train_dir = output_dir / 'train'
    output_train_dir.mkdir(parents=True, exist_ok=True)
    
    # Load everything ONCE
    print(f"\n📂 Loading data...")
    X_mmap, y, attack_types = load_data_lazy(data_dir)
    print(f"   X (mmap): {X_mmap.shape}")
    print(f"   y: {y.shape}")
    print(f"   attack_types: {attack_types.shape}")
    
    # Count rare classes
    rare_counts = {cls: (attack_types == cls).sum() for cls in rare_classes}
    print(f"\n📊 Rare class distribution:")
    for cls, count in rare_counts.items():
        print(f"   {cls}: {count} samples")
    
    # Create output arrays lists (we'll concatenate at end)
    X_output_list = []
    y_output_list = []
    at_output_list = []
    
    # Step 1: Process non-rare samples
    print(f"\n🔄 Step 1: Processing non-rare samples...")
    mask_non_rare = np.ones(len(attack_types), dtype=bool)
    for cls in rare_classes:
        mask_non_rare = mask_non_rare & (attack_types != cls)
    
    X_non_rare = np.array(X_mmap[mask_non_rare])  # Convert mmap slice to array
    y_non_rare = y[mask_non_rare]
    at_non_rare = attack_types[mask_non_rare]
    
    X_output_list.append(X_non_rare)
    y_output_list.append(y_non_rare)
    at_output_list.append(at_non_rare)
    
    print(f"   Non-rare samples: {len(X_non_rare)}")
    
    # Step 2: Augment rare classes
    print(f"\n🔄 Step 2: Augmenting rare classes...")
    
    augmentation_summary = {}
    
    for cls in rare_classes:
        print(f"\n   {cls}:")
        
        # Get samples for this class
        mask_cls = attack_types == cls
        X_cls_orig = np.array(X_mmap[mask_cls])  # Convert from mmap
        y_cls_orig = y[mask_cls]
        
        original_count = len(X_cls_orig)
        print(f"   Original: {original_count} samples")
        
        # Augment
        X_aug_list = [X_cls_orig]
        y_aug_list = [y_cls_orig]
        
        # Jitter
        print(f"      → Jitter (1×)")
        X_jitter = jitter_augment(X_cls_orig, noise_std=0.01, seed=seed)
        X_aug_list.append(X_jitter)
        y_aug_list.append(y_cls_orig)
        
        # Mixup (multiple rounds to reach augment_factor)
        n_mixup_rounds = augment_factor - 2  # Already have original + jitter
        for i in range(n_mixup_rounds):
            print(f"      → Mixup round {i+1} (1×)")
            X_current = np.vstack(X_aug_list)
            X_mixed = mixup_augment(X_current, alpha=0.2, seed=seed + i)
            X_aug_list.append(X_mixed)
            y_aug_list.append(np.full(len(X_mixed), y_cls_orig[0]))
        
        # Combine
        X_cls_aug = np.vstack(X_aug_list)
        y_cls_aug = np.hstack(y_aug_list)
        at_cls_aug = np.full(len(X_cls_aug), cls)
        
        print(f"      Final: {len(X_cls_aug)} samples ({len(X_cls_aug) / original_count:.1f}×)")
        
        X_output_list.append(X_cls_aug)
        y_output_list.append(y_cls_aug)
        at_output_list.append(at_cls_aug)
        
        augmentation_summary[cls] = {
            'original': original_count,
            'augmented': len(X_cls_aug),
            'multiplier': len(X_cls_aug) / original_count
        }
        
        # Clear to free memory
        del X_cls_orig, y_cls_orig, X_aug_list, y_aug_list
    
    # Step 3: Combine and save
    print(f"\n💾 Step 3: Saving augmented data...")
    
    X_final = np.vstack(X_output_list)
    y_final = np.hstack(y_output_list)
    at_final = np.hstack(at_output_list)
    
    # Shuffle
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    at_final = at_final[indices]
    
    print(f"   Saving X.npy: {X_final.shape}")
    np.save(output_train_dir / 'X.npy', X_final)
    
    print(f"   Saving y.npy: {y_final.shape}")
    np.save(output_train_dir / 'y.npy', y_final)
    
    print(f"   Saving attack_types.npy: {at_final.shape}")
    np.save(output_train_dir / 'attack_types.npy', at_final)
    
    # Step 4: Compute class weights
    print(f"\n⚖️  Computing class weights (capped at 50×)...")
    
    counter = Counter(at_final)
    total = len(at_final)
    n_classes = len(counter)
    
    weights = {}
    for cls, count in counter.items():
        weight = total / (n_classes * count)
        weights[cls] = min(weight, 50.0)  # Cap at 50×
    
    # Normalize
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    print(f"   Rare class weights:")
    for cls in rare_classes:
        if cls in weights:
            print(f"   {cls}: {weights[cls]:.2f}×")
    
    # Step 5: Save metadata
    weights_path = output_dir / 'class_weights.json'
    classes = sorted(weights.keys())
    weights_data = {
        'class_to_idx': {cls: i for i, cls in enumerate(classes)},
        'idx_to_class': {str(i): cls for i, cls in enumerate(classes)},
        'weights_dict': weights,
        'weights_list': [weights[cls] for cls in classes]
    }
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    # Save summary
    summary = {
        'method': 'ultra_lite_augmentation',
        'augmentation': augmentation_summary,
        'total_samples': len(X_final),
        'seed': seed,
        'class_weights': weights
    }
    summary_path = output_dir / 'augmentation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Copy val/test
    for subdir in ['val', 'test']:
        src = data_dir / subdir
        dst = output_dir / subdir
        if src.exists() and not dst.exists():
            print(f"   Copying {subdir}/...")
            shutil.copytree(src, dst)
    
    print(f"\n" + "=" * 70)
    print(f"✅ AUGMENTATION COMPLETE")
    print(f"=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(X_final)}")
    print(f"Class weights: {weights_path}")
    print(f"\nAugmentation summary:")
    for cls, info in augmentation_summary.items():
        print(f"   {cls}: {info['original']} → {info['augmented']} ({info['multiplier']:.1f}×)")


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-lite augmentation (jitter + mixup)"
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed/cic_ids_2017_v2',
        help='Data directory'
    )
    parser.add_argument(
        '--rare-classes', type=str, nargs='+',
        default=['Bot', 'SSH-Patator'],
        help='Classes to augment'
    )
    parser.add_argument(
        '--augment-factor', type=int, default=3,
        help='Augmentation factor (3 = 3x samples)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else \
        data_dir.parent / f'{data_dir.name}_augmented'
    
    augment_simple(
        data_dir=data_dir,
        output_dir=output_dir,
        rare_classes=args.rare_classes,
        augment_factor=args.augment_factor,
        seed=args.seed
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

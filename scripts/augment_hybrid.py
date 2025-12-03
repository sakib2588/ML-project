#!/usr/bin/env python3
"""
Hybrid Augmentation Script for Rare Classes

Safer augmentation pipeline that doesn't create unrealistic synthetic samples:
1. Jitter - Add Gaussian noise for realistic variations
2. Mixup - Interpolate within same class
3. Capped SMOTE - Limited synthetic generation (≤5× per iteration)
4. t-SNE Validation - Verify synthetic overlaps with real

Usage:
    python scripts/augment_hybrid.py \
        --data-dir data/processed/cic_ids_2017_v2 \
        --rare-classes Bot SSH-Patator \
        --target-samples 500 \
        --validate-tsne
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases
ArrayLike = NDArray[Any]

# Optional imports with proper typing
VISUALIZATION_AVAILABLE = False
SMOTE_AVAILABLE = False

try:
    from sklearn.manifold import TSNE as _TSNE
    import matplotlib.pyplot as _plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    _TSNE = None  # type: ignore
    _plt = None  # type: ignore

try:
    from imblearn.over_sampling import SMOTE as _SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE = None  # type: ignore
    print("⚠️  imbalanced-learn not installed. SMOTE disabled. Run: pip install imbalanced-learn")


def load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data with memory-mapping for large files."""
    train_dir = data_dir / 'train'
    
    # Load X with memory mapping
    X_path = train_dir / 'X.npy'
    if not X_path.exists():
        raise FileNotFoundError(f"X.npy not found in {train_dir}")
    X = np.load(X_path, mmap_mode='r')
    
    # Load y
    y_path = train_dir / 'y.npy'
    if not y_path.exists():
        raise FileNotFoundError(f"y.npy not found in {train_dir}")
    y = np.load(y_path)
    
    # Load attack_types
    at_path = train_dir / 'attack_types.npy'
    if not at_path.exists():
        raise FileNotFoundError(f"attack_types.npy not found in {train_dir}")
    attack_types = np.load(at_path, allow_pickle=True)
    
    return X, y, attack_types


def jitter_augment(X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise per-feature to create realistic variations.
    
    This is the safest augmentation - creates minor variations without
    fundamentally changing the sample structure.
    
    Args:
        X: Input samples (n_samples, window, features)
        noise_std: Standard deviation of noise relative to feature std
        
    Returns:
        Jittered samples
    """
    # Compute per-feature standard deviation
    feature_std = np.std(X, axis=(0, 1), keepdims=True)
    feature_std = np.maximum(feature_std, 1e-8)  # Avoid division by zero
    
    # Add scaled noise
    noise = np.random.normal(0, noise_std, X.shape)
    X_jittered = X + noise * feature_std
    
    return X_jittered


def mixup_augment(X: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Interpolate between samples within same class.
    
    Creates smooth interpolations that are realistic variations.
    
    Args:
        X: Input samples (n_samples, window, features)
        alpha: Beta distribution parameter (lower = closer to original)
        
    Returns:
        Mixed samples
    """
    n = len(X)
    if n < 2:
        return X.copy()
    
    # Sample mixing coefficients from Beta distribution
    lam = np.random.beta(alpha, alpha, size=n)
    lam = lam.reshape(-1, 1, 1)  # Reshape for broadcasting
    
    # Random permutation for mixing partners
    indices = np.random.permutation(n)
    
    # Mixup
    X_mixed = lam * X + (1 - lam) * X[indices]
    
    return X_mixed


def capped_smote(X_flat: ArrayLike, y: ArrayLike, 
                 target_class: str, target_count: int,
                 max_multiplier: int = 5,
                 random_state: int = 42) -> Tuple[ArrayLike, ArrayLike]:
    """
    SMOTE with capped expansion to prevent unrealistic synthetic samples.
    
    Args:
        X_flat: Flattened features (n_samples, n_features)
        y: Class labels (attack types)
        target_class: Class to augment
        target_count: Desired final count
        max_multiplier: Maximum synthetic samples per iteration (e.g., 5× original)
        random_state: Random seed
        
    Returns:
        Augmented X_flat, y
    """
    if not SMOTE_AVAILABLE or _SMOTE is None:
        print(f"   ⚠️  SMOTE not available, skipping capped SMOTE")
        return X_flat, y
    
    # Ensure numpy arrays
    X_current = np.asarray(X_flat).copy()
    y_current = np.asarray(y).copy()
    
    current_count = int((y_current == target_class).sum())
    
    if current_count >= target_count:
        return X_current, y_current
    
    iteration = 0
    max_iterations = 10  # Safety limit
    
    while int((y_current == target_class).sum()) < target_count and iteration < max_iterations:
        current = int((y_current == target_class).sum())
        remaining = target_count - current
        
        # Cap this iteration's synthetic generation
        this_iteration_target = min(
            current + int(current * (max_multiplier - 1)),  # At most 5× current
            current + remaining  # Or whatever we need
        )
        
        if this_iteration_target <= current:
            break
        
        # Adaptive k_neighbors
        k = min(5, current - 1)
        if k < 1:
            k = 1
        
        try:
            smote = _SMOTE(
                sampling_strategy={target_class: this_iteration_target},  # type: ignore[arg-type]
                k_neighbors=k,
                random_state=random_state + iteration
            )
            result = smote.fit_resample(X_current, y_current)  # type: ignore[union-attr]
            X_resampled = np.asarray(result[0])
            y_resampled = np.asarray(result[1])
            X_current = X_resampled
            y_current = y_resampled
        except Exception as e:
            print(f"   ⚠️  SMOTE iteration {iteration} failed: {e}")
            break
        
        iteration += 1
    
    return X_current, y_current


def validate_augmentation_tsne(X_real: ArrayLike, X_synthetic: ArrayLike,
                                class_name: str, output_dir: Path) -> bool:
    """
    Visualize real vs synthetic samples with t-SNE.
    
    Returns True if synthetic appears to overlap with real (good).
    Returns False if synthetic forms separate cluster (bad).
    """
    if not VISUALIZATION_AVAILABLE or _plt is None or _TSNE is None:
        print(f"   ⚠️  matplotlib/sklearn not available, skipping t-SNE validation")
        return True  # Assume OK if we can't check
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten for t-SNE
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_synthetic_flat = X_synthetic.reshape(len(X_synthetic), -1)
    
    # Combine
    X_all = np.vstack([X_real_flat, X_synthetic_flat])
    labels = ['Real'] * len(X_real) + ['Synthetic'] * len(X_synthetic)
    
    # Subsample if too many points
    max_points = 2000
    if len(X_all) > max_points:
        indices = np.random.choice(len(X_all), max_points, replace=False)
        X_all = X_all[indices]
        labels = [labels[i] for i in indices]
    
    print(f"   Running t-SNE for {class_name} ({len(X_all)} points)...")
    
    # t-SNE
    perplexity = min(30, len(X_all) - 1)
    tsne = _TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_embedded = tsne.fit_transform(X_all)
    
    # Plot
    _plt.figure(figsize=(10, 8))
    colors = {'Real': 'blue', 'Synthetic': 'red'}
    
    for label in ['Real', 'Synthetic']:
        mask = np.array(labels) == label
        _plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                   c=colors[label], label=label, alpha=0.6, s=20)
    
    _plt.title(f'{class_name}: Real vs Synthetic (t-SNE)')
    _plt.xlabel('t-SNE 1')
    _plt.ylabel('t-SNE 2')
    _plt.legend()
    
    output_path = output_dir / f'tsne_validation_{class_name}.png'
    _plt.savefig(output_path, dpi=150, bbox_inches='tight')
    _plt.close()
    
    print(f"   Saved: {output_path}")
    
    # Compute overlap score (simple heuristic)
    # Real centroid
    real_mask = np.array(labels) == 'Real'
    synth_mask = np.array(labels) == 'Synthetic'
    
    real_centroid = X_embedded[real_mask].mean(axis=0)
    synth_centroid = X_embedded[synth_mask].mean(axis=0)
    
    # Distance between centroids vs spread
    centroid_dist = np.linalg.norm(real_centroid - synth_centroid)
    real_spread = np.std(X_embedded[real_mask])
    
    overlap_ratio = centroid_dist / (real_spread + 1e-8)
    
    if overlap_ratio < 2.0:
        print(f"   ✅ Good overlap (ratio={overlap_ratio:.2f} < 2.0)")
        return True
    else:
        print(f"   ⚠️  Poor overlap (ratio={overlap_ratio:.2f} >= 2.0) - synthetic may be unrealistic!")
        return False


def hybrid_augment(X: np.ndarray, y: np.ndarray, attack_types: np.ndarray,
                   rare_classes: List[str], target_samples: int,
                   validate_tsne: bool = True,
                   output_dir: Optional[Path] = None,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full hybrid augmentation pipeline:
    1. Jitter (all rare-class samples)
    2. Mixup (within each rare class)
    3. Capped SMOTE (limited synthetic generation)
    
    Args:
        X: Features (n_samples, window, features)
        y: Binary labels
        attack_types: Attack type strings
        rare_classes: List of classes to augment
        target_samples: Target count per class
        validate_tsne: Whether to generate t-SNE plots
        output_dir: Where to save validation plots
        seed: Random seed
        
    Returns:
        Augmented X, y, attack_types
    """
    np.random.seed(seed)
    
    print("\n" + "=" * 60)
    print("HYBRID AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Track original rare samples for t-SNE validation
    original_rare = {}
    
    # Start with copies
    X_aug = np.array(X)  # Convert from mmap to regular array
    y_aug = np.array(y)
    at_aug = np.array(attack_types)
    
    for cls in rare_classes:
        mask = at_aug == cls
        original_count = mask.sum()
        
        if original_count == 0:
            print(f"\n⚠️  {cls}: No samples found, skipping")
            continue
        
        print(f"\n{'─' * 40}")
        print(f"Augmenting: {cls}")
        print(f"Original: {original_count} samples")
        print(f"Target: {target_samples} samples")
        
        # Store original for validation
        X_cls_original = X_aug[mask].copy()
        original_rare[cls] = X_cls_original
        
        # Step 1: Jitter
        print(f"\n   Step 1: Jitter (noise_std=0.01)")
        X_jittered = jitter_augment(X_cls_original, noise_std=0.01)
        
        # Combine original + jittered
        X_combined = np.vstack([X_cls_original, X_jittered])
        y_combined = np.hstack([y_aug[mask], y_aug[mask]])
        at_combined = np.hstack([at_aug[mask], at_aug[mask]])
        
        current_count = len(X_combined)
        print(f"   After jitter: {current_count} samples")
        
        # Step 2: Mixup (if still under target)
        if current_count < target_samples:
            print(f"\n   Step 2: Mixup (alpha=0.2)")
            n_mixup = min(current_count, target_samples - current_count)
            
            # Generate mixup samples
            X_mixup = mixup_augment(X_combined[:n_mixup], alpha=0.2)
            y_mixup = y_combined[:n_mixup]
            at_mixup = at_combined[:n_mixup]
            
            X_combined = np.vstack([X_combined, X_mixup])
            y_combined = np.hstack([y_combined, y_mixup])
            at_combined = np.hstack([at_combined, at_mixup])
            
            current_count = len(X_combined)
            print(f"   After mixup: {current_count} samples")
        
        # Step 3: Capped SMOTE (if still under target)
        if current_count < target_samples and SMOTE_AVAILABLE:
            print(f"\n   Step 3: Capped SMOTE (max 5× per iteration)")
            
            # Flatten for SMOTE
            X_flat = X_combined.reshape(len(X_combined), -1)
            
            # Create temporary labels for SMOTE (just this class vs "other")
            temp_labels = np.array([cls] * len(X_combined))
            
            X_smote_flat, _ = capped_smote(
                X_flat, temp_labels, 
                target_class=cls, 
                target_count=target_samples,
                max_multiplier=5,
                random_state=seed
            )
            
            # Reshape back
            original_shape = X_combined.shape
            X_combined = X_smote_flat.reshape(-1, original_shape[1], original_shape[2])
            y_combined = np.full(len(X_combined), y_aug[mask][0])
            at_combined = np.full(len(X_combined), cls)
            
            current_count = len(X_combined)
            print(f"   After SMOTE: {current_count} samples")
        
        # Remove original samples from main arrays, add augmented
        non_cls_mask = at_aug != cls
        X_aug = np.vstack([X_aug[non_cls_mask], X_combined])
        y_aug = np.hstack([y_aug[non_cls_mask], y_combined])
        at_aug = np.hstack([at_aug[non_cls_mask], at_combined])
        
        print(f"\n   Final {cls} count: {(at_aug == cls).sum()}")
    
    # t-SNE Validation
    if validate_tsne and output_dir:
        print(f"\n{'=' * 60}")
        print("t-SNE VALIDATION")
        print("=" * 60)
        
        all_valid = True
        for cls in rare_classes:
            if cls in original_rare and (at_aug == cls).sum() > len(original_rare[cls]):
                X_real = original_rare[cls]
                
                # Get synthetic samples (all except original count)
                cls_mask = at_aug == cls
                X_all_cls = X_aug[cls_mask]
                X_synthetic = X_all_cls[len(X_real):]
                
                if len(X_synthetic) > 0:
                    valid = validate_augmentation_tsne(
                        X_real, X_synthetic, cls, output_dir
                    )
                    all_valid = all_valid and valid
        
        if not all_valid:
            print("\n⚠️  WARNING: Some classes show poor t-SNE overlap!")
            print("   Consider reducing target_samples or tweaking parameters.")
    
    # Shuffle final dataset
    indices = np.random.permutation(len(X_aug))
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]
    at_aug = at_aug[indices]
    
    return X_aug, y_aug, at_aug


def compute_class_weights(attack_types: np.ndarray, 
                          max_weight: float = 50.0) -> Dict[str, float]:
    """
    Compute balanced class weights with cap to prevent training instability.
    
    Args:
        attack_types: Array of attack type strings
        max_weight: Maximum weight (prevents gradient explosion)
        
    Returns:
        Dictionary of class -> weight
    """
    counter = Counter(attack_types)
    total = len(attack_types)
    n_classes = len(counter)
    
    weights = {}
    for cls, count in counter.items():
        # Balanced weight formula
        weight = total / (n_classes * count)
        # Cap the weight
        weights[cls] = min(weight, max_weight)
    
    # Normalize so minimum weight is 1.0
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    return weights


def main():
    parser = argparse.ArgumentParser(description="Hybrid augmentation for rare classes")
    parser.add_argument('--data-dir', type=str, default='data/processed/cic_ids_2017_v2',
                        help='Data directory')
    parser.add_argument('--rare-classes', type=str, nargs='+', 
                        default=['Bot', 'SSH-Patator'],
                        help='Classes to augment')
    parser.add_argument('--target-samples', type=int, default=500,
                        help='Target samples per rare class')
    parser.add_argument('--validate-tsne', action='store_true',
                        help='Generate t-SNE validation plots')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as data-dir with _augmented suffix)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup of original data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze without saving')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir.parent / f'{data_dir.name}_augmented'
    reports_dir = Path('reports')
    
    print("\n" + "=" * 60)
    print("HYBRID RARE-CLASS AUGMENTATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rare classes: {args.rare_classes}")
    print(f"Target samples: {args.target_samples}")
    print(f"t-SNE validation: {args.validate_tsne}")
    
    # Load data
    print("\n📂 Loading data...")
    X, y, attack_types = load_data(data_dir)
    print(f"   Loaded: {X.shape}")
    
    # Show current distribution
    print("\n📊 Current distribution:")
    counter = Counter(attack_types)
    for cls in args.rare_classes:
        count = counter.get(cls, 0)
        status = "✅" if count >= args.target_samples else "⚠️ RARE"
        print(f"   {status} {cls}: {count} samples")
    
    # Run hybrid augmentation
    X_aug, y_aug, at_aug = hybrid_augment(
        X, y, attack_types,
        rare_classes=args.rare_classes,
        target_samples=args.target_samples,
        validate_tsne=args.validate_tsne,
        output_dir=reports_dir if args.validate_tsne else None,
        seed=args.seed
    )
    
    # Show final distribution
    print("\n📊 Final distribution:")
    counter_final = Counter(at_aug)
    for cls in args.rare_classes:
        count = counter_final.get(cls, 0)
        original = counter.get(cls, 0)
        print(f"   {cls}: {original} → {count} samples")
    
    # Compute class weights
    print("\n⚖️  Computing class weights (capped at 50×)...")
    class_weights = compute_class_weights(at_aug, max_weight=50.0)
    for cls in args.rare_classes:
        if cls in class_weights:
            print(f"   {cls}: {class_weights[cls]:.2f}×")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes saved")
        return 0
    
    # Save augmented data
    print("\n💾 Saving augmented data...")
    
    # Create output directory structure
    output_train_dir = output_dir / 'train'
    output_train_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy non-training data (val, test)
    for subdir in ['val', 'test']:
        src = data_dir / subdir
        dst = output_dir / subdir
        if src.exists() and not dst.exists():
            print(f"   Copying {subdir}/...")
            shutil.copytree(src, dst)
    
    # Save augmented training data
    print(f"   Saving X.npy ({X_aug.shape})...")
    np.save(output_train_dir / 'X.npy', X_aug)
    
    print(f"   Saving y.npy ({y_aug.shape})...")
    np.save(output_train_dir / 'y.npy', y_aug)
    
    print(f"   Saving attack_types.npy ({at_aug.shape})...")
    np.save(output_train_dir / 'attack_types.npy', at_aug)
    
    # Save class weights
    weights_path = output_dir / 'class_weights.json'
    classes = sorted(class_weights.keys())
    weights_data = {
        'class_to_idx': {cls: i for i, cls in enumerate(classes)},
        'idx_to_class': {str(i): cls for i, cls in enumerate(classes)},
        'weights_dict': class_weights,
        'weights_list': [class_weights[cls] for cls in classes]
    }
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    print(f"   Saved class weights: {weights_path}")
    
    # Save augmentation metadata
    metadata = {
        'source_dir': str(data_dir),
        'rare_classes': args.rare_classes,
        'target_samples': args.target_samples,
        'seed': args.seed,
        'original_counts': {cls: counter.get(cls, 0) for cls in args.rare_classes},
        'final_counts': {cls: counter_final.get(cls, 0) for cls in args.rare_classes},
        'total_original': len(X),
        'total_final': len(X_aug)
    }
    metadata_path = output_dir / 'augmentation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata: {metadata_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Total samples: {len(X)} → {len(X_aug)}")
    
    if args.validate_tsne:
        print(f"\n📊 t-SNE plots saved to: {reports_dir}/")
        print("   Review plots to verify synthetic samples overlap with real!")
    
    print("\n📋 Next steps:")
    print(f"   1. Review t-SNE plots in {reports_dir}/")
    print(f"   2. Train teacher: python scripts/train_teacher_balanced.py --data-dir {output_dir}")
    print(f"   3. Verify Bot/SSH-Patator recall > 60%")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

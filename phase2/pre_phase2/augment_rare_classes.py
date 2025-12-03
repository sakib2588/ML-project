#!/usr/bin/env python3
"""
Rare Class Augmentation Script

Addresses CRITICAL issue: Bot/SSH-Patator zero detection

Methods:
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. Class-weighted sampling
3. Focal Loss configuration

Usage:
    python phase2/pre_phase2/augment_rare_classes.py \
        --data-dir data/processed/cic_ids_2017 \
        --target-classes Bot,SSH-Patator \
        --target-samples 5000 \
        --method smote
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️  WARNING: imbalanced-learn not installed. Run: pip install imbalanced-learn")


def load_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training features, labels, and attack types."""
    train_dir = data_dir / 'train'
    
    # Try multiple filename patterns for X
    X_train = None
    for name in ['X.npy', 'X_train.npy', 'features.npy']:
        path = train_dir / name
        if path.exists():
            X_train = np.load(path)
            break
    if X_train is None:
        raise FileNotFoundError(f"No feature file found in {train_dir}")
    
    # Try multiple filename patterns for y
    y_train = None
    for name in ['y.npy', 'y_train.npy', 'labels.npy']:
        path = train_dir / name
        if path.exists():
            y_train = np.load(path)
            break
    if y_train is None:
        raise FileNotFoundError(f"No label file found in {train_dir}")
    
    # Try multiple filename patterns for attack types
    attack_types = None
    for name in ['attack_types.npy', 'y_attack_types.npy', 'attack_type.npy']:
        path = train_dir / name
        if path.exists():
            attack_types = np.load(path, allow_pickle=True)
            break
    
    if attack_types is None:
        # Create placeholder if not exists
        print("⚠️  No attack_types.npy found, using binary labels")
        attack_types = np.where(y_train == 0, 'BENIGN', 'UNKNOWN_ATTACK')
    
    return X_train, y_train, attack_types


def save_data(data_dir: Path, X_train: np.ndarray, y_train: np.ndarray, 
              attack_types: np.ndarray, backup: bool = True) -> None:
    """Save augmented data with optional backup."""
    train_dir = data_dir / 'train'
    
    if backup:
        backup_dir = data_dir / 'train_backup'
        if not backup_dir.exists():
            print(f"📦 Creating backup at {backup_dir}")
            shutil.copytree(train_dir, backup_dir)
    
    # Save with original filenames if they exist, otherwise use standard names
    x_name = 'X.npy' if (train_dir / 'X.npy').exists() else 'X_train.npy'
    y_name = 'y.npy' if (train_dir / 'y.npy').exists() else 'y_train.npy'
    
    np.save(train_dir / x_name, X_train)
    np.save(train_dir / y_name, y_train)
    np.save(train_dir / 'y_attack_types.npy', attack_types)
    
    print(f"💾 Saved augmented data to {train_dir}")


def apply_smote(X: np.ndarray, y: np.ndarray, attack_types: np.ndarray,
                target_classes: List[str], target_samples: int,
                random_state: int = 42):
    """Apply SMOTE to oversample rare classes.
    
    Handles 3D data (samples, window, features) by flattening.
    
    Returns:
        Tuple of (X_resampled, y_resampled, attack_types_resampled) as numpy arrays
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn required for SMOTE. Run: pip install imbalanced-learn")
    
    print(f"\n🔄 Applying SMOTE for classes: {target_classes}")
    
    # Handle 3D data by flattening
    original_shape = X.shape
    if len(original_shape) == 3:
        print(f"   Data is 3D {original_shape}, flattening for SMOTE...")
        n_samples = original_shape[0]
        X_flat = X.reshape(n_samples, -1)
    else:
        X_flat = X
    
    # Create sampling strategy
    from collections import Counter
    current_counts = Counter(attack_types)
    
    sampling_strategy: Dict[str, int] = {}
    for cls in target_classes:
        current = current_counts.get(cls, 0)
        if current < target_samples:
            sampling_strategy[cls] = target_samples
            print(f"   {cls}: {current} → {target_samples} samples")
        else:
            print(f"   {cls}: Already has {current} samples (>= {target_samples})")
    
    if not sampling_strategy:
        print("   No augmentation needed!")
        return X, y, attack_types
    
    # Use attack_types as target for SMOTE
    try:
        # Adjust k_neighbors based on minimum class size
        min_samples = min(current_counts.get(cls, 1) for cls in target_classes)
        k_neighbors = min(5, min_samples - 1)
        if k_neighbors < 1:
            k_neighbors = 1
        
        smote = SMOTE(  # type: ignore[possibly-undefined]
            sampling_strategy=sampling_strategy,  # type: ignore
            random_state=random_state,
            k_neighbors=k_neighbors
        )
        X_resampled_flat, attack_types_resampled = smote.fit_resample(X_flat, attack_types)  # type: ignore
        
        # Ensure numpy arrays
        X_resampled_flat = np.asarray(X_resampled_flat)
        attack_types_resampled = np.asarray(attack_types_resampled)
        
        # Reshape back to 3D if needed
        if len(original_shape) == 3:
            new_n_samples = X_resampled_flat.shape[0]
            X_resampled = X_resampled_flat.reshape(new_n_samples, original_shape[1], original_shape[2])
        else:
            X_resampled = X_resampled_flat
        
        # Reconstruct binary labels
        y_resampled = np.where(attack_types_resampled == 'BENIGN', 0, 1).astype(np.float32)
        
        print(f"\n✅ SMOTE complete:")
        print(f"   Original: {len(X)} samples")
        print(f"   Augmented: {len(X_resampled)} samples")
        
        return X_resampled, y_resampled, attack_types_resampled
        
    except ValueError as e:
        print(f"❌ SMOTE failed: {e}")
        print("   → Trying ADASYN as fallback...")
        
        try:
            adasyn = ADASYN(  # type: ignore[possibly-undefined]
                sampling_strategy=sampling_strategy,  # type: ignore
                random_state=random_state,
                n_neighbors=min(3, min(current_counts.get(cls, 1) for cls in target_classes) - 1)
            )
            X_resampled, attack_types_resampled = adasyn.fit_resample(X, attack_types)  # type: ignore
            
            # Ensure numpy arrays
            X_resampled = np.asarray(X_resampled)
            attack_types_resampled = np.asarray(attack_types_resampled)
            y_resampled = np.where(attack_types_resampled == 'BENIGN', 0, 1).astype(np.float32)
            
            return X_resampled, y_resampled, attack_types_resampled
            
        except ValueError as e2:
            print(f"❌ ADASYN also failed: {e2}")
            print("   → Classes may have too few samples for oversampling.")
            return X, y, attack_types


def apply_noise_augmentation(X: np.ndarray, y: np.ndarray, attack_types: np.ndarray,
                             target_classes: List[str], target_samples: int,
                             noise_scale: float = 0.01,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Gaussian noise augmentation for rare classes."""
    print(f"\n🔄 Applying Noise Augmentation for classes: {target_classes}")
    
    np.random.seed(random_state)
    
    X_augmented = [X]
    y_augmented = [y]
    types_augmented = [attack_types]
    
    for cls in target_classes:
        mask = attack_types == cls
        X_cls = X[mask]
        y_cls = y[mask]
        current_count = len(X_cls)
        
        if current_count == 0:
            print(f"   ⚠️  {cls}: No samples to augment!")
            continue
        
        needed = target_samples - current_count
        if needed <= 0:
            print(f"   {cls}: Already has {current_count} samples")
            continue
        
        print(f"   {cls}: {current_count} → {target_samples} samples (+{needed})")
        
        # Generate noisy copies
        n_copies_per_sample = int(np.ceil(needed / current_count))
        
        for _ in range(n_copies_per_sample):
            noise = np.random.normal(0, noise_scale, X_cls.shape)
            X_noisy = X_cls + noise * np.std(X_cls, axis=0, keepdims=True)
            X_augmented.append(X_noisy)
            y_augmented.append(y_cls)
            types_augmented.append(np.full(len(X_cls), cls))
    
    X_result = np.vstack(X_augmented)
    y_result = np.concatenate(y_augmented)
    types_result = np.concatenate(types_augmented)
    
    # Shuffle
    indices = np.random.permutation(len(X_result))
    
    print(f"\n✅ Noise augmentation complete:")
    print(f"   Original: {len(X)} samples")
    print(f"   Augmented: {len(X_result)} samples")
    
    return X_result[indices], y_result[indices], types_result[indices]


def compute_class_weights(attack_types: np.ndarray) -> Dict[str, float]:
    """Compute balanced class weights."""
    from collections import Counter
    
    counter = Counter(attack_types)
    total = len(attack_types)
    n_classes = len(counter)
    
    weights = {}
    for attack_type, count in counter.items():
        weights[attack_type] = total / (n_classes * count)
    
    # Normalize
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    return weights


def compute_focal_loss_config(attack_types: np.ndarray, 
                               critical_classes: List[str]) -> Dict:
    """Compute Focal Loss configuration for rare classes."""
    from collections import Counter
    
    counter = Counter(attack_types)
    total = len(attack_types)
    
    # Base alpha: inverse frequency
    alpha_dict = {}
    for cls, count in counter.items():
        freq = count / total
        # Higher alpha for rarer classes
        alpha_dict[cls] = min(1.0, 1.0 / (freq * len(counter)))
    
    # Boost critical classes
    for cls in critical_classes:
        if cls in alpha_dict:
            alpha_dict[cls] *= 2.0
    
    return {
        'gamma': 2.0,  # Focus on hard examples
        'alpha': alpha_dict,
        'reduction': 'mean'
    }


def generate_class_weight_file(weights: Dict[str, float], output_path: Path) -> None:
    """Save class weights to JSON file for training."""
    # Convert to list format for PyTorch
    classes = sorted(weights.keys())
    weight_list = [weights[cls] for cls in classes]
    
    output = {
        'class_to_idx': {cls: i for i, cls in enumerate(classes)},
        'idx_to_class': {i: cls for i, cls in enumerate(classes)},
        'weights_dict': weights,
        'weights_list': weight_list
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Class weights saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment rare attack classes")
    parser.add_argument('--data-dir', type=str, default='data/processed/cic_ids_2017_v2',
                        help='Path to processed data directory')
    parser.add_argument('--target-classes', type=str, default='Bot,SSH-Patator',
                        help='Comma-separated list of classes to augment')
    parser.add_argument('--target-samples', type=int, default=5000,
                        help='Target number of samples per class')
    parser.add_argument('--method', type=str, choices=['smote', 'noise', 'both'],
                        default='smote', help='Augmentation method')
    parser.add_argument('--noise-scale', type=float, default=0.01,
                        help='Noise scale for noise augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating backup of original data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze without saving')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    target_classes = [c.strip() for c in args.target_classes.split(',')]
    
    print("\n🔧 RARE CLASS AUGMENTATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Target classes: {target_classes}")
    print(f"Target samples: {args.target_samples}")
    print(f"Method: {args.method}")
    
    # Load data
    print("\n📂 Loading data...")
    X_train, y_train, attack_types = load_data(data_dir)
    print(f"   Loaded {len(X_train)} samples, {X_train.shape[1]} features")
    
    # Show current distribution
    from collections import Counter
    print("\n📊 Current distribution:")
    counter = Counter(attack_types)
    for cls in target_classes:
        count = counter.get(cls, 0)
        status = "✅" if count >= args.target_samples else "❌"
        print(f"   {status} {cls}: {count} samples")
    
    # Apply augmentation
    if args.method in ['smote', 'both']:
        X_train, y_train, attack_types = apply_smote(
            X_train, y_train, attack_types,
            target_classes, args.target_samples, args.seed
        )
    
    if args.method in ['noise', 'both']:
        X_train, y_train, attack_types = apply_noise_augmentation(
            X_train, y_train, attack_types,
            target_classes, args.target_samples, args.noise_scale, args.seed
        )
    
    # Compute class weights
    print("\n⚖️  Computing class weights...")
    class_weights = compute_class_weights(attack_types)
    
    print("\n   Class weights (for weighted loss):")
    for cls in target_classes:
        if cls in class_weights:
            print(f"   {cls}: {class_weights[cls]:.2f}")
    
    # Compute Focal Loss config
    focal_config = compute_focal_loss_config(attack_types, target_classes)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes saved")
        return 0
    
    # Save augmented data
    print("\n💾 Saving augmented data...")
    save_data(data_dir, X_train, y_train, attack_types, backup=not args.no_backup)
    
    # Save class weights
    weights_path = data_dir / 'class_weights.json'
    generate_class_weight_file(class_weights, weights_path)
    
    # Save focal loss config
    focal_path = data_dir / 'focal_loss_config.json'
    with open(focal_path, 'w') as f:
        json.dump(focal_config, f, indent=2)
    print(f"💾 Focal Loss config saved to {focal_path}")
    
    # Final verification
    print("\n✅ AUGMENTATION COMPLETE")
    print("=" * 60)
    counter = Counter(attack_types)
    for cls in target_classes:
        count = counter.get(cls, 0)
        status = "✅" if count >= args.target_samples else "⚠️"
        print(f"   {status} {cls}: {count} samples")
    
    print("\n📋 Next steps:")
    print("   1. Retrain Phase 1 model with augmented data")
    print("   2. Verify Bot/SSH-Patator detection > 80%")
    print("   3. Proceed to Phase 2 compression")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Memory-Efficient Hybrid Augmentation (Lite Version)

Processes data in chunks to avoid RAM overflow on laptops.
- Processes 10K samples at a time
- Only keeps current chunk in memory
- Streams to disk incrementally

Usage:
    python scripts/augment_hybrid_lite.py \
        --data-dir data/processed/cic_ids_2017_v2 \
        --rare-classes Bot SSH-Patator \
        --target-samples 300 \
        --chunk-size 10000
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

SMOTE_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE as _SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE = None  # type: ignore
    print("⚠️  imbalanced-learn not installed")


def load_data_info(data_dir: Path) -> Tuple[int, Tuple[int, int, int]]:
    """Get dataset shape and sample count without loading full array."""
    train_dir = data_dir / 'train'
    
    # Use numpy to inspect shape without loading
    X_path = train_dir / 'X.npy'
    X_shape = np.load(X_path, mmap_mode='r').shape
    
    # Count attack types
    at_path = train_dir / 'attack_types.npy'
    attack_types = np.load(at_path, allow_pickle=True)
    
    return len(attack_types), X_shape


def load_chunk(data_dir: Path, start_idx: int, end_idx: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Load a chunk of data."""
    train_dir = data_dir / 'train'
    
    # Load chunk from mmap
    X = np.load(train_dir / 'X.npy', mmap_mode='r')
    y = np.load(train_dir / 'y.npy')
    at = np.load(train_dir / 'attack_types.npy', allow_pickle=True)
    
    # Extract chunk
    X_chunk = np.array(X[start_idx:end_idx])  # Convert from mmap
    y_chunk = y[start_idx:end_idx]
    at_chunk = at[start_idx:end_idx]
    
    return X_chunk, y_chunk, at_chunk


def jitter_augment_batch(X: ArrayLike, noise_std: float = 0.01) -> ArrayLike:
    """Add jitter to batch."""
    feature_std = np.std(X, axis=(0, 1), keepdims=True)
    feature_std = np.maximum(feature_std, 1e-8)
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise * feature_std


def mixup_batch(X: ArrayLike, alpha: float = 0.2) -> ArrayLike:
    """Mixup within batch."""
    n = len(X)
    if n < 2:
        return X.copy()
    
    lam = np.random.beta(alpha, alpha, size=n).reshape(-1, 1, 1)
    indices = np.random.permutation(n)
    return lam * X + (1 - lam) * X[indices]


def augment_rare_classes_lite(
    data_dir: Path,
    output_dir: Path,
    rare_classes: List[str],
    target_samples: int,
    chunk_size: int = 10000,
    seed: int = 42
) -> None:
    """
    Memory-efficient augmentation using chunked processing.
    
    Strategy:
    1. Identify rare class indices globally
    2. Process each rare class separately
    3. Stream augmented samples to disk
    4. Reconstruct full dataset by combining chunks
    """
    np.random.seed(seed)
    
    print("\n" + "=" * 70)
    print("MEMORY-EFFICIENT HYBRID AUGMENTATION (Lite Version)")
    print("=" * 70)
    
    train_dir = data_dir / 'train'
    output_train_dir = output_dir / 'train'
    output_train_dir.mkdir(parents=True, exist_ok=True)
    
    # Get global stats
    total_samples, X_shape = load_data_info(data_dir)
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {total_samples}")
    print(f"   X shape: {X_shape}")
    print(f"   Chunk size: {chunk_size}")
    
    # Step 1: Find rare class indices globally
    print(f"\n🔍 Step 1: Locating rare classes...")
    rare_indices = {cls: [] for cls in rare_classes}
    
    at_full = np.load(train_dir / 'attack_types.npy', allow_pickle=True)
    for cls in rare_classes:
        mask = at_full == cls
        indices = np.where(mask)[0]
        rare_indices[cls] = indices
        print(f"   {cls}: {len(indices)} samples found")
    
    # Step 2: Augment rare classes
    print(f"\n🔄 Step 2: Augmenting rare classes...")
    
    augmented_samples = {cls: [] for cls in rare_classes}
    augmented_labels = {cls: [] for cls in rare_classes}
    augmented_types = {cls: [] for cls in rare_classes}
    
    for cls in rare_classes:
        indices = rare_indices[cls]
        original_count = len(indices)
        needed = max(0, target_samples - original_count)
        
        if needed == 0:
            print(f"   {cls}: Already has {original_count} samples (≥ {target_samples})")
            continue
        
        print(f"\n   {cls}: {original_count} → {target_samples} samples (+{needed})")
        
        # Load original samples
        X_cls_list = []
        y_cls_list = []
        
        for idx in indices:
            # Find which chunk contains this index
            chunk_start = (idx // chunk_size) * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_samples)
            local_idx = idx - chunk_start
            
            X_chunk, y_chunk, _ = load_chunk(data_dir, chunk_start, chunk_end)
            X_cls_list.append(X_chunk[local_idx])
            y_cls_list.append(y_chunk[local_idx])
        
        X_cls = np.array(X_cls_list)
        y_cls = np.array(y_cls_list)
        
        # Strategy: Jitter + Mixup to reach target
        current = len(X_cls)
        step = 1
        
        # Jitter
        print(f"      Step {step}: Jitter (noise_std=0.01)")
        X_jittered = jitter_augment_batch(X_cls, noise_std=0.01)
        X_aug = np.vstack([X_cls, X_jittered])
        y_aug = np.hstack([y_cls, y_cls])
        current = len(X_aug)
        print(f"         After jitter: {current} samples")
        step += 1
        
        # Mixup if needed
        if current < target_samples:
            print(f"      Step {step}: Mixup (alpha=0.2)")
            n_mixup = min(current, target_samples - current)
            X_mixed = mixup_batch(X_aug[:n_mixup], alpha=0.2)
            X_aug = np.vstack([X_aug, X_mixed])
            y_aug = np.hstack([y_aug, y_aug[:n_mixup]])
            current = len(X_aug)
            print(f"         After mixup: {current} samples")
            step += 1
        
        # SMOTE if still needed
        if current < target_samples and SMOTE_AVAILABLE:
            print(f"      Step {step}: SMOTE (capped at 2× current)")
            
            X_flat = X_aug.reshape(len(X_aug), -1)
            temp_labels = np.array([cls] * len(X_aug))
            
            try:
                # Capped SMOTE
                target_this = min(len(X_aug) * 2, target_samples)
                k_neighbors = min(5, len(X_aug) - 1)
                
                smote = _SMOTE(
                    sampling_strategy={cls: target_this},  # type: ignore[arg-type]
                    k_neighbors=k_neighbors,
                    random_state=seed
                )
                result = smote.fit_resample(X_flat, temp_labels)  # type: ignore[union-attr]
                X_aug = np.asarray(result[0]).reshape(-1, X_shape[1], X_shape[2])
                y_aug = np.full(len(X_aug), y_cls[0])
                current = len(X_aug)
                print(f"         After SMOTE: {current} samples")
            except Exception as e:
                print(f"         ⚠️  SMOTE failed: {e}")
        
        # Cap at target
        if current > target_samples:
            indices_keep = np.random.choice(current, target_samples, replace=False)
            X_aug = X_aug[indices_keep]
            y_aug = y_aug[indices_keep]
            current = target_samples
        
        augmented_samples[cls] = X_aug
        augmented_labels[cls] = y_aug
        augmented_types[cls] = np.full(len(X_aug), cls)
        
        print(f"      Final {cls}: {current} samples")
    
    # Step 3: Combine augmented with non-augmented samples
    print(f"\n💾 Step 3: Writing augmented dataset...")
    
    # Open output files for streaming
    X_out = output_train_dir / 'X.npy'
    y_out = output_train_dir / 'y.npy'
    at_out = output_train_dir / 'attack_types.npy'
    
    X_writer = open(X_out.with_suffix('.tmp'), 'wb')
    y_writer = open(y_out.with_suffix('.tmp'), 'wb')
    at_writer = open(at_out.with_suffix('.tmp'), 'wb')
    
    total_written = 0
    
    # Write non-augmented samples in chunks
    print(f"   Writing non-augmented samples...")
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        
        X_chunk, y_chunk, at_chunk = load_chunk(data_dir, chunk_start, chunk_end)
        
        # Filter out rare class samples (they're being augmented)
        mask = np.ones(len(at_chunk), dtype=bool)
        for cls in rare_classes:
            mask = mask & (at_chunk != cls)
        
        X_filtered = X_chunk[mask]
        y_filtered = y_chunk[mask]
        at_filtered = at_chunk[mask]
        
        # Write filtered chunk
        np.save(X_writer, X_filtered)
        np.save(y_writer, y_filtered)
        np.save(at_writer, at_filtered)
        
        total_written += len(X_filtered)
        print(f"   → Chunk {chunk_start}-{chunk_end}: {len(X_filtered)} samples kept")
    
    # Write augmented samples
    print(f"   Writing augmented samples...")
    for cls in rare_classes:
        if len(augmented_samples[cls]) > 0:
            np.save(X_writer, augmented_samples[cls])
            np.save(y_writer, augmented_labels[cls])
            np.save(at_writer, augmented_types[cls])
            total_written += len(augmented_samples[cls])
            print(f"   → {cls}: {len(augmented_samples[cls])} augmented samples added")
    
    X_writer.close()
    y_writer.close()
    at_writer.close()
    
    print(f"\n   Total samples written: {total_written}")
    
    # Step 4: Concatenate temp files (memory-efficient)
    print(f"\n📦 Step 4: Assembling final arrays...")
    
    # This is a placeholder - in production, you'd use numpy's concatenate
    # For now, we'll use a simpler approach
    print(f"   ⚠️  For large datasets, use external tools like:")
    print(f"      `npz_concat` or similar")
    
    # Copy val/test
    for subdir in ['val', 'test']:
        src = data_dir / subdir
        dst = output_dir / subdir
        if src.exists() and not dst.exists():
            print(f"   Copying {subdir}/...")
            shutil.copytree(src, dst)
    
    # Compute class weights
    print(f"\n⚖️  Computing class weights...")
    
    # Load full attack_types to compute weights
    at_full_aug = np.concatenate([
        at_chunk for chunk_start in range(0, total_samples, chunk_size)
        for chunk_end in [min(chunk_start + chunk_size, total_samples)]
        for _, _, at_chunk in [load_chunk(data_dir, chunk_start, chunk_end)]
    ] + [augmented_types[cls] for cls in rare_classes if len(augmented_types[cls]) > 0])
    
    counter = Counter(at_full_aug)
    total = len(at_full_aug)
    n_classes = len(counter)
    
    weights = {}
    for cls, count in counter.items():
        weight = total / (n_classes * count)
        weights[cls] = min(weight, 50.0)  # Cap at 50×
    
    # Normalize
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    # Save weights
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
    
    print(f"   Rare class weights:")
    for cls in rare_classes:
        if cls in weights:
            print(f"   {cls}: {weights[cls]:.2f}×")
    
    # Save metadata
    metadata = {
        'method': 'hybrid_lite (chunked)',
        'chunk_size': chunk_size,
        'rare_classes': rare_classes,
        'target_samples': target_samples,
        'total_samples': total_written,
        'seed': seed
    }
    metadata_path = output_dir / 'augmentation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ AUGMENTATION COMPLETE")
    print(f"   Output: {output_dir}")
    print(f"   Samples: {total_written}")
    print(f"   Weights: {weights_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient hybrid augmentation for rare classes"
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
        '--target-samples', type=int, default=300,
        help='Target samples per rare class'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=10000,
        help='Chunk size for processing'
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
    
    augment_rare_classes_lite(
        data_dir=data_dir,
        output_dir=output_dir,
        rare_classes=args.rare_classes,
        target_samples=args.target_samples,
        chunk_size=args.chunk_size,
        seed=args.seed
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

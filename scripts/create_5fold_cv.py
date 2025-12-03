#!/usr/bin/env python3
"""
Create 5-Fold Cross-Validation Splits for Research-Grade Evaluation.

This script:
1. Loads the preprocessed CIC-IDS-2017 data
2. Creates 5 stratified train/val/test splits
3. Applies SMOTE to each fold's training data
4. Saves each fold separately for independent training

For research paper: "5-fold stratified cross-validation was used to ensure
robust evaluation across different data partitions."
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import json
import shutil
import gc

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    SOURCE_DIR = Path("/home/sakib/ids-compression/data/processed/cic_ids_2017_v2")
    OUTPUT_BASE = Path("/home/sakib/ids-compression/data/processed")
    
    # CV settings
    N_FOLDS = 5
    RANDOM_STATE = 42
    VAL_RATIO = 0.15  # 15% of training data for validation
    
    # SMOTE settings
    SMOTE_TARGETS = {
        "Bot": 1000,
        "SSH-Patator": 1000
    }
    SMOTE_K_NEIGHBORS = 5

def log(msg: str):
    """Print timestamped message"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_full_dataset():
    """Load and combine train/val/test into full dataset using memory mapping."""
    log("Loading full dataset (memory-mapped for low RAM)...")
    
    train_dir = Config.SOURCE_DIR / "train"
    val_dir = Config.SOURCE_DIR / "val"
    test_dir = Config.SOURCE_DIR / "test"
    
    # First pass: get shapes
    shapes = []
    for split_dir in [train_dir, val_dir, test_dir]:
        X = np.load(split_dir / "X.npy", mmap_mode='r')
        shapes.append(X.shape[0])
        del X
    
    total_samples = sum(shapes)
    log(f"Total samples: {total_samples}")
    
    # Load attack_types (small, fits in memory)
    attack_parts = []
    for split_dir in [train_dir, val_dir, test_dir]:
        attack_parts.append(np.load(split_dir / "attack_types.npy", allow_pickle=True))
    attack_full = np.concatenate(attack_parts, axis=0)
    del attack_parts
    
    # Load y (small, fits in memory)
    y_parts = []
    for split_dir in [train_dir, val_dir, test_dir]:
        y_parts.append(np.load(split_dir / "y.npy"))
    y_full = np.concatenate(y_parts, axis=0)
    del y_parts
    
    log(f"Attack types loaded: {len(attack_full)}")
    log(f"Class distribution:")
    for cls, count in sorted(Counter(attack_full).items(), key=lambda x: -x[1]):
        log(f"  {cls}: {count}")
    
    # Return paths instead of full X array
    return shapes, y_full, attack_full

def apply_smote(X_train: np.ndarray, y_train: np.ndarray, 
                attack_types: np.ndarray, fold_num: int) -> tuple:
    """Apply SMOTE to rare classes in training data."""
    from imblearn.over_sampling import SMOTE
    
    log(f"  Applying SMOTE to Fold {fold_num}...")
    
    # Check which classes need augmentation
    counts = Counter(attack_types)
    classes_to_augment = []
    
    for cls, target in Config.SMOTE_TARGETS.items():
        if cls in counts and counts[cls] < target:
            classes_to_augment.append(cls)
            log(f"    {cls}: {counts[cls]} -> {target}")
    
    if not classes_to_augment:
        log("    No SMOTE needed")
        return X_train, y_train, attack_types
    
    # Flatten 3D to 2D for SMOTE
    n_samples, window_size, n_features = X_train.shape
    X_flat = X_train.reshape(n_samples, -1)
    
    # Build sampling strategy as dict
    sampling_strategy: dict = {}
    for cls in classes_to_augment:
        if cls in counts:
            sampling_strategy[cls] = int(Config.SMOTE_TARGETS[cls])
    
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,  # type: ignore
        k_neighbors=min(Config.SMOTE_K_NEIGHBORS, min(counts[c] for c in classes_to_augment) - 1),
        random_state=Config.RANDOM_STATE + fold_num
    )
    
    X_resampled, attack_resampled = smote.fit_resample(X_flat, attack_types)  # type: ignore
    
    # Convert back to numpy if needed
    if hasattr(X_resampled, 'values'):
        X_resampled = X_resampled.values
    if hasattr(attack_resampled, 'values'):
        attack_resampled = attack_resampled.values
    
    # Reshape back to 3D
    X_resampled = np.array(X_resampled).reshape(-1, window_size, n_features)
    
    # Recreate binary labels
    attack_resampled_arr = np.array(attack_resampled)
    y_resampled = (attack_resampled_arr != "BENIGN").astype(np.float32)
    
    log(f"    SMOTE complete: {n_samples} -> {len(X_resampled)}")
    
    return X_resampled, y_resampled, attack_resampled

def create_fold_lowmem(shapes: list, y_full: np.ndarray, attack_full: np.ndarray,
                       train_idx: np.ndarray, test_idx: np.ndarray, fold_num: int):
    """Create a single fold with minimal memory usage - loads X only when needed."""
    
    fold_dir = Config.OUTPUT_BASE / f"fold_{fold_num}"
    
    log(f"\n{'='*60}")
    log(f"Creating Fold {fold_num}")
    log(f"{'='*60}")
    
    # Split indices
    y_trainval = y_full[train_idx]
    attack_trainval = attack_full[train_idx]
    
    y_test = y_full[test_idx]
    attack_test = attack_full[test_idx]
    
    # Further split train into train/val
    n_trainval = len(train_idx)
    n_val = int(n_trainval * Config.VAL_RATIO)
    
    indices = np.arange(n_trainval)
    np.random.seed(Config.RANDOM_STATE + fold_num)
    np.random.shuffle(indices)
    
    val_local_idx = indices[:n_val]
    train_local_idx = indices[n_val:]
    
    # Get global indices for train/val/test
    train_global_idx = train_idx[train_local_idx]
    val_global_idx = train_idx[val_local_idx]
    test_global_idx = test_idx
    
    y_train = y_full[train_global_idx]
    attack_train = attack_full[train_global_idx]
    y_val = y_full[val_global_idx]
    attack_val = attack_full[val_global_idx]
    
    log(f"  Train: {len(train_global_idx)}, Val: {len(val_global_idx)}, Test: {len(test_global_idx)}")
    
    # Now load X data in chunks and extract needed indices
    log(f"  Loading X data for this fold...")
    
    train_dir = Config.SOURCE_DIR / "train"
    val_dir = Config.SOURCE_DIR / "val"  
    test_dir = Config.SOURCE_DIR / "test"
    
    # Load all X files with memory mapping
    X_files = [
        np.load(train_dir / "X.npy", mmap_mode='r'),
        np.load(val_dir / "X.npy", mmap_mode='r'),
        np.load(test_dir / "X.npy", mmap_mode='r')
    ]
    
    # Calculate offsets
    offsets = [0]
    for s in shapes[:-1]:
        offsets.append(offsets[-1] + s)
    
    def get_X_by_indices(indices):
        """Get X values for given global indices."""
        results = []
        for idx in indices:
            # Find which file this index belongs to
            for file_idx, (offset, shape) in enumerate(zip(offsets, shapes)):
                if idx < offset + shape:
                    local_idx = idx - offset
                    results.append(X_files[file_idx][local_idx])
                    break
        return np.array(results)
    
    # Extract X for each split
    X_train = get_X_by_indices(train_global_idx)
    X_val = get_X_by_indices(val_global_idx)
    X_test = get_X_by_indices(test_global_idx)
    
    # Close memory-mapped files
    del X_files
    gc.collect()
    
    log(f"  Before SMOTE:")
    log(f"    Train: {X_train.shape}")
    log(f"    Val: {X_val.shape}")
    log(f"    Test: {X_test.shape}")
    
    # Apply SMOTE to training data only
    X_train_aug, y_train_aug, attack_train_aug = apply_smote(
        X_train, y_train, attack_train, fold_num
    )
    
    # Free original training data
    del X_train, y_train, attack_train
    gc.collect()
    
    log(f"  After SMOTE:")
    log(f"    Train: {X_train_aug.shape}")
    
    # Save fold
    for split_name, (X, y, attack) in [
        ("train", (X_train_aug, y_train_aug, attack_train_aug)),
        ("val", (X_val, y_val, attack_val)),
        ("test", (X_test, y_test, attack_test))
    ]:
        split_dir = fold_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(split_dir / "X.npy", X)
        np.save(split_dir / "y.npy", y)
        np.save(split_dir / "attack_types.npy", attack)
    
    # Save metadata
    metadata = {
        "fold": fold_num,
        "created": datetime.now().isoformat(),
        "train_samples": len(X_train_aug),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "smote_applied": True,
        "smote_targets": Config.SMOTE_TARGETS,
        "class_distribution": {
            "train": dict(Counter(attack_train_aug)),
            "val": dict(Counter(attack_val)),
            "test": dict(Counter(attack_test))
        }
    }
    
    with open(fold_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    log(f"  ✓ Fold {fold_num} saved to {fold_dir}")
    
    # Free memory after saving
    del X_train_aug, y_train_aug, attack_train_aug
    del X_val, y_val, attack_val
    del X_test, y_test, attack_test
    gc.collect()
    
    return metadata


def create_fold(X_full: np.ndarray, y_full: np.ndarray, attack_full: np.ndarray,
                train_idx: np.ndarray, test_idx: np.ndarray, fold_num: int):
    """Create a single fold with train/val/test splits and SMOTE."""
    import gc
    
    fold_dir = Config.OUTPUT_BASE / f"fold_{fold_num}"
    
    log(f"\n{'='*60}")
    log(f"Creating Fold {fold_num}")
    log(f"{'='*60}")
    
    # Split train/test - use indexing to avoid copies
    X_trainval = X_full[train_idx].copy()
    y_trainval = y_full[train_idx].copy()
    attack_trainval = attack_full[train_idx].copy()
    
    X_test = X_full[test_idx].copy()
    y_test = y_full[test_idx].copy()
    attack_test = attack_full[test_idx].copy()
    
    # Further split train into train/val
    n_trainval = len(X_trainval)
    n_val = int(n_trainval * Config.VAL_RATIO)
    
    # Stratified val split
    indices = np.arange(n_trainval)
    np.random.seed(Config.RANDOM_STATE + fold_num)
    np.random.shuffle(indices)
    
    val_idx = indices[:n_val]
    train_idx_inner = indices[n_val:]
    
    X_train = X_trainval[train_idx_inner]
    y_train = y_trainval[train_idx_inner]
    attack_train = attack_trainval[train_idx_inner]
    
    X_val = X_trainval[val_idx]
    y_val = y_trainval[val_idx]
    attack_val = attack_trainval[val_idx]
    
    # Free trainval memory
    del X_trainval, y_trainval, attack_trainval
    gc.collect()
    
    log(f"  Before SMOTE:")
    log(f"    Train: {X_train.shape}")
    log(f"    Val: {X_val.shape}")
    log(f"    Test: {X_test.shape}")
    
    # Apply SMOTE to training data only
    X_train_aug, y_train_aug, attack_train_aug = apply_smote(
        X_train, y_train, attack_train, fold_num
    )
    
    log(f"  After SMOTE:")
    log(f"    Train: {X_train_aug.shape}")
    
    # Save fold
    for split_name, (X, y, attack) in [
        ("train", (X_train_aug, y_train_aug, attack_train_aug)),
        ("val", (X_val, y_val, attack_val)),
        ("test", (X_test, y_test, attack_test))
    ]:
        split_dir = fold_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(split_dir / "X.npy", X)
        np.save(split_dir / "y.npy", y)
        np.save(split_dir / "attack_types.npy", attack)
    
    # Save metadata
    metadata = {
        "fold": fold_num,
        "created": datetime.now().isoformat(),
        "train_samples": len(X_train_aug),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "smote_applied": True,
        "smote_targets": Config.SMOTE_TARGETS,
        "class_distribution": {
            "train": dict(Counter(attack_train_aug)),
            "val": dict(Counter(attack_val)),
            "test": dict(Counter(attack_test))
        }
    }
    
    with open(fold_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    log(f"  ✓ Fold {fold_num} saved to {fold_dir}")
    
    # Free memory after saving
    del X_train, y_train, attack_train
    del X_train_aug, y_train_aug, attack_train_aug
    del X_val, y_val, attack_val
    del X_test, y_test, attack_test
    gc.collect()
    
    return metadata

def main():
    log("=" * 70)
    log("5-FOLD CROSS-VALIDATION DATASET CREATION")
    log("=" * 70)
    log(f"Source: {Config.SOURCE_DIR}")
    log(f"Output: {Config.OUTPUT_BASE}/fold_1 through fold_5")
    log(f"SMOTE targets: {Config.SMOTE_TARGETS}")
    
    # Load metadata only (not full X array)
    shapes, y_full, attack_full = load_full_dataset()
    total_X_samples = sum(shapes)
    total_samples = len(attack_full)  # Use attack_types length as ground truth
    
    # Handle shape mismatch - truncate to smaller size
    if total_X_samples != total_samples:
        log(f"WARNING: X samples ({total_X_samples}) != attack_types ({total_samples})")
        log(f"  Truncating to {total_samples} samples")
        # Adjust shapes proportionally
        diff = total_X_samples - total_samples
        # Take from the first split (train)
        shapes[0] = shapes[0] - diff
        total_X_samples = sum(shapes)
        log(f"  Adjusted shapes: {shapes}")
    
    # Truncate y_full to match
    if len(y_full) != total_samples:
        y_full = y_full[:total_samples]
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS,
        shuffle=True,
        random_state=Config.RANDOM_STATE
    )
    
    # Create indices for the full dataset
    all_indices = np.arange(total_samples)
    
    # Create each fold
    all_metadata = {}
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(all_indices, attack_full), 1):
        metadata = create_fold_lowmem(
            shapes, y_full, attack_full,
            train_idx, test_idx, fold_num
        )
        all_metadata[f"fold_{fold_num}"] = metadata
        
        # Force garbage collection between folds
        gc.collect()
        log(f"  Memory cleaned up after Fold {fold_num}")
    
    # Save summary
    summary_path = Config.OUTPUT_BASE / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_folds": Config.N_FOLDS,
            "created": datetime.now().isoformat(),
            "total_samples": total_samples,
            "smote_targets": Config.SMOTE_TARGETS,
            "folds": all_metadata
        }, f, indent=2)
    
    log(f"\n{'='*70}")
    log("5-FOLD CV CREATION COMPLETE")
    log(f"{'='*70}")
    log(f"Summary saved to: {summary_path}")
    log(f"\nFold directories created:")
    for i in range(1, Config.N_FOLDS + 1):
        log(f"  - {Config.OUTPUT_BASE}/fold_{i}/")
    
    log(f"\nNext step: Run training on each fold")
    log(f"  python scripts/train_all_folds.py")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Create Rare-Class Holdout Set (Memory-Efficient Version)
Uses memory-mapping to avoid loading 2.5GB into RAM.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import gc


def create_holdout_set(data_dir, holdout_classes, holdout_samples_per_class, output_dir, seed=42):
    np.random.seed(seed)
    
    print("=" * 60)
    print("CREATING RARE-CLASS HOLDOUT SET (Memory-Efficient)")
    print("=" * 60)
    
    train_dir = Path(data_dir) / "train"
    output_dir = Path(output_dir)
    
    # Step 1: Load small files only
    print("\nStep 1: Loading metadata files...")
    attack_types = np.load(train_dir / "attack_types.npy", allow_pickle=True)
    y_train = np.load(train_dir / "y.npy")
    
    print(f"  attack_types: {attack_types.shape}")
    print(f"  y_train: {y_train.shape}")
    
    # Memory-map X (doesn't load into RAM)
    X_mmap = np.load(train_dir / "X.npy", mmap_mode='r')
    X_shape = X_mmap.shape
    print(f"  X_train shape: {X_shape} (memory-mapped)")
    
    # Align lengths
    min_len = min(X_shape[0], len(attack_types), len(y_train))
    attack_types = attack_types[:min_len]
    y_train = y_train[:min_len]
    
    # Step 2: Find holdout indices
    print("\nStep 2: Finding holdout indices...")
    holdout_indices = []
    holdout_info = {}
    
    for cls in holdout_classes:
        cls_indices = np.where(attack_types == cls)[0]
        available = len(cls_indices)
        n_holdout = min(holdout_samples_per_class, available)
        
        if available == 0:
            print(f"  WARNING: No samples for '{cls}'")
            continue
            
        selected = np.random.choice(cls_indices, size=n_holdout, replace=False)
        holdout_indices.extend(selected.tolist())
        holdout_info[cls] = {"available": available, "holdout": n_holdout}
        print(f"  {cls}: {n_holdout}/{available} selected")
    
    holdout_indices = np.array(sorted(holdout_indices))
    holdout_set = set(holdout_indices)
    remaining_indices = np.array([i for i in range(min_len) if i not in holdout_set])
    
    print(f"\nHoldout: {len(holdout_indices)} samples")
    print(f"Remaining: {len(remaining_indices)} samples")
    
    # Step 3: Save holdout data
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nStep 3: Saving holdout data...")
    
    X_holdout = X_mmap[holdout_indices].copy()
    np.save(output_dir / "X_rare.npy", X_holdout)
    np.save(output_dir / "y_rare.npy", y_train[holdout_indices])
    np.save(output_dir / "attack_types_rare.npy", attack_types[holdout_indices])
    print(f"  Saved to: {output_dir}")
    
    del X_holdout
    gc.collect()
    
    # Step 4: Backup original
    print("\nStep 4: Backing up original data...")
    backup_dir = Path(data_dir) / "train_original_backup"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)
        shutil.copy2(train_dir / "X.npy", backup_dir / "X.npy")
        shutil.copy2(train_dir / "y.npy", backup_dir / "y.npy")
        shutil.copy2(train_dir / "attack_types.npy", backup_dir / "attack_types.npy")
        print(f"  Backup created: {backup_dir}")
    else:
        print(f"  Backup exists: {backup_dir}")
    
    # Step 5: Create new training data
    print("\nStep 5: Creating updated training data (chunked)...")
    
    np.save(train_dir / "y.npy", y_train[remaining_indices])
    np.save(train_dir / "attack_types.npy", attack_types[remaining_indices])
    
    # Process X in chunks
    new_n = len(remaining_indices)
    new_shape = (new_n,) + X_shape[1:]
    X_new = np.zeros(new_shape, dtype=X_mmap.dtype)
    
    chunk_size = 50000
    for i in range(0, new_n, chunk_size):
        end = min(i + chunk_size, new_n)
        X_new[i:end] = X_mmap[remaining_indices[i:end]]
        print(f"  Progress: {end:,}/{new_n:,}", end='\r')
    
    print(f"\n  Saving X.npy...")
    X_new_path = train_dir / "X_new.npy"
    np.save(X_new_path, X_new)
    
    del X_new, X_mmap
    gc.collect()
    
    (train_dir / "X.npy").unlink()
    X_new_path.rename(train_dir / "X.npy")
    
    # Step 6: Save metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "holdout_classes": holdout_classes,
        "holdout_samples_per_class": holdout_samples_per_class,
        "seed": seed,
        "holdout_info": {k: {"available": int(v["available"]), "holdout": int(v["holdout"])} for k, v in holdout_info.items()},
        "total_holdout": len(holdout_indices),
        "original_samples": min_len,
        "new_samples": new_n
    }
    
    with open(output_dir / "holdout_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    for cls, info in holdout_info.items():
        print(f"  {cls}: {info['holdout']} holdout samples")
    print(f"\nTraining: {new_n:,} samples (reduced by {len(holdout_indices)})")
    
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/cic_ids_2017_v2")
    parser.add_argument("--holdout-classes", nargs="+", default=["Bot", "SSH-Patator"])
    parser.add_argument("--holdout-samples", type=int, default=40)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "holdout"
    
    create_holdout_set(data_dir, args.holdout_classes, args.holdout_samples, output_dir, args.seed)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

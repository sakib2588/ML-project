#!/usr/bin/env python3
"""
Complete Pre-Phase 2 Pipeline
=============================

Fully automated script that:
1. Checks if SMOTE augmentation is done
2. Runs SMOTE if needed (Bot: 69→1000, SSH-Patator: 73→1000)
3. Retrains Phase 1 model with Focal Loss + balanced sampling
4. Evaluates per-class detection rates
5. Verifies Bot/SSH-Patator >80%

Run and leave overnight:
    python scripts/complete_prephase2_pipeline.py

Expected runtime: 8-12 hours (CPU)
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional, Any

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Paths
    DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cic_ids_2017_v2"
    AUGMENTED_DIR = PROJECT_ROOT / "data" / "processed" / "cic_ids_2017_v2_augmented"
    OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase1_balanced"
    LOG_FILE = OUTPUT_DIR / "pipeline_log.txt"
    
    # Targets
    RARE_CLASSES = ["Bot", "SSH-Patator"]
    TARGET_SAMPLES = 1000
    TARGET_DETECTION_RATE = 0.80  # 80%
    
    # Training
    EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    PATIENCE = 15
    WEIGHT_DECAY = 0.01
    
    # Seeds for reproducibility
    RANDOM_STATE = 42


def log(msg: str, level: str = "INFO"):
    """Log to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {msg}"
    print(formatted)
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def header(title: str):
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


# ============================================================
# STEP 1: CHECK/RUN SMOTE
# ============================================================
def check_augmentation_status() -> Tuple[bool, Optional[Path]]:
    """Check if augmentation is already done."""
    header("STEP 1: CHECKING AUGMENTATION STATUS")
    
    metadata_file = Config.AUGMENTED_DIR / "augmentation_metadata.json"
    
    if not metadata_file.exists():
        log("❌ No augmentation found")
        return False, None
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    log(f"✓ Found augmentation from: {metadata.get('timestamp', 'unknown')}")
    
    # Verify rare class counts
    class_dist = metadata.get("class_distribution", {})
    
    all_good = True
    for cls in Config.RARE_CLASSES:
        if cls in class_dist:
            count = class_dist[cls].get("count", 0)
            log(f"  {cls}: {count} samples")
            if count < Config.TARGET_SAMPLES:
                log(f"  ⚠️ {cls} below target ({Config.TARGET_SAMPLES})")
                all_good = False
        else:
            log(f"  ⚠️ {cls} not found")
            all_good = False
    
    if all_good:
        log("✓ Augmentation verified")
        return True, Config.AUGMENTED_DIR
    else:
        log("❌ Augmentation insufficient")
        return False, None


def run_smote_augmentation() -> Tuple[bool, Optional[Path]]:
    """Run SMOTE augmentation."""
    header("RUNNING SMOTE AUGMENTATION")
    
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log("Installing imbalanced-learn...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "imbalanced-learn"], check=True)
        from imblearn.over_sampling import SMOTE
    
    # Load training data
    log("Loading training data...")
    train_dir = Config.DATA_DIR / "train"
    
    X = np.load(train_dir / "X.npy")
    y = np.load(train_dir / "y.npy")
    attack_types = np.load(train_dir / "attack_types.npy", allow_pickle=True)
    
    log(f"Loaded: X={X.shape}, y={y.shape}, attack_types={attack_types.shape}")
    
    # Check for shape mismatch and fix
    if len(X) != len(attack_types):
        log(f"⚠️ Shape mismatch: X={len(X)}, attack_types={len(attack_types)}")
        # Truncate to smaller size
        min_len = min(len(X), len(attack_types))
        X = X[:min_len]
        y = y[:min_len]
        attack_types = attack_types[:min_len]
        log(f"Truncated to {min_len} samples")
    
    # Current distribution
    counts = Counter(attack_types)
    log("\nCurrent distribution:")
    for cls in sorted(counts.keys()):
        marker = "⚠️" if counts[cls] < 100 else "✓"
        log(f"  {marker} {cls}: {counts[cls]:,}")
    
    # Flatten for SMOTE
    n_samples, window_size, n_features = X.shape
    X_flat = X.reshape(n_samples, -1)
    log(f"\nFlattened: {X_flat.shape}")
    
    # Sampling strategy
    sampling_strategy = {}
    for cls in Config.RARE_CLASSES:
        current = counts.get(cls, 0)
        if current < Config.TARGET_SAMPLES and current >= 6:
            sampling_strategy[cls] = Config.TARGET_SAMPLES
            log(f"  {cls}: {current} → {Config.TARGET_SAMPLES}")
    
    if not sampling_strategy:
        log("No augmentation needed")
        return True, Config.DATA_DIR
    
    # SMOTE
    min_samples = min(counts.get(cls, 999) for cls in sampling_strategy.keys())
    k_neighbors = min(5, min_samples - 1)
    k_neighbors = max(1, k_neighbors)
    
    log(f"\nApplying SMOTE (k={k_neighbors})...")
    log("This may take 30-60 minutes...")
    
    start = time.time()
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,  # type: ignore
        random_state=Config.RANDOM_STATE,
        k_neighbors=k_neighbors
    )
    
    try:
        X_resampled, types_resampled = smote.fit_resample(X_flat, attack_types)  # type: ignore
        X_resampled = np.asarray(X_resampled)
        types_resampled = np.asarray(types_resampled)
        elapsed = time.time() - start
        log(f"✓ SMOTE completed in {elapsed/60:.1f} minutes")
    except Exception as e:
        log(f"❌ SMOTE failed: {e}")
        return False, None
    
    # Reshape
    X_aug = X_resampled.reshape(-1, window_size, n_features)
    y_aug = np.array([0 if a == "BENIGN" else 1 for a in types_resampled], dtype=np.float32)
    
    log(f"Augmented shape: {X_aug.shape}")
    
    # Save
    output_dir = Config.AUGMENTED_DIR
    train_out = output_dir / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    
    np.save(train_out / "X.npy", X_aug)
    np.save(train_out / "y.npy", y_aug)
    np.save(train_out / "attack_types.npy", types_resampled)
    
    # Copy val/test
    for split in ["val", "test"]:
        src = Config.DATA_DIR / split
        dst = output_dir / split
        if src.exists():
            dst.mkdir(exist_ok=True)
            for f in ["X.npy", "y.npy", "attack_types.npy"]:
                if (src / f).exists():
                    shutil.copy(src / f, dst / f)
    
    # Metadata
    new_counts = Counter(types_resampled)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "method": "SMOTE",
        "k_neighbors": k_neighbors,
        "original_samples": int(n_samples),
        "augmented_samples": int(len(X_aug)),
        "class_distribution": {
            str(cls): {"count": int(new_counts.get(cls, 0))}
            for cls in sorted(str(k) for k in new_counts.keys())
        }
    }
    
    with open(output_dir / "augmentation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    log(f"✓ Saved to {output_dir}")
    log(f"  Original: {n_samples:,} → Augmented: {len(X_aug):,}")
    
    return True, output_dir


# ============================================================
# STEP 2: RETRAIN MODEL
# ============================================================
def check_existing_model() -> Tuple[bool, Dict[str, Any]]:
    """Check if model already trained and meets targets."""
    model_path = Config.OUTPUT_DIR / "best_model.pt"
    results_path = Config.OUTPUT_DIR / "results.json"
    
    if not model_path.exists() or not results_path.exists():
        return False, {}
    
    with open(results_path) as f:
        results = json.load(f)
    
    per_class = results.get("per_class_detection_rates", {})
    
    all_pass = True
    for cls in Config.RARE_CLASSES:
        if cls in per_class:
            dr = per_class[cls].get("detection_rate", 0)
            if dr < Config.TARGET_DETECTION_RATE:
                all_pass = False
    
    return all_pass, results


def retrain_model(data_dir: Path) -> Tuple[bool, Dict]:
    """Retrain Phase 1 model with Focal Loss."""
    header("STEP 2: RETRAINING MODEL")
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    
    # Check existing
    already_done, existing_results = check_existing_model()
    if already_done:
        log("✓ Model already trained and meets targets")
        return True, existing_results
    
    # Import model
    try:
        from src.models.ds_cnn import DS_1D_CNN as DSCNN
    except ImportError:
        log("❌ Cannot import model")
        return False, {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    
    # Load data
    log("Loading augmented data...")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    X_train = np.load(train_dir / "X.npy")
    y_train = np.load(train_dir / "y.npy")
    attack_types_train = np.load(train_dir / "attack_types.npy", allow_pickle=True)
    
    X_val = np.load(val_dir / "X.npy")
    y_val = np.load(val_dir / "y.npy")
    
    X_test = np.load(test_dir / "X.npy")
    y_test = np.load(test_dir / "y.npy")
    attack_types_test = np.load(test_dir / "attack_types.npy", allow_pickle=True)
    
    log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Weighted sampler
    log("Creating weighted sampler...")
    counts = Counter(attack_types_train)
    total = len(attack_types_train)
    n_classes = len(counts)
    
    class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
    min_w = min(class_weights.values())
    class_weights = {k: v / min_w for k, v in class_weights.items()}
    
    sample_weights = np.array([class_weights[a] for a in attack_types_train])
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Tensors - Model expects (batch, window=15, features=65) - NO permute needed
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Model - binary classification (num_classes=1 outputs single logit)
    log("Creating model...")
    model = DSCNN(num_classes=1, input_shape=(15, 65)).to(device)
    
    # Focal Loss for binary classification
    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # inputs: [batch] or [batch, 1] - logits
            # targets: [batch] - binary 0/1
            inputs = inputs.view(-1)  # Flatten to [batch]
            targets = targets.view(-1).float()  # Ensure float for BCE
            
            bce = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
            pt = torch.exp(-bce)
            focal = self.alpha * (1 - pt) ** self.gamma * bce
            return focal.mean()
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    
    log(f"\nTraining for {Config.EPOCHS} epochs...")
    log(f"Batch: {Config.BATCH_SIZE}, LR: {Config.LEARNING_RATE}, Patience: {Config.PATIENCE}")
    
    start_time = time.time()
    
    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = (time.time() - start_time) / 60
            log(f"Epoch {epoch+1}/{Config.EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f} | {elapsed:.1f}m")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.OUTPUT_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                log(f"Early stopping at epoch {epoch+1}")
                break
    
    total_time = (time.time() - start_time) / 3600
    log(f"\nTraining completed in {total_time:.2f} hours")
    
    # Load best and evaluate
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / "best_model.pt"))
    
    log("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, y_test, attack_types_test, device)
    
    # Save results
    with open(Config.OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return True, results


def evaluate_model(model, test_loader, y_test, attack_types, device) -> Dict:
    """Evaluate model with per-class metrics."""
    import torch
    
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    y_true = np.array(y_test)
    
    # Overall
    accuracy = np.mean(all_preds == y_true)
    
    # Per-class
    unique_attacks = np.unique(attack_types)
    per_class = {}
    
    for attack in unique_attacks:
        mask = attack_types == attack
        if np.sum(mask) == 0:
            continue
        
        attack_true = y_true[mask]
        attack_preds = all_preds[mask]
        total = int(np.sum(mask))
        
        if attack == "BENIGN":
            detected = int(np.sum((attack_true == 0) & (attack_preds == 0)))
            dr = detected / np.sum(attack_true == 0) if np.sum(attack_true == 0) > 0 else 0
        else:
            detected = int(np.sum((attack_true == 1) & (attack_preds == 1)))
            dr = detected / np.sum(attack_true == 1) if np.sum(attack_true == 1) > 0 else 0
        
        per_class[attack] = {
            "total": total,
            "detected": detected,
            "detection_rate": float(dr)
        }
    
    return {
        "accuracy": float(accuracy),
        "per_class_detection_rates": per_class,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# STEP 3: VERIFY TARGETS
# ============================================================
def verify_targets(results: Dict) -> bool:
    """Verify detection targets are met."""
    header("STEP 3: VERIFYING TARGETS")
    
    per_class = results.get("per_class_detection_rates", {})
    
    all_pass = True
    
    log("\n--- RARE CLASSES (Target: >80%) ---")
    for cls in Config.RARE_CLASSES:
        if cls in per_class:
            dr = per_class[cls]["detection_rate"]
            total = per_class[cls]["total"]
            detected = per_class[cls]["detected"]
            
            status = "✓ PASS" if dr >= Config.TARGET_DETECTION_RATE else "❌ FAIL"
            log(f"  {status} {cls}: {dr:.1%} ({detected}/{total})")
            
            if dr < Config.TARGET_DETECTION_RATE:
                all_pass = False
        else:
            log(f"  ⚠️ {cls}: Not in test set")
    
    log("\n--- CRITICAL CLASSES (Target: >98%) ---")
    for cls in ["DDoS", "PortScan"]:
        if cls in per_class:
            dr = per_class[cls]["detection_rate"]
            total = per_class[cls]["total"]
            detected = per_class[cls]["detected"]
            
            status = "✓" if dr >= 0.98 else "⚠️"
            log(f"  {status} {cls}: {dr:.1%} ({detected}/{total})")
    
    accuracy = results.get("accuracy", 0)
    log(f"\n--- OVERALL ACCURACY: {accuracy:.2%} ---")
    
    return all_pass


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n")
    header("COMPLETE PRE-PHASE 2 PIPELINE")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Data: {Config.DATA_DIR}")
    log(f"Output: {Config.OUTPUT_DIR}")
    
    total_start = time.time()
    
    # Step 1: Check/Run SMOTE
    smote_done, data_dir = check_augmentation_status()
    
    if not smote_done:
        smote_done, data_dir = run_smote_augmentation()
        if not smote_done or data_dir is None:
            log("❌ SMOTE failed, exiting")
            return 1
    
    # Ensure data_dir is not None
    if data_dir is None:
        log("❌ No data directory, exiting")
        return 1
    
    # Step 2: Retrain
    train_success, results = retrain_model(data_dir)
    
    if not train_success:
        log("❌ Training failed, exiting")
        return 1
    
    # Step 3: Verify
    targets_met = verify_targets(results)
    
    # Summary
    total_time = (time.time() - total_start) / 3600
    
    print("\n")
    header("PIPELINE COMPLETE")
    log(f"Total time: {total_time:.2f} hours")
    log(f"Results: {Config.OUTPUT_DIR / 'results.json'}")
    log(f"Model: {Config.OUTPUT_DIR / 'best_model.pt'}")
    
    if targets_met:
        log("\n✅ ALL TARGETS MET - READY FOR PHASE 2!")
        log("Next: cd phase2 && bash run_pipeline.sh")
        return 0
    else:
        log("\n⚠️ SOME TARGETS NOT MET")
        log("Consider: increasing target samples or training epochs")
        return 1


if __name__ == "__main__":
    sys.exit(main())

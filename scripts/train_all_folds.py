#!/usr/bin/env python3
"""
Train all 5 folds sequentially with the balanced model.
Each fold uses SMOTE-augmented training data + Focal Loss + Weighted Sampler.

For research paper: Reports mean ± std across all folds for statistical significance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import time
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    DATA_BASE = Path("/home/sakib/ids-compression/data/processed")
    OUTPUT_BASE = Path("/home/sakib/ids-compression/experiments/5fold_cv")
    
    # Training
    EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    
    # Targets
    BOT_TARGET = 0.80
    SSH_TARGET = 0.80
    DDOS_TARGET = 0.98
    PORTSCAN_TARGET = 0.98

def log(msg: str, fold_num: int = 0):
    """Print timestamped message"""
    prefix = f"[Fold {fold_num}]" if fold_num > 0 else "[MAIN]"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {prefix} {msg}")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def train_fold(fold_num: int) -> dict:
    """Train a single fold and return results."""
    from src.models.ds_cnn import DS_1D_CNN as DSCNN
    
    log(f"Starting training...", fold_num)
    
    fold_dir = Config.DATA_BASE / f"fold_{fold_num}"
    output_dir = Config.OUTPUT_BASE / f"fold_{fold_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}", fold_num)
    
    # Load data
    log("Loading data...", fold_num)
    X_train = np.load(fold_dir / "train/X.npy")
    y_train = np.load(fold_dir / "train/y.npy")
    attack_train = np.load(fold_dir / "train/attack_types.npy", allow_pickle=True)
    
    X_val = np.load(fold_dir / "val/X.npy")
    y_val = np.load(fold_dir / "val/y.npy")
    
    X_test = np.load(fold_dir / "test/X.npy")
    y_test = np.load(fold_dir / "test/y.npy")
    attack_test = np.load(fold_dir / "test/attack_types.npy", allow_pickle=True)
    
    log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", fold_num)
    
    # Weighted sampler
    log("Creating weighted sampler...", fold_num)
    counts = Counter(attack_train)
    total = len(attack_train)
    n_classes = len(counts)
    class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
    min_w = min(class_weights.values())
    class_weights = {k: v / min_w for k, v in class_weights.items()}
    sample_weights = np.array([class_weights[a] for a in attack_train], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),  # Convert to list for WeightedRandomSampler
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoaders
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=Config.BATCH_SIZE,
        sampler=sampler
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
    
    # Model
    log("Creating model...", fold_num)
    model = DSCNN(num_classes=1, input_shape=(15, 65)).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    log(f"Training for {Config.EPOCHS} epochs...", fold_num)
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = (time.time() - start_time) / 60
            log(f"Epoch {epoch+1}/{Config.EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}m", fold_num)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                log(f"Early stopping at epoch {epoch+1}", fold_num)
                break
    
    total_time = (time.time() - start_time) / 3600
    log(f"Training completed in {total_time:.2f} hours", fold_num)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
    all_preds = np.array(all_preds)
    
    # Per-class metrics
    results = {
        "fold": fold_num,
        "training_time_hours": total_time,
        "best_val_loss": best_val_loss,
        "overall_accuracy": float(np.mean(all_preds == y_test)),
        "per_class": {}
    }
    
    for cls in np.unique(attack_test):
        mask = attack_test == cls
        if mask.sum() > 0:
            cls_preds = all_preds[mask]
            cls_labels = y_test[mask]
            detection_rate = float(np.mean(cls_preds == cls_labels))
            results["per_class"][cls] = {
                "detection_rate": detection_rate,
                "total": int(mask.sum()),
                "correct": int((cls_preds == cls_labels).sum())
            }
    
    # Log key metrics
    log(f"\n--- Results ---", fold_num)
    for cls in ["Bot", "SSH-Patator", "DDoS", "PortScan"]:
        if cls in results["per_class"]:
            dr = results["per_class"][cls]["detection_rate"]
            total = results["per_class"][cls]["total"]
            correct = results["per_class"][cls]["correct"]
            status = "✓" if (cls in ["Bot", "SSH-Patator"] and dr >= 0.80) or \
                          (cls in ["DDoS", "PortScan"] and dr >= 0.98) else "✗"
            log(f"  {status} {cls}: {dr*100:.1f}% ({correct}/{total})", fold_num)
    log(f"  Overall Accuracy: {results['overall_accuracy']*100:.2f}%", fold_num)
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def aggregate_results(all_results: list) -> dict:
    """Aggregate results across all folds for research paper."""
    
    # Collect per-class detection rates
    class_rates = {}
    for result in all_results:
        for cls, metrics in result["per_class"].items():
            if cls not in class_rates:
                class_rates[cls] = []
            class_rates[cls].append(metrics["detection_rate"])
    
    # Compute mean ± std
    summary = {
        "n_folds": len(all_results),
        "per_class_summary": {},
        "overall_accuracy": {
            "mean": float(np.mean([r["overall_accuracy"] for r in all_results])),
            "std": float(np.std([r["overall_accuracy"] for r in all_results]))
        }
    }
    
    for cls, rates in class_rates.items():
        summary["per_class_summary"][cls] = {
            "mean": float(np.mean(rates)),
            "std": float(np.std(rates)),
            "min": float(np.min(rates)),
            "max": float(np.max(rates))
        }
    
    return summary

def main():
    log("=" * 70)
    log("5-FOLD CROSS-VALIDATION TRAINING")
    log("=" * 70)
    
    Config.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Check if folds exist
    for i in range(1, 6):
        fold_dir = Config.DATA_BASE / f"fold_{i}"
        if not fold_dir.exists():
            log(f"ERROR: Fold {i} not found at {fold_dir}")
            log("Please run: python scripts/create_5fold_cv.py")
            return
    
    # Train all folds
    all_results = []
    total_start = time.time()
    
    for fold_num in range(1, 6):
        log(f"\n{'='*70}")
        log(f"FOLD {fold_num}/5")
        log(f"{'='*70}")
        
        results = train_fold(fold_num)
        all_results.append(results)
    
    # Aggregate results
    summary = aggregate_results(all_results)
    
    # Print final summary
    log(f"\n{'='*70}")
    log("5-FOLD CROSS-VALIDATION COMPLETE")
    log(f"{'='*70}")
    log(f"Total time: {(time.time() - total_start) / 3600:.2f} hours")
    log(f"\n--- RESEARCH PAPER RESULTS (Mean ± Std) ---")
    
    for cls in ["Bot", "SSH-Patator", "DDoS", "PortScan", "BENIGN"]:
        if cls in summary["per_class_summary"]:
            m = summary["per_class_summary"][cls]["mean"] * 100
            s = summary["per_class_summary"][cls]["std"] * 100
            log(f"  {cls}: {m:.1f}% ± {s:.1f}%")
    
    oa_m = summary["overall_accuracy"]["mean"] * 100
    oa_s = summary["overall_accuracy"]["std"] * 100
    log(f"  Overall: {oa_m:.1f}% ± {oa_s:.1f}%")
    
    # Save summary
    summary_path = Config.OUTPUT_BASE / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to: {summary_path}")
    
    log(f"\n✅ Ready for Phase 2 compression with 5-fold validated model!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Task DS-CNN Training Script

This script implements the correct approach: training a model with TWO classification heads:
1. Binary head: Is this BENIGN or ATTACK? (for deployment)
2. Attack-type head: Which of the 10 classes is this? (auxiliary supervision)

The multi-task learning forces the model to learn discriminative features for EACH attack type,
which paradoxically improves binary classification performance as well.

Research reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., CVPR 2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import time
import sys
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    DATA_BASE = Path("/home/sakib/ids-compression/data/processed")
    OUTPUT_BASE = Path("/home/sakib/ids-compression/experiments/multitask_cv")
    
    # Training
    EPOCHS = 80
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20
    
    # Multi-task loss weights
    BINARY_LOSS_WEIGHT = 0.9    # Primary task (high priority for BENIGN recall)
    ATTACK_TYPE_LOSS_WEIGHT = 0.1  # Auxiliary task (lower to avoid confusing binary task)
    
    # Focal Loss parameters
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 1.0
    
    # Class mapping (from data exploration)
    CLASS_NAMES = [
        "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
        "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", "PortScan", "SSH-Patator"
    ]
    NUM_CLASSES = 10
    
    # Targets for research
    TARGETS = {
        "Bot": 0.80,
        "SSH-Patator": 0.80,
        "DDoS": 0.98,
        "PortScan": 0.98
    }

def log(msg: str, fold_num: int = 0):
    """Print timestamped message with flush for immediate output."""
    prefix = f"[Fold {fold_num}]" if fold_num > 0 else "[MAIN]"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {prefix} {msg}", flush=True)


# ============================================================================
# Multi-Task DS-CNN Model
# ============================================================================
class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable conv block: depthwise -> pointwise -> BN -> ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                    padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class MultiTaskDSCNN(nn.Module):
    """
    Depthwise Separable CNN with Multi-Task Heads.
    
    Architecture:
    - Shared backbone: 3 DS-Conv blocks
    - Two classification heads:
      1. Binary head: BENIGN vs ATTACK (num_outputs=1, sigmoid)
      2. Attack-type head: 10-way classification (num_outputs=10, softmax)
    """
    def __init__(self, input_shape=(15, 65), conv_channels=(32, 64, 64), 
                 classifier_hidden=64, dropout=0.2, num_attack_types=10):
        super().__init__()
        
        self.window_len, self.n_features = input_shape
        self.num_attack_types = num_attack_types
        
        # Build shared backbone
        layers = []
        in_ch = self.n_features
        for out_ch in conv_channels:
            layers.append(DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size=3, dropout=dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared feature extraction
        last_ch = conv_channels[-1]
        self.feature_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(last_ch, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # HEAD 1: Binary classifier (BENIGN=0 vs ATTACK=1)
        self.binary_head = nn.Linear(classifier_hidden, 1)
        
        # HEAD 2: Attack-type classifier (10 classes)
        self.attack_type_head = nn.Linear(classifier_hidden, num_attack_types)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass returning both heads' outputs.
        
        Args:
            x: (batch, window_len, n_features)
        
        Returns:
            binary_logits: (batch, 1) - for binary classification
            attack_type_logits: (batch, 10) - for 10-way classification
        """
        # Transpose to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2).contiguous()
        
        # Shared backbone
        x = self.backbone(x)
        x = self.global_pool(x)
        
        # Shared features
        features = self.feature_fc(x)
        
        # Two heads
        binary_logits = self.binary_head(features)
        attack_type_logits = self.attack_type_head(features)
        
        return binary_logits, attack_type_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Functions
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance."""
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


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Total Loss = binary_weight * Focal(binary_pred, binary_target) 
               + attack_type_weight * CE(attack_pred, attack_target) [only for attack samples]
    
    IMPORTANT: Class weights are moderated using sqrt() to prevent extreme imbalance correction.
    Raw weights like 500x for Bot would cause the model to become overly aggressive about predicting
    attacks, leading to high false alarm rates. sqrt(500) ≈ 22 provides meaningful but moderate
    correction. This is a standard technique in imbalanced learning research.
    """
    def __init__(self, binary_weight=0.7, attack_type_weight=0.3, 
                 class_weights=None, focal_gamma=2.0):
        super().__init__()
        self.binary_weight = binary_weight
        self.attack_type_weight = attack_type_weight
        self.focal_gamma = focal_gamma
        
        # Binary focal loss (handles benign vs attack imbalance)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        
        # Attack-type cross entropy with MODERATED class weights
        # Using sqrt to reduce extreme weight ratios (e.g., 500 -> 22)
        if class_weights is not None:
            # Moderate the weights: sqrt brings extreme values closer to 1
            moderated_weights = torch.sqrt(class_weights)
            moderated_weights = moderated_weights / moderated_weights.min()
            self.attack_type_ce = nn.CrossEntropyLoss(weight=moderated_weights, reduction='mean')
        else:
            self.attack_type_ce = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, binary_logits, attack_type_logits, binary_targets, attack_type_targets):
        """
        Compute multi-task loss.
        
        Args:
            binary_logits: (batch, 1) - binary classification logits
            attack_type_logits: (batch, 10) - 10-way classification logits
            binary_targets: (batch,) - 0=BENIGN, 1=ATTACK
            attack_type_targets: (batch,) - 0-9 class indices
        
        Returns:
            total_loss, binary_loss, attack_type_loss
        """
        # 1. Binary focal loss (all samples)
        binary_loss = self.focal_loss(binary_logits, binary_targets)
        
        # 2. Attack-type CE loss (ONLY for ATTACK samples, not BENIGN)
        # This prevents the model from being confused about what class BENIGN belongs to
        attack_mask = binary_targets == 1  # Only attack samples
        if attack_mask.sum() > 0:
            attack_type_loss = self.attack_type_ce(
                attack_type_logits[attack_mask], 
                attack_type_targets[attack_mask]
            )
        else:
            # No attack samples in batch, set loss to 0
            attack_type_loss = torch.tensor(0.0, device=binary_logits.device)
        
        # Combined loss
        total_loss = self.binary_weight * binary_loss + self.attack_type_weight * attack_type_loss
        
        return total_loss, binary_loss, attack_type_loss


# ============================================================================
# Dataset
# ============================================================================
class MultiTaskDataset(Dataset):
    """Dataset that returns (X, binary_label, attack_type_label)"""
    def __init__(self, X: np.ndarray, y_binary: np.ndarray, attack_types: np.ndarray, class_to_idx: dict):
        self.X = torch.FloatTensor(X)
        self.y_binary = torch.FloatTensor(y_binary)
        
        # Convert attack type strings to indices
        self.y_attack_type = torch.LongTensor([class_to_idx[a] for a in attack_types])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_binary[idx], self.y_attack_type[idx]


# ============================================================================
# Training Function
# ============================================================================
def train_fold(fold_num: int, class_to_idx: dict, idx_to_class: dict) -> dict:
    """Train a single fold with multi-task learning."""
    
    log(f"Starting multi-task training...", fold_num)
    
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
    attack_val = np.load(fold_dir / "val/attack_types.npy", allow_pickle=True)
    
    X_test = np.load(fold_dir / "test/X.npy")
    y_test = np.load(fold_dir / "test/y.npy")
    attack_test = np.load(fold_dir / "test/attack_types.npy", allow_pickle=True)
    
    log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", fold_num)
    
    # Class distribution
    train_dist = Counter(attack_train)
    log(f"Train class distribution:", fold_num)
    for cls, count in sorted(train_dist.items()):
        log(f"  {cls}: {count}", fold_num)
    
    # Compute class weights for attack-type loss (inverse frequency)
    total = len(attack_train)
    n_classes = Config.NUM_CLASSES
    class_weights_np = np.ones(n_classes, dtype=np.float32)
    for cls, count in train_dist.items():
        idx = class_to_idx[cls]
        class_weights_np[idx] = total / (n_classes * count)
    # Normalize
    class_weights_np = class_weights_np / class_weights_np.min()
    class_weights_tensor = torch.FloatTensor(class_weights_np).to(device)
    log(f"Raw class weights: {dict(zip(Config.CLASS_NAMES, class_weights_np.tolist()))}", fold_num)
    log(f"After sqrt moderation: {dict(zip(Config.CLASS_NAMES, np.sqrt(class_weights_np).tolist()))}", fold_num)
    
    # Weighted sampler (by BINARY class to ensure balanced benign/attack batches)
    # This ensures the model doesn't ignore BENIGN samples
    # OPTIMIZED: Vectorized instead of slow list comprehension
    n_benign = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    binary_weights = np.where(y_train == 0, 1.0 / n_benign, 1.0 / n_attack).astype(np.float64)
    binary_weights = binary_weights / binary_weights.max()  # Normalize to [0, 1]
    log(f"[Fold {fold_num}] Sampler weights created: BENIGN={n_benign}, ATTACK={n_attack}", fold_num)
    
    sampler = WeightedRandomSampler(
        weights=binary_weights.tolist(),  # Convert numpy array to list for compatibility
        num_samples=len(binary_weights),
        replacement=True
    )
    
    # Datasets
    train_dataset = MultiTaskDataset(X_train, y_train, attack_train, class_to_idx)
    val_dataset = MultiTaskDataset(X_val, y_val, attack_val, class_to_idx)
    test_dataset = MultiTaskDataset(X_test, y_test, attack_test, class_to_idx)
    
    # Free memory
    del X_train, y_train, attack_train
    del X_val, y_val, attack_val
    gc.collect()
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    log("Creating multi-task model...", fold_num)
    model = MultiTaskDSCNN(
        input_shape=(15, 65),
        conv_channels=(32, 64, 64),
        classifier_hidden=64,
        dropout=0.2,
        num_attack_types=Config.NUM_CLASSES
    ).to(device)
    log(f"Model parameters: {model.count_parameters():,}", fold_num)
    
    # Multi-task loss
    criterion = MultiTaskLoss(
        binary_weight=Config.BINARY_LOSS_WEIGHT,
        attack_type_weight=Config.ATTACK_TYPE_LOSS_WEIGHT,
        class_weights=class_weights_tensor,
        focal_gamma=Config.FOCAL_GAMMA
    )
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    log(f"Training for up to {Config.EPOCHS} epochs...", fold_num)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_attack_acc": []}
    
    for epoch in range(Config.EPOCHS):
        # ========== TRAIN ==========
        model.train()
        train_loss = 0
        train_binary_loss = 0
        train_attack_loss = 0
        
        for X_batch, y_binary, y_attack in train_loader:
            X_batch = X_batch.to(device)
            y_binary = y_binary.to(device)
            y_attack = y_attack.to(device)
            
            optimizer.zero_grad()
            binary_logits, attack_logits = model(X_batch)
            
            total_loss, b_loss, a_loss = criterion(binary_logits, attack_logits, y_binary, y_attack)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_binary_loss += b_loss.item()
            train_attack_loss += a_loss.item()
        
        train_loss /= len(train_loader)
        train_binary_loss /= len(train_loader)
        train_attack_loss /= len(train_loader)
        
        # ========== VALIDATE ==========
        model.eval()
        val_loss = 0
        val_binary_correct = 0
        val_attack_correct = 0
        val_total = 0
        val_benign_correct = 0
        val_benign_total = 0
        val_attack_as_attack = 0
        val_attack_total = 0
        
        with torch.no_grad():
            for X_batch, y_binary, y_attack in val_loader:
                X_batch = X_batch.to(device)
                y_binary = y_binary.to(device)
                y_attack = y_attack.to(device)
                
                binary_logits, attack_logits = model(X_batch)
                total_loss, _, _ = criterion(binary_logits, attack_logits, y_binary, y_attack)
                val_loss += total_loss.item()
                
                # Binary accuracy
                binary_preds = (torch.sigmoid(binary_logits.squeeze()) > 0.5).long()
                val_binary_correct += (binary_preds == y_binary.long()).sum().item()
                
                # BENIGN recall (how many BENIGN samples predicted as BENIGN)
                benign_mask = y_binary == 0
                if benign_mask.sum() > 0:
                    val_benign_correct += ((binary_preds == 0) & benign_mask).sum().item()
                    val_benign_total += benign_mask.sum().item()
                
                # ATTACK recall (how many ATTACK samples predicted as ATTACK)
                attack_mask = y_binary == 1
                if attack_mask.sum() > 0:
                    val_attack_as_attack += ((binary_preds == 1) & attack_mask).sum().item()
                    val_attack_total += attack_mask.sum().item()
                
                # Attack-type accuracy
                attack_preds = attack_logits.argmax(dim=1)
                val_attack_correct += (attack_preds == y_attack).sum().item()
                
                val_total += len(X_batch)
        
        val_loss /= len(val_loader)
        val_binary_acc = val_binary_correct / val_total
        val_attack_acc = val_attack_correct / val_total
        val_benign_recall = val_benign_correct / val_benign_total if val_benign_total > 0 else 0
        val_attack_recall = val_attack_as_attack / val_attack_total if val_attack_total > 0 else 0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_binary_acc)
        history["val_attack_acc"].append(val_attack_acc)
        
        # Log progress (with BENIGN and ATTACK recalls)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (time.time() - start_time) / 60
            log(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                f"TL: {train_loss:.4f} (B:{train_binary_loss:.4f} A:{train_attack_loss:.4f}) | "
                f"VL: {val_loss:.4f} | BenignR: {val_benign_recall*100:.1f}% | "
                f"AttackR: {val_attack_recall*100:.1f}% | {elapsed:.1f}m", fold_num)
        
        scheduler.step(val_loss)
        
        # Save best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_binary_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_binary_acc,
                'val_attack_acc': val_attack_acc
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                log(f"Early stopping at epoch {epoch+1}", fold_num)
                break
    
    total_time = (time.time() - start_time) / 3600
    log(f"Training completed in {total_time:.2f} hours", fold_num)
    
    # ========== EVALUATE ON TEST SET ==========
    log("Evaluating on test set...", fold_num)
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Collect predictions
    all_binary_preds = []
    all_attack_preds = []
    all_binary_targets = []
    all_attack_targets = []
    
    with torch.no_grad():
        for X_batch, y_binary, y_attack in test_loader:
            X_batch = X_batch.to(device)
            binary_logits, attack_logits = model(X_batch)
            
            binary_preds = (torch.sigmoid(binary_logits.squeeze()) > 0.5).cpu().numpy().astype(int)
            attack_preds = attack_logits.argmax(dim=1).cpu().numpy()
            
            all_binary_preds.extend(binary_preds)
            all_attack_preds.extend(attack_preds)
            all_binary_targets.extend(y_binary.numpy().astype(int))
            all_attack_targets.extend(y_attack.numpy())
    
    all_binary_preds = np.array(all_binary_preds)
    all_attack_preds = np.array(all_attack_preds)
    all_binary_targets = np.array(all_binary_targets)
    all_attack_targets = np.array(all_attack_targets)
    
    # Overall metrics
    binary_accuracy = np.mean(all_binary_preds == all_binary_targets)
    attack_type_accuracy = np.mean(all_attack_preds == all_attack_targets)
    
    # Per-class metrics (using ATTACK-TYPE predictions for proper evaluation)
    log(f"\n{'='*50}", fold_num)
    log(f"TEST RESULTS", fold_num)
    log(f"{'='*50}", fold_num)
    log(f"Binary Accuracy: {binary_accuracy*100:.2f}%", fold_num)
    log(f"10-Way Attack-Type Accuracy: {attack_type_accuracy*100:.2f}%", fold_num)
    
    results = {
        "fold": fold_num,
        "training_time_hours": total_time,
        "best_epoch": checkpoint['epoch'],
        "binary_accuracy": float(binary_accuracy),
        "attack_type_accuracy": float(attack_type_accuracy),
        "per_class": {}
    }
    
    log(f"\nPer-Class Recall (True Positives / Actual):", fold_num)
    targets_met = 0
    targets_total = len(Config.TARGETS)
    
    for cls_idx, cls_name in enumerate(Config.CLASS_NAMES):
        mask = all_attack_targets == cls_idx
        if mask.sum() > 0:
            total = int(mask.sum())
            
            # CRITICAL FIX: Use BINARY HEAD for BENIGN, ATTACK-TYPE HEAD for attacks
            if cls_name == "BENIGN":
                # For BENIGN: check if binary head predicted 0 (not attack)
                binary_preds_for_benign = all_binary_preds[mask]
                correct = int((binary_preds_for_benign == 0).sum())
                recall = correct / total
            else:
                # For attacks: check if attack-type head predicted the correct class
                cls_preds = all_attack_preds[mask]
                correct = int((cls_preds == cls_idx).sum())
                recall = correct / total
            
            # Check target
            target = Config.TARGETS.get(cls_name)
            if cls_name == "BENIGN":
                # BENIGN should have very high recall (>95%)
                if recall < 0.95:
                    status = "⚠️"
                    log(f"  {status} {cls_name}: {recall*100:.1f}% ({correct}/{total}) — CRITICAL (should be >95%)", fold_num)
                else:
                    log(f"  ✓ {cls_name}: {recall*100:.1f}% ({correct}/{total})", fold_num)
            elif target:
                met = recall >= target
                if met:
                    targets_met += 1
                status = "✓" if met else "✗"
                target_str = f" (target: {target*100:.0f}%)"
                log(f"  {status} {cls_name}: {recall*100:.1f}% ({correct}/{total}){target_str}", fold_num)
            else:
                log(f"    {cls_name}: {recall*100:.1f}% ({correct}/{total})", fold_num)
            
            results["per_class"][cls_name] = {
                "recall": float(recall),
                "total": total,
                "correct": correct
            }
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # CRITICAL: Clean up model/optimizer memory before returning
    # This prevents memory accumulation across folds
    del model, optimizer, scheduler, criterion
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    
    return results


def aggregate_results(all_results: list) -> dict:
    """Aggregate results across all folds for research paper."""
    
    # Collect per-class recalls
    class_recalls = {}
    for result in all_results:
        for cls, metrics in result["per_class"].items():
            if cls not in class_recalls:
                class_recalls[cls] = []
            class_recalls[cls].append(metrics["recall"])
    
    # Compute mean ± std
    summary = {
        "n_folds": len(all_results),
        "per_class_summary": {},
        "binary_accuracy": {
            "mean": float(np.mean([r["binary_accuracy"] for r in all_results])),
            "std": float(np.std([r["binary_accuracy"] for r in all_results]))
        },
        "attack_type_accuracy": {
            "mean": float(np.mean([r["attack_type_accuracy"] for r in all_results])),
            "std": float(np.std([r["attack_type_accuracy"] for r in all_results]))
        }
    }
    
    for cls, recalls in class_recalls.items():
        summary["per_class_summary"][cls] = {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls)),
            "min": float(np.min(recalls)),
            "max": float(np.max(recalls))
        }
    
    return summary


def main():
    log("=" * 70)
    log("MULTI-TASK 5-FOLD CROSS-VALIDATION TRAINING")
    log("=" * 70)
    log("")
    log("Strategy: Binary classification + 10-way attack-type classification")
    log(f"Loss weights: Binary={Config.BINARY_LOSS_WEIGHT}, Attack-Type={Config.ATTACK_TYPE_LOSS_WEIGHT}")
    log("")
    
    Config.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(Config.CLASS_NAMES)}
    idx_to_class = {idx: cls for idx, cls in enumerate(Config.CLASS_NAMES)}
    log(f"Class mapping: {class_to_idx}")
    
    # Check if folds exist
    for i in range(1, 6):
        fold_dir = Config.DATA_BASE / f"fold_{i}"
        if not fold_dir.exists():
            log(f"ERROR: Fold {i} not found at {fold_dir}")
            log("Please run: python scripts/create_5fold_cv_ultra_light.py")
            return
    
    # Train all folds
    all_results = []
    total_start = time.time()
    
    for fold_num in range(1, 6):
        log(f"\n{'='*70}")
        log(f"FOLD {fold_num}/5")
        log(f"{'='*70}")
        
        results = train_fold(fold_num, class_to_idx, idx_to_class)
        all_results.append(results)
        
        # CRITICAL: Proper memory cleanup between folds to prevent crashes on long runs
        # This is essential for overnight training - memory leaks will cause fold 3-5 to fail
        log(f"Cleaning up memory after fold {fold_num}...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        log(f"Memory cleanup complete.")
    
    # Aggregate results
    summary = aggregate_results(all_results)
    
    # Print final summary
    log(f"\n{'='*70}")
    log("MULTI-TASK 5-FOLD CROSS-VALIDATION COMPLETE")
    log(f"{'='*70}")
    log(f"Total time: {(time.time() - total_start) / 3600:.2f} hours")
    
    log(f"\n{'='*50}")
    log("RESEARCH PAPER RESULTS (Mean ± Std)")
    log(f"{'='*50}")
    
    # Binary accuracy
    ba_m = summary["binary_accuracy"]["mean"] * 100
    ba_s = summary["binary_accuracy"]["std"] * 100
    log(f"Binary Accuracy: {ba_m:.1f}% ± {ba_s:.1f}%")
    
    # Attack-type accuracy
    aa_m = summary["attack_type_accuracy"]["mean"] * 100
    aa_s = summary["attack_type_accuracy"]["std"] * 100
    log(f"10-Way Attack-Type Accuracy: {aa_m:.1f}% ± {aa_s:.1f}%")
    
    log(f"\nPer-Class Recall (10-Way Classification):")
    targets_met = 0
    targets_total = len(Config.TARGETS)
    
    for cls in Config.CLASS_NAMES:
        if cls in summary["per_class_summary"]:
            m = summary["per_class_summary"][cls]["mean"] * 100
            s = summary["per_class_summary"][cls]["std"] * 100
            
            target = Config.TARGETS.get(cls)
            if target:
                met = m / 100 >= target
                if met:
                    targets_met += 1
                status = "✓" if met else "✗"
                log(f"  {status} {cls}: {m:.1f}% ± {s:.1f}% (target: {target*100:.0f}%)")
            else:
                log(f"    {cls}: {m:.1f}% ± {s:.1f}%")
    
    log(f"\nTargets Met: {targets_met}/{targets_total}")
    
    # Save summary
    summary_path = Config.OUTPUT_BASE / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to: {summary_path}")
    
    if targets_met == targets_total:
        log(f"\n✅ ALL TARGETS MET! Ready for Phase 2 compression!")
    else:
        log(f"\n⚠️ {targets_total - targets_met} targets not met. Consider tuning hyperparameters.")


if __name__ == "__main__":
    main()

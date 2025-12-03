#!/usr/bin/env python3
"""
Phase 2 - Stage 0: Baseline Student Training
=============================================

Train student models WITHOUT knowledge distillation as baseline.
Uses the same multi-task loss as Phase 1:
- Binary focal loss (BENIGN vs ATTACK)
- Attack-type CE loss (10-way, only on attack samples)

Usage:
    python train_student_baseline.py --size 5000
    python train_student_baseline.py --size 50000 --seed 42
    python train_student_baseline.py --all  # Run all sizes with quick seeds
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase2.models_multitask import build_student, MultiTaskStudentDSCNN
from phase2.config import (
    ROOT_DIR, DATA_DIR, ARTIFACTS_DIR,
    SEEDS, QUICK_SEEDS, STUDENT_PARAM_TARGETS,
    BASELINE_CONFIG, ACCURACY_TARGETS, CRITICAL_CLASSES
)


# ============ CONFIG ============
class Config:
    # Data
    DATA_DIR = DATA_DIR
    OUTPUT_DIR = ARTIFACTS_DIR / "stage0_baseline"
    
    # Training
    EPOCHS = BASELINE_CONFIG.epochs
    QUICK_EPOCHS = BASELINE_CONFIG.quick_epochs
    BATCH_SIZE = BASELINE_CONFIG.batch_size
    LR = BASELINE_CONFIG.lr
    WEIGHT_DECAY = BASELINE_CONFIG.weight_decay
    EARLY_STOP_PATIENCE = BASELINE_CONFIG.early_stop_patience
    
    # Loss weights
    BINARY_LOSS_WEIGHT = 0.9
    ATTACK_TYPE_LOSS_WEIGHT = 0.1
    FOCAL_GAMMA = BASELINE_CONFIG.focal_loss_gamma
    
    # Classes (auto-detected from data)
    # Will be populated during training from actual data
    CLASS_NAMES = None  # Set dynamically
    
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============ LOGGING ============
def log(msg: str, experiment: str = ""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{experiment}] " if experiment else ""
    print(f"[{timestamp}] {prefix}{msg}")


# ============ FOCAL LOSS ============
class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()


# ============ MULTI-TASK LOSS ============
class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    - Binary focal loss for BENIGN vs ATTACK
    - Cross-entropy for attack type (ONLY on attack samples)
    """
    def __init__(
        self, 
        binary_weight: float = 0.9,
        attack_type_weight: float = 0.1,
        focal_gamma: float = 2.0,
        attack_class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.binary_weight = binary_weight
        self.attack_type_weight = attack_type_weight
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.attack_class_weights = attack_class_weights
        
    def forward(
        self, 
        binary_logits: torch.Tensor,
        attack_type_logits: torch.Tensor,
        y_binary: torch.Tensor,
        y_attack_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute multi-task loss.
        
        CRITICAL: Attack-type CE is computed ONLY on attack samples (y_binary == 1)
        """
        # Binary loss (all samples)
        binary_loss = self.focal_loss(binary_logits, y_binary)
        
        # Attack-type loss (ONLY on attack samples)
        attack_mask = (y_binary == 1)
        if attack_mask.sum() > 0:
            attack_logits_masked = attack_type_logits[attack_mask]
            attack_labels_masked = y_attack_type[attack_mask]
            
            if self.attack_class_weights is not None:
                attack_type_loss = F.cross_entropy(
                    attack_logits_masked, 
                    attack_labels_masked,
                    weight=self.attack_class_weights
                )
            else:
                attack_type_loss = F.cross_entropy(
                    attack_logits_masked, 
                    attack_labels_masked
                )
        else:
            attack_type_loss = torch.tensor(0.0, device=binary_logits.device)
        
        # Combined loss
        total_loss = self.binary_weight * binary_loss + self.attack_type_weight * attack_type_loss
        
        return total_loss, binary_loss, attack_type_loss


# ============ DATASET ============
class MultiTaskDataset(Dataset):
    """Dataset that returns (X, binary_label, attack_type_label)"""
    def __init__(self, X: np.ndarray, y_binary: np.ndarray, attack_types: np.ndarray, class_to_idx: dict):
        self.X = torch.FloatTensor(X)
        self.y_binary = torch.FloatTensor(y_binary)
        self.y_attack_type = torch.LongTensor([class_to_idx[a] for a in attack_types])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_binary[idx], self.y_attack_type[idx]


# ============ DATA LOADING ============
def load_fold_data(data_dir: Path) -> Dict:
    """Load data for a single fold/split."""
    X_train = np.load(data_dir / "train/X.npy")
    y_train = np.load(data_dir / "train/y.npy")
    attack_train = np.load(data_dir / "train/attack_types.npy", allow_pickle=True)
    
    X_val = np.load(data_dir / "val/X.npy")
    y_val = np.load(data_dir / "val/y.npy")
    attack_val = np.load(data_dir / "val/attack_types.npy", allow_pickle=True)
    
    X_test = np.load(data_dir / "test/X.npy")
    y_test = np.load(data_dir / "test/y.npy")
    attack_test = np.load(data_dir / "test/attack_types.npy", allow_pickle=True)
    
    # Ensure all arrays have same length (take minimum)
    def align_arrays(X, y, attack):
        n = min(len(X), len(y), len(attack))
        return X[:n], y[:n], attack[:n]
    
    X_train, y_train, attack_train = align_arrays(X_train, y_train, attack_train)
    X_val, y_val, attack_val = align_arrays(X_val, y_val, attack_val)
    X_test, y_test, attack_test = align_arrays(X_test, y_test, attack_test)
    
    return {
        'train': (X_train, y_train, attack_train),
        'val': (X_val, y_val, attack_val),
        'test': (X_test, y_test, attack_test)
    }


# ============ EVALUATION ============
def evaluate_model(
    model: MultiTaskStudentDSCNN,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str]
) -> Dict:
    """Evaluate model and return metrics."""
    model.eval()
    
    all_binary_preds = []
    all_binary_targets = []
    all_attack_preds = []
    all_attack_targets = []
    all_attack_types = []
    
    with torch.no_grad():
        for X, y_binary, y_attack in loader:
            X = X.to(device)
            binary_logits, attack_logits = model(X)
            
            # Binary predictions
            binary_probs = torch.sigmoid(binary_logits.squeeze())
            binary_preds = (binary_probs > 0.5).cpu().numpy()
            
            # Attack type predictions
            attack_preds = attack_logits.argmax(dim=1).cpu().numpy()
            
            all_binary_preds.extend(binary_preds)
            all_binary_targets.extend(y_binary.numpy())
            all_attack_preds.extend(attack_preds)
            all_attack_targets.extend(y_attack.numpy())
            all_attack_types.extend([idx_to_class[i] for i in y_attack.numpy()])
    
    all_binary_preds = np.array(all_binary_preds)
    all_binary_targets = np.array(all_binary_targets)
    all_attack_preds = np.array(all_attack_preds)
    all_attack_targets = np.array(all_attack_targets)
    
    # Binary metrics
    binary_acc = (all_binary_preds == all_binary_targets).mean() * 100
    
    # Per-class recall using CORRECT head
    # BENIGN: use binary predictions (predict 0 for BENIGN)
    # Attack classes: use attack-type predictions
    per_class_recall = {}
    
    # Build class_to_idx from idx_to_class
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    unique_classes = set(all_attack_types)
    
    for cls in sorted(unique_classes):
        mask = np.array([t == cls for t in all_attack_types])
        if mask.sum() == 0:
            continue
        
        if cls == 'BENIGN':
            # BENIGN recall from binary head (predict 0)
            recall = ((all_binary_preds[mask] == 0).sum() / mask.sum()) * 100
        else:
            # Attack class recall from attack-type head
            cls_idx = class_to_idx.get(cls, -1)
            if cls_idx >= 0:
                recall = ((all_attack_preds[mask] == cls_idx).sum() / mask.sum()) * 100
            else:
                recall = 0.0
        
        per_class_recall[cls] = recall
    
    # F1 macro (from attack-type head for attack classes)
    attack_mask = all_binary_targets == 1
    if attack_mask.sum() > 0:
        f1_macro = f1_score(
            all_attack_targets[attack_mask],
            all_attack_preds[attack_mask],
            average='macro'
        ) * 100
    else:
        f1_macro = 0.0
    
    # False Alarm Rate = FP / (FP + TN)
    # FP: BENIGN predicted as attack
    benign_mask = all_binary_targets == 0
    if benign_mask.sum() > 0:
        false_alarms = (all_binary_preds[benign_mask] == 1).sum()
        far = false_alarms / benign_mask.sum() * 100
    else:
        far = 0.0
    
    return {
        'binary_acc': binary_acc,
        'f1_macro': f1_macro,
        'far': far,
        'per_class_recall': per_class_recall
    }


# ============ TRAINING ============
def train_student(
    params_target: int,
    seed: int,
    epochs: Optional[int] = None,
    experiment_name: Optional[str] = None
) -> Dict:
    """Train a student model from scratch."""
    
    if epochs is None:
        epochs = Config.EPOCHS
    
    if experiment_name is None:
        experiment_name = f"student_{params_target//1000}k_seed{seed}"
    
    log(f"=" * 60, experiment_name)
    log(f"Training student: {params_target} params, seed {seed}", experiment_name)
    log(f"=" * 60, experiment_name)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}", experiment_name)
    
    # Load data
    log("Loading data...", experiment_name)
    data = load_fold_data(Config.DATA_DIR)
    X_train, y_train, attack_train = data['train']
    X_val, y_val, attack_val = data['val']
    X_test, y_test, attack_test = data['test']
    
    log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", experiment_name)
    
    # Get unique classes from train data
    unique_classes = sorted(set(attack_train))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    n_classes = len(unique_classes)
    
    log(f"Classes ({n_classes}): {unique_classes}", experiment_name)
    
    # Compute class weights
    train_dist = Counter(attack_train)
    total = len(attack_train)
    class_weights_np = np.ones(n_classes, dtype=np.float32)
    for cls, count in train_dist.items():
        idx = class_to_idx[cls]
        class_weights_np[idx] = total / (n_classes * count)
    class_weights_np = class_weights_np / class_weights_np.min()
    class_weights_tensor = torch.FloatTensor(class_weights_np).to(device)
    
    # Create datasets
    train_dataset = MultiTaskDataset(X_train, y_train, attack_train, class_to_idx)
    val_dataset = MultiTaskDataset(X_val, y_val, attack_val, class_to_idx)
    test_dataset = MultiTaskDataset(X_test, y_test, attack_test, class_to_idx)
    
    # Balanced sampler (by binary class)
    n_train = len(X_train)
    n_benign = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    binary_weights = np.where(y_train == 0, 1.0 / n_benign, 1.0 / n_attack).astype(np.float64)
    binary_weights = binary_weights / binary_weights.max()
    
    sampler = WeightedRandomSampler(
        weights=binary_weights.tolist(),
        num_samples=n_train,
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (window, features)
    model = build_student(
        params_target=params_target,
        input_shape=input_shape,
        num_attack_types=n_classes
    ).to(device)
    
    actual_params = model.count_parameters()
    log(f"Model: {actual_params:,} params (target: {params_target:,})", experiment_name)
    log(f"Config: channels={model.conv_channels}, hidden={model.classifier_hidden}, SE={model.use_se}", experiment_name)
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        binary_weight=Config.BINARY_LOSS_WEIGHT,
        attack_type_weight=Config.ATTACK_TYPE_LOSS_WEIGHT,
        focal_gamma=Config.FOCAL_GAMMA,
        attack_class_weights=class_weights_tensor
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    output_dir = Config.OUTPUT_DIR / f"{params_target//1000}k" / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for X, y_binary, y_attack in train_loader:
            X = X.to(device)
            y_binary = y_binary.to(device)
            y_attack = y_attack.to(device)
            
            optimizer.zero_grad()
            binary_logits, attack_logits = model(X)
            loss, _, _ = criterion(binary_logits, attack_logits, y_binary, y_attack)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device, idx_to_class)
        history['val_metrics'].append(val_metrics)
        
        # Progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                f"Val Acc: {val_metrics['binary_acc']:.2f}% - "
                f"F1: {val_metrics['f1_macro']:.2f}% - "
                f"BENIGN: {val_metrics['per_class_recall'].get('BENIGN', 0):.1f}%",
                experiment_name)
        
        # Early stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'config': model.get_config(),
                'class_to_idx': class_to_idx
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                log(f"Early stopping at epoch {epoch+1}", experiment_name)
                break
    
    # Load best model and evaluate on test
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, device, idx_to_class)
    
    # Check critical constraints
    ddos_recall = test_metrics['per_class_recall'].get('DDoS', 0)
    portscan_recall = test_metrics['per_class_recall'].get('PortScan', 0)
    benign_recall = test_metrics['per_class_recall'].get('BENIGN', 0)
    
    constraints_met = (
        ddos_recall >= ACCURACY_TARGETS.critical_recall_min and
        portscan_recall >= ACCURACY_TARGETS.critical_recall_min
    )
    
    # Results
    results = {
        'params_target': params_target,
        'actual_params': actual_params,
        'seed': seed,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'constraints_met': constraints_met,
        'model_config': model.get_config()
    }
    
    log(f"\n{'='*50}", experiment_name)
    log(f"RESULTS - {experiment_name}", experiment_name)
    log(f"{'='*50}", experiment_name)
    log(f"Parameters: {actual_params:,}", experiment_name)
    log(f"Best epoch: {best_epoch}", experiment_name)
    log(f"Binary Accuracy: {test_metrics['binary_acc']:.2f}%", experiment_name)
    log(f"F1-Macro: {test_metrics['f1_macro']:.2f}%", experiment_name)
    log(f"FAR: {test_metrics['far']:.2f}%", experiment_name)
    log(f"\nPer-class Recall:", experiment_name)
    for cls, recall in sorted(test_metrics['per_class_recall'].items()):
        marker = "⚠️" if cls in CRITICAL_CLASSES and recall < ACCURACY_TARGETS.critical_recall_min else ""
        log(f"  {cls}: {recall:.2f}% {marker}", experiment_name)
    log(f"\nConstraints met: {'✅' if constraints_met else '❌'}", experiment_name)
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description="Train baseline student models")
    parser.add_argument('--size', type=int, choices=STUDENT_PARAM_TARGETS,
                        help='Target parameter count (5000, 50000, 200000)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--quick', action='store_true', help='Use quick epochs')
    parser.add_argument('--all', action='store_true', help='Run all sizes with quick seeds')
    
    args = parser.parse_args()
    
    if args.all:
        # Run all sizes with quick seeds
        log("Running all student sizes with quick seeds...")
        all_results = []
        
        for size in STUDENT_PARAM_TARGETS:
            for seed in QUICK_SEEDS:
                epochs = Config.QUICK_EPOCHS if args.quick else None
                result = train_student(size, seed, epochs)
                all_results.append(result)
        
        # Summary
        log("\n" + "=" * 70)
        log("SUMMARY - All Baseline Students")
        log("=" * 70)
        
        for size in STUDENT_PARAM_TARGETS:
            size_results = [r for r in all_results if r['params_target'] == size]
            accs = [r['test_metrics']['binary_acc'] for r in size_results]
            f1s = [r['test_metrics']['f1_macro'] for r in size_results]
            ddos = [r['test_metrics']['per_class_recall'].get('DDoS', 0) for r in size_results]
            portscan = [r['test_metrics']['per_class_recall'].get('PortScan', 0) for r in size_results]
            benign = [r['test_metrics']['per_class_recall'].get('BENIGN', 0) for r in size_results]
            
            log(f"\n{size//1000}K params ({size_results[0]['actual_params']:,} actual):")
            log(f"  Binary Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
            log(f"  F1-Macro:   {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
            log(f"  DDoS:       {np.mean(ddos):.2f}% ± {np.std(ddos):.2f}%")
            log(f"  PortScan:   {np.mean(portscan):.2f}% ± {np.std(portscan):.2f}%")
            log(f"  BENIGN:     {np.mean(benign):.2f}% ± {np.std(benign):.2f}%")
        
        # Save summary
        with open(Config.OUTPUT_DIR / "summary.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
    elif args.size:
        epochs = Config.QUICK_EPOCHS if args.quick else args.epochs
        train_student(args.size, args.seed, epochs)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

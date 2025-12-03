#!/usr/bin/env python
"""
Phase 2 - Stage 0: Baseline Student Training
=============================================

Train baseline student model with N=7 seeds for statistical rigor.
This establishes the accuracy anchor for compression stages.

Usage:
    # Single seed (for testing)
    python train_baseline.py --seed 42 --quick
    
    # All seeds (full run)
    python train_baseline.py --all-seeds
    
    # Specific seed with full epochs
    python train_baseline.py --seed 42 --epochs 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import math
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from phase2.config import (
    SEEDS, DATA_DIR, ARTIFACTS_DIR, 
    BASELINE_CONFIG, STUDENT_CONFIG, ACCURACY_TARGETS
)
from phase2.utils import (
    set_seed, get_env_info, get_device, clear_memory,
    load_data, create_dataloader, get_balanced_sampler,
    compute_metrics, Metrics, save_checkpoint, count_parameters,
    get_model_size_mb, compute_flops, ExperimentLogger
)
from phase2.models import create_student


# ============ LOSS FUNCTIONS ============
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance handling."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============ LEARNING RATE SCHEDULERS ============
class WarmupCosineScheduler:
    """Cosine annealing with warmup."""
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============ EVALUATION ============
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_types: Optional[np.ndarray] = None
) -> Metrics:
    """Evaluate model on data loader."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(
        all_labels, all_preds, all_probs,
        attack_types=attack_types,
        class_names=['Benign', 'Attack']
    )
    
    return metrics


# ============ TRAINING ============
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train_baseline(
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    quick: bool = False,
    verbose: bool = True
) -> dict:
    """
    Train baseline student model.
    
    Args:
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        quick: Use fewer epochs for quick testing
        verbose: Print progress
    
    Returns:
        Dictionary with model path, metrics, and metadata
    """
    # Setup
    set_seed(seed)
    device = get_device()
    
    # Output directory
    out_dir = ARTIFACTS_DIR / "stage0" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = ExperimentLogger(out_dir, f"baseline_seed{seed}")
    logger.log(f"Starting baseline training with seed={seed}")
    logger.log(f"Device: {device}")
    
    # Load data
    logger.log("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types_test = load_data(DATA_DIR)
    
    n_features = X_train.shape[2]
    logger.log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.log(f"Train attack ratio: {y_train.mean()*100:.1f}%")
    
    # Create model
    model = create_student(n_features=n_features).to(device)
    n_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    logger.log(f"Model: {n_params:,} params, {model_size:.2f} MB")
    
    # Create data loaders with balanced sampling
    train_sampler = get_balanced_sampler(y_train)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=True
    )
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size, shuffle=False)
    
    # Loss function with adaptive alpha
    attack_ratio = float(y_train.mean())
    criterion = FocalLoss(alpha=attack_ratio, gamma=BASELINE_CONFIG.focal_loss_gamma)
    logger.log(f"Focal Loss: alpha={attack_ratio:.3f}, gamma={BASELINE_CONFIG.focal_loss_gamma}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    actual_epochs = BASELINE_CONFIG.quick_epochs if quick else epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=actual_epochs)
    
    # Training loop
    best_f1 = 0.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(actual_epochs):
        current_lr = scheduler.step(epoch)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_metrics.accuracy,
            'val_f1_macro': val_metrics.f1_macro,
            'val_detection_rate': val_metrics.detection_rate,
            'val_far': val_metrics.false_alarm_rate,
            'lr': current_lr
        })
        
        logger.log_metrics(seed, 'stage0', epoch + 1, val_metrics)
        
        if verbose:
            print(f"Epoch {epoch+1:3d}/{actual_epochs} | Loss: {train_loss:.4f} | "
                  f"Acc: {val_metrics.accuracy:.1f}% | F1: {val_metrics.f1_macro:.1f}% | "
                  f"DR: {val_metrics.detection_rate:.1f}% | FAR: {val_metrics.false_alarm_rate:.1f}%")
        
        # Save best
        if val_metrics.f1_macro > best_f1:
            best_f1 = val_metrics.f1_macro
            best_metrics = val_metrics
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics, out_dir / "best.pth",
                config={'seed': seed, 'epochs': actual_epochs, 'lr': lr}
            )
            
            if verbose:
                print(f"   ★ New best F1: {best_f1:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= BASELINE_CONFIG.early_stop_patience:
            logger.log(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    logger.log("Evaluating on test set...")
    checkpoint = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, attack_types=attack_types_test)
    
    # Results
    results = {
        'seed': seed,
        'stage': 'stage0_baseline',
        'n_params': n_params,
        'model_size_mb': model_size,
        'best_epoch': best_epoch,
        'training_time_minutes': training_time / 60,
        'test_metrics': test_metrics.to_dict(),
        'best_val_f1': best_f1,
        'history': history,
        'env_info': get_env_info(),
        'config': {
            'epochs': actual_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
        }
    }
    
    # Save results
    with open(out_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING COMPLETE - Seed {seed}")
    print(f"{'='*60}")
    print(f"Test Accuracy:       {test_metrics.accuracy:.2f}%")
    print(f"Test F1-Macro:       {test_metrics.f1_macro:.2f}%")
    print(f"Test Detection Rate: {test_metrics.detection_rate:.2f}%")
    print(f"Test FAR:            {test_metrics.false_alarm_rate:.2f}%")
    print(f"Test AUC:            {test_metrics.auc:.2f}%")
    print(f"DDoS Recall:         {test_metrics.ddos_recall:.2f}%")
    print(f"PortScan Recall:     {test_metrics.portscan_recall:.2f}%")
    print(f"Training Time:       {training_time/60:.1f} minutes")
    print(f"Model saved to:      {out_dir / 'best.pth'}")
    
    # Check targets
    print(f"\n{'='*60}")
    print("TARGET CHECKS")
    print(f"{'='*60}")
    
    checks = [
        ("Critical recall (DDoS) > 98%", test_metrics.ddos_recall >= 98.0),
        ("Critical recall (PortScan) > 98%", test_metrics.portscan_recall >= 98.0),
        ("FAR ≤ 1.5%", test_metrics.false_alarm_rate <= 1.5),
    ]
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    logger.log(f"Training complete. Results saved to {out_dir}")
    
    return results


def train_all_seeds(
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    quick: bool = False
) -> dict:
    """Train baseline model for all seeds and aggregate results."""
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'#'*70}\n")
        
        results = train_baseline(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            quick=quick
        )
        all_results.append(results)
        
        # Clear memory between seeds
        clear_memory()
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS ACROSS ALL SEEDS")
    print(f"{'='*70}")
    
    metrics_to_aggregate = ['accuracy', 'f1_macro', 'detection_rate', 
                           'false_alarm_rate', 'auc', 'ddos_recall', 'portscan_recall']
    
    aggregated = {}
    for metric in metrics_to_aggregate:
        values = [r['test_metrics'][metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci = 1.96 * std_val / np.sqrt(len(values))
        
        aggregated[metric] = {
            'mean': mean_val,
            'std': std_val,
            'ci_95': ci,
            'values': values
        }
        
        print(f"{metric:20s}: {mean_val:.2f}% ± {std_val:.2f}% (95% CI: ±{ci:.2f}%)")
    
    # Save aggregated results
    summary = {
        'stage': 'stage0_baseline',
        'n_seeds': len(SEEDS),
        'seeds': SEEDS,
        'aggregated_metrics': aggregated,
        'individual_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = ARTIFACTS_DIR / "stage0" / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return summary


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Stage 0: Baseline Student Training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all-seeds', action='store_true', help='Train all seeds')
    parser.add_argument('--epochs', type=int, default=BASELINE_CONFIG.epochs, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=BASELINE_CONFIG.batch_size, help='Batch size')
    parser.add_argument('--lr', type=float, default=BASELINE_CONFIG.lr, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=BASELINE_CONFIG.weight_decay, help='Weight decay')
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer epochs')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PHASE 2 - STAGE 0: BASELINE STUDENT TRAINING")
    print(f"{'='*70}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    print(f"Epochs: {BASELINE_CONFIG.quick_epochs if args.quick else args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    if args.all_seeds:
        train_all_seeds(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            quick=args.quick
        )
    else:
        train_baseline(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            quick=args.quick
        )


if __name__ == '__main__':
    main()

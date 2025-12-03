#!/usr/bin/env python
"""
Phase 2 - Stage 3: KD Fine-tuning After Pruning
=================================================

Recover accuracy lost to pruning using Knowledge Distillation.
Uses teacher's soft targets to guide the pruned student.

Usage:
    # Fine-tune after pruning (single seed)
    python fine_tune_kd.py --seed 42 --schedule uniform_50
    
    # Fine-tune all schedules
    python fine_tune_kd.py --seed 42 --all-schedules
    
    # All seeds with best schedule
    python fine_tune_kd.py --all-seeds --schedule nonuniform
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import math
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase2.config import (
    SEEDS, DATA_DIR, ARTIFACTS_DIR,
    KD_CONFIG, ACCURACY_TARGETS
)
from phase2.utils import (
    set_seed, get_device, load_data, create_dataloader,
    get_balanced_sampler, compute_metrics, Metrics,
    save_checkpoint, count_parameters, get_model_size_mb,
    ExperimentLogger, clear_memory
)
from phase2.models import create_student, create_teacher


# ============ KD LOSS ============
class KDLoss(nn.Module):
    """Knowledge Distillation Loss."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# ============ SCHEDULER ============
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ============ EVALUATION ============
def evaluate(model, loader, device, attack_types=None):
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
    
    return compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs),
        attack_types=attack_types, class_names=['Benign', 'Attack']
    )


# ============ KD FINE-TUNING ============
def fine_tune_kd(
    seed: int,
    schedule_name: str = "uniform_50",
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    temperature: float = 4.0,
    alpha: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Fine-tune pruned model using Knowledge Distillation.
    """
    if epochs is None:
        epochs = KD_CONFIG.ft_epochs
    if lr is None:
        lr = KD_CONFIG.ft_lr
    
    set_seed(seed)
    device = get_device()
    
    out_dir = ARTIFACTS_DIR / "stage3" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger(out_dir, f"kd_ft_{schedule_name}_seed{seed}")
    logger.log(f"Starting KD fine-tune: seed={seed}, schedule={schedule_name}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    
    # Load teacher logits
    train_logits = np.load(ARTIFACTS_DIR / "teacher" / "train_logits.npy")
    val_logits = np.load(ARTIFACTS_DIR / "teacher" / "val_logits.npy")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(train_logits)
    )
    train_sampler = get_balanced_sampler(y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    
    val_loader = create_dataloader(X_val, y_val, 256, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, 256, shuffle=False)
    
    # Load pruned model
    pruned_path = ARTIFACTS_DIR / "stage2" / f"seed{seed}" / f"pruned_{schedule_name}.pth"
    if not pruned_path.exists():
        logger.log(f"ERROR: Pruned model not found at {pruned_path}")
        raise FileNotFoundError(f"Pruned model not found: {pruned_path}")
    
    checkpoint = torch.load(pruned_path, map_location='cpu')
    
    if 'model' in checkpoint and checkpoint['model'] is not None:
        model = checkpoint['model']
        logger.log("Loaded pruned model architecture from checkpoint.")
    else:
        logger.log("Checkpoint missing full model. Reconstructing student architecture.")
        model = create_student(n_features=n_features)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as exc:
            logger.log(f"State dict mismatch: {exc}")
            raise RuntimeError(
                "Pruned model checkpoint does not contain architecture information. "
                "Re-run pruning with the updated pipeline to save the full model."
            )
    
    model = model.to(device)
    
    n_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    logger.log(f"Model: {n_params:,} params, {model_size:.3f} MB")
    
    # Get baseline metrics (before KD-FT)
    pre_ft_metrics = evaluate(model, val_loader, device)
    logger.log(f"Pre-FT Val F1: {pre_ft_metrics.f1_macro:.2f}%")
    
    # Get Stage 1 baseline for comparison
    stage1_path = ARTIFACTS_DIR / "stage1" / f"seed{seed}" / "results.json"
    stage1_f1 = None
    if stage1_path.exists():
        with open(stage1_path) as f:
            stage1_f1 = json.load(f)['test_metrics']['f1_macro']
    
    # KD Loss
    criterion = KDLoss(temperature=temperature, alpha=alpha)
    
    # Optimizer - lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, total_epochs=epochs)
    
    # Training loop
    best_f1 = 0.0
    best_metrics = None
    patience_counter = 0
    patience = 10
    history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        current_lr = scheduler.step(epoch)
        
        # Train
        model.train()
        total_loss = 0
        
        for X, y, teacher_logits in train_loader:
            X, y = X.to(device), y.to(device)
            teacher_logits = teacher_logits.to(device)
            
            optimizer.zero_grad()
            student_logits = model(X)
            
            loss = criterion(student_logits, teacher_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        history.append({
            'epoch': epoch + 1,
            'loss': total_loss / len(train_loader),
            'val_f1': val_metrics.f1_macro,
            'val_acc': val_metrics.accuracy,
            'val_dr': val_metrics.detection_rate,
            'val_far': val_metrics.false_alarm_rate
        })
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"F1: {val_metrics.f1_macro:.1f}% | DR: {val_metrics.detection_rate:.1f}%")
        
        # Save best
        if val_metrics.f1_macro > best_f1:
            best_f1 = val_metrics.f1_macro
            best_metrics = val_metrics
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics, out_dir / f"best_{schedule_name}.pth",
                config={'seed': seed, 'schedule': schedule_name, 'T': temperature, 'alpha': alpha}
            )
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.log(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Load best and evaluate on test
    checkpoint = torch.load(out_dir / f"best_{schedule_name}.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, attack_types=attack_types)
    
    # Results
    results = {
        'seed': seed,
        'stage': 'stage3_kd_ft',
        'schedule_name': schedule_name,
        'hyperparams': {'temperature': temperature, 'alpha': alpha, 'lr': lr, 'epochs': epochs},
        'n_params': n_params,
        'model_size_mb': model_size,
        'training_time_minutes': training_time / 60,
        'pre_ft_val_f1': pre_ft_metrics.f1_macro,
        'post_ft_val_f1': best_f1,
        'test_metrics': test_metrics.to_dict(),
        'stage1_f1': stage1_f1,
        'recovery': test_metrics.f1_macro - pre_ft_metrics.f1_macro if pre_ft_metrics else None,
        'vs_stage1': test_metrics.f1_macro - stage1_f1 if stage1_f1 else None,
        'history': history
    }
    
    with open(out_dir / f"results_{schedule_name}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"KD FINE-TUNE COMPLETE - Seed {seed}, Schedule: {schedule_name}")
    print(f"{'='*60}")
    print(f"Test Accuracy:       {test_metrics.accuracy:.2f}%")
    print(f"Test F1-Macro:       {test_metrics.f1_macro:.2f}%")
    print(f"Test Detection Rate: {test_metrics.detection_rate:.2f}%")
    print(f"Test FAR:            {test_metrics.false_alarm_rate:.2f}%")
    print(f"DDoS Recall:         {test_metrics.ddos_recall:.2f}%")
    print(f"PortScan Recall:     {test_metrics.portscan_recall:.2f}%")
    
    print(f"\nRecovery: Pre-FT {pre_ft_metrics.f1_macro:.2f}% → Post-FT {test_metrics.f1_macro:.2f}% "
          f"(+{test_metrics.f1_macro - pre_ft_metrics.f1_macro:.2f}%)")
    
    if stage1_f1:
        delta = test_metrics.f1_macro - stage1_f1
        print(f"vs Stage 1: {stage1_f1:.2f}% → {test_metrics.f1_macro:.2f}% ({delta:+.2f}%)")
        
        if abs(delta) <= ACCURACY_TARGETS.kd_ft_recovery_max:
            print(f"✓ GO: F1 within ±{ACCURACY_TARGETS.kd_ft_recovery_max}% of Stage 1")
        else:
            print(f"✗ NO-GO: F1 drop ({delta:.2f}%) > {ACCURACY_TARGETS.kd_ft_recovery_max}%")
    
    # Critical recall check
    if test_metrics.passes_critical_recall():
        print("✓ Critical recall (DDoS, PortScan) > 98%")
    else:
        print(f"✗ Critical recall failed: DDoS={test_metrics.ddos_recall:.1f}%, PortScan={test_metrics.portscan_recall:.1f}%")
    
    return results


def fine_tune_all_schedules(seed: int = 42):
    """Fine-tune all available pruning schedules."""
    
    prune_dir = ARTIFACTS_DIR / "stage2" / f"seed{seed}"
    schedules = []
    
    # Find all pruned models
    for f in prune_dir.glob("pruned_*.pth"):
        schedule_name = f.stem.replace("pruned_", "")
        schedules.append(schedule_name)
    
    if not schedules:
        print(f"No pruned models found in {prune_dir}")
        return
    
    print(f"Found schedules: {schedules}")
    
    results = []
    for schedule in schedules:
        print(f"\n{'='*50}")
        print(f"Fine-tuning: {schedule}")
        print(f"{'='*50}")
        
        result = fine_tune_kd(seed=seed, schedule_name=schedule)
        results.append(result)
        clear_memory()
    
    # Summary
    print(f"\n{'='*70}")
    print("FINE-TUNE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Schedule':<20} {'Pre-FT F1':>10} {'Post-FT F1':>12} {'Recovery':>10}")
    print("-" * 55)
    
    for r in results:
        recovery = r['test_metrics']['f1_macro'] - r['pre_ft_val_f1']
        print(f"{r['schedule_name']:<20} {r['pre_ft_val_f1']:>9.2f}% {r['test_metrics']['f1_macro']:>11.2f}% {recovery:>+9.2f}%")


def fine_tune_all_seeds(schedule_name: str = "nonuniform"):
    """Fine-tune all seeds with specified schedule."""
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"# KD FINE-TUNE SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'#'*70}")
        
        result = fine_tune_kd(seed=seed, schedule_name=schedule_name)
        all_results.append(result)
        clear_memory()
    
    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATED KD-FT RESULTS (Schedule: {schedule_name})")
    print(f"{'='*70}")
    
    metrics = ['accuracy', 'f1_macro', 'detection_rate', 'false_alarm_rate', 'ddos_recall', 'portscan_recall']
    
    for metric in metrics:
        values = [r['test_metrics'][metric] for r in all_results]
        print(f"{metric:20s}: {np.mean(values):.2f}% ± {np.std(values):.2f}%")
    
    # vs Stage 1
    deltas = [r['vs_stage1'] for r in all_results if r['vs_stage1'] is not None]
    if deltas:
        print(f"\nΔ vs Stage 1: {np.mean(deltas):+.2f}% ± {np.std(deltas):.2f}%")
    
    # Save summary
    summary = {
        'stage': 'stage3_kd_ft',
        'schedule_name': schedule_name,
        'n_seeds': len(SEEDS),
        'individual_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(ARTIFACTS_DIR / "stage3" / f"summary_{schedule_name}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Stage 3: KD Fine-tuning After Pruning')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all-seeds', action='store_true', help='Run all seeds')
    parser.add_argument('--schedule', type=str, default='uniform_50', help='Pruning schedule name')
    parser.add_argument('--all-schedules', action='store_true', help='Fine-tune all schedules')
    parser.add_argument('--epochs', type=int, default=KD_CONFIG.ft_epochs, help='Epochs')
    parser.add_argument('--lr', type=float, default=KD_CONFIG.ft_lr, help='Learning rate')
    parser.add_argument('--T', type=float, default=4.0, help='Temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PHASE 2 - STAGE 3: KD FINE-TUNING AFTER PRUNING")
    print(f"{'='*70}")
    
    if args.all_schedules:
        fine_tune_all_schedules(seed=args.seed)
    elif args.all_seeds:
        fine_tune_all_seeds(schedule_name=args.schedule)
    else:
        fine_tune_kd(
            seed=args.seed,
            schedule_name=args.schedule,
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.T,
            alpha=args.alpha
        )


if __name__ == '__main__':
    main()

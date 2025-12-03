#!/usr/bin/env python
"""
Phase 2 - Train Teacher Model
==============================

Train a larger teacher model for Knowledge Distillation.
Teacher has ~4x parameters of student.

Usage:
    python train_teacher.py --epochs 100 --device cuda:0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from phase2.config import (
    SEEDS, ARTIFACTS_DIR, KD_CONFIG, 
    ACCURACY_TARGETS, DATA_DIR, HARDWARE_TARGETS
)
from phase2.utils import (
    set_reproducibility, MetricsTracker, DataLoaderFactory, FocalLoss
)
from phase2.models import DSCNNTeacher


def train_teacher(
    n_features: int = 78,
    n_classes: int = 8,
    epochs: int = 100,
    device: str = 'cpu',
    seed: int = 42
):
    """Train the teacher model."""
    
    # Setup
    set_reproducibility(seed)
    torch_device = torch.device(device)
    
    # Output directory
    output_dir = ARTIFACTS_DIR / "teacher"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("TEACHER MODEL TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {torch_device}")
    print(f"  Seed: {seed}")
    print(f"  Epochs: {epochs}")
    print(f"  Output: {output_dir}")
    
    # Create data loaders
    print(f"\nLoading data...")
    factory = DataLoaderFactory(DATA_DIR, seed=seed)
    train_loader = factory.get_train_loader(batch_size=256)
    val_loader = factory.get_val_loader(batch_size=512)
    test_loader = factory.get_test_loader(batch_size=512)
    
    # Get data stats
    for batch in train_loader:
        x, y = batch
        n_features = x.shape[1]
        n_classes = len(torch.unique(y))
        print(f"  Features: {n_features}")
        print(f"  Classes: {n_classes}")
        break
    
    # Create teacher model
    print(f"\nCreating teacher model...")
    teacher = DSCNNTeacher(n_features=n_features, n_classes=n_classes)
    teacher = teacher.to(torch_device)
    
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params: {n_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = FocalLoss(gamma=2.0, alpha=None)
    optimizer = optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Metrics tracker
    tracker = MetricsTracker()
    
    # Training loop
    best_val_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining...")
    
    for epoch in range(epochs):
        teacher.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(torch_device), y.to(torch_device)
            
            optimizer.zero_grad()
            logits = teacher(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        teacher.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(torch_device)
                logits = teacher(x)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())
        
        metrics = tracker.compute_metrics(np.array(all_labels), np.array(all_preds))
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}, "
                  f"val_f1={metrics['f1_macro']:.2f}%, "
                  f"val_acc={metrics['accuracy']:.2f}%")
        
        # Early stopping
        if metrics['f1_macro'] > best_val_f1:
            best_val_f1 = metrics['f1_macro']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'n_params': n_params
            }, output_dir / "best_teacher.pt")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    print(f"\nFinal evaluation...")
    checkpoint = torch.load(output_dir / "best_teacher.pt", weights_only=False)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    
    teacher.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(torch_device)
            logits = teacher(x)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    test_metrics = tracker.compute_metrics(np.array(all_labels), np.array(all_preds))
    
    print(f"\n{'='*70}")
    print("TEACHER RESULTS")
    print(f"{'='*70}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Accuracy:   {test_metrics['accuracy']:.2f}%")
    print(f"  F1-Macro:   {test_metrics['f1_macro']:.2f}%")
    print(f"  DR:         {test_metrics['detection_rate']:.2f}%")
    print(f"  FAR:        {test_metrics['false_alarm_rate']:.2f}%")
    
    # Save results
    results = {
        'seed': seed,
        'n_params': n_params,
        'best_epoch': checkpoint['epoch'],
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "teacher_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save logits for KD training
    print(f"\nCaching teacher logits for KD...")
    cache_teacher_logits(teacher, train_loader, val_loader, device, output_dir)
    
    print(f"\nTeacher training complete!")
    print(f"Model saved to: {output_dir / 'best_teacher.pt'}")
    
    return teacher


def cache_teacher_logits(teacher, train_loader, val_loader, device, output_dir):
    """Cache teacher logits for faster KD training."""
    
    teacher.eval()
    
    # Cache training logits
    train_logits = []
    train_labels = []
    
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            logits = teacher(x)
            train_logits.append(logits.cpu())
            train_labels.append(y)
    
    train_logits = torch.cat(train_logits, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    torch.save({
        'logits': train_logits,
        'labels': train_labels
    }, output_dir / "train_logits.pt")
    
    # Cache validation logits
    val_logits = []
    val_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = teacher(x)
            val_logits.append(logits.cpu())
            val_labels.append(y)
    
    val_logits = torch.cat(val_logits, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    torch.save({
        'logits': val_logits,
        'labels': val_labels
    }, output_dir / "val_logits.pt")
    
    print(f"  Cached {len(train_logits)} train logits")
    print(f"  Cached {len(val_logits)} val logits")


def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_teacher(
        epochs=args.epochs,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

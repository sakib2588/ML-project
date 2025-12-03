#!/usr/bin/env python
"""
Run 200K Student Baseline Sweep
================================
Trains 200K model for all 3 seeds and saves results.
Results saved to: artifacts/stage0_baseline/200k/seed{N}/

Usage:
    python run_200k_sweep.py
    
    # Or run in background with logging:
    nohup python run_200k_sweep.py > logs/200k_sweep.log 2>&1 &
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import json
import time
import functools
import numpy as np

# Force flush output for nohup compatibility
print = functools.partial(print, flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from datetime import datetime

# Import from phase2
from phase2.models_multitask import build_student
from phase2.config import DATA_DIR

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    """Load preprocessed data."""
    # DATA_DIR already points to data/processed/cic_ids_2017_v2
    processed_dir = DATA_DIR
    
    # Class name to index mapping
    CLASS_NAMES = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 
                   'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 
                   'PortScan', 'SSH-Patator']
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    # Data is in train/X.npy format
    X_train = np.load(processed_dir / "train" / "X.npy")
    y_train = np.load(processed_dir / "train" / "y.npy")
    X_val = np.load(processed_dir / "val" / "X.npy")
    y_val = np.load(processed_dir / "val" / "y.npy")
    X_test = np.load(processed_dir / "test" / "X.npy")
    y_test = np.load(processed_dir / "test" / "y.npy")
    
    # Load attack types (stored as strings) and convert to indices
    attack_types_str = np.load(processed_dir / "train" / "attack_types.npy", allow_pickle=True)
    attack_types = np.array([name_to_idx.get(str(a), 0) for a in attack_types_str], dtype=np.int64)
    
    # Filter to match attack_types length
    n_samples = len(attack_types)
    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]
    
    print(f"Data loaded: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_test.shape[0]:,} test")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, attack_types

def get_balanced_sampler(y_train, attack_types):
    """Create balanced sampler for training."""
    # Combine binary and attack type for stratification
    combined = y_train * 100 + attack_types
    class_counts = np.bincount(combined.astype(int))
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
    weights = 1.0 / class_counts[combined.astype(int)]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
    return sampler

def evaluate_model(model, loader, device, attack_types_test=None):
    """Evaluate model and return metrics."""
    model.eval()
    all_binary_preds = []
    all_binary_labels = []
    all_attack_preds = []
    all_attack_labels = []
    
    with torch.no_grad():
        for X, y_bin, y_atk in loader:
            X = X.to(device)
            binary_logits, attack_logits = model(X)
            
            binary_preds = (torch.sigmoid(binary_logits) >= 0.5).long().squeeze()
            attack_preds = attack_logits.argmax(dim=1)
            
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_labels.extend(y_bin.numpy())
            all_attack_preds.extend(attack_preds.cpu().numpy())
            all_attack_labels.extend(y_atk.numpy())
    
    binary_preds = np.array(all_binary_preds)
    binary_labels = np.array(all_binary_labels)
    attack_preds = np.array(all_attack_preds)
    attack_labels = np.array(all_attack_labels)
    
    # Binary accuracy
    binary_acc = (binary_preds == binary_labels).mean() * 100
    
    # FAR (False Alarm Rate) - benign classified as attack
    benign_mask = binary_labels == 0
    far = (binary_preds[benign_mask] == 1).mean() * 100 if benign_mask.sum() > 0 else 0
    
    # Per-class recall for attack types
    class_names = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 
                   'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 
                   'PortScan', 'SSH-Patator']
    
    per_class_recall = {}
    for i, name in enumerate(class_names):
        mask = attack_labels == i
        if mask.sum() > 0:
            recall = (attack_preds[mask] == i).mean() * 100
            per_class_recall[name] = recall
        else:
            per_class_recall[name] = 0.0
    
    # F1-Macro
    from sklearn.metrics import f1_score
    f1_macro = f1_score(attack_labels, attack_preds, average='macro') * 100
    
    return {
        'binary_acc': binary_acc,
        'f1_macro': f1_macro,
        'far': far,
        'per_class_recall': per_class_recall
    }

def train_200k_model(seed, epochs=25, batch_size=512, lr=1e-3):
    """Train a single 200K model with given seed."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING 200K MODEL - SEED {seed}")
    print(f"{'='*70}")
    
    # Setup
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Output directory
    out_dir = Path(f"artifacts/stage0_baseline/200k/seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    if (out_dir / "results.json").exists():
        print(f"Seed {seed} already completed, skipping...")
        return None
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types = load_data()
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.LongTensor(attack_types)
    )
    
    sampler = get_balanced_sampler(y_train, attack_types)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    # For val/test, we need attack_types too
    # Class name to index mapping (same as in load_data)
    CLASS_NAMES = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 
                   'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 
                   'PortScan', 'SSH-Patator']
    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    # Load and convert val attack types
    if (DATA_DIR / "val" / "attack_types.npy").exists():
        val_str = np.load(DATA_DIR / "val" / "attack_types.npy", allow_pickle=True)
        attack_types_val = np.array([name_to_idx.get(str(a), 0) for a in val_str], dtype=np.int64)
    else:
        attack_types_val = np.zeros(len(y_val), dtype=np.int64)
    
    # Load and convert test attack types
    if (DATA_DIR / "test" / "attack_types.npy").exists():
        test_str = np.load(DATA_DIR / "test" / "attack_types.npy", allow_pickle=True)
        attack_types_test = np.array([name_to_idx.get(str(a), 0) for a in test_str], dtype=np.int64)
    else:
        attack_types_test = np.zeros(len(y_test), dtype=np.int64)
    
    print(f"Val attack types: {len(attack_types_val)}, Test attack types: {len(attack_types_test)}")
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
        torch.LongTensor(attack_types_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
        torch.LongTensor(attack_types_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    print("\nBuilding 200K model...")
    model = build_student(params_target=200000)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Loss functions
    binary_criterion = nn.BCEWithLogitsLoss()
    attack_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for X, y_bin, y_atk in train_loader:
            X = X.to(device)
            y_bin = y_bin.to(device).float().unsqueeze(1)
            y_atk = y_atk.to(device)
            
            optimizer.zero_grad()
            
            binary_logits, attack_logits = model(X)
            
            # Combined loss
            loss_binary = binary_criterion(binary_logits, y_bin)
            loss_attack = attack_criterion(attack_logits, y_atk)
            loss = 0.5 * loss_binary + 0.5 * loss_attack
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        
        # Quick validation (binary acc only for speed)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y_bin, _ in val_loader:
                X = X.to(device)
                binary_logits, _ = model(X)
                preds = (torch.sigmoid(binary_logits) >= 0.5).long().squeeze()
                val_correct += (preds.cpu() == y_bin).sum().item()
                val_total += len(y_bin)
        
        val_acc = val_correct / val_total * 100
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'val_acc': val_acc
        })
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, out_dir / "best_model.pt")
    
    training_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {training_time:.1f} minutes")
    print(f"Best epoch: {best_epoch} with val acc: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nTEST RESULTS:")
    print(f"  Binary Accuracy: {test_metrics['binary_acc']:.2f}%")
    print(f"  F1-Macro:        {test_metrics['f1_macro']:.2f}%")
    print(f"  FAR:             {test_metrics['far']:.2f}%")
    print(f"  DDoS Recall:     {test_metrics['per_class_recall'].get('DDoS', 0):.2f}%")
    print(f"  PortScan Recall: {test_metrics['per_class_recall'].get('PortScan', 0):.2f}%")
    
    # Check constraints
    ddos_ok = test_metrics['per_class_recall'].get('DDoS', 0) >= 98
    portscan_ok = test_metrics['per_class_recall'].get('PortScan', 0) >= 98
    constraints_met = 1.0 if (ddos_ok and portscan_ok) else 0.0
    
    # Save results
    results = {
        'params_target': 200000,
        'actual_params': n_params,
        'seed': seed,
        'best_epoch': best_epoch,
        'training_time_minutes': training_time,
        'test_metrics': test_metrics,
        'constraints_met': constraints_met,
        'model_config': model.get_config(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(out_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {out_dir / 'results.json'}")
    
    return results

def main():
    """Run 200K sweep for all seeds."""
    print("="*70)
    print("200K STUDENT BASELINE SWEEP")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    seeds = [0, 7, 42]
    all_results = []
    
    for seed in seeds:
        result = train_200k_model(seed, epochs=25)
        if result:
            all_results.append(result)
    
    # Summary
    if all_results:
        print("\n" + "="*70)
        print("SWEEP COMPLETE - SUMMARY")
        print("="*70)
        
        for r in all_results:
            print(f"\nSeed {r['seed']}:")
            print(f"  Binary Acc: {r['test_metrics']['binary_acc']:.2f}%")
            print(f"  F1-Macro:   {r['test_metrics']['f1_macro']:.2f}%")
            print(f"  DDoS:       {r['test_metrics']['per_class_recall'].get('DDoS', 0):.2f}%")
            print(f"  PortScan:   {r['test_metrics']['per_class_recall'].get('PortScan', 0):.2f}%")
        
        # Aggregate
        binary_accs = [r['test_metrics']['binary_acc'] for r in all_results]
        f1s = [r['test_metrics']['f1_macro'] for r in all_results]
        
        print(f"\nAGGREGATE ({len(all_results)} seeds):")
        print(f"  Binary Acc: {np.mean(binary_accs):.2f}% ± {np.std(binary_accs):.2f}%")
        print(f"  F1-Macro:   {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
    
    print(f"\nFinished: {datetime.now()}")

if __name__ == "__main__":
    main()

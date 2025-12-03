#!/usr/bin/env python
"""
Phase 2 - Stage 1: Multi-Task Knowledge Distillation Training
==============================================================

Train student model using teacher's soft targets for both:
1. Binary classification (BENIGN vs ATTACK)
2. Attack-type classification (10-way)

KD Loss combines:
- Hard label loss (focal + CE) from ground truth
- Soft label loss (KL divergence from teacher predictions)

Usage:
    # Train student with KD from Phase 1 teacher
    python train_kd.py --size 50000 --seed 42
    
    # Hyperparameter sweep
    python train_kd.py --size 50000 --sweep
    
    # All seeds with best hyperparams
    python train_kd.py --size 50000 --all-seeds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
from datetime import datetime
from collections import Counter
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score

from phase2.config import (
    SEEDS, QUICK_SEEDS, DATA_DIR, ARTIFACTS_DIR, PHASE1_CV_DIR,
    KD_CONFIG, ACCURACY_TARGETS, STUDENT_PARAM_TARGETS
)
from phase2.models_multitask import build_student, MultiTaskStudentDSCNN


# ============ CONFIG ============
class Config:
    DATA_DIR = DATA_DIR
    OUTPUT_DIR = ARTIFACTS_DIR / "stage1_kd"
    TEACHER_PATH = PHASE1_CV_DIR / "fold_1" / "best_model.pt"
    
    EPOCHS = KD_CONFIG.epochs
    BATCH_SIZE = KD_CONFIG.batch_size
    LR = KD_CONFIG.lr
    WEIGHT_DECAY = KD_CONFIG.weight_decay
    EARLY_STOP_PATIENCE = 15
    
    TEMPERATURE = KD_CONFIG.temperature
    ALPHA = KD_CONFIG.alpha
    BINARY_WEIGHT = 0.9
    ATTACK_WEIGHT = 0.1
    FOCAL_GAMMA = 2.0
    
    TEMPERATURE_SWEEP = KD_CONFIG.temperature_sweep
    ALPHA_SWEEP = KD_CONFIG.alpha_sweep

Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============ LOGGING ============
def log(msg: str, exp: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{exp}] " if exp else ""
    print(f"[{ts}] {prefix}{msg}")


# ============ TEACHER MODEL (Phase 1 architecture) ============
class DepthwiseSeparableConv1d(nn.Module):
    """DS-Conv matching Phase 1 architecture (bias=False)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.pointwise(self.depthwise(x)))))


class MultiTaskTeacher(nn.Module):
    """Phase 1 teacher model architecture."""
    def __init__(self, input_shape=(15, 65), conv_channels=(32, 64, 64), 
                 classifier_hidden=64, dropout=0.2, num_attack_types=10):
        super().__init__()
        self.window_len, self.n_features = input_shape
        
        layers = []
        in_ch = self.n_features
        for out_ch in conv_channels:
            layers.append(DepthwiseSeparableConv1d(in_ch, out_ch, 3, dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_channels[-1], classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.binary_head = nn.Linear(classifier_hidden, 1)
        self.attack_type_head = nn.Linear(classifier_hidden, num_attack_types)
    
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.global_pool(self.backbone(x))
        features = self.feature_fc(x)
        return self.binary_head(features), self.attack_type_head(features)


def load_teacher(path: Path, device: torch.device) -> MultiTaskTeacher:
    """Load Phase 1 teacher."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', {})
    
    teacher = MultiTaskTeacher(
        input_shape=(15, 65),
        conv_channels=(32, 64, 64),
        classifier_hidden=64,
        num_attack_types=config.get('num_attack_types', 10)
    )
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.to(device).eval()
    
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ============ LOSS FUNCTIONS ============
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        inputs, targets = inputs.view(-1), targets.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class MultiTaskKDLoss(nn.Module):
    """Multi-task KD loss for binary + attack-type heads."""
    def __init__(self, temperature=4.0, alpha=0.5, binary_weight=0.9, 
                 attack_weight=0.1, focal_gamma=2.0, attack_class_weights=None):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.binary_weight = binary_weight
        self.attack_weight = attack_weight
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.attack_class_weights = attack_class_weights
    
    def forward(self, s_binary, s_attack, t_binary, t_attack, y_binary, y_attack):
        # Hard label loss
        hard_binary = self.focal_loss(s_binary, y_binary)
        
        attack_mask = (y_binary == 1)
        if attack_mask.sum() > 0:
            hard_attack = F.cross_entropy(
                s_attack[attack_mask], y_attack[attack_mask],
                weight=self.attack_class_weights
            ) if self.attack_class_weights is not None else F.cross_entropy(
                s_attack[attack_mask], y_attack[attack_mask]
            )
        else:
            hard_attack = torch.tensor(0.0, device=s_binary.device)
        
        hard_loss = self.binary_weight * hard_binary + self.attack_weight * hard_attack
        
        # Soft label loss (KD)
        T = self.T
        soft_binary = F.binary_cross_entropy(
            torch.sigmoid(s_binary / T).squeeze(),
            torch.sigmoid(t_binary / T).squeeze(),
            reduction='mean'
        ) * (T ** 2)
        
        soft_attack = F.kl_div(
            F.log_softmax(s_attack / T, dim=1),
            F.softmax(t_attack / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)
        
        soft_loss = self.binary_weight * soft_binary + self.attack_weight * soft_attack
        
        # Combined
        total = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return total, {
            'total': total.item(), 'hard': hard_loss.item(), 'soft': soft_loss.item(),
            'hard_binary': hard_binary.item(), 'hard_attack': hard_attack.item(),
            'soft_binary': soft_binary.item(), 'soft_attack': soft_attack.item()
        }


# ============ DATASET ============
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_binary, attack_types, class_to_idx):
        self.X = torch.FloatTensor(X)
        self.y_binary = torch.FloatTensor(y_binary)
        self.y_attack = torch.LongTensor([class_to_idx[a] for a in attack_types])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_binary[idx], self.y_attack[idx]


# ============ DATA LOADING ============
def load_data(data_dir):
    def load_split(split):
        X = np.load(data_dir / split / "X.npy")
        y = np.load(data_dir / split / "y.npy")
        attack = np.load(data_dir / split / "attack_types.npy", allow_pickle=True)
        n = min(len(X), len(y), len(attack))
        return X[:n], y[:n], attack[:n]
    
    return {s: load_split(s) for s in ['train', 'val', 'test']}


# ============ EVALUATION ============
def evaluate(model, loader, device, idx_to_class):
    model.eval()
    binary_preds, binary_targets = [], []
    attack_preds, attack_targets, attack_types = [], [], []
    
    with torch.no_grad():
        for X, y_b, y_a in loader:
            b_out, a_out = model(X.to(device))
            binary_preds.extend((torch.sigmoid(b_out.squeeze()) > 0.5).cpu().numpy())
            binary_targets.extend(y_b.numpy())
            attack_preds.extend(a_out.argmax(dim=1).cpu().numpy())
            attack_targets.extend(y_a.numpy())
            attack_types.extend([idx_to_class[i] for i in y_a.numpy()])
    
    binary_preds = np.array(binary_preds)
    binary_targets = np.array(binary_targets)
    attack_preds = np.array(attack_preds)
    attack_targets = np.array(attack_targets)
    
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    per_class_recall = {}
    for cls in set(attack_types):
        mask = np.array([t == cls for t in attack_types])
        if mask.sum() == 0:
            continue
        if cls == 'BENIGN':
            per_class_recall[cls] = ((binary_preds[mask] == 0).sum() / mask.sum()) * 100
        else:
            idx = class_to_idx.get(cls, -1)
            per_class_recall[cls] = ((attack_preds[mask] == idx).sum() / mask.sum()) * 100 if idx >= 0 else 0.0
    
    attack_mask = binary_targets == 1
    f1_macro = f1_score(attack_targets[attack_mask], attack_preds[attack_mask], average='macro') * 100 if attack_mask.sum() > 0 else 0.0
    benign_mask = binary_targets == 0
    far = (binary_preds[benign_mask] == 1).sum() / benign_mask.sum() * 100 if benign_mask.sum() > 0 else 0.0
    
    return {
        'binary_acc': (binary_preds == binary_targets).mean() * 100,
        'f1_macro': f1_macro,
        'far': far,
        'per_class_recall': per_class_recall
    }


# ============ TRAINING ============
def train_kd(params_target, seed, temperature=None, alpha=None, epochs=None, exp_name=None):
    if temperature is None: temperature = Config.TEMPERATURE
    if alpha is None: alpha = Config.ALPHA
    if epochs is None: epochs = Config.EPOCHS
    if exp_name is None: exp_name = f"kd_{params_target//1000}k_T{temperature}_a{alpha}_s{seed}"
    
    log(f"{'='*60}", exp_name)
    log(f"KD Training: {params_target} params, T={temperature}, α={alpha}, seed={seed}", exp_name)
    log(f"{'='*60}", exp_name)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher
    if not Config.TEACHER_PATH.exists():
        raise FileNotFoundError(f"Teacher not found: {Config.TEACHER_PATH}")
    teacher = load_teacher(Config.TEACHER_PATH, device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    log(f"Teacher: {teacher_params:,} params (frozen)", exp_name)
    
    # Load data
    data = load_data(Config.DATA_DIR)
    X_train, y_train, attack_train = data['train']
    X_val, y_val, attack_val = data['val']
    X_test, y_test, attack_test = data['test']
    log(f"Data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}", exp_name)
    
    unique_classes = sorted(set(attack_train))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    n_classes = len(unique_classes)
    
    # Class weights
    train_dist = Counter(attack_train)
    class_weights = np.array([len(attack_train) / (n_classes * train_dist[c]) for c in unique_classes], dtype=np.float32)
    class_weights = torch.FloatTensor(class_weights / class_weights.min()).to(device)
    
    # Datasets
    train_ds = MultiTaskDataset(X_train, y_train, attack_train, class_to_idx)
    val_ds = MultiTaskDataset(X_val, y_val, attack_val, class_to_idx)
    test_ds = MultiTaskDataset(X_test, y_test, attack_test, class_to_idx)
    
    # Sampler
    n_benign, n_attack = (y_train == 0).sum(), (y_train == 1).sum()
    weights = np.where(y_train == 0, 1.0 / n_benign, 1.0 / n_attack)
    weights = (weights / weights.max()).astype(np.float64)
    sampler = WeightedRandomSampler(weights.tolist(), len(X_train), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)
    
    # Student
    student = build_student(params_target, (X_train.shape[1], X_train.shape[2]), n_classes).to(device)
    student_params = student.count_parameters()
    log(f"Student: {student_params:,} params, compression: {teacher_params/student_params:.1f}x", exp_name)
    
    # Loss & optimizer
    criterion = MultiTaskKDLoss(temperature, alpha, Config.BINARY_WEIGHT, Config.ATTACK_WEIGHT, 
                                Config.FOCAL_GAMMA, class_weights)
    optimizer = torch.optim.AdamW(student.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    out_dir = Config.OUTPUT_DIR / f"{params_target//1000}k" / f"T{temperature}_a{alpha}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1, best_epoch, patience = 0.0, 0, 0
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        for X, y_b, y_a in train_loader:
            X, y_b, y_a = X.to(device), y_b.to(device), y_a.to(device)
            
            with torch.no_grad():
                t_binary, t_attack = teacher(X)
            
            s_binary, s_attack = student(X)
            loss, _ = criterion(s_binary, s_attack, t_binary, t_attack, y_b, y_a)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        val_metrics = evaluate(student, val_loader, device, idx_to_class)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - "
                f"Acc: {val_metrics['binary_acc']:.1f}% - F1: {val_metrics['f1_macro']:.1f}%", exp_name)
        
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            patience = 0
            torch.save({
                'model_state_dict': student.state_dict(),
                'epoch': epoch, 'val_metrics': val_metrics,
                'config': student.get_config(), 'class_to_idx': class_to_idx,
                'kd_config': {'temperature': temperature, 'alpha': alpha}
            }, out_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= Config.EARLY_STOP_PATIENCE:
                log(f"Early stopping at epoch {epoch+1}", exp_name)
                break
    
    # Test
    checkpoint = torch.load(out_dir / "best_model.pt")
    student.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(student, test_loader, device, idx_to_class)
    
    ddos = test_metrics['per_class_recall'].get('DDoS', 0)
    portscan = test_metrics['per_class_recall'].get('PortScan', 0)
    constraints_met = ddos >= 98 and portscan >= 98
    
    results = {
        'params_target': params_target, 'actual_params': student_params,
        'teacher_params': teacher_params, 'compression': teacher_params / student_params,
        'seed': seed, 'temperature': temperature, 'alpha': alpha,
        'best_epoch': best_epoch, 'test_metrics': test_metrics, 'constraints_met': constraints_met
    }
    
    log(f"\n{'='*50}", exp_name)
    log(f"RESULTS", exp_name)
    log(f"Params: {student_params:,} ({teacher_params/student_params:.1f}x compression)", exp_name)
    log(f"Acc: {test_metrics['binary_acc']:.2f}%, F1: {test_metrics['f1_macro']:.2f}%, FAR: {test_metrics['far']:.2f}%", exp_name)
    for cls, recall in sorted(test_metrics['per_class_recall'].items()):
        marker = "⚠️" if cls in ['DDoS', 'PortScan'] and recall < 98 else ""
        log(f"  {cls}: {recall:.2f}% {marker}", exp_name)
    log(f"Constraints: {'✅' if constraints_met else '❌'}", exp_name)
    
    with open(out_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Multi-Task Knowledge Distillation')
    parser.add_argument('--size', type=int, choices=STUDENT_PARAM_TARGETS, default=50000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--sweep', action='store_true', help='Run T/alpha sweep')
    parser.add_argument('--all-seeds', action='store_true', help='Train all seeds')
    parser.add_argument('--quick', action='store_true', help='Quick run (25 epochs)')
    
    args = parser.parse_args()
    
    if args.sweep:
        log("Running KD hyperparameter sweep...")
        results = []
        epochs = 25 if args.quick else 50
        
        for T in Config.TEMPERATURE_SWEEP:
            for alpha in Config.ALPHA_SWEEP:
                for seed in QUICK_SEEDS[:2]:
                    try:
                        r = train_kd(args.size, seed, T, alpha, epochs)
                        results.append(r)
                    except Exception as e:
                        log(f"Error T={T}, α={alpha}, seed={seed}: {e}")
        
        log("\n" + "=" * 60)
        log("SWEEP SUMMARY")
        for T in Config.TEMPERATURE_SWEEP:
            for alpha in Config.ALPHA_SWEEP:
                rs = [r for r in results if r['temperature'] == T and r['alpha'] == alpha]
                if rs:
                    f1s = [r['test_metrics']['f1_macro'] for r in rs]
                    log(f"T={T}, α={alpha}: F1={np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
        
        with open(Config.OUTPUT_DIR / "sweep_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=float)
    
    elif args.all_seeds:
        log("Training all seeds...")
        results = []
        epochs = 25 if args.quick else args.epochs
        
        for seed in SEEDS:
            r = train_kd(args.size, seed, args.temperature, args.alpha, epochs)
            results.append(r)
        
        log("\n" + "=" * 60)
        log(f"ALL SEEDS SUMMARY ({len(SEEDS)} seeds)")
        f1s = [r['test_metrics']['f1_macro'] for r in results]
        ddos = [r['test_metrics']['per_class_recall'].get('DDoS', 0) for r in results]
        portscan = [r['test_metrics']['per_class_recall'].get('PortScan', 0) for r in results]
        benign = [r['test_metrics']['per_class_recall'].get('BENIGN', 0) for r in results]
        
        log(f"F1-Macro:  {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
        log(f"DDoS:      {np.mean(ddos):.2f}% ± {np.std(ddos):.2f}%")
        log(f"PortScan:  {np.mean(portscan):.2f}% ± {np.std(portscan):.2f}%")
        log(f"BENIGN:    {np.mean(benign):.2f}% ± {np.std(benign):.2f}%")
        
        with open(Config.OUTPUT_DIR / f"{args.size//1000}k_all_seeds.json", 'w') as f:
            json.dump(results, f, indent=2, default=float)
    
    else:
        epochs = 25 if args.quick else args.epochs
        train_kd(args.size, args.seed, args.temperature, args.alpha, epochs)


if __name__ == '__main__':
    main()

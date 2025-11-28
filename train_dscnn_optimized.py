"""
Optimized DS-CNN Training for 85%+ Attack Recall

Strategies:
1. Aggressive Focal Loss (gamma=3.0, alpha=0.85)
2. Class-weighted sampling (oversample attacks)
3. Deeper architecture with more capacity
4. Threshold optimization
5. Longer training with cosine annealing
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss with high gamma for hard example mining"""
    def __init__(self, alpha=0.85, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting for class imbalance
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class OptimizedDSCNN(nn.Module):
    """
    Optimized DS-CNN with:
    - More filters for better feature extraction
    - SE blocks for channel attention
    - Residual connections
    - Dropout for regularization
    """
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        window_len, n_features = input_shape
        
        # Input projection
        self.input_conv = nn.Conv1d(n_features, 64, 1)
        self.input_bn = nn.BatchNorm1d(64)
        
        # DS Conv blocks with SE attention
        self.block1 = DepthwiseSeparableConv(64, 128, kernel_size=3)
        self.se1 = SEBlock(128)
        self.drop1 = nn.Dropout(0.2)
        
        self.block2 = DepthwiseSeparableConv(128, 128, kernel_size=3)
        self.se2 = SEBlock(128)
        self.drop2 = nn.Dropout(0.2)
        
        self.block3 = DepthwiseSeparableConv(128, 256, kernel_size=3)
        self.se3 = SEBlock(256)
        self.drop3 = nn.Dropout(0.3)
        
        self.block4 = DepthwiseSeparableConv(256, 256, kernel_size=3)
        self.se4 = SEBlock(256)
        self.drop4 = nn.Dropout(0.3)
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # 256*2 from GAP+GMP
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch, window, features) -> (batch, features, window)
        x = x.transpose(1, 2)
        
        # Input projection
        x = F.gelu(self.input_bn(self.input_conv(x)))
        
        # Conv blocks
        x = self.drop1(self.se1(self.block1(x)))
        x = self.drop2(self.se2(self.block2(x)))
        x = self.drop3(self.se3(self.block3(x)))
        x = self.drop4(self.se4(self.block4(x)))
        
        # Pooling
        gap = self.gap(x).squeeze(-1)
        gmp = self.gmp(x).squeeze(-1)
        x = torch.cat([gap, gmp], dim=1)
        
        return self.classifier(x)


def load_data(data_path):
    """Load preprocessed data"""
    X_train = np.load(os.path.join(data_path, 'train', 'X.npy'))
    y_train = np.load(os.path.join(data_path, 'train', 'y.npy'))
    X_val = np.load(os.path.join(data_path, 'val', 'X.npy'))
    y_val = np.load(os.path.join(data_path, 'val', 'y.npy'))
    X_test = np.load(os.path.join(data_path, 'test', 'X.npy'))
    y_test = np.load(os.path.join(data_path, 'test', 'y.npy'))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_weighted_sampler(y_train):
    """Create weighted sampler to oversample attacks"""
    class_counts = np.bincount(y_train)
    # Give 3x more weight to attack class
    weights = np.array([1.0, 3.0])
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )
    return sampler


def find_optimal_threshold(model, val_loader, device):
    """Find threshold that maximizes attack recall while keeping FAR reasonable"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_threshold = 0.5
    best_score = 0
    
    # Search for threshold that gives best recall with FAR < 5%
    for threshold in np.arange(0.1, 0.6, 0.02):
        preds = (all_probs >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Score: maximize recall while keeping FAR < 5%
        if far < 0.05:
            score = recall
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    return best_threshold


def evaluate(model, data_loader, device, threshold=0.5):
    """Evaluate model with custom threshold"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    detection_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 0
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'auc': auc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_path)
    
    logger.info(f"Train: {X_train.shape}, Attack ratio: {y_train.mean()*100:.1f}%")
    logger.info(f"Val: {X_val.shape}, Attack ratio: {y_val.mean()*100:.1f}%")
    logger.info(f"Test: {X_test.shape}, Attack ratio: {y_test.mean()*100:.1f}%")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    # Weighted sampler for class imbalance
    sampler = create_weighted_sampler(y_train)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = OptimizedDSCNN(input_shape).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training
    best_val_recall = 0
    best_model_state = None
    best_epoch = 0
    history = {'train_loss': [], 'val_dr': [], 'val_f1': []}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Optimized DS-CNN for {args.epochs} epochs")
    logger.info(f"Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        
        history['train_loss'].append(train_loss)
        history['val_dr'].append(val_metrics['detection_rate'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val F1: {val_metrics['f1_macro']:.2f}% | "
            f"DR: {val_metrics['detection_rate']:.2f}% | "
            f"FAR: {val_metrics['false_alarm_rate']:.2f}%"
        )
        
        # Save best model (by detection rate)
        if val_metrics['detection_rate'] > best_val_recall:
            best_val_recall = val_metrics['detection_rate']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Find optimal threshold
    logger.info("\nFinding optimal classification threshold...")
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
    
    # Final evaluation with optimal threshold
    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL TEST RESULTS (threshold={optimal_threshold:.2f})")
    logger.info(f"{'='*70}")
    
    test_metrics = evaluate(model, test_loader, device, threshold=optimal_threshold)
    
    logger.info(f"Accuracy:       {test_metrics['accuracy']:.2f}%")
    logger.info(f"F1-Macro:       {test_metrics['f1_macro']:.2f}%")
    logger.info(f"Detection Rate: {test_metrics['detection_rate']:.2f}%")
    logger.info(f"False Alarm:    {test_metrics['false_alarm_rate']:.2f}%")
    logger.info(f"AUC:            {test_metrics['auc']:.2f}%")
    logger.info(f"Training Time:  {training_time:.1f}s")
    logger.info(f"Best Epoch:     {best_epoch}")
    logger.info(f"Threshold:      {optimal_threshold:.2f}")
    
    # Also show with default threshold for comparison
    test_metrics_default = evaluate(model, test_loader, device, threshold=0.5)
    logger.info(f"\n[With default threshold 0.5]")
    logger.info(f"Detection Rate: {test_metrics_default['detection_rate']:.2f}%")
    logger.info(f"False Alarm:    {test_metrics_default['false_alarm_rate']:.2f}%")
    
    # Publication readiness check
    logger.info(f"\n{'='*70}")
    logger.info("PUBLICATION READINESS CHECK")
    logger.info(f"{'='*70}")
    
    checks = [
        ("Accuracy > 90%", test_metrics['accuracy'] >= 90),
        ("Attack Recall > 80%", test_metrics['detection_rate'] >= 80),
        ("Attack Recall > 85%", test_metrics['detection_rate'] >= 85),
        ("FAR < 10%", test_metrics['false_alarm_rate'] <= 10),
        ("FAR < 5%", test_metrics['false_alarm_rate'] <= 5),
        ("F1-Macro > 85%", test_metrics['f1_macro'] >= 85),
    ]
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {name}: {status}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'model_name': 'DS_CNN_Optimized',
        'n_parameters': n_params,
        'optimal_threshold': optimal_threshold,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'test_metrics': test_metrics,
        'test_metrics_default_threshold': test_metrics_default,
        'config': {
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr
        }
    }
    
    with open(os.path.join(args.output_dir, 'dscnn_optimized_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': best_model_state,
        'optimal_threshold': optimal_threshold,
        'config': results['config']
    }, os.path.join(args.output_dir, 'dscnn_optimized_best.pt'))
    
    logger.info(f"\nResults saved to {args.output_dir}")
    
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/processed/cic_ids_2017_nn_v2')
    parser.add_argument('--output-dir', default='experiments/dscnn_optimized')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--focal-alpha', type=float, default=0.85)
    parser.add_argument('--focal-gamma', type=float, default=3.0)
    
    args = parser.parse_args()
    train_model(args)

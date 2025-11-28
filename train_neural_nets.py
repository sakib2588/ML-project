"""
Enhanced Neural Network Training for IDS

This script trains DS-CNN, MLP, and LSTM models on the enhanced preprocessed data
with the following improvements:

1. Focal Loss for handling class imbalance
2. Learning rate scheduling (cosine annealing with warm restarts)
3. Early stopping with patience
4. Proper model initialization
5. IDS-specific metrics (Detection Rate, FAR, F1)
6. Model architecture optimizations

Author: Research-grade implementation for IDS publication
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (attacks)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal term
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class EnhancedMLP(nn.Module):
    """
    Enhanced MLP with residual connections and better regularization.
    """
    def __init__(
        self, 
        input_shape: Tuple[int, int],  # (window_size, n_features)
        num_classes: int = 2,
        hidden_sizes: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.4
    ):
        super().__init__()
        
        window_size, n_features = input_shape
        input_dim = window_size * n_features
        
        self.flatten = nn.Flatten()
        
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.features(x)
        return self.classifier(x)


class EnhancedDSCNN(nn.Module):
    """
    Enhanced Depthwise Separable CNN with SE blocks and residual connections.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, n_features)
        num_classes: int = 2,
        channels: Tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        window_size, n_features = input_shape
        
        # Initial projection: (batch, window, features) -> (batch, features, window)
        # Then conv1d operates on window dimension
        
        self.input_bn = nn.BatchNorm1d(n_features)
        
        # Build conv blocks
        blocks = []
        in_ch = n_features
        
        for i, out_ch in enumerate(channels):
            blocks.append(self._make_ds_block(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        
        self.conv_blocks = nn.Sequential(*blocks)
        
        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )
        
        self._init_weights()
    
    def _make_ds_block(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        """Depthwise separable conv block with SE attention."""
        padding = kernel_size // 2
        return nn.Sequential(
            # Depthwise conv
            nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # SE Block
            SEBlock(out_ch)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_features)
        x = x.transpose(1, 2)  # (batch, n_features, window_size)
        x = self.input_bn(x)
        x = self.conv_blocks(x)
        x = self.gap(x).squeeze(-1)  # (batch, channels[-1])
        return self.classifier(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        s = x.mean(dim=-1)  # (batch, channels)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class EnhancedLSTM(nn.Module):
    """
    Enhanced LSTM with attention mechanism.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, n_features)
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        window_size, n_features = input_shape
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_size // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_size*2)
        
        return self.classifier(context)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_processed_data(data_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load preprocessed data from disk."""
    data_dir = Path(data_path)
    
    splits = {}
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / split / 'X.npy')
        y = np.load(data_dir / split / 'y.npy')
        splits[split] = (X, y)
        logger.info(f"Loaded {split}: X={X.shape}, y={y.shape}, attack_ratio={100*np.mean(y):.1f}%")
    
    return splits


def create_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 128
) -> Dict[str, DataLoader]:
    """Create PyTorch dataloaders."""
    loaders = {}
    
    for split, (X, y) in splits.items():
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if DEVICE.type == 'cuda' else False
        )
    
    return loaders


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total, 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Dict[str, float]:
    """Evaluate model and compute IDS metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            
            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'loss': total_loss / len(all_labels),
        'accuracy': 100.0 * accuracy_score(all_labels, all_preds),
        'f1_macro': 100.0 * f1_score(all_labels, all_preds, average='macro'),
        'precision': 100.0 * precision_score(all_labels, all_preds, zero_division=0),
        'recall': 100.0 * recall_score(all_labels, all_preds, zero_division=0),
        'detection_rate': 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Attack recall
        'false_alarm_rate': 100.0 * fp / (fp + tn) if (fp + tn) > 0 else 0.0,  # FPR
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }
    
    # Try to compute AUC
    try:
        metrics['auc'] = 100.0 * roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = 0.0
    
    return metrics


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    model_name: str,
    epochs: int = 50,
    lr: float = 0.001,
    focal_alpha: float = 0.25,  # Lower alpha for more balanced data
    focal_gamma: float = 2.0,
    patience: int = 15,  # More patience
    output_dir: str = 'experiments/nn_enhanced'
) -> Dict[str, any]:
    """Full training loop with early stopping."""
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    model = model.to(DEVICE)
    
    # Loss function
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler: Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, scheduler)
        
        # Validate
        val_metrics = evaluate(model, loaders['val'], criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)
        
        # Logging
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val F1: {val_metrics['f1_macro']:.2f}% | "
            f"DR: {val_metrics['detection_rate']:.2f}% | "
            f"FAR: {val_metrics['false_alarm_rate']:.2f}%"
        )
        
        # Check for improvement
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_path / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(output_path / 'best_model.pt'))
    
    # Final evaluation on test set
    test_metrics = evaluate(model, loaders['test'], criterion)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL TEST RESULTS - {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:       {test_metrics['accuracy']:.2f}%")
    logger.info(f"F1-Macro:       {test_metrics['f1_macro']:.2f}%")
    logger.info(f"Detection Rate: {test_metrics['detection_rate']:.2f}%")
    logger.info(f"False Alarm:    {test_metrics['false_alarm_rate']:.2f}%")
    logger.info(f"AUC:            {test_metrics['auc']:.2f}%")
    logger.info(f"Training Time:  {training_time:.1f}s")
    logger.info(f"Best Epoch:     {best_epoch}")
    
    # Save results
    results = {
        'model_name': model_name,
        'best_epoch': best_epoch,
        'training_time_seconds': training_time,
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'test_metrics': test_metrics,
        'history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_f1': [m['f1_macro'] for m in history['val_metrics']],
            'val_dr': [m['detection_rate'] for m in history['val_metrics']],
        }
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main(
    data_path: str = 'data/processed/cic_ids_2017_nn',
    output_dir: str = 'experiments/nn_enhanced',
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    models_to_train: Optional[List[str]] = None
):
    """Train all neural network models."""
    
    logger.info("=" * 70)
    logger.info("ENHANCED NEURAL NETWORK TRAINING")
    logger.info("=" * 70)
    
    # Load data
    splits = load_processed_data(data_path)
    loaders = create_dataloaders(splits, batch_size=batch_size)
    
    # Get input shape
    X_train = splits['train'][0]
    input_shape = X_train.shape[1:]  # (window_size, n_features)
    logger.info(f"Input shape: {input_shape}")
    
    # Define models
    all_models = {
        'MLP': EnhancedMLP(input_shape, num_classes=2, hidden_sizes=(512, 256, 128), dropout=0.4),
        'DS_CNN': EnhancedDSCNN(input_shape, num_classes=2, channels=(64, 128, 256), dropout=0.3),
        'LSTM': EnhancedLSTM(input_shape, num_classes=2, hidden_size=128, num_layers=2, dropout=0.3),
    }
    
    if models_to_train:
        models = {k: v for k, v in all_models.items() if k in models_to_train}
    else:
        models = all_models
    
    all_results = {}
    
    for model_name, model in models.items():
        results = train_model(
            model=model,
            loaders=loaders,
            model_name=model_name,
            epochs=epochs,
            lr=lr,
            output_dir=output_dir
        )
        all_results[model_name] = results
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL MODELS")
    print("=" * 80)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1-Macro':>10} {'Det.Rate':>10} {'FAR':>10} {'AUC':>10} {'Time':>10}")
    print("-" * 80)
    
    for name, res in all_results.items():
        tm = res['test_metrics']
        print(f"{name:<15} {tm['accuracy']:>9.2f}% {tm['f1_macro']:>9.2f}% {tm['detection_rate']:>9.2f}% {tm['false_alarm_rate']:>9.2f}% {tm['auc']:>9.2f}% {res['training_time_seconds']:>9.1f}s")
    
    print("=" * 80)
    
    # Save summary
    summary_path = Path(output_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Enhanced Neural Networks for IDS')
    parser.add_argument('--data-path', default='data/processed/cic_ids_2017_nn', help='Path to processed data')
    parser.add_argument('--output-dir', default='experiments/nn_enhanced', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--models', nargs='+', choices=['MLP', 'DS_CNN', 'LSTM'], help='Models to train')
    parser.add_argument('--quick', action='store_true', help='Quick mode (10 epochs)')
    
    args = parser.parse_args()
    
    epochs = 10 if args.quick else args.epochs
    
    main(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        models_to_train=args.models
    )

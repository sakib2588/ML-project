#!/usr/bin/env python
"""
Phase 2 - Quantization-Aware Training for Multi-Task Models
============================================================

Train INT8-aware weights to minimize post-quantization accuracy loss.
Supports multi-task DS-CNN with binary + attack heads.

Usage:
    # QAT from baseline model
    python qat_multitask.py --model-path <path> --epochs 20
    
    # QAT with specific bits
    python qat_multitask.py --model-path <path> --bits 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import copy
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from phase2.config import DATA_DIR, ARTIFACTS_DIR, QAT_CONFIG
from phase2.models_multitask import MultiTaskStudentDSCNN, build_student


# ============ FAKE QUANTIZATION MODULES ============
class FakeQuantize(nn.Module):
    """
    Differentiable fake quantization for QAT.
    
    Uses straight-through estimator (STE) for gradient computation.
    """
    
    def __init__(self, bits: int = 8, symmetric: bool = True, per_channel: bool = False):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        # Running statistics
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        
        self.momentum = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update running statistics
            with torch.no_grad():
                x_min = x.min()
                x_max = x.max()
                
                if self.num_batches_tracked == 0:
                    self.running_min = x_min
                    self.running_max = x_max
                else:
                    self.running_min = (1 - self.momentum) * self.running_min + self.momentum * x_min
                    self.running_max = (1 - self.momentum) * self.running_max + self.momentum * x_max
                
                self.num_batches_tracked += 1
                
                # Update scale and zero_point
                if self.symmetric:
                    abs_max = max(abs(self.running_min), abs(self.running_max))
                    self.scale = abs_max / (2 ** (self.bits - 1) - 1)
                else:
                    self.scale = (self.running_max - self.running_min) / (2 ** self.bits - 1)
                    self.zero_point = self.running_min
        
        # Simulate quantization
        scale = self.scale.clamp(min=1e-8)
        
        if self.symmetric:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
            x_q = torch.round(x / scale)
            x_q = torch.clamp(x_q, qmin, qmax)
            x_dq = x_q * scale
        else:
            qmin = 0
            qmax = 2 ** self.bits - 1
            x_q = torch.round((x - self.zero_point) / scale)
            x_q = torch.clamp(x_q, qmin, qmax)
            x_dq = x_q * scale + self.zero_point
        
        # Straight-through estimator: forward uses quantized, backward uses original
        return x + (x_dq - x).detach()


class QATConv1d(nn.Module):
    """Conv1d with fake quantization on weights and activations."""
    
    def __init__(self, conv: nn.Conv1d, bits: int = 8):
        super().__init__()
        self.conv = conv
        self.weight_quant = FakeQuantize(bits=bits, symmetric=True)
        self.act_quant = FakeQuantize(bits=bits, symmetric=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights
        weight_q = self.weight_quant(self.conv.weight)
        
        # Convolution with quantized weights
        out = F.conv1d(x, weight_q, self.conv.bias,
                       self.conv.stride, self.conv.padding,
                       self.conv.dilation, self.conv.groups)
        
        # Quantize activations
        return self.act_quant(out)


class QATLinear(nn.Module):
    """Linear with fake quantization on weights and activations."""
    
    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        self.linear = linear
        self.weight_quant = FakeQuantize(bits=bits, symmetric=True)
        self.act_quant = FakeQuantize(bits=bits, symmetric=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_quant(self.linear.weight)
        out = F.linear(x, weight_q, self.linear.bias)
        return self.act_quant(out)


# ============ QAT MODEL WRAPPER ============
class QATMultiTaskModel(nn.Module):
    """
    Wrapper that adds fake quantization to MultiTaskStudentDSCNN.
    
    Replaces Conv1d and Linear layers with QAT versions.
    """
    
    def __init__(self, base_model: MultiTaskStudentDSCNN, bits: int = 8):
        super().__init__()
        self.base = copy.deepcopy(base_model)
        self.bits = bits
        
        # Input quantization
        self.input_quant = FakeQuantize(bits=bits, symmetric=False)
        
        # Replace layers with QAT versions
        self._replace_layers()
        
    def _replace_layers(self):
        """Replace Conv1d and Linear with QAT versions."""
        
        def replace_module(parent, name, module):
            if isinstance(module, nn.Conv1d):
                setattr(parent, name, QATConv1d(module, self.bits))
            elif isinstance(module, nn.Linear):
                setattr(parent, name, QATLinear(module, self.bits))
        
        # Replace in backbone
        for name, module in list(self.base.named_children()):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                replace_module(self.base, name, module)
            elif isinstance(module, nn.ModuleList):
                for i, m in enumerate(module):
                    if isinstance(m, (nn.Conv1d, nn.Linear)):
                        module[i] = QATConv1d(m, self.bits) if isinstance(m, nn.Conv1d) else QATLinear(m, self.bits)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Quantize input
        x = self.input_quant(x)
        
        # Forward through base model
        return self.base(x)
    
    def get_quantized_state_dict(self) -> Dict:
        """Extract state dict from base model (without QAT wrappers)."""
        state_dict = {}
        
        for name, module in self.base.named_modules():
            if isinstance(module, QATConv1d):
                state_dict[f"{name}.weight"] = module.conv.weight.data
                if module.conv.bias is not None:
                    state_dict[f"{name}.bias"] = module.conv.bias.data
            elif isinstance(module, QATLinear):
                state_dict[f"{name}.weight"] = module.linear.weight.data
                if module.linear.bias is not None:
                    state_dict[f"{name}.bias"] = module.linear.bias.data
            elif hasattr(module, 'weight'):
                state_dict[f"{name}.weight"] = module.weight.data
                if hasattr(module, 'bias') and module.bias is not None:
                    state_dict[f"{name}.bias"] = module.bias.data
        
        return state_dict


# ============ QAT TRAINER ============
class MultiTaskQATTrainer:
    """
    Quantization-Aware Training for multi-task models.
    """
    
    def __init__(
        self,
        model: MultiTaskStudentDSCNN,
        device: torch.device,
        bits: int = 8
    ):
        self.device = device
        self.bits = bits
        
        # Wrap model with QAT
        self.qat_model = QATMultiTaskModel(model, bits=bits).to(device)
        self.original_model = model
        
    def calibrate(self, dataloader: DataLoader, n_batches: int = 100):
        """
        Calibration pass to initialize quantization parameters.
        """
        self.qat_model.eval()
        
        i = -1
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                    
                X = batch[0].to(self.device)
                _ = self.qat_model(X)
        
        print(f"[QAT] Calibration complete ({min(i+1, n_batches)} batches)")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-4,
        patience: int = 8,
        verbose: bool = True
    ) -> Dict:
        """
        QAT training loop with multi-task loss.
        """
        optimizer = torch.optim.AdamW(
            self.qat_model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        
        # Multi-task losses
        bce_loss = nn.BCEWithLogitsLoss()
        ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        best_f1 = 0.0
        patience_counter = 0
        best_state = None
        history = []
        
        for epoch in range(epochs):
            self.qat_model.train()
            total_loss = 0
            
            for batch in train_loader:
                if len(batch) == 3:
                    X, y_binary, y_attack = batch
                else:
                    X, y_binary = batch
                    y_attack = torch.zeros_like(y_binary)
                
                X = X.to(self.device)
                y_binary = y_binary.to(self.device).float()
                y_attack = y_attack.to(self.device)
                
                optimizer.zero_grad()
                
                binary_out, attack_out = self.qat_model(X)
                
                # Binary loss
                loss_binary = bce_loss(binary_out.squeeze(), y_binary)
                
                # Attack loss (only for attack samples)
                attack_mask = y_binary > 0.5
                if attack_mask.sum() > 0:
                    loss_attack = ce_loss(attack_out[attack_mask], y_attack[attack_mask])
                else:
                    loss_attack = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                loss = 0.9 * loss_binary + 0.1 * loss_attack
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.qat_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            # Validate
            val_metrics = self._evaluate(val_loader)
            combined_f1 = 0.5 * val_metrics['binary_f1'] + 0.5 * val_metrics['attack_f1']
            
            history.append({
                'epoch': epoch + 1,
                'loss': total_loss / len(train_loader),
                'binary_f1': val_metrics['binary_f1'],
                'attack_f1': val_metrics['attack_f1'],
                'lr': scheduler.get_last_lr()[0]
            })
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - "
                      f"Binary F1: {val_metrics['binary_f1']:.1f}% - "
                      f"Attack F1: {val_metrics['attack_f1']:.1f}%")
            
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                patience_counter = 0
                best_state = copy.deepcopy(self.qat_model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best
        if best_state is not None:
            self.qat_model.load_state_dict(best_state)
        
        return {
            'epochs_trained': len(history),
            'best_combined_f1': best_f1,
            'history': history
        }
    
    def _evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate QAT model."""
        self.qat_model.eval()
        
        all_binary_preds = []
        all_binary_labels = []
        all_attack_preds = []
        all_attack_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    X, y_binary, y_attack = batch
                else:
                    X, y_binary = batch
                    y_attack = torch.zeros_like(y_binary)
                
                X = X.to(self.device)
                binary_out, attack_out = self.qat_model(X)
                
                binary_pred = (torch.sigmoid(binary_out) >= 0.5).long().squeeze()
                attack_pred = attack_out.argmax(dim=1)
                
                all_binary_preds.extend(binary_pred.cpu().numpy())
                all_binary_labels.extend(y_binary.numpy())
                all_attack_preds.extend(attack_pred.cpu().numpy())
                all_attack_labels.extend(y_attack.numpy())
        
        from sklearn.metrics import f1_score
        
        binary_f1 = f1_score(all_binary_labels, all_binary_preds, average='binary') * 100
        attack_f1 = f1_score(all_attack_labels, all_attack_preds, average='macro') * 100
        
        return {
            'binary_f1': binary_f1,
            'attack_f1': attack_f1,
            'binary_acc': (np.array(all_binary_preds) == np.array(all_binary_labels)).mean() * 100,
            'attack_acc': (np.array(all_attack_preds) == np.array(all_attack_labels)).mean() * 100
        }
    
    def export_model(self) -> MultiTaskStudentDSCNN:
        """Export the base model with QAT-trained weights."""
        # Create a fresh model
        model = copy.deepcopy(self.original_model)
        
        # Copy weights from QAT model
        qat_state = self.qat_model.base.state_dict()
        
        # Filter out QAT-specific keys
        clean_state = {}
        for k, v in qat_state.items():
            # Remove QAT wrapper keys
            if '.conv.weight' in k:
                clean_key = k.replace('.conv.weight', '.weight')
                clean_state[clean_key] = v
            elif '.conv.bias' in k:
                clean_key = k.replace('.conv.bias', '.bias')
                clean_state[clean_key] = v
            elif '.linear.weight' in k:
                clean_key = k.replace('.linear.weight', '.weight')
                clean_state[clean_key] = v
            elif '.linear.bias' in k:
                clean_key = k.replace('.linear.bias', '.bias')
                clean_state[clean_key] = v
            elif 'quant' not in k:
                clean_state[k] = v
        
        model.load_state_dict(clean_state, strict=False)
        return model


# ============ DATA LOADING ============
def load_multitask_data(data_dir: Path) -> Tuple:
    """Load preprocessed data for multi-task training."""
    processed_dir = data_dir / "processed" / "cic_ids_2017"
    
    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    attack_types_train = np.load(processed_dir / "attack_types.npy")
    
    # Filter to match attack_types length
    n_attack = len(attack_types_train)
    X_train = X_train[:n_attack]
    y_train = y_train[:n_attack]
    
    return X_train, y_train, attack_types_train, X_val, y_val, X_test, y_test


def create_multitask_loader(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_attack: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
    balanced: bool = False
) -> DataLoader:
    """Create DataLoader for multi-task training."""
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y_binary),
        torch.LongTensor(y_attack)
    )
    
    if balanced and shuffle:
        class_counts = np.bincount(y_binary)
        weights = 1.0 / class_counts[y_binary]
        sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(y_binary),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============ MAIN CLI ============
def main():
    parser = argparse.ArgumentParser(description='QAT for Multi-Task Model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--bits', type=int, default=8, help='Quantization bits')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"MULTI-TASK QAT - {args.bits}-bit Quantization")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, attack_types, X_val, y_val, X_test, y_test = load_multitask_data(DATA_DIR)
    
    n_features = X_train.shape[2]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create loaders
    train_loader = create_multitask_loader(
        X_train, y_train, attack_types,
        batch_size=args.batch_size, balanced=True
    )
    val_loader = create_multitask_loader(
        X_val, y_val, np.zeros_like(y_val),
        batch_size=args.batch_size * 2, shuffle=False
    )
    test_loader = create_multitask_loader(
        X_test, y_test, np.zeros_like(y_test),
        batch_size=args.batch_size * 2, shuffle=False
    )
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer model size
    init_conv_shape = state_dict['initial_conv.weight'].shape
    channels_0 = init_conv_shape[0]
    
    if channels_0 <= 16:
        params_target = 5000
    elif channels_0 <= 48:
        params_target = 50000
    else:
        params_target = 200000
    
    model = build_student(params_target=params_target)
    model.load_state_dict(state_dict)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params:,} params")
    
    # Create QAT trainer
    trainer = MultiTaskQATTrainer(model, device, bits=args.bits)
    
    # Calibration
    print("\nCalibrating quantization parameters...")
    trainer.calibrate(train_loader, n_batches=100)
    
    # Get pre-QAT baseline
    pre_metrics = trainer._evaluate(val_loader)
    print(f"\nPre-QAT - Binary F1: {pre_metrics['binary_f1']:.2f}%, "
          f"Attack F1: {pre_metrics['attack_f1']:.2f}%")
    
    # QAT Training
    print(f"\nStarting QAT training ({args.epochs} epochs)...")
    train_results = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=8,
        verbose=True
    )
    
    # Final evaluation
    post_metrics = trainer._evaluate(val_loader)
    test_metrics = trainer._evaluate(test_loader)
    
    print(f"\n{'='*60}")
    print("QAT RESULTS")
    print(f"{'='*60}")
    print(f"Val  - Binary F1: {pre_metrics['binary_f1']:.2f}% -> {post_metrics['binary_f1']:.2f}% "
          f"(Δ {post_metrics['binary_f1'] - pre_metrics['binary_f1']:+.2f}%)")
    print(f"Val  - Attack F1: {pre_metrics['attack_f1']:.2f}% -> {post_metrics['attack_f1']:.2f}% "
          f"(Δ {post_metrics['attack_f1'] - pre_metrics['attack_f1']:+.2f}%)")
    print(f"Test - Binary F1: {test_metrics['binary_f1']:.2f}%")
    print(f"Test - Attack F1: {test_metrics['attack_f1']:.2f}%")
    
    # Export model
    exported_model = trainer.export_model()
    
    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.model_path).parent / "qat"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    save_path = out_dir / f"qat_{args.bits}bit.pt"
    torch.save({
        'model_state_dict': exported_model.state_dict(),
        'bits': args.bits,
        'pre_qat_metrics': pre_metrics,
        'post_qat_metrics': post_metrics,
        'test_metrics': test_metrics,
        'training_history': train_results['history']
    }, save_path)
    
    print(f"\nSaved to: {save_path}")
    
    # Save results JSON
    results = {
        'model_path': str(args.model_path),
        'bits': args.bits,
        'epochs_trained': train_results['epochs_trained'],
        'pre_qat': pre_metrics,
        'post_qat': post_metrics,
        'test': test_metrics,
        'model_size_bytes': n_params * 4,  # float32
        'quantized_size_bytes': n_params,  # int8
        'compression_ratio': 4.0,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(out_dir / f"qat_{args.bits}bit_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

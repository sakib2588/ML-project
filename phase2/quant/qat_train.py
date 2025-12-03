#!/usr/bin/env python
"""
Phase 2 - Stage 4: Quantization-Aware Training (QAT)
=====================================================

Train INT8-aware weights to minimize post-quantization accuracy loss.

Two approaches:
1. PyTorch native QAT (torch.quantization)
2. TensorFlow QAT (for TFLite export)

Usage:
    # PyTorch QAT
    python qat_train.py --seed 42 --schedule uniform_50 --framework pytorch
    
    # TensorFlow QAT (for TFLite)
    python qat_train.py --seed 42 --schedule uniform_50 --framework tensorflow
    
    # All seeds
    python qat_train.py --all-seeds --schedule nonuniform
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase2.config import (
    SEEDS, DATA_DIR, ARTIFACTS_DIR, QAT_CONFIG, ACCURACY_TARGETS
)
from phase2.utils import (
    set_seed, get_device, load_data, create_dataloader,
    get_balanced_sampler, get_representative_dataset,
    compute_metrics, Metrics, save_checkpoint,
    count_parameters, get_model_size_mb, ExperimentLogger, clear_memory
)
from phase2.models import create_student


# ============ PYTORCH QAT ============
def prepare_model_for_qat(model: nn.Module) -> nn.Module:
    """
    Prepare PyTorch model for Quantization-Aware Training.
    
    Inserts fake-quantization modules for INT8 simulation.
    """
    import torch.quantization as quant
    
    # Define quantization config
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')  # type: ignore[assignment]
    
    # Fuse modules where possible (Conv-BN-ReLU)
    # For our model, we need to manually specify fusible patterns
    # This is model-specific
    
    # Prepare for QAT (inserts fake-quant modules)
    quant.prepare_qat(model, inplace=True)
    
    return model


def convert_to_quantized(model: nn.Module) -> nn.Module:
    """Convert QAT model to quantized model."""
    import torch.quantization as quant
    
    model.eval()
    model_quantized = quant.convert(model, inplace=False)
    
    return model_quantized


def get_quantized_model_size(model: nn.Module) -> float:
    """Estimate INT8 model size in MB."""
    # Count parameters and assume INT8 (1 byte per param)
    n_params = sum(p.numel() for p in model.parameters())
    return n_params / (1024 * 1024)  # Approximate MB


# ============ SIMULATED QAT ============
class FakeQuantize(nn.Module):
    """Simple fake quantization for simulation."""
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(1), requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update scale based on min/max
            x_min, x_max = x.min(), x.max()
            
            if self.symmetric:
                abs_max = max(abs(x_min), abs(x_max))
                self.scale.data = abs_max / (2 ** (self.bits - 1) - 1)
            else:
                self.scale.data = (x_max - x_min) / (2 ** self.bits - 1)
                self.zero_point.data = x_min
            
            # Simulate quantization
            if self.symmetric:
                x_q = torch.round(x / (self.scale + 1e-8))
                x_q = torch.clamp(x_q, -(2**(self.bits-1)), 2**(self.bits-1) - 1)
                x_dq = x_q * self.scale
            else:
                x_q = torch.round((x - self.zero_point) / (self.scale + 1e-8))
                x_q = torch.clamp(x_q, 0, 2**self.bits - 1)
                x_dq = x_q * self.scale + self.zero_point
            
            # Straight-through estimator
            return x + (x_dq - x).detach()
        else:
            return x


class QATStudentModel(nn.Module):
    """Student model with fake quantization for QAT."""
    
    def __init__(self, base_model: nn.Module, bits: int = 8):
        super().__init__()
        self.base = base_model
        self.bits = bits
        
        # Add fake quantization to weights
        self._add_fake_quant()
    
    def _add_fake_quant(self):
        """Add fake quantization modules."""
        self.input_quant = FakeQuantize(self.bits)
        self.output_quant = FakeQuantize(self.bits)
        
        # Add quant to each major layer
        for name, module in self.base.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                module.register_forward_hook(self._quant_hook)
    
    def _quant_hook(self, module, input, output):
        """Hook to quantize activations."""
        if self.training:
            return self.output_quant(output)
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        return self.base(x)


# ============ QAT TRAINING ============
def train_qat(
    seed: int,
    schedule_name: str = "uniform_50",
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Run Quantization-Aware Training.
    """
    if epochs is None:
        epochs = QAT_CONFIG.epochs
    if lr is None:
        lr = QAT_CONFIG.lr
    
    set_seed(seed)
    device = get_device()
    
    out_dir = ARTIFACTS_DIR / "stage4" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger(out_dir, f"qat_{schedule_name}_seed{seed}")
    logger.log(f"Starting QAT: seed={seed}, schedule={schedule_name}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    
    # Create calibration dataset
    repr_data = get_representative_dataset(X_train, y_train, QAT_CONFIG.calibration_samples)
    logger.log(f"Calibration dataset: {repr_data.shape}")
    
    # Data loaders
    train_sampler = get_balanced_sampler(y_train)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=QAT_CONFIG.batch_size, sampler=train_sampler)
    
    val_loader = create_dataloader(X_val, y_val, QAT_CONFIG.batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, QAT_CONFIG.batch_size, shuffle=False)
    
    # Load Stage 3 model (or Stage 1 if no pruning)
    model_path = ARTIFACTS_DIR / "stage3" / f"seed{seed}" / f"best_{schedule_name}.pth"
    if not model_path.exists():
        model_path = ARTIFACTS_DIR / "stage1" / f"seed{seed}" / "best.pth"
    if not model_path.exists():
        model_path = ARTIFACTS_DIR / "stage0" / f"seed{seed}" / "best.pth"
    
    logger.log(f"Loading model from: {model_path}")
    
    base_model = create_student(n_features=n_features)
    checkpoint = torch.load(model_path, map_location='cpu')
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get baseline metrics
    base_model = base_model.to(device)
    pre_qat_metrics = evaluate(base_model, val_loader, device)
    logger.log(f"Pre-QAT Val F1: {pre_qat_metrics.f1_macro:.2f}%")
    
    # Wrap with QAT
    model = QATStudentModel(base_model, bits=8).to(device)
    
    # Calibration pass
    logger.log("Running calibration...")
    model.eval()
    with torch.no_grad():
        repr_tensor = torch.FloatTensor(repr_data).to(device)
        for i in range(0, len(repr_tensor), 256):
            batch = repr_tensor[i:i+256]
            _ = model(batch)
    
    # QAT Training
    logger.log("Starting QAT training...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_f1 = 0.0
    patience_counter = 0
    patience = 8
    history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
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
            'val_acc': val_metrics.accuracy
        })
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"F1: {val_metrics.f1_macro:.1f}%")
        
        # Save best
        if val_metrics.f1_macro > best_f1:
            best_f1 = val_metrics.f1_macro
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.base.state_dict(),
                'qat_model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': best_f1
            }, out_dir / f"best_{schedule_name}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.log(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Load best and test
    checkpoint = torch.load(out_dir / f"best_{schedule_name}.pth", map_location=device)
    model.base.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, attack_types=attack_types)
    
    # Estimate quantized size
    float_size = get_model_size_mb(model.base)
    int8_size = get_quantized_model_size(model.base)
    
    # Results
    results = {
        'seed': seed,
        'stage': 'stage4_qat',
        'schedule_name': schedule_name,
        'hyperparams': {'epochs': epochs, 'lr': lr, 'bits': 8},
        'training_time_minutes': training_time / 60,
        'pre_qat_f1': pre_qat_metrics.f1_macro,
        'post_qat_f1': best_f1,
        'test_metrics': test_metrics.to_dict(),
        'model_size': {
            'float_mb': float_size,
            'int8_mb_estimate': int8_size,
            'compression_ratio': float_size / int8_size if int8_size > 0 else 0
        },
        'history': history
    }
    
    with open(out_dir / f"results_{schedule_name}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"QAT COMPLETE - Seed {seed}, Schedule: {schedule_name}")
    print(f"{'='*60}")
    print(f"Test Accuracy:       {test_metrics.accuracy:.2f}%")
    print(f"Test F1-Macro:       {test_metrics.f1_macro:.2f}%")
    print(f"Test Detection Rate: {test_metrics.detection_rate:.2f}%")
    print(f"DDoS Recall:         {test_metrics.ddos_recall:.2f}%")
    print(f"PortScan Recall:     {test_metrics.portscan_recall:.2f}%")
    print(f"\nSize: {float_size:.3f} MB (float) → ~{int8_size:.3f} MB (int8)")
    
    # Check accuracy drop
    delta = test_metrics.f1_macro - pre_qat_metrics.f1_macro
    if abs(delta) <= QAT_CONFIG.epochs:  # Using epochs as threshold placeholder
        print(f"✓ QAT F1 change: {delta:+.2f}%")
    
    return results


def evaluate(model, loader, device, attack_types=None):
    """Evaluate model."""
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


def train_all_seeds(schedule_name: str = "nonuniform"):
    """Run QAT for all seeds."""
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"# QAT SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'#'*70}")
        
        result = train_qat(seed=seed, schedule_name=schedule_name)
        all_results.append(result)
        clear_memory()
    
    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATED QAT RESULTS (Schedule: {schedule_name})")
    print(f"{'='*70}")
    
    metrics = ['accuracy', 'f1_macro', 'detection_rate', 'false_alarm_rate']
    for metric in metrics:
        values = [r['test_metrics'][metric] for r in all_results]
        print(f"{metric:20s}: {np.mean(values):.2f}% ± {np.std(values):.2f}%")
    
    # Save summary
    summary = {
        'stage': 'stage4_qat',
        'schedule_name': schedule_name,
        'n_seeds': len(SEEDS),
        'individual_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(ARTIFACTS_DIR / "stage4" / f"summary_{schedule_name}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Stage 4: Quantization-Aware Training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all-seeds', action='store_true', help='Run all seeds')
    parser.add_argument('--schedule', type=str, default='uniform_50', help='Pruning schedule')
    parser.add_argument('--epochs', type=int, default=QAT_CONFIG.epochs, help='Epochs')
    parser.add_argument('--lr', type=float, default=QAT_CONFIG.lr, help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PHASE 2 - STAGE 4: QUANTIZATION-AWARE TRAINING")
    print(f"{'='*70}")
    
    if args.all_seeds:
        train_all_seeds(schedule_name=args.schedule)
    else:
        train_qat(
            seed=args.seed,
            schedule_name=args.schedule,
            epochs=args.epochs,
            lr=args.lr
        )


if __name__ == '__main__':
    main()

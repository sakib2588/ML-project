#!/usr/bin/env python
"""
Phase 2 - Stage 2: Structured Pruning
======================================

Structured (filter/channel) pruning for model compression.
Includes per-layer sensitivity analysis and various pruning schedules.

Steps:
1. Per-layer sensitivity analysis
2. Generate non-uniform schedule based on sensitivity
3. Apply pruning with selected schedule
4. Save pruned model for KD fine-tuning

Usage:
    # Run sensitivity analysis first
    python prune_model.py --mode sensitivity --seed 42
    
    # Apply uniform pruning
    python prune_model.py --mode prune --schedule uniform --ratio 0.5 --seed 42
    
    # Apply non-uniform pruning (based on sensitivity)
    python prune_model.py --mode prune --schedule nonuniform --seed 42
    
    # Run all pruning schedules
    python prune_model.py --mode all --seed 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import copy
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase2.config import (
    SEEDS, DATA_DIR, ARTIFACTS_DIR, PRUNING_CONFIG, ACCURACY_TARGETS
)
from phase2.utils import (
    set_seed, get_device, load_data, create_dataloader,
    compute_metrics, Metrics, count_parameters, get_model_size_mb,
    compute_flops, ExperimentLogger, clear_memory
)
from phase2.models import create_student, StudentModel


# ============ PRUNING UTILITIES ============
def get_prunable_layers(model: nn.Module) -> Dict[str, nn.Conv1d]:
    """Get all prunable convolutional layers."""
    prunable = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            # Skip depthwise convolutions (groups > 1)
            if module.groups == 1 or module.groups == module.out_channels:
                prunable[name] = module
    
    return prunable


def compute_filter_importance(layer: nn.Conv1d, method: str = 'l1') -> np.ndarray:
    """
    Compute importance score for each filter.
    
    Args:
        layer: Conv1d layer
        method: 'l1' (L1-norm), 'l2' (L2-norm), 'random'
    
    Returns:
        importance: Array of shape (out_channels,)
    """
    weight = layer.weight.data.cpu().numpy()  # (out_ch, in_ch, k)
    
    if method == 'l1':
        # L1-norm of each filter
        importance = np.abs(weight).sum(axis=(1, 2))
    elif method == 'l2':
        # L2-norm of each filter
        importance = np.sqrt((weight ** 2).sum(axis=(1, 2)))
    elif method == 'random':
        importance = np.random.rand(weight.shape[0])
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    return importance


def prune_conv_layer(
    layer: nn.Conv1d,
    prune_ratio: float,
    method: str = 'l1'
) -> Tuple[nn.Conv1d, np.ndarray]:
    """
    Prune a convolutional layer by removing filters.
    
    Returns:
        new_layer: Pruned layer
        kept_indices: Indices of kept filters
    """
    if prune_ratio <= 0:
        return layer, np.arange(layer.out_channels)
    
    importance = compute_filter_importance(layer, method)
    n_keep = max(1, int(layer.out_channels * (1 - prune_ratio)))
    
    # Keep top-k filters by importance
    kept_indices = np.argsort(importance)[-n_keep:]
    kept_indices = np.sort(kept_indices)  # Maintain order
    
    # Create new layer
    new_layer = nn.Conv1d(
        in_channels=layer.in_channels,
        out_channels=n_keep,
        kernel_size=layer.kernel_size[0],
        stride=layer.stride[0],
        padding=layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
        dilation=layer.dilation[0],
        groups=min(layer.groups, n_keep),
        bias=layer.bias is not None
    )
    
    # Copy weights
    new_layer.weight.data = layer.weight.data[kept_indices].clone()
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data[kept_indices].clone()  # type: ignore[union-attr]
    
    return new_layer, kept_indices


def adjust_downstream_layer(
    layer: nn.Module,
    kept_indices: np.ndarray,
    is_conv: bool = True
) -> nn.Module:
    """Adjust a downstream layer after pruning its input channels."""
    
    if is_conv and isinstance(layer, nn.Conv1d):
        if layer.groups > 1:
            # Depthwise conv - adjust groups and channels
            new_layer = nn.Conv1d(
                in_channels=len(kept_indices),
                out_channels=len(kept_indices),
                kernel_size=layer.kernel_size[0],
                stride=layer.stride[0],
                padding=layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
                groups=len(kept_indices),
                bias=layer.bias is not None
            )
            new_layer.weight.data = layer.weight.data[kept_indices].clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[kept_indices].clone()  # type: ignore[union-attr]
        else:
            # Regular conv - adjust input channels
            new_layer = nn.Conv1d(
                in_channels=len(kept_indices),
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size[0],
                stride=layer.stride[0],
                padding=layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
                bias=layer.bias is not None
            )
            new_layer.weight.data = layer.weight.data[:, kept_indices].clone()
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone()  # type: ignore[union-attr]
        return new_layer
    
    elif isinstance(layer, nn.BatchNorm1d):
        new_layer = nn.BatchNorm1d(len(kept_indices))
        new_layer.weight.data = layer.weight.data[kept_indices].clone()
        new_layer.bias.data = layer.bias.data[kept_indices].clone()
        if layer.running_mean is not None:
            new_layer.running_mean = layer.running_mean[kept_indices].clone()
        if layer.running_var is not None:
            new_layer.running_var = layer.running_var[kept_indices].clone()
        return new_layer
    
    elif isinstance(layer, nn.Linear):
        new_layer = nn.Linear(len(kept_indices), layer.out_features, bias=layer.bias is not None)
        new_layer.weight.data = layer.weight.data[:, kept_indices].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        return new_layer
    
    return layer


# ============ STRUCTURED PRUNER FOR STUDENT MODEL ============
class StudentPruner:
    """
    Structured pruner specifically for StudentModel architecture.
    
    Handles the multi-scale conv blocks, SE blocks, and attention.
    """
    
    def __init__(self, model: StudentModel, method: str = 'l1'):
        self.model = model
        self.method = method
        self.layer_info = self._analyze_model()
    
    def _analyze_model(self) -> Dict:
        """Analyze model structure for pruning."""
        info = {
            'stem': {'out_ch': self.model.stem[0].out_channels},
            'stage1': {'out_ch': 128},  # MultiScaleConv output
            'stage2': {'out_ch': 128},
            'stage3': {'out_ch': 256},
        }
        return info
    
    def get_prunable_stages(self) -> List[str]:
        """Get list of prunable stages."""
        return ['stem', 'stage1', 'stage2', 'stage3']
    
    def prune_stage(
        self,
        stage_name: str,
        prune_ratio: float
    ) -> Tuple[int, int]:
        """
        Prune a specific stage.
        
        Returns:
            (original_channels, pruned_channels)
        """
        if stage_name == 'stem':
            return self._prune_stem(prune_ratio)
        elif stage_name == 'stage1':
            return self._prune_stage1(prune_ratio)
        elif stage_name == 'stage2':
            return self._prune_stage2(prune_ratio)
        elif stage_name == 'stage3':
            return self._prune_stage3(prune_ratio)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    def _prune_stem(self, prune_ratio: float) -> Tuple[int, int]:
        """Prune stem conv."""
        conv = self.model.stem[0]
        assert isinstance(conv, nn.Conv1d), "Expected Conv1d"
        original_ch = conv.out_channels
        
        # Prune conv
        new_conv, kept_idx = prune_conv_layer(conv, prune_ratio, self.method)
        self.model.stem[0] = new_conv
        
        # Adjust BN
        self.model.stem[1] = adjust_downstream_layer(
            self.model.stem[1], kept_idx, is_conv=False
        )
        
        # Adjust stage1 input (MultiScaleConv)
        msc = self.model.stage1[0]
        conv1 = getattr(msc, 'conv1')
        conv3 = getattr(msc, 'conv3')
        conv5 = getattr(msc, 'conv5')
        conv7 = getattr(msc, 'conv7')
        assert isinstance(conv1, nn.Conv1d) and isinstance(conv3, nn.Conv1d)
        assert isinstance(conv5, nn.Conv1d) and isinstance(conv7, nn.Conv1d)
        msc.conv1 = self._adjust_conv_input(conv1, kept_idx)
        msc.conv3 = self._adjust_conv_input(conv3, kept_idx)
        msc.conv5 = self._adjust_conv_input(conv5, kept_idx)
        msc.conv7 = self._adjust_conv_input(conv7, kept_idx)
        
        return original_ch, len(kept_idx)
    
    def _prune_stage1(self, prune_ratio: float) -> Tuple[int, int]:
        """Prune stage1 (MultiScaleConv + SE)."""
        msc = self.model.stage1[0]
        bn = getattr(msc, 'bn')
        assert isinstance(bn, nn.BatchNorm1d), "Expected BatchNorm1d"
        original_ch = bn.num_features
        
        # Each branch contributes ch/4 channels
        ch_per_branch = original_ch // 4
        n_keep_per_branch = max(1, int(ch_per_branch * (1 - prune_ratio)))
        
        # Prune each branch
        all_kept = []
        convs = [getattr(msc, 'conv1'), getattr(msc, 'conv3'), getattr(msc, 'conv5'), getattr(msc, 'conv7')]
        for i, conv in enumerate(convs):
            assert isinstance(conv, nn.Conv1d), f"Expected Conv1d for branch {i}"
            importance = compute_filter_importance(conv, self.method)
            kept = np.argsort(importance)[-n_keep_per_branch:]
            kept = np.sort(kept)
            all_kept.extend(kept + i * ch_per_branch)
            
            # Create new conv
            new_conv = nn.Conv1d(conv.in_channels, n_keep_per_branch, 
                                conv.kernel_size[0], padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding)
            new_conv.weight.data = conv.weight.data[kept].clone()
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[kept].clone()  # type: ignore[union-attr]
            
            if i == 0: msc.conv1 = new_conv
            elif i == 1: msc.conv3 = new_conv
            elif i == 2: msc.conv5 = new_conv
            else: msc.conv7 = new_conv
        
        new_ch = n_keep_per_branch * 4
        kept_idx = np.array(all_kept)
        
        # Adjust BN
        msc.bn = nn.BatchNorm1d(new_ch)
        
        # Adjust SE block
        se = self.model.stage1[1]
        se.fc1 = nn.Linear(new_ch, new_ch // 4)
        se.fc2 = nn.Linear(new_ch // 4, new_ch)
        
        # Adjust stage2 input
        stage2_pw = getattr(self.model.stage2_conv, 'pointwise')
        assert isinstance(stage2_pw, nn.Conv1d), "Expected Conv1d"
        self.model.stage2_conv.depthwise = self._create_depthwise(new_ch, 3)
        self.model.stage2_conv.pointwise = nn.Conv1d(new_ch, stage2_pw.out_channels, 1)
        
        return original_ch, new_ch
    
    def _prune_stage2(self, prune_ratio: float) -> Tuple[int, int]:
        """Prune stage2."""
        conv = getattr(self.model.stage2_conv, 'pointwise')
        assert isinstance(conv, nn.Conv1d), "Expected Conv1d"
        original_ch = conv.out_channels
        
        # Prune pointwise conv output
        new_conv, kept_idx = prune_conv_layer(conv, prune_ratio, self.method)
        self.model.stage2_conv.pointwise = new_conv
        
        # Adjust BN
        self.model.stage2_conv.bn = nn.BatchNorm1d(len(kept_idx))
        
        # Adjust attention
        attn = self.model.stage2_attn
        red_ch = max(1, len(kept_idx) // 8)
        attn.query = nn.Conv1d(len(kept_idx), red_ch, 1)
        attn.key = nn.Conv1d(len(kept_idx), red_ch, 1)
        attn.value = nn.Conv1d(len(kept_idx), len(kept_idx), 1)
        attn.scale = red_ch ** -0.5
        
        # Adjust stage3 input
        stage3_pw = getattr(self.model.stage3[0], 'pointwise')
        assert isinstance(stage3_pw, nn.Conv1d), "Expected Conv1d"
        self.model.stage3[0].depthwise = self._create_depthwise(len(kept_idx), 3)
        self.model.stage3[0].pointwise = nn.Conv1d(len(kept_idx), stage3_pw.out_channels, 1)
        
        return original_ch, len(kept_idx)
    
    def _prune_stage3(self, prune_ratio: float) -> Tuple[int, int]:
        """Prune stage3."""
        conv = getattr(self.model.stage3[0], 'pointwise')
        assert isinstance(conv, nn.Conv1d), "Expected Conv1d"
        original_ch = conv.out_channels
        
        # Prune pointwise conv
        new_conv, kept_idx = prune_conv_layer(conv, prune_ratio, self.method)
        self.model.stage3[0].pointwise = new_conv
        
        # Adjust BN
        self.model.stage3[0].bn = nn.BatchNorm1d(len(kept_idx))
        
        # Adjust SE
        se = self.model.stage3[1]
        se.fc1 = nn.Linear(len(kept_idx), len(kept_idx) // 4)
        se.fc2 = nn.Linear(len(kept_idx) // 4, len(kept_idx))
        
        # Adjust classifier (gap + gmp = 2x channels)
        self.model.fc1 = nn.Linear(len(kept_idx) * 2, self.model.fc1.out_features)
        
        return original_ch, len(kept_idx)
    
    def _adjust_conv_input(self, conv: nn.Conv1d, kept_idx: np.ndarray) -> nn.Conv1d:
        """Adjust conv input channels."""
        new_conv = nn.Conv1d(
            len(kept_idx), conv.out_channels, conv.kernel_size[0],
            padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        )
        new_conv.weight.data = conv.weight.data[:, kept_idx].clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()  # type: ignore[union-attr]
        return new_conv
    
    def _create_depthwise(self, channels: int, kernel_size: int) -> nn.Conv1d:
        """Create new depthwise conv."""
        return nn.Conv1d(channels, channels, kernel_size, 
                        padding=kernel_size//2, groups=channels)
    
    def apply_schedule(self, schedule: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
        """
        Apply pruning schedule to model.
        
        Args:
            schedule: Dict mapping stage_name to prune_ratio
        
        Returns:
            Dict mapping stage_name to (original_ch, pruned_ch)
        """
        results = {}
        
        for stage_name, prune_ratio in schedule.items():
            if prune_ratio > 0:
                orig, pruned = self.prune_stage(stage_name, prune_ratio)
                results[stage_name] = (orig, pruned)
        
        return results


# ============ EVALUATION ============
def evaluate_quick(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Metrics:
    """Quick evaluation."""
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
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )


def finetune_quick(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4
) -> Metrics:
    """Quick fine-tune after pruning."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    
    return evaluate_quick(model, val_loader, device)


# ============ SENSITIVITY ANALYSIS ============
def run_sensitivity_analysis(
    seed: int = 42,
    prune_pcts: Optional[List[int]] = None,
    finetune_epochs: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Run per-layer sensitivity analysis.
    
    For each prunable stage, test different pruning ratios and measure accuracy drop.
    """
    if prune_pcts is None:
        prune_pcts = PRUNING_CONFIG.sensitivity_prune_pcts
    
    set_seed(seed)
    device = get_device()
    
    out_dir = ARTIFACTS_DIR / "stage2" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, _, _, _ = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    
    # Create data loaders (smaller for sensitivity)
    n_samples = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    train_loader = create_dataloader(X_train[indices], y_train[indices], batch_size=128)
    val_loader = create_dataloader(X_val, y_val, batch_size=256, shuffle=False)
    
    # Load baseline model
    baseline_path = ARTIFACTS_DIR / "stage1" / f"seed{seed}" / "best.pth"
    if not baseline_path.exists():
        baseline_path = ARTIFACTS_DIR / "stage0" / f"seed{seed}" / "best.pth"
    
    if verbose:
        print(f"Loading model from: {baseline_path}")
    
    # Get baseline accuracy
    base_model = create_student(n_features=n_features).to(device)
    checkpoint = torch.load(baseline_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    baseline_metrics = evaluate_quick(base_model, val_loader, device)
    baseline_f1 = baseline_metrics.f1_macro
    
    if verbose:
        print(f"Baseline F1: {baseline_f1:.2f}%")
    
    # Sensitivity analysis
    results = {'baseline_f1': baseline_f1, 'stages': {}}
    stages = ['stem', 'stage1', 'stage2', 'stage3']
    
    for stage in stages:
        if verbose:
            print(f"\nAnalyzing {stage}...")
        
        stage_results = []
        
        for pct in prune_pcts:
            # Fresh copy of model
            model = create_student(n_features=n_features).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Prune only this stage
            pruner = StudentPruner(model, method='l1')
            orig_ch, pruned_ch = pruner.prune_stage(stage, pct / 100.0)
            
            # Fine-tune
            metrics_after = finetune_quick(model, train_loader, val_loader, device, finetune_epochs)
            
            delta_f1 = metrics_after.f1_macro - baseline_f1
            
            stage_results.append({
                'prune_pct': pct,
                'f1_after': metrics_after.f1_macro,
                'delta_f1': delta_f1,
                'channels': (orig_ch, pruned_ch)
            })
            
            if verbose:
                print(f"  {pct}%: F1={metrics_after.f1_macro:.2f}% (Δ={delta_f1:+.2f}%)")
            
            clear_memory()
        
        results['stages'][stage] = stage_results
    
    # Save results
    with open(out_dir / "sensitivity.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nSensitivity results saved to: {out_dir / 'sensitivity.json'}")
    
    return results


def generate_nonuniform_schedule(
    sensitivity_results: Dict,
    target_drop: float = 1.0
) -> Dict[str, float]:
    """
    Generate non-uniform pruning schedule based on sensitivity.
    
    Less sensitive layers get pruned more aggressively.
    """
    schedules = {}
    
    for stage, results in sensitivity_results['stages'].items():
        # Find max pruning that keeps delta_f1 > -target_drop
        max_pct = 0
        for r in results:
            if r['delta_f1'] >= -target_drop:
                max_pct = max(max_pct, r['prune_pct'])
        
        schedules[stage] = max_pct / 100.0
    
    return schedules


# ============ APPLY PRUNING ============
def apply_pruning(
    seed: int,
    schedule: Dict[str, float],
    schedule_name: str = "custom",
    verbose: bool = True
) -> Dict:
    """Apply pruning schedule to model."""
    set_seed(seed)
    device = get_device()
    
    out_dir = ARTIFACTS_DIR / "stage2" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    
    val_loader = create_dataloader(X_val, y_val, batch_size=256, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=256, shuffle=False)
    
    # Load model (prefer KD model)
    model_path = ARTIFACTS_DIR / "stage1" / f"seed{seed}" / "best.pth"
    if not model_path.exists():
        model_path = ARTIFACTS_DIR / "stage0" / f"seed{seed}" / "best.pth"
    
    model = create_student(n_features=n_features).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Metrics before pruning
    original_params = count_parameters(model)
    original_size = get_model_size_mb(model)
    original_metrics = evaluate_quick(model, val_loader, device)
    
    if verbose:
        print(f"Before pruning:")
        print(f"  Params: {original_params:,}")
        print(f"  Size: {original_size:.3f} MB")
        print(f"  Val F1: {original_metrics.f1_macro:.2f}%")
        print(f"\nApplying schedule: {schedule}")
    
    # Apply pruning
    pruner = StudentPruner(model, method='l1')
    prune_results = pruner.apply_schedule(schedule)
    
    # Metrics after pruning
    pruned_params = count_parameters(model)
    pruned_size = get_model_size_mb(model)
    pruned_metrics = evaluate_quick(model, val_loader, device)
    
    if verbose:
        print(f"\nAfter pruning:")
        print(f"  Params: {pruned_params:,} ({pruned_params/original_params*100:.1f}%)")
        print(f"  Size: {pruned_size:.3f} MB ({pruned_size/original_size*100:.1f}%)")
        print(f"  Val F1: {pruned_metrics.f1_macro:.2f}% (Δ={pruned_metrics.f1_macro - original_metrics.f1_macro:+.2f}%)")
    
    # Test evaluation
    test_metrics = evaluate_quick(model, test_loader, device)
    
    # Save pruned model (include full module for architecture fidelity)
    model_path = out_dir / f"pruned_{schedule_name}.pth"
    pruned_model_cpu = copy.deepcopy(model).cpu()
    torch.save({
        'model_state_dict': pruned_model_cpu.state_dict(),
        'model': pruned_model_cpu,
        'schedule': schedule,
        'schedule_name': schedule_name,
        'original_params': original_params,
        'pruned_params': pruned_params,
        'prune_results': prune_results
    }, model_path)
    
    # Results
    results = {
        'seed': seed,
        'schedule_name': schedule_name,
        'schedule': schedule,
        'original': {
            'params': original_params,
            'size_mb': original_size,
            'val_f1': original_metrics.f1_macro
        },
        'pruned': {
            'params': pruned_params,
            'size_mb': pruned_size,
            'val_f1': pruned_metrics.f1_macro,
            'test_f1': test_metrics.f1_macro,
            'test_dr': test_metrics.detection_rate,
            'test_far': test_metrics.false_alarm_rate
        },
        'compression': {
            'param_ratio': pruned_params / original_params,
            'size_ratio': pruned_size / original_size,
            'f1_drop': original_metrics.f1_macro - pruned_metrics.f1_macro
        },
        'model_path': str(model_path)
    }
    
    with open(out_dir / f"pruning_{schedule_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nPruned model saved to: {model_path}")
    
    # Check go/no-go
    f1_drop = original_metrics.f1_macro - pruned_metrics.f1_macro
    if f1_drop <= ACCURACY_TARGETS.prune_immediate_drop_max:
        print(f"✓ GO: Immediate F1 drop ({f1_drop:.2f}%) ≤ {ACCURACY_TARGETS.prune_immediate_drop_max}%")
    else:
        print(f"✗ NO-GO: Immediate F1 drop ({f1_drop:.2f}%) > {ACCURACY_TARGETS.prune_immediate_drop_max}%")
    
    return results


# ============ ITERATIVE PRUNING (NEW - Enhanced per feedback) ============
def apply_iterative_pruning(
    seed: int,
    target_ratio: float = 0.5,
    n_steps: int = 10,
    finetune_epochs_per_step: int = 5,
    max_f1_drop: float = 3.0,
    verbose: bool = True
) -> Dict:
    """
    Apply iterative (gradual) pruning with fine-tuning between steps.
    
    Research shows iterative pruning outperforms one-shot by +0.5-1.0% F1.
    Reference: "Rethinking the Value of Network Pruning", ICLR 2019
    
    Args:
        seed: Random seed
        target_ratio: Final target pruning ratio (e.g., 0.5 = 50% pruned)
        n_steps: Number of pruning steps
        finetune_epochs_per_step: Fine-tuning epochs between pruning steps
        max_f1_drop: Stop if F1 drops more than this %
        verbose: Print progress
    
    Returns:
        Dict with pruning history and final model info
    """
    set_seed(seed)
    device = get_device()
    
    out_dir = ARTIFACTS_DIR / "stage2" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print("ITERATIVE PRUNING")
        print(f"{'='*70}")
        print(f"Target ratio: {target_ratio*100:.0f}%")
        print(f"Steps: {n_steps}")
        print(f"Fine-tune epochs/step: {finetune_epochs_per_step}")
        print(f"Max F1 drop threshold: {max_f1_drop}%")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    
    train_loader = create_dataloader(X_train, y_train, batch_size=128)
    val_loader = create_dataloader(X_val, y_val, batch_size=256, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=256, shuffle=False)
    
    # Load baseline model
    model_path = ARTIFACTS_DIR / "stage1" / f"seed{seed}" / "best.pth"
    if not model_path.exists():
        model_path = ARTIFACTS_DIR / "stage0" / f"seed{seed}" / "best.pth"
    
    model = create_student(n_features=n_features).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Baseline metrics
    baseline_metrics = evaluate_quick(model, val_loader, device)
    baseline_f1 = baseline_metrics.f1_macro
    baseline_params = count_parameters(model)
    baseline_size = get_model_size_mb(model)
    
    if verbose:
        print(f"\nBaseline: F1={baseline_f1:.2f}%, Params={baseline_params:,}, Size={baseline_size:.3f}MB")
    
    # Pruning history
    history = {
        'baseline_f1': baseline_f1,
        'baseline_params': baseline_params,
        'baseline_size_mb': baseline_size,
        'steps': [],
        'per_attack_degradation': {}
    }
    
    # Calculate step ratio
    step_ratio = target_ratio / n_steps
    current_total_ratio = 0.0
    
    stages = ['stem', 'stage1', 'stage2', 'stage3']
    
    for step in range(n_steps):
        current_total_ratio += step_ratio
        step_schedule = {stage: step_ratio for stage in stages}
        
        if verbose:
            print(f"\n--- Step {step+1}/{n_steps}: Cumulative prune {current_total_ratio*100:.1f}% ---")
        
        # Create pruner for current model
        pruner = StudentPruner(model, method='l1')
        
        # Apply incremental pruning
        try:
            prune_results = pruner.apply_schedule(step_schedule)
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Pruning failed at step {step+1}: {e}")
            break
        
        # Evaluate after pruning (before fine-tuning)
        metrics_pruned = evaluate_quick(model, val_loader, device)
        
        # Fine-tune
        if verbose:
            print(f"  Pre-FT F1: {metrics_pruned.f1_macro:.2f}%")
        
        metrics_finetuned = finetune_quick(
            model, train_loader, val_loader, device,
            epochs=finetune_epochs_per_step, lr=1e-4
        )
        
        current_params = count_parameters(model)
        current_size = get_model_size_mb(model)
        f1_drop_from_baseline = baseline_f1 - metrics_finetuned.f1_macro
        
        step_info = {
            'step': step + 1,
            'cumulative_ratio': current_total_ratio,
            'f1_pre_finetune': metrics_pruned.f1_macro,
            'f1_post_finetune': metrics_finetuned.f1_macro,
            'f1_drop_from_baseline': f1_drop_from_baseline,
            'params': current_params,
            'size_mb': current_size,
            'param_ratio': current_params / baseline_params
        }
        history['steps'].append(step_info)
        
        if verbose:
            print(f"  Post-FT F1: {metrics_finetuned.f1_macro:.2f}% "
                  f"(Δ from baseline: {-f1_drop_from_baseline:+.2f}%)")
            print(f"  Params: {current_params:,} ({current_params/baseline_params*100:.1f}%)")
        
        # Check early stopping condition
        if f1_drop_from_baseline > max_f1_drop:
            if verbose:
                print(f"\n⚠️ EARLY STOP: F1 drop ({f1_drop_from_baseline:.2f}%) > threshold ({max_f1_drop}%)")
            break
        
        clear_memory()
    
    # Final test evaluation
    test_metrics = evaluate_quick(model, test_loader, device)
    
    # Final results
    final_params = count_parameters(model)
    final_size = get_model_size_mb(model)
    
    history['final'] = {
        'params': final_params,
        'size_mb': final_size,
        'param_ratio': final_params / baseline_params,
        'size_ratio': final_size / baseline_size,
        'test_f1': test_metrics.f1_macro,
        'test_dr': test_metrics.detection_rate,
        'test_far': test_metrics.false_alarm_rate,
        'f1_drop': baseline_f1 - test_metrics.f1_macro
    }
    
    # Save model
    model_path_out = out_dir / f"pruned_iterative_{int(target_ratio*100)}.pth"
    pruned_model_cpu = copy.deepcopy(model).cpu()
    torch.save({
        'model_state_dict': pruned_model_cpu.state_dict(),
        'model': pruned_model_cpu,
        'history': history,
        'target_ratio': target_ratio,
        'n_steps': n_steps
    }, model_path_out)
    
    # Save history
    with open(out_dir / f"iterative_pruning_{int(target_ratio*100)}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    if verbose:
        print(f"\n{'='*70}")
        print("ITERATIVE PRUNING COMPLETE")
        print(f"{'='*70}")
        print(f"Final Params: {final_params:,} ({final_params/baseline_params*100:.1f}% of original)")
        print(f"Final Size: {final_size:.3f} MB ({final_size/baseline_size*100:.1f}% of original)")
        print(f"Test F1: {test_metrics.f1_macro:.2f}%")
        print(f"F1 Drop: {baseline_f1 - test_metrics.f1_macro:.2f}%")
        print(f"Model saved: {model_path_out}")
    
    return history


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Stage 2: Structured Pruning')
    parser.add_argument('--mode', choices=['sensitivity', 'prune', 'iterative', 'all'],
                        default='sensitivity', help='Mode to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--schedule', type=str, default='uniform',
                        choices=['uniform', 'nonuniform', 'conservative'],
                        help='Pruning schedule type')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Pruning ratio (for uniform/iterative)')
    parser.add_argument('--n-steps', type=int, default=10,
                        help='Number of iterative pruning steps')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='Fine-tune epochs per iterative step')
    parser.add_argument('--max-drop', type=float, default=3.0,
                        help='Max F1 drop before early stopping')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PHASE 2 - STAGE 2: STRUCTURED PRUNING")
    print(f"{'='*70}")
    
    if args.mode == 'sensitivity':
        run_sensitivity_analysis(seed=args.seed)
    
    elif args.mode == 'prune':
        schedule = None
        schedule_name = args.schedule

        if args.schedule == 'uniform':
            schedule = {stage: args.ratio for stage in ['stem', 'stage1', 'stage2', 'stage3']}
            schedule_name = f"uniform_{int(args.ratio*100)}"
        elif args.schedule == 'nonuniform':
            # Load sensitivity results
            sens_path = ARTIFACTS_DIR / "stage2" / f"seed{args.seed}" / "sensitivity.json"
            if sens_path.exists():
                with open(sens_path) as f:
                    sens_results = json.load(f)
                schedule = generate_nonuniform_schedule(sens_results)
                schedule_name = "nonuniform"
            else:
                print("Sensitivity results not found. Run --mode sensitivity first.")
                return
        elif args.schedule == 'conservative':
            # Conservative schedule (less aggressive)
            schedule = {'stem': 0.2, 'stage1': 0.3, 'stage2': 0.3, 'stage3': 0.4}
            schedule_name = "conservative"
        
        if schedule is None:
            raise ValueError("Pruning schedule could not be determined")
        
        apply_pruning(seed=args.seed, schedule=schedule, schedule_name=schedule_name)
    
    elif args.mode == 'iterative':
        # NEW: Iterative pruning mode
        apply_iterative_pruning(
            seed=args.seed,
            target_ratio=args.ratio,
            n_steps=args.n_steps,
            finetune_epochs_per_step=args.finetune_epochs,
            max_f1_drop=args.max_drop,
            verbose=True
        )
    
    elif args.mode == 'all':
        # Run everything
        print("Running sensitivity analysis...")
        sens_results = run_sensitivity_analysis(seed=args.seed)
        
        print("\nApplying uniform 30%...")
        apply_pruning(args.seed, {s: 0.3 for s in ['stem', 'stage1', 'stage2', 'stage3']}, "uniform_30")
        
        print("\nApplying uniform 50%...")
        apply_pruning(args.seed, {s: 0.5 for s in ['stem', 'stage1', 'stage2', 'stage3']}, "uniform_50")
        
        print("\nApplying non-uniform...")
        nonuniform = generate_nonuniform_schedule(sens_results)
        apply_pruning(args.seed, nonuniform, "nonuniform")
        
        # NEW: Also run iterative pruning
        print("\nApplying iterative 50% (10 steps)...")
        apply_iterative_pruning(
            seed=args.seed,
            target_ratio=0.5,
            n_steps=10,
            finetune_epochs_per_step=5,
            max_f1_drop=3.0,
            verbose=True
        )


if __name__ == '__main__':
    main()

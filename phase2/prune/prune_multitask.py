#!/usr/bin/env python
"""
Phase 2 - Pruning for Multi-Task Models
========================================

Structured pruning for MultiTaskStudentDSCNN models.
Prunes shared backbone while preserving both heads.

Usage:
    # Sensitivity analysis
    python prune_multitask.py --model-path <path> --mode sensitivity
    
    # Apply pruning with specific ratio
    python prune_multitask.py --model-path <path> --mode prune --ratio 0.3
    
    # Iterative pruning with fine-tuning
    python prune_multitask.py --model-path <path> --mode iterative --target-ratio 0.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import copy
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from phase2.config import (
    DATA_DIR, ARTIFACTS_DIR, PRUNING_CONFIG, ACCURACY_TARGETS, SEEDS
)
from phase2.models_multitask import MultiTaskStudentDSCNN, build_student


# ============ MULTI-TASK PRUNER ============
class MultiTaskPruner:
    """
    Structured pruning for MultiTaskStudentDSCNN.
    
    Prunable layers in the shared backbone:
    - initial_conv (Conv1d)
    - dw_convs (multi-scale depthwise convolutions)
    - pw_convs (pointwise convolutions)
    
    Protected layers (not pruned):
    - binary_head (binary classification output)
    - attack_head (10-class output)
    - se_* layers (if present)
    """
    
    def __init__(
        self,
        model: MultiTaskStudentDSCNN,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model
        self.device = device
        
        # Find prunable layers in shared backbone
        self.prunable_layers = self._find_prunable_layers()
        print(f"[Pruner] Found {len(self.prunable_layers)} prunable layers")
        
    def _find_prunable_layers(self) -> Dict[str, nn.Module]:
        """Identify layers that can be safely pruned."""
        prunable = {}
        
        for name, module in self.model.named_modules():
            # Skip heads - these must not be pruned
            if 'binary_head' in name or 'attack_head' in name:
                continue
            
            # Skip SE attention layers (they depend on channel count)
            if 'se_' in name:
                continue
                
            # Prune Conv1d layers (except final output layers)
            if isinstance(module, nn.Conv1d):
                prunable[name] = module
                
        return prunable
    
    def get_layer_importance(
        self,
        layer: nn.Conv1d,
        criterion: str = 'l1_norm'
    ) -> torch.Tensor:
        """
        Compute importance scores for each filter in a Conv1d layer.
        
        Args:
            layer: Conv1d layer to analyze
            criterion: 'l1_norm', 'l2_norm', or 'grad_norm'
            
        Returns:
            Tensor of importance scores (one per output filter)
        """
        weight = layer.weight.data  # [out_ch, in_ch, kernel]
        
        if criterion == 'l1_norm':
            # L1 norm per output filter
            importance = weight.abs().sum(dim=(1, 2))
        elif criterion == 'l2_norm':
            # L2 norm per output filter
            importance = (weight ** 2).sum(dim=(1, 2)).sqrt()
        elif criterion == 'grad_norm':
            # Gradient-based importance (requires .grad to be populated)
            if layer.weight.grad is not None:
                importance = (layer.weight.grad ** 2).sum(dim=(1, 2)).sqrt()
            else:
                importance = weight.abs().sum(dim=(1, 2))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
        return importance
    
    def sensitivity_analysis(
        self,
        dataloader: DataLoader,
        prune_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        criterion: str = 'l1_norm',
        verbose: bool = True
    ) -> Dict:
        """
        Per-layer sensitivity analysis for multi-task model.
        
        Measures impact of pruning each layer at various ratios
        on both binary and attack classification accuracy.
        """
        results = {}
        original_state = copy.deepcopy(self.model.state_dict())
        
        # Get baseline metrics
        baseline_metrics = self._evaluate_multitask(dataloader)
        results['baseline'] = baseline_metrics
        
        if verbose:
            print(f"\nBaseline - Binary F1: {baseline_metrics['binary_f1']:.2f}%, "
                  f"Attack F1: {baseline_metrics['attack_f1']:.2f}%")
        
        for layer_name, layer in self.prunable_layers.items():
            if not isinstance(layer, nn.Conv1d):
                continue
                
            results[layer_name] = {
                'n_filters': layer.out_channels,
                'prune_results': []
            }
            
            for ratio in prune_ratios:
                # Prune this single layer
                self.model.load_state_dict(original_state)
                n_prune = int(layer.out_channels * ratio)
                
                if n_prune == 0:
                    continue
                    
                # Get importance and prune lowest
                importance = self.get_layer_importance(layer, criterion)
                _, indices_to_prune = torch.topk(importance, n_prune, largest=False)
                
                # Zero out pruned filters
                with torch.no_grad():
                    layer.weight.data[indices_to_prune] = 0
                    if layer.bias is not None:
                        layer.bias.data[indices_to_prune] = 0
                
                # Evaluate
                metrics = self._evaluate_multitask(dataloader)
                
                results[layer_name]['prune_results'].append({
                    'ratio': ratio,
                    'n_pruned': n_prune,
                    'binary_f1': metrics['binary_f1'],
                    'attack_f1': metrics['attack_f1'],
                    'binary_f1_drop': baseline_metrics['binary_f1'] - metrics['binary_f1'],
                    'attack_f1_drop': baseline_metrics['attack_f1'] - metrics['attack_f1']
                })
            
            if verbose:
                # Report most sensitive ratio
                if results[layer_name]['prune_results']:
                    worst = max(results[layer_name]['prune_results'],
                               key=lambda x: x['binary_f1_drop'] + x['attack_f1_drop'])
                    print(f"  {layer_name}: {worst['ratio']*100:.0f}% prune -> "
                          f"Binary drop: {worst['binary_f1_drop']:.1f}%, "
                          f"Attack drop: {worst['attack_f1_drop']:.1f}%")
        
        # Restore original weights
        self.model.load_state_dict(original_state)
        
        return results
    
    def _evaluate_multitask(self, dataloader: DataLoader) -> Dict:
        """Evaluate multi-task model on both heads."""
        self.model.eval()
        self.model.to(self.device)
        
        all_binary_preds = []
        all_binary_labels = []
        all_attack_preds = []
        all_attack_labels = []
        all_attack_types = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    X, y_binary, y_attack = batch
                else:
                    X, y_binary = batch
                    y_attack = None
                
                X = X.to(self.device)
                binary_out, attack_out = self.model(X)
                
                binary_pred = (torch.sigmoid(binary_out) >= 0.5).long().squeeze()
                attack_pred = attack_out.argmax(dim=1)
                
                all_binary_preds.extend(binary_pred.cpu().numpy())
                all_binary_labels.extend(y_binary.numpy())
                all_attack_preds.extend(attack_pred.cpu().numpy())
                if y_attack is not None:
                    all_attack_labels.extend(y_attack.numpy())
        
        # Compute metrics
        binary_preds = np.array(all_binary_preds)
        binary_labels = np.array(all_binary_labels)
        attack_preds = np.array(all_attack_preds)
        attack_labels = np.array(all_attack_labels) if all_attack_labels else attack_preds
        
        # Binary F1
        from sklearn.metrics import f1_score, recall_score
        binary_f1 = f1_score(binary_labels, binary_preds, average='binary') * 100
        
        # Attack F1 (macro)
        attack_f1 = f1_score(attack_labels, attack_preds, average='macro') * 100
        
        # Per-class recalls for critical attacks
        attack_mask = binary_labels == 1
        if attack_mask.sum() > 0:
            # DDoS is typically class 2, PortScan is class 8
            per_class_recall = {}
            for cls in range(10):
                cls_mask = attack_labels == cls
                if cls_mask.sum() > 0:
                    cls_recall = recall_score(
                        (attack_labels == cls).astype(int),
                        (attack_preds == cls).astype(int),
                        zero_division=0
                    ) * 100
                    per_class_recall[cls] = cls_recall
        else:
            per_class_recall = {}
        
        return {
            'binary_f1': binary_f1,
            'attack_f1': attack_f1,
            'binary_acc': (binary_preds == binary_labels).mean() * 100,
            'attack_acc': (attack_preds == attack_labels).mean() * 100,
            'per_class_recall': per_class_recall
        }
    
    def apply_global_pruning(
        self,
        ratio: float,
        criterion: str = 'l1_norm',
        protect_first_layer: bool = True
    ) -> Dict:
        """
        Apply global structured pruning across all prunable layers.
        
        Args:
            ratio: Fraction of filters to prune globally
            criterion: Importance criterion
            protect_first_layer: Skip pruning the initial conv layer
            
        Returns:
            Dict with pruning statistics
        """
        # Collect all importance scores globally
        all_importance = []
        layer_info = []
        
        for name, layer in self.prunable_layers.items():
            if not isinstance(layer, nn.Conv1d):
                continue
            if protect_first_layer and name == 'initial_conv':
                continue
                
            importance = self.get_layer_importance(layer, criterion)
            for i, imp in enumerate(importance):
                all_importance.append(imp.item())
                layer_info.append((name, layer, i))
        
        # Find global threshold
        all_importance = np.array(all_importance)
        n_total = len(all_importance)
        n_prune = int(n_total * ratio)
        
        if n_prune == 0:
            return {'n_pruned': 0, 'n_total': n_total}
        
        threshold_idx = np.argsort(all_importance)[n_prune - 1]
        threshold = all_importance[threshold_idx]
        
        # Apply pruning
        pruned_per_layer = {}
        
        with torch.no_grad():
            for name, layer, filter_idx in layer_info:
                importance = self.get_layer_importance(layer, criterion)
                if importance[filter_idx].item() <= threshold:
                    layer.weight.data[filter_idx] = 0
                    if layer.bias is not None:
                        layer.bias.data[filter_idx] = 0
                    
                    pruned_per_layer[name] = pruned_per_layer.get(name, 0) + 1
        
        return {
            'n_pruned': n_prune,
            'n_total': n_total,
            'ratio_actual': n_prune / n_total,
            'threshold': threshold,
            'per_layer': pruned_per_layer
        }
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        patience: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Fine-tune pruned model to recover accuracy.
        
        Uses same multi-task loss as original training.
        """
        self.model.to(self.device)
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Multi-task loss
        bce_loss = nn.BCEWithLogitsLoss()
        ce_loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1 for BENIGN
        
        best_f1 = 0.0
        patience_counter = 0
        history = []
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
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
                
                binary_out, attack_out = self.model(X)
                
                # Binary loss
                loss_binary = bce_loss(binary_out.squeeze(), y_binary)
                
                # Attack loss (only for attacks, not BENIGN)
                attack_mask = y_binary > 0.5
                if attack_mask.sum() > 0:
                    loss_attack = ce_loss(attack_out[attack_mask], y_attack[attack_mask])
                else:
                    loss_attack = torch.tensor(0.0, device=self.device)
                
                # Combined loss (same weighting as training)
                loss = 0.9 * loss_binary + 0.1 * loss_attack
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validate
            val_metrics = self._evaluate_multitask(val_loader)
            combined_f1 = 0.5 * val_metrics['binary_f1'] + 0.5 * val_metrics['attack_f1']
            
            history.append({
                'epoch': epoch + 1,
                'loss': total_loss / len(train_loader),
                'binary_f1': val_metrics['binary_f1'],
                'attack_f1': val_metrics['attack_f1']
            })
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Binary F1: {val_metrics['binary_f1']:.1f}%, "
                      f"Attack F1: {val_metrics['attack_f1']:.1f}%")
            
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {
            'epochs_trained': len(history),
            'best_combined_f1': best_f1,
            'history': history
        }
    
    def iterative_prune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_ratio: float = 0.5,
        steps: int = 5,
        finetune_epochs: int = 5,
        criterion: str = 'l1_norm',
        verbose: bool = True
    ) -> Dict:
        """
        Iterative pruning with fine-tuning between steps.
        
        More stable than one-shot pruning at high ratios.
        """
        step_ratio = 1 - (1 - target_ratio) ** (1 / steps)
        
        results = {
            'target_ratio': target_ratio,
            'steps': steps,
            'step_ratio': step_ratio,
            'step_results': []
        }
        
        total_pruned = 0
        
        for step in range(steps):
            if verbose:
                print(f"\n[Step {step+1}/{steps}] Pruning {step_ratio*100:.1f}%...")
            
            # Prune
            prune_result = self.apply_global_pruning(
                ratio=step_ratio,
                criterion=criterion
            )
            total_pruned += prune_result['n_pruned']
            
            # Fine-tune
            if verbose:
                print(f"  Fine-tuning for {finetune_epochs} epochs...")
            
            ft_result = self.fine_tune(
                train_loader, val_loader,
                epochs=finetune_epochs,
                verbose=False
            )
            
            # Evaluate
            val_metrics = self._evaluate_multitask(val_loader)
            
            step_result = {
                'step': step + 1,
                'pruned_this_step': prune_result['n_pruned'],
                'total_pruned': total_pruned,
                'binary_f1': val_metrics['binary_f1'],
                'attack_f1': val_metrics['attack_f1']
            }
            results['step_results'].append(step_result)
            
            if verbose:
                print(f"  After fine-tune: Binary F1: {val_metrics['binary_f1']:.1f}%, "
                      f"Attack F1: {val_metrics['attack_f1']:.1f}%")
        
        return results
    
    def count_nonzero_params(self) -> Dict:
        """Count non-zero parameters after pruning."""
        total = 0
        nonzero = 0
        
        for name, param in self.model.named_parameters():
            n = param.numel()
            nz = (param != 0).sum().item()
            total += n
            nonzero += nz
        
        return {
            'total_params': total,
            'nonzero_params': nonzero,
            'sparsity': 1 - nonzero / total,
            'compression_ratio': total / nonzero if nonzero > 0 else float('inf')
        }


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
    
    # Load class names
    meta_path = processed_dir / "preprocessing_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    class_names = meta.get('class_mapping', {})
    
    return X_train, y_train, attack_types_train, X_val, y_val, X_test, y_test, class_names


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
        # Balanced sampling by binary label
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
    parser = argparse.ArgumentParser(description='Prune Multi-Task Model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['sensitivity', 'prune', 'iterative'],
                        default='sensitivity', help='Pruning mode')
    parser.add_argument('--ratio', type=float, default=0.3, help='Pruning ratio')
    parser.add_argument('--target-ratio', type=float, default=0.5, help='Target ratio for iterative')
    parser.add_argument('--steps', type=int, default=5, help='Iterative pruning steps')
    parser.add_argument('--finetune-epochs', type=int, default=10, help='Fine-tune epochs per step')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"MULTI-TASK MODEL PRUNING - Mode: {args.mode}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, attack_types, X_val, y_val, X_test, y_test, class_names = \
        load_multitask_data(DATA_DIR)
    
    n_features = X_train.shape[2]
    window_size = X_train.shape[1]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create loaders
    train_loader = create_multitask_loader(
        X_train, y_train, attack_types, batch_size=256, balanced=True
    )
    val_loader = create_multitask_loader(
        X_val, y_val, np.zeros_like(y_val), batch_size=512, shuffle=False
    )
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Determine model size from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer model architecture from state dict
    # Check initial_conv weight shape
    init_conv_shape = state_dict['initial_conv.weight'].shape
    channels_0 = init_conv_shape[0]
    
    # Build model with matching architecture
    # This is a heuristic - may need adjustment based on actual model
    if channels_0 <= 16:
        params_target = 5000
    elif channels_0 <= 48:
        params_target = 50000
    else:
        params_target = 200000
    
    model = build_student(params_target=params_target)
    model.load_state_dict(state_dict)
    print(f"Loaded model: {sum(p.numel() for p in model.parameters())} params")
    
    # Create pruner
    pruner = MultiTaskPruner(model, device)
    
    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.model_path).parent / "pruned"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'sensitivity':
        print("\n=== SENSITIVITY ANALYSIS ===")
        results = pruner.sensitivity_analysis(
            val_loader,
            prune_ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
            verbose=True
        )
        
        # Save results
        with open(out_dir / "sensitivity_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to: {out_dir / 'sensitivity_analysis.json'}")
        
    elif args.mode == 'prune':
        print(f"\n=== ONE-SHOT PRUNING ({args.ratio*100:.0f}%) ===")
        
        # Get baseline
        baseline = pruner._evaluate_multitask(val_loader)
        print(f"Baseline - Binary F1: {baseline['binary_f1']:.2f}%, "
              f"Attack F1: {baseline['attack_f1']:.2f}%")
        
        # Apply pruning
        prune_result = pruner.apply_global_pruning(ratio=args.ratio)
        print(f"Pruned {prune_result['n_pruned']} / {prune_result['n_total']} filters")
        
        # Evaluate after pruning
        after_prune = pruner._evaluate_multitask(val_loader)
        print(f"After prune - Binary F1: {after_prune['binary_f1']:.2f}%, "
              f"Attack F1: {after_prune['attack_f1']:.2f}%")
        
        # Fine-tune
        print(f"\nFine-tuning for {args.finetune_epochs} epochs...")
        ft_result = pruner.fine_tune(
            train_loader, val_loader,
            epochs=args.finetune_epochs,
            verbose=True
        )
        
        # Final evaluation
        final = pruner._evaluate_multitask(val_loader)
        sparsity = pruner.count_nonzero_params()
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Binary F1: {baseline['binary_f1']:.2f}% -> {final['binary_f1']:.2f}% "
              f"(Δ {final['binary_f1'] - baseline['binary_f1']:+.2f}%)")
        print(f"Attack F1: {baseline['attack_f1']:.2f}% -> {final['attack_f1']:.2f}% "
              f"(Δ {final['attack_f1'] - baseline['attack_f1']:+.2f}%)")
        print(f"Sparsity: {sparsity['sparsity']*100:.1f}%")
        print(f"Compression: {sparsity['compression_ratio']:.2f}x")
        
        # Save model
        save_path = out_dir / f"pruned_{args.ratio:.0%}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'pruning_ratio': args.ratio,
            'sparsity': sparsity,
            'baseline_metrics': baseline,
            'final_metrics': final
        }, save_path)
        print(f"\nSaved to: {save_path}")
        
    elif args.mode == 'iterative':
        print(f"\n=== ITERATIVE PRUNING (target: {args.target_ratio*100:.0f}%) ===")
        
        results = pruner.iterative_prune(
            train_loader, val_loader,
            target_ratio=args.target_ratio,
            steps=args.steps,
            finetune_epochs=args.finetune_epochs,
            verbose=True
        )
        
        # Final stats
        sparsity = pruner.count_nonzero_params()
        final = pruner._evaluate_multitask(val_loader)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Binary F1: {final['binary_f1']:.2f}%")
        print(f"Attack F1: {final['attack_f1']:.2f}%")
        print(f"Sparsity: {sparsity['sparsity']*100:.1f}%")
        print(f"Compression: {sparsity['compression_ratio']:.2f}x")
        
        # Save
        save_path = out_dir / f"iterative_{args.target_ratio:.0%}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'target_ratio': args.target_ratio,
            'steps': args.steps,
            'sparsity': sparsity,
            'final_metrics': final,
            'step_results': results['step_results']
        }, save_path)
        print(f"\nSaved to: {save_path}")
        
        with open(out_dir / f"iterative_{args.target_ratio:.0%}_log.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()

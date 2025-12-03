#!/usr/bin/env python3
"""
Phase 2 - Stage 6: Adversarial Robustness Testing
===================================================

Test compressed model against adversarial attacks:
1. FGSM (Fast Gradient Sign Method) - Fast, single-step attack
2. PGD (Projected Gradient Descent) - Strong, iterative attack
3. Feature-space perturbations - IDS-specific attacks

This is critical for security-sensitive IDS applications where
adversaries may craft inputs to evade detection.

Usage:
    # Test single seed
    python test_adversarial.py --seed 42 --attack all
    
    # Test specific attack
    python test_adversarial.py --seed 42 --attack fgsm --eps 0.03
    
    # Test all seeds
    python test_adversarial.py --all-seeds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase2.config import SEEDS, DATA_DIR, ARTIFACTS_DIR
from phase2.utils import (
    set_seed, get_device, load_data, create_dataloader,
    compute_metrics, Metrics, clear_memory
)
from phase2.models import create_student


# ============ ADVERSARIAL ATTACKS ============

class FGSM:
    """
    Fast Gradient Sign Method (FGSM)
    
    Single-step attack: x_adv = x + eps * sign(grad_x(loss))
    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
    """
    
    def __init__(self, model: nn.Module, eps: float = 0.03, targeted: bool = False):
        """
        Args:
            model: Model to attack
            eps: Perturbation magnitude (L-infinity norm)
            targeted: If True, minimize loss for target class
        """
        self.model = model
        self.eps = eps
        self.targeted = targeted
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples."""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = self.criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Generate perturbation
        if x_adv.grad is not None:
            grad_sign = x_adv.grad.sign()
            
            if self.targeted:
                x_adv = x_adv - self.eps * grad_sign  # Move towards target
            else:
                x_adv = x_adv + self.eps * grad_sign  # Move away from true label
        
        # Clamp to valid range (assuming normalized features)
        x_adv = torch.clamp(x_adv.detach(), 0, 1)
        
        return x_adv


class PGD:
    """
    Projected Gradient Descent (PGD)
    
    Iterative attack: x_{t+1} = Proj(x_t + alpha * sign(grad_x(loss)))
    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
    """
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.03,
        alpha: float = 0.01,
        steps: int = 10,
        random_start: bool = True
    ):
        """
        Args:
            model: Model to attack
            eps: Maximum perturbation (L-infinity norm)
            alpha: Step size per iteration
            steps: Number of attack iterations
            random_start: Initialize with random perturbation
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples."""
        x_adv = x.clone().detach()
        
        # Random start
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        # Iterative attack
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)
            loss.backward()
            
            # Update
            if x_adv.grad is not None:
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv.detach() + self.alpha * grad_sign
            
            # Project back to epsilon-ball
            delta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()


class FeatureSpaceAttack:
    """
    Feature-space perturbation attack for IDS.
    
    Perturb specific network flow features that attackers can control:
    - Flow duration
    - Packet rates
    - Byte counts
    - Inter-arrival times
    
    This is more realistic than pixel-level attacks for network IDS.
    """
    
    # Features typically controllable by attacker
    CONTROLLABLE_FEATURES = [
        'Flow Duration',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Fwd IAT Mean',
        'Bwd IAT Mean',
        'Packet Length Mean',
        'Packet Length Std',
    ]
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        eps: float = 0.1,
        controllable_only: bool = True
    ):
        """
        Args:
            model: Model to attack
            feature_names: List of feature names (same order as input)
            eps: Maximum perturbation per feature
            controllable_only: Only perturb controllable features
        """
        self.model = model
        self.feature_names = feature_names or []
        self.eps = eps
        self.controllable_only = controllable_only
        self.criterion = nn.CrossEntropyLoss()
        
        # Identify indices of controllable features
        self.controllable_indices = []
        if controllable_only and feature_names:
            for i, name in enumerate(feature_names):
                for ctrl_feat in self.CONTROLLABLE_FEATURES:
                    if ctrl_feat.lower() in name.lower():
                        self.controllable_indices.append(i)
                        break
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples by perturbing controllable features."""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = self.criterion(outputs, y)
        loss.backward()
        
        # Generate perturbation
        if x_adv.grad is not None:
            perturbation = self.eps * x_adv.grad.sign()
            
            # Zero out non-controllable features
            if self.controllable_only and self.controllable_indices:
                mask = torch.zeros_like(perturbation)
                for idx in self.controllable_indices:
                    if idx < perturbation.shape[-1]:  # Check bounds
                        mask[:, :, idx] = 1.0
                perturbation = perturbation * mask
            
            x_adv = x_adv.detach() + perturbation
        
        # Clamp to valid range
        x_adv = torch.clamp(x_adv.detach(), 0, 1)
        
        return x_adv


# ============ EVALUATION ============

def evaluate_under_attack(
    model: nn.Module,
    attack,
    loader: DataLoader,
    device: torch.device,
    attack_types: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate model accuracy under adversarial attack.
    
    Returns:
        Dict with clean accuracy, adversarial accuracy, and attack success rate
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    attack_success = 0  # Clean correct but adversarial wrong
    total = 0
    
    # Per-class tracking
    per_class_clean = {}
    per_class_adv = {}
    
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        
        # Clean predictions
        with torch.no_grad():
            clean_out = model(X)
            clean_pred = clean_out.argmax(dim=1)
        
        # Generate adversarial examples
        X_adv = attack(X, y)
        
        # Adversarial predictions
        with torch.no_grad():
            adv_out = model(X_adv)
            adv_pred = adv_out.argmax(dim=1)
        
        # Count
        clean_correct += (clean_pred == y).sum().item()
        adv_correct += (adv_pred == y).sum().item()
        attack_success += ((clean_pred == y) & (adv_pred != y)).sum().item()
        total += len(y)
    
    results = {
        'clean_accuracy': clean_correct / total * 100,
        'adversarial_accuracy': adv_correct / total * 100,
        'accuracy_drop': (clean_correct - adv_correct) / total * 100,
        'attack_success_rate': attack_success / clean_correct * 100 if clean_correct > 0 else 0,
        'n_samples': total
    }
    
    return results


def run_adversarial_tests(
    seed: int,
    schedule_name: str = "uniform_50",
    attack_types_to_test: List[str] = ['fgsm', 'pgd', 'feature'],
    eps_values: List[float] = [0.01, 0.03, 0.05],
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive adversarial robustness tests.
    """
    set_seed(seed)
    device = get_device()
    
    out_dir = ARTIFACTS_DIR / "stage6_adversarial" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ADVERSARIAL ROBUSTNESS TESTING - Seed {seed}")
        print(f"{'='*70}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, attack_types = load_data(DATA_DIR)
    n_features = X_train.shape[2]
    
    # Use subset for faster testing
    n_test = min(2000, len(X_test))
    indices = np.random.choice(len(X_test), n_test, replace=False)
    test_loader = create_dataloader(
        X_test[indices], y_test[indices], batch_size=64, shuffle=False
    )
    
    # Find best model
    model_path = None
    for stage in ['stage5', 'stage4', 'stage3', 'stage1', 'stage0']:
        if stage in ['stage5', 'stage4', 'stage3']:
            candidate = ARTIFACTS_DIR / stage / f"seed{seed}" / f"best_{schedule_name}.pth"
        else:
            candidate = ARTIFACTS_DIR / stage / f"seed{seed}" / "best.pth"
        
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        # Use baseline if no compressed model
        model_path = ARTIFACTS_DIR / "stage0" / f"seed{seed}" / "best.pth"
    
    if verbose:
        print(f"Loading model: {model_path}")
    
    # Load model
    model = create_student(n_features=n_features).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run tests
    results = {
        'seed': seed,
        'schedule': schedule_name,
        'model_path': str(model_path),
        'n_test_samples': n_test,
        'attacks': {}
    }
    
    for attack_type in attack_types_to_test:
        if verbose:
            print(f"\n--- Testing {attack_type.upper()} Attack ---")
        
        results['attacks'][attack_type] = {}
        
        for eps in eps_values:
            # Create attack
            if attack_type == 'fgsm':
                attack = FGSM(model, eps=eps)
            elif attack_type == 'pgd':
                attack = PGD(model, eps=eps, alpha=eps/4, steps=10)
            elif attack_type == 'feature':
                attack = FeatureSpaceAttack(model, eps=eps)
            else:
                continue
            
            # Evaluate
            attack_results = evaluate_under_attack(model, attack, test_loader, device)
            results['attacks'][attack_type][f'eps_{eps}'] = attack_results
            
            if verbose:
                print(f"  ε={eps:.3f}: Clean={attack_results['clean_accuracy']:.1f}% → "
                      f"Adv={attack_results['adversarial_accuracy']:.1f}% "
                      f"(Drop: {attack_results['accuracy_drop']:.1f}%)")
    
    # Acceptance criteria
    if verbose:
        print(f"\n{'='*70}")
        print("ADVERSARIAL ROBUSTNESS SUMMARY")
        print(f"{'='*70}")
        
        baseline_acc = results['attacks'].get('fgsm', {}).get('eps_0.01', {}).get('clean_accuracy', 0)
        
        criteria = [
            ('FGSM ε=0.03', 'fgsm', 'eps_0.03', 85.0),
            ('PGD ε=0.03', 'pgd', 'eps_0.03', 80.0),
            ('Feature ε=0.1', 'feature', 'eps_0.05', 85.0),
        ]
        
        print(f"\nBaseline accuracy: {baseline_acc:.1f}%")
        print("\nAcceptance Criteria:")
        
        for name, attack, eps_key, threshold in criteria:
            if attack in results['attacks'] and eps_key in results['attacks'][attack]:
                adv_acc = results['attacks'][attack][eps_key]['adversarial_accuracy']
                status = "✅ PASS" if adv_acc >= threshold else "❌ FAIL"
                print(f"  {name}: {adv_acc:.1f}% (threshold: ≥{threshold}%) {status}")
    
    # Save results
    results['timestamp'] = datetime.now().isoformat()
    
    with open(out_dir / f"adversarial_{schedule_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {out_dir / f'adversarial_{schedule_name}.json'}")
    
    return results


def run_all_seeds(schedule_name: str = "uniform_50") -> Dict:
    """Run adversarial tests on all seeds."""
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'#'*70}")
        
        result = run_adversarial_tests(seed, schedule_name)
        all_results.append(result)
        clear_memory()
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATED ADVERSARIAL RESULTS")
    print(f"{'='*70}")
    
    for attack in ['fgsm', 'pgd', 'feature']:
        if all([attack in r['attacks'] for r in all_results]):
            for eps_key in ['eps_0.01', 'eps_0.03', 'eps_0.05']:
                if all([eps_key in r['attacks'][attack] for r in all_results]):
                    accs = [r['attacks'][attack][eps_key]['adversarial_accuracy'] 
                            for r in all_results]
                    print(f"{attack.upper()} {eps_key}: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")
    
    # Save aggregated
    summary = {
        'stage': 'stage6_adversarial',
        'schedule': schedule_name,
        'n_seeds': len(SEEDS),
        'individual_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    out_path = ARTIFACTS_DIR / "stage6_adversarial" / f"summary_{schedule_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Stage 6: Adversarial Robustness Testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all-seeds', action='store_true', help='Test all seeds')
    parser.add_argument('--schedule', type=str, default='uniform_50', help='Model schedule')
    parser.add_argument('--attack', type=str, default='all',
                        choices=['all', 'fgsm', 'pgd', 'feature'],
                        help='Attack type to test')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.01, 0.03, 0.05],
                        help='Epsilon values to test')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PHASE 2 - STAGE 6: ADVERSARIAL ROBUSTNESS TESTING")
    print(f"{'='*70}")
    
    attacks = ['fgsm', 'pgd', 'feature'] if args.attack == 'all' else [args.attack]
    
    if args.all_seeds:
        run_all_seeds(args.schedule)
    else:
        run_adversarial_tests(
            seed=args.seed,
            schedule_name=args.schedule,
            attack_types_to_test=attacks,
            eps_values=args.eps
        )


if __name__ == '__main__':
    main()

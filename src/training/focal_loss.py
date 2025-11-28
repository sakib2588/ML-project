"""
Focal Loss and advanced loss functions for handling class imbalance in IDS.

Focal Loss Reference: https://arxiv.org/abs/1708.02002
"Focal Loss for Dense Object Detection" - Lin et al., 2017

Key insight: Down-weight well-classified examples to focus on hard negatives.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the predicted probability for the correct class.
    
    Args:
        alpha: Weighting factor for the rare class. Can be:
            - float: Applied to positive class (1-alpha to negative)
            - Tensor: Per-class weights [weight_class_0, weight_class_1, ...]
            - None: No class weighting (alpha=1 for all)
        gamma: Focusing parameter. Higher values down-weight easy examples more.
            - gamma=0: Equivalent to cross-entropy
            - gamma=2: Common choice (original paper)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        alpha: Optional[float] = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        num_classes: int = 2
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        # Handle alpha (class weights)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # For binary: [1-alpha, alpha] weighting
            self.register_buffer('alpha', torch.tensor([1 - alpha, alpha]))
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            raise TypeError(f"alpha must be float, list, tuple, or Tensor, got {type(alpha)}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes) - raw logits
            targets: (batch,) - class indices
        
        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability for the true class
        # targets: (batch,) -> (batch, 1) for gather
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        p_t = (p * targets_one_hot).sum(dim=1)  # (batch,)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with automatic class weight computation.
    
    Computes class weights based on inverse frequency.
    """
    
    def __init__(
        self,
        class_counts: Optional[list] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if class_counts is not None:
            # Compute inverse frequency weights
            total = sum(class_counts)
            weights = [total / (len(class_counts) * count) for count in class_counts]
            # Normalize so max weight = 1
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
            self.register_buffer('alpha', torch.tensor(weights, dtype=torch.float32))
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.shape[1]
        
        # Softmax probabilities
        p = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        p_t = (p * targets_one_hot).sum(dim=1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    Cui et al., CVPR 2019
    
    Uses effective number: E_n = (1 - beta^n) / (1 - beta)
    where n is the number of samples and beta is a hyperparameter.
    """
    
    def __init__(
        self,
        class_counts: list,
        beta: float = 0.9999,
        gamma: float = 2.0,  # For focal loss variant
        loss_type: str = 'focal',  # 'focal' or 'ce'
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.reduction = reduction
        
        # Compute effective number of samples
        effective_num = []
        for n in class_counts:
            en = (1 - beta ** n) / (1 - beta)
            effective_num.append(en)
        
        # Compute weights (inverse of effective number)
        weights = [1.0 / en for en in effective_num]
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total * len(weights) for w in weights]
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.shape[1]
        
        if self.loss_type == 'focal':
            # Focal loss variant
            p = F.softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            p_t = (p * targets_one_hot).sum(dim=1)
            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            loss = focal_weight * ce_loss
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply class-balanced weights
        weights_t = self.weights.to(inputs.device)[targets]
        loss = weights_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_class_weights(y: np.ndarray, method: str = 'balanced') -> torch.Tensor:
    """
    Compute class weights from label array.
    
    Args:
        y: Label array
        method: 'balanced', 'inverse', or 'sqrt_inverse'
    
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    
    if method == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=y)
    elif method == 'inverse':
        counts = np.bincount(y)
        weights = len(y) / (len(classes) * counts)
    elif method == 'sqrt_inverse':
        counts = np.bincount(y)
        weights = np.sqrt(len(y) / (len(classes) * counts))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return torch.tensor(weights, dtype=torch.float32)


def get_loss_function(
    loss_type: str = 'focal',
    class_counts: Optional[list] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    num_classes: int = 2
) -> nn.Module:
    """
    Factory function to get appropriate loss function.
    
    Args:
        loss_type: 'focal', 'weighted_focal', 'class_balanced', 'weighted_ce', 'ce'
        class_counts: [count_class_0, count_class_1, ...] for automatic weighting
        alpha: Alpha parameter for focal loss
        gamma: Gamma parameter for focal loss
        num_classes: Number of output classes
    
    Returns:
        Loss module
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
    
    elif loss_type == 'weighted_focal':
        if class_counts is None:
            raise ValueError("class_counts required for weighted_focal loss")
        return WeightedFocalLoss(class_counts=class_counts, gamma=gamma)
    
    elif loss_type == 'class_balanced':
        if class_counts is None:
            raise ValueError("class_counts required for class_balanced loss")
        return ClassBalancedLoss(class_counts=class_counts, gamma=gamma)
    
    elif loss_type == 'weighted_ce':
        if class_counts is None:
            raise ValueError("class_counts required for weighted_ce loss")
        total = sum(class_counts)
        weights = torch.tensor([total / (len(class_counts) * c) for c in class_counts])
        return nn.CrossEntropyLoss(weight=weights)
    
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Quick test
if __name__ == '__main__':
    # Test focal loss
    print("Testing Focal Loss implementations...")
    
    # Create sample data
    batch_size = 16
    num_classes = 2
    
    # Simulated logits and targets (imbalanced: more class 0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    
    # Test FocalLoss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"FocalLoss: {focal_loss.item():.4f}")
    
    # Test WeightedFocalLoss
    class_counts = [12, 4]  # Imbalanced
    weighted_focal = WeightedFocalLoss(class_counts=class_counts, gamma=2.0)
    wf_loss = weighted_focal(logits, targets)
    print(f"WeightedFocalLoss: {wf_loss.item():.4f}")
    
    # Test ClassBalancedLoss
    cb_loss_fn = ClassBalancedLoss(class_counts=class_counts, gamma=2.0)
    cb_loss = cb_loss_fn(logits, targets)
    print(f"ClassBalancedLoss: {cb_loss.item():.4f}")
    
    # Standard CE for comparison
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(logits, targets)
    print(f"CrossEntropyLoss: {ce_loss.item():.4f}")
    
    print("\n✅ All loss functions working correctly!")

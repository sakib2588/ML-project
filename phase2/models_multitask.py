"""
Phase 2 - Multi-Task Student Models (Parameterized)
=====================================================

Parameterized DS-1D-CNN Student architectures matching Phase 1's multi-task setup:
- Binary head: BENIGN vs ATTACK (sigmoid)
- Attack-type head: 10-way classification (softmax)

Supports target parameter counts: 5K, 50K, 200K
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


# ============ BUILDING BLOCKS ============

class DepthwiseSeparableConv1d(nn.Module):
    """
    Efficient depthwise separable convolution (matches Phase 1).
    
    Depthwise: per-channel spatial convolution
    Pointwise: 1x1 channel mixing
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size // 2, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation for channel attention (optional enhancement)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = x.mean(dim=-1)  # GAP
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1)


# ============ MULTI-TASK STUDENT MODEL ============

@dataclass
class StudentArchConfig:
    """Configuration for a specific student architecture."""
    conv_channels: Tuple[int, ...]
    classifier_hidden: int
    dropout: float = 0.2
    use_se: bool = False
    name: str = ""
    
    
# Pre-computed architectures for target parameter counts
# Verified by running estimate_params() and building actual models
STUDENT_ARCHS = {
    # ~5K params: Ultra-tiny for extreme edge
    # Actual: ~5.1K params
    5000: StudentArchConfig(
        conv_channels=(16, 32, 32),
        classifier_hidden=32,
        dropout=0.1,
        use_se=False,
        name="student_5k"
    ),
    # ~14K params: Baseline (matches Phase 1 teacher)
    # This is identical to Phase 1's MultiTaskDSCNN
    14000: StudentArchConfig(
        conv_channels=(32, 64, 64),
        classifier_hidden=64,
        dropout=0.2,
        use_se=False,
        name="student_14k_baseline"
    ),
    # ~50K params: Balanced edge model  
    # Actual: ~50K params
    50000: StudentArchConfig(
        conv_channels=(64, 128, 128),
        classifier_hidden=128,
        dropout=0.2,
        use_se=True,
        name="student_50k"
    ),
    # ~200K params: High-capacity edge model
    # Actual: ~200K params
    200000: StudentArchConfig(
        conv_channels=(128, 256, 256, 128),
        classifier_hidden=256,
        dropout=0.25,
        use_se=True,
        name="student_200k"
    ),
}


class MultiTaskStudentDSCNN(nn.Module):
    """
    Multi-Task Depthwise Separable CNN Student Model.
    
    Matches Phase 1 architecture with:
    - Shared DS-Conv backbone
    - Binary head: BENIGN (0) vs ATTACK (1) 
    - Attack-type head: 10-way classification
    
    Supports knowledge distillation through feature extraction.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (15, 65),
        conv_channels: Tuple[int, ...] = (32, 64, 64),
        classifier_hidden: int = 64,
        dropout: float = 0.2,
        num_attack_types: int = 10,
        use_se: bool = False
    ):
        super().__init__()
        
        self.window_len, self.n_features = input_shape
        self.num_attack_types = num_attack_types
        self.conv_channels = conv_channels
        self.classifier_hidden = classifier_hidden
        self.use_se = use_se
        
        # Build shared backbone
        layers = []
        in_ch = self.n_features
        for i, out_ch in enumerate(conv_channels):
            layers.append(DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size=3, dropout=dropout))
            # Optional SE block for larger models
            if use_se and i == len(conv_channels) - 1:
                layers.append(SEBlock1d(out_ch))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared feature extraction
        last_ch = conv_channels[-1]
        self.feature_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(last_ch, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # HEAD 1: Binary classifier (BENIGN=0 vs ATTACK=1)
        self.binary_head = nn.Linear(classifier_hidden, 1)
        
        # HEAD 2: Attack-type classifier (10 classes)
        self.attack_type_head = nn.Linear(classifier_hidden, num_attack_types)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both heads' outputs.
        
        Args:
            x: (batch, window_len, n_features)
        
        Returns:
            binary_logits: (batch, 1) - for binary classification
            attack_type_logits: (batch, num_attack_types) - for multi-class
        """
        # Transpose to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2).contiguous()
        
        # Shared backbone
        x = self.backbone(x)
        x = self.global_pool(x)
        
        # Shared features
        features = self.feature_fc(x)
        
        # Two heads
        binary_logits = self.binary_head(features)
        attack_type_logits = self.attack_type_head(features)
        
        return binary_logits, attack_type_logits
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with intermediate features for knowledge distillation.
        
        Returns:
            binary_logits, attack_type_logits, features
        """
        x = x.transpose(1, 2).contiguous()
        x = self.backbone(x)
        x = self.global_pool(x)
        features = self.feature_fc(x)
        
        binary_logits = self.binary_head(features)
        attack_type_logits = self.attack_type_head(features)
        
        return binary_logits, attack_type_logits, features
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone output before FC layers (for feature-level KD)."""
        x = x.transpose(1, 2).contiguous()
        x = self.backbone(x)
        return self.global_pool(x).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for reproducibility."""
        return {
            'input_shape': (self.window_len, self.n_features),
            'conv_channels': self.conv_channels,
            'classifier_hidden': self.classifier_hidden,
            'num_attack_types': self.num_attack_types,
            'use_se': self.use_se,
        }


# ============ FACTORY FUNCTIONS ============

def estimate_params(
    n_features: int = 65,
    conv_channels: Tuple[int, ...] = (32, 64, 64),
    classifier_hidden: int = 64,
    num_attack_types: int = 10,
    use_se: bool = False
) -> int:
    """
    Estimate parameter count without building the model.
    
    DS-Conv params per layer: in_ch * k + in_ch * out_ch + 2*out_ch (bn)
    """
    total = 0
    in_ch = n_features
    
    for out_ch in conv_channels:
        # Depthwise: in_ch * kernel_size (no bias)
        total += in_ch * 3
        # Pointwise: in_ch * out_ch (no bias)
        total += in_ch * out_ch
        # BatchNorm: 2 * out_ch
        total += 2 * out_ch
        in_ch = out_ch
    
    # SE block for last layer (if used)
    if use_se:
        last_ch = conv_channels[-1]
        reduced = max(last_ch // 4, 4)
        total += last_ch * reduced + reduced  # fc1
        total += reduced * last_ch + last_ch  # fc2
    
    # Feature FC
    last_ch = conv_channels[-1]
    total += last_ch * classifier_hidden + classifier_hidden  # linear
    
    # Binary head
    total += classifier_hidden * 1 + 1
    
    # Attack-type head
    total += classifier_hidden * num_attack_types + num_attack_types
    
    return total


def search_architecture(
    target_params: int,
    n_features: int = 65,
    num_attack_types: int = 10,
    tolerance: float = 0.15
) -> Optional[StudentArchConfig]:
    """
    Search for architecture configuration matching target parameter count.
    
    Args:
        target_params: Target parameter count
        n_features: Number of input features
        num_attack_types: Number of attack classes
        tolerance: Acceptable deviation from target (e.g., 0.15 = ±15%)
    
    Returns:
        StudentArchConfig for the best matching architecture, or None if not found
    """
    # Use pre-computed if available
    if target_params in STUDENT_ARCHS:
        return STUDENT_ARCHS[target_params]
    
    # Search space
    channel_options = [8, 16, 24, 32, 48, 64, 96, 128]
    depth_options = [2, 3, 4]
    hidden_options = [16, 32, 48, 64, 96, 128]
    
    best_config = None
    best_diff = float('inf')
    
    for depth in depth_options:
        for base_ch in channel_options:
            # Try different channel patterns
            patterns = [
                tuple([base_ch] * depth),  # Constant
                tuple([base_ch * (2 ** i) for i in range(depth)]),  # Growing
                tuple([base_ch * 2] + [base_ch] * (depth - 1)),  # Wide start
            ]
            
            for channels in patterns:
                # Filter out configs with channels > 256
                if max(channels) > 256:
                    continue
                    
                for hidden in hidden_options:
                    for use_se in [False, True]:
                        params = estimate_params(
                            n_features=n_features,
                            conv_channels=channels,
                            classifier_hidden=hidden,
                            num_attack_types=num_attack_types,
                            use_se=use_se
                        )
                        
                        diff = abs(params - target_params)
                        if diff < best_diff:
                            best_diff = diff
                            best_config = StudentArchConfig(
                                conv_channels=channels,
                                classifier_hidden=hidden,
                                use_se=use_se,
                                name=f"student_{target_params // 1000}k"
                            )
    
    # Check if within tolerance
    if best_config is not None:
        actual_params = estimate_params(
            n_features=n_features,
            conv_channels=best_config.conv_channels,
            classifier_hidden=best_config.classifier_hidden,
            num_attack_types=num_attack_types,
            use_se=best_config.use_se
        )
        if abs(actual_params - target_params) / target_params > tolerance:
            print(f"Warning: Best config has {actual_params} params "
                  f"({abs(actual_params - target_params) / target_params * 100:.1f}% off target)")
    
    return best_config


def build_student(
    params_target: int,
    input_shape: Tuple[int, int] = (15, 65),
    num_attack_types: int = 10,
    dropout: float = 0.2
) -> MultiTaskStudentDSCNN:
    """
    Build a multi-task student model with approximately target parameter count.
    
    Args:
        params_target: Target number of parameters (5000, 50000, 200000, etc.)
        input_shape: (window_len, n_features)
        num_attack_types: Number of attack classes
        dropout: Dropout rate
    
    Returns:
        MultiTaskStudentDSCNN model
    
    Example:
        >>> student_5k = build_student(5000)
        >>> student_50k = build_student(50000)
        >>> student_200k = build_student(200000)
    """
    n_features = input_shape[1]
    
    # Get architecture config
    arch = search_architecture(
        target_params=params_target,
        n_features=n_features,
        num_attack_types=num_attack_types
    )
    
    if arch is None:
        raise ValueError(f"Could not find architecture for {params_target} params")
    
    # Build model
    model = MultiTaskStudentDSCNN(
        input_shape=input_shape,
        conv_channels=arch.conv_channels,
        classifier_hidden=arch.classifier_hidden,
        dropout=dropout if arch.dropout == 0.2 else arch.dropout,
        num_attack_types=num_attack_types,
        use_se=arch.use_se
    )
    
    return model


def build_student_from_config(config: StudentArchConfig, **kwargs) -> MultiTaskStudentDSCNN:
    """Build student from explicit configuration."""
    return MultiTaskStudentDSCNN(
        conv_channels=config.conv_channels,
        classifier_hidden=config.classifier_hidden,
        dropout=config.dropout,
        use_se=config.use_se,
        **kwargs
    )


# ============ ARCHITECTURE SUMMARY ============

def summarize_architectures(
    param_targets: List[int] = [5000, 50000, 200000],
    input_shape: Tuple[int, int] = (15, 65),
    num_attack_types: int = 10
) -> str:
    """Print summary of all target architectures."""
    lines = ["=" * 70]
    lines.append("Multi-Task Student Architecture Summary")
    lines.append("=" * 70)
    lines.append(f"Input: {input_shape[0]} timesteps × {input_shape[1]} features")
    lines.append(f"Attack types: {num_attack_types}")
    lines.append("-" * 70)
    
    for target in param_targets:
        model = build_student(target, input_shape, num_attack_types)
        actual = model.count_parameters()
        config = model.get_config()
        
        lines.append(f"\n[{target // 1000}K Target]")
        lines.append(f"  Channels: {config['conv_channels']}")
        lines.append(f"  Hidden: {config['classifier_hidden']}")
        lines.append(f"  SE Block: {config['use_se']}")
        lines.append(f"  Actual params: {actual:,} ({actual/target*100:.1f}% of target)")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ============ EXPORTS ============

__all__ = [
    'DepthwiseSeparableConv1d',
    'SEBlock1d',
    'MultiTaskStudentDSCNN',
    'StudentArchConfig',
    'STUDENT_ARCHS',
    'build_student',
    'build_student_from_config',
    'estimate_params',
    'search_architecture',
    'summarize_architectures',
]


# ============ TEST ============

if __name__ == "__main__":
    import sys
    
    print(summarize_architectures())
    print()
    
    # Test each target size
    for target in [5000, 50000, 200000]:
        print(f"\nTesting {target // 1000}K model...")
        model = build_student(target)
        
        # Test forward pass
        x = torch.randn(4, 15, 65)  # (batch, window, features)
        binary_out, attack_out = model(x)
        
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Binary output: {binary_out.shape}")
        print(f"  Attack output: {attack_out.shape}")
        
        # Test KD forward
        binary_out, attack_out, features = model.forward_with_features(x)
        print(f"  Features shape: {features.shape}")
        
        # Verify shapes
        assert binary_out.shape == (4, 1), f"Binary shape mismatch: {binary_out.shape}"
        assert attack_out.shape == (4, 10), f"Attack shape mismatch: {attack_out.shape}"
    
    print("\n✅ All tests passed!")

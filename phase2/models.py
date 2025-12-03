"""
Phase 2 - Student and Teacher Models
=====================================

DS-1D-CNN Student (edge deployment) and Teacher (KD source) architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# ============ BUILDING BLOCKS ============
class MultiScaleConv(nn.Module):
    """Multi-scale temporal convolution (kernel sizes 1,3,5,7)."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 4 == 0, "out_ch must be divisible by 4"
        ch = out_ch // 4
        
        self.conv1 = nn.Conv1d(in_ch, ch, kernel_size=1)
        self.conv3 = nn.Conv1d(in_ch, ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_ch, ch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_ch, ch, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([
            self.conv1(x),
            self.conv3(x),
            self.conv5(x),
            self.conv7(x)
        ], dim=1)
        return F.relu(self.bn(out))


class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size // 2, groups=in_ch
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.bn(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = x.mean(dim=-1)  # Global average pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1)


class TemporalAttention(nn.Module):
    """Attention over time steps."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // reduction, 1)
        self.key = nn.Conv1d(channels, channels // reduction, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.scale = (channels // reduction) ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.size()
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn = torch.bmm(Q.transpose(1, 2), K) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(V, attn.transpose(1, 2))
        return out + x  # Residual


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for teacher model."""
    
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.size()
        
        qkv = self.qkv(x).reshape(B, 3, self.n_heads, self.head_dim, T)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = torch.einsum('bhdt,bhds->bhts', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhts,bhds->bhdt', attn, v)
        out = out.reshape(B, C, T)
        
        return self.proj(out) + x


# ============ STUDENT MODEL ============
class StudentModel(nn.Module):
    """
    DS-1D-CNN Student Model for Edge Deployment.
    
    Architecture:
    - Stem: 1x1 conv projection
    - Stage 1: Multi-scale conv + SE
    - Stage 2: DS-Conv + Temporal attention
    - Stage 3: DS-Conv + SE
    - Global pooling (avg + max)
    - Classifier
    
    Target: ~70-90K parameters
    """
    
    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        stem_channels: int = 64,
        stage1_channels: int = 128,
        stage2_channels: int = 128,
        stage3_channels: int = 256,
        classifier_hidden: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, stem_channels, kernel_size=1),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU()
        )
        
        # Stage 1: Multi-scale
        self.stage1 = nn.Sequential(
            MultiScaleConv(stem_channels, stage1_channels),
            SEBlock(stage1_channels)
        )
        
        # Stage 2: DS-Conv + Attention
        self.stage2_conv = DepthwiseSeparableConv(stage1_channels, stage2_channels)
        self.stage2_attn = TemporalAttention(stage2_channels)
        
        # Stage 3: Higher-level
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(stage2_channels, stage3_channels),
            SEBlock(stage3_channels)
        )
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(stage3_channels * 2, classifier_hidden)
        self.fc2 = nn.Linear(classifier_hidden, n_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) - batch, time, features
        Returns:
            logits: (B, n_classes)
        """
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2_conv(x)
        x = self.stage2_attn(x)
        x = self.stage3(x)
        
        # Global pooling
        gap = self.gap(x).squeeze(-1)
        gmp = self.gmp(x).squeeze(-1)
        x = torch.cat([gap, gmp], dim=1)
        
        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features (for KD)."""
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2_conv(x)
        x = self.stage2_attn(x)
        x = self.stage3(x)
        
        gap = self.gap(x).squeeze(-1)
        gmp = self.gmp(x).squeeze(-1)
        return torch.cat([gap, gmp], dim=1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
        }


# ============ TEACHER MODEL ============
class TeacherModel(nn.Module):
    """
    Teacher Model for Knowledge Distillation.
    
    Larger architecture (3-5x student params) with:
    - Wider channels
    - Multi-head attention
    - Additional stages
    
    Target: ~300-500K parameters
    """
    
    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        stem_channels: int = 128,
        stage1_channels: int = 256,
        stage2_channels: int = 256,
        stage3_channels: int = 512,
        classifier_hidden: int = 256,
        dropout: float = 0.2,
        n_attention_heads: int = 4
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Stem (wider)
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, stem_channels, kernel_size=1),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(),
            nn.Conv1d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU()
        )
        
        # Stage 1: Multi-scale (wider)
        self.stage1 = nn.Sequential(
            MultiScaleConv(stem_channels, stage1_channels),
            SEBlock(stage1_channels),
            MultiScaleConv(stage1_channels, stage1_channels),
            SEBlock(stage1_channels)
        )
        
        # Stage 2: Multi-head attention
        self.stage2_conv = DepthwiseSeparableConv(stage1_channels, stage2_channels)
        self.stage2_attn = MultiHeadAttention(stage2_channels, n_attention_heads)
        self.stage2_conv2 = DepthwiseSeparableConv(stage2_channels, stage2_channels)
        
        # Stage 3: Higher-level (wider)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(stage2_channels, stage3_channels),
            SEBlock(stage3_channels),
            DepthwiseSeparableConv(stage3_channels, stage3_channels),
            SEBlock(stage3_channels)
        )
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Classifier (larger)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(stage3_channels * 2, classifier_hidden)
        self.fc2 = nn.Linear(classifier_hidden, classifier_hidden // 2)
        self.fc3 = nn.Linear(classifier_hidden // 2, n_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) - batch, time, features
        Returns:
            logits: (B, n_classes)
        """
        x = x.transpose(1, 2)
        
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2_conv(x)
        x = self.stage2_attn(x)
        x = self.stage2_conv2(x)
        x = self.stage3(x)
        
        # Global pooling
        gap = self.gap(x).squeeze(-1)
        gmp = self.gmp(x).squeeze(-1)
        x = torch.cat([gap, gmp], dim=1)
        
        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features (for KD)."""
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2_conv(x)
        x = self.stage2_attn(x)
        x = self.stage2_conv2(x)
        x = self.stage3(x)
        
        gap = self.gap(x).squeeze(-1)
        gmp = self.gmp(x).squeeze(-1)
        return torch.cat([gap, gmp], dim=1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
        }


# ============ FACTORY FUNCTIONS ============
def create_student(n_features: int, n_classes: int = 2, **kwargs) -> StudentModel:
    """Create student model with default config."""
    return StudentModel(n_features=n_features, n_classes=n_classes, **kwargs)

def create_teacher(n_features: int, n_classes: int = 2, **kwargs) -> TeacherModel:
    """Create teacher model with default config."""
    return TeacherModel(n_features=n_features, n_classes=n_classes, **kwargs)


# Backward-compatible aliases used across the pipeline
DSCNNStudent = StudentModel
DSCNNTeacher = TeacherModel

__all__ = [
    'MultiScaleConv', 'DepthwiseSeparableConv', 'SEBlock', 'TemporalAttention',
    'MultiHeadAttention', 'StudentModel', 'TeacherModel',
    'DSCNNStudent', 'DSCNNTeacher',
    'create_student', 'create_teacher'
]


# ============ TEST ============
if __name__ == "__main__":
    # Test models
    n_features = 65
    window_size = 15
    batch_size = 4
    
    # Create dummy input
    x = torch.randn(batch_size, window_size, n_features)
    
    # Student
    student = create_student(n_features)
    student_out = student(x)
    student_params = student.count_parameters()
    print(f"Student: {student_params:,} params, output shape: {student_out.shape}")
    
    # Teacher
    teacher = create_teacher(n_features)
    teacher_out = teacher(x)
    teacher_params = teacher.count_parameters()
    print(f"Teacher: {teacher_params:,} params, output shape: {teacher_out.shape}")
    
    # Ratio
    print(f"Teacher/Student ratio: {teacher_params/student_params:.1f}x")

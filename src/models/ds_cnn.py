"""
Research-grade Depthwise Separable 1D CNN (DS_1D_CNN).

Design goals:
- Depthwise separable conv blocks (parameter-efficient)
- Configurable activations, BN toggles, optional SE blocks
- Same-padding to preserve sequence length by default
- Weight initialization and activation hooks for research/debugging
- to_config/from_config for reproducibility
- Defaults target ~80K params (approx) for common window sizes; tune conv_channels/classifier_hidden to match exact budget
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .base_model import BaseIDSModel

IntSeq = Sequence[int]
PathLike = Union[str, os.PathLike]  # not used directly but useful for future save/load


# -------------------------
# Helper blocks
# -------------------------
class _SEBlock(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0 for SE block")
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        s = x.mean(dim=-1)  # (batch, channels)
        s = self.act(self.fc1(s))
        s = self.sig(self.fc2(s))
        return x * s.unsqueeze(-1)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable conv block: depthwise -> pointwise -> BN? -> Activation
    Keeps sequence length by default using same-padding.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive integers")

        if padding is None:
            padding = (kernel_size - 1) // 2  # same-padding

        # depthwise conv
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        # pointwise conv
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.use_bn = bool(use_bn)
        self.bn = nn.BatchNorm1d(out_channels) if self.use_bn else nn.Identity()

        activation = (activation or "relu").lower()
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # introspection
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -------------------------
# Main model
# -------------------------
class DS_1D_CNN(BaseIDSModel):
    """
    Depthwise Separable 1D CNN.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        (window_length, n_features). n_features is interpreted as conv channels.
    num_classes : int
    conv_channels : Sequence[int]
        Output channels for each conv block (e.g., [32, 64, 64]).
    kernel_size : Union[int, Sequence[int]]
        Single kernel size or per-block sizes.
    dropout_rate : float
    use_se : bool
        Whether to include Squeeze-and-Excitation blocks (off by default).
    use_bn : bool
        BatchNorm after pointwise conv (recommended True).
    activation : str
        Activation used inside conv blocks.
    classifier_hidden : int
        Hidden units in classifier dense layer.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 2,
        conv_channels: IntSeq = (32, 64, 64),
        kernel_size: Union[int, IntSeq] = 3,
        dropout_rate: float = 0.2,
        use_se: bool = False,
        use_bn: bool = True,
        activation: str = "relu",
        classifier_hidden: int = 64,
        device: Optional[torch.device] = None,
    ):
        # Validate input_shape format
        if not (isinstance(input_shape, (tuple, list)) and len(input_shape) == 2):
            raise ValueError("input_shape must be (window_length, n_features)")

        super().__init__(input_shape=input_shape, num_classes=num_classes, device=device)

        window_len, n_features = input_shape
        self.window_length = int(window_len)
        self.n_features = int(n_features)
        self.activation_name = activation
        self.use_se = bool(use_se)
        self.use_bn = bool(use_bn)
        self.dropout_rate = float(dropout_rate)

        self.conv_channels = list(map(int, conv_channels)) if conv_channels else [self.n_features]
        num_blocks = len(self.conv_channels)

        # Normalize kernel_size to list
        if isinstance(kernel_size, int):
            kernel_sizes = [int(kernel_size)] * num_blocks
        else:
            kernel_sizes = list(map(int, kernel_size))
            if len(kernel_sizes) != num_blocks:
                raise ValueError("If kernel_size is a sequence it must match length of conv_channels")

        # Build convolutional blocks
        conv_blocks: List[nn.Module] = []
        in_ch = self.n_features
        for i, out_ch in enumerate(self.conv_channels):
            ks = kernel_sizes[i]
            block = nn.Sequential(
                DepthwiseSeparableConv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=ks,
                    padding=(ks - 1) // 2,
                    use_bn=self.use_bn,
                    activation=self.activation_name,
                )
            )
            if self.use_se:
                block.add_module("se", _SEBlock(out_ch))
            if self.dropout_rate and self.dropout_rate > 0:
                block.add_module("drop", nn.Dropout(self.dropout_rate))
            conv_blocks.append(block)
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Global pooling -> classifier
        last_ch = self.conv_channels[-1] if len(self.conv_channels) > 0 else self.n_features
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (batch, channels, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # (batch, last_ch)
            nn.Linear(last_ch, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate if self.dropout_rate > 0 else 0.0),
            nn.Linear(classifier_hidden, self.num_classes),
        )

        # Activation hooks
        self._activation_hooks: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Weight initialization
        self._init_weights()

        self.logger.info(
            f"DS_1D_CNN initialized; input_shape={input_shape}, conv_channels={self.conv_channels}, "
            f"kernel_sizes={kernel_sizes}, use_se={self.use_se}, dropout={self.dropout_rate}, classifier_hidden={classifier_hidden}"
        )

    # -------------------------
    # Weight init
    # -------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------
    # Activation hooks
    # -------------------------
    def register_activation_hook(self, layer_idx: int) -> str:
        """
        Register a forward hook on a conv block (by index). Returns the hook key.
        Use get_activation(key) to retrieve last activation (detached to CPU).
        """
        if not (0 <= layer_idx < len(self.conv_blocks)):
            raise IndexError("layer_idx out of range for conv_blocks")

        target = self.conv_blocks[layer_idx]
        # Choose final module inside the block to capture post-activation output
        hook_module = target[-1] if isinstance(target, nn.Sequential) and len(target) > 0 else target

        key = f"conv_block_{layer_idx}"

        def _hook(module, inp, outp):
            try:
                self._activation_hooks[key] = outp.detach().cpu()
            except Exception:
                # best-effort fallback if detach fails
                self._activation_hooks[key] = outp

        handle = hook_module.register_forward_hook(_hook)
        self._hook_handles.append(handle)
        self.logger.debug(f"Registered activation hook for key={key} on module={hook_module}")
        return key

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Override BaseIDSModel.get_activation(layer_name).
        Returns the stored activation (or None if not present).
        """
        return self._activation_hooks.get(layer_name)

    def clear_activation_hooks(self) -> None:
        """Remove hooks and clear stored activations."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []
        self._activation_hooks = {}

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input: (batch, window_length, n_features).
        Internally transposes to (batch, channels, seq_len) for Conv1d.
        """
        if x.ndim != 3:
            raise ValueError("Input tensor must be 3D: (batch, window_length, n_features)")

        # transpose -> (batch, channels, seq_len)
        x = x.transpose(1, 2).contiguous()
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    # -------------------------
    # Config / introspection
    # -------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[torch.device] = None) -> "DS_1D_CNN":
        """Recreate model from a config dict produced by to_config()."""
        cfg = dict(cfg)
        input_shape = tuple(cfg.pop("input_shape"))
        return cls(input_shape=input_shape, device=device, **cfg)

    def get_model_info(self, input_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        base = super().get_model_info(input_tensor)
        base.update({
            "architecture": "DS_1D_CNN",
            "conv_channels": tuple(self.conv_channels),
            "use_se": self.use_se,
            "dropout_rate": self.dropout_rate,
            "use_bn": self.use_bn,
        })
        return base

    # optional convenience
    def param_count(self) -> int:
        """Return number of trainable parameters (alias of BaseIDSModel.count_parameters)."""
        return self.count_parameters()

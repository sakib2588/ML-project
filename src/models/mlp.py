"""
SmallMLP: Research-grade MLP baseline for windowed IDS data.

Design decisions (OL):
- Final layer returns logits (no softmax) to work with nn.CrossEntropyLoss.
- Default hidden_sizes (128,64,32) chosen to match ~small baseline parameter budget.
- Keep forward() minimal — no metrics; use hooks/metrics utilities instead.
- Reproducible initialization with optional seed.
- Device-aware: model and inputs moved to self.device.
- Shape checks to fail fast on bad inputs.
- Activation hooks for inspectability.
- Lightweight save/load for reproducible experiments.
"""
from __future__ import annotations

from typing import Tuple, Optional, Sequence, Dict, Any, Callable
import logging
import math

import torch
import torch.nn as nn

from .base_model import BaseIDSModel

# ---- helpers ----
ActivationFactory = Callable[[], nn.Module]

def _get_activation(name: str) -> ActivationFactory:
    name = (name or "relu").lower()
    if name == "relu":
        return lambda: nn.ReLU(inplace=True)
    if name == "gelu":
        return lambda: nn.GELU()
    if name == "tanh":
        return lambda: nn.Tanh()
    if name == "leaky_relu":
        return lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True)
    raise ValueError(f"Unsupported activation='{name}' — choose relu / gelu / tanh / leaky_relu.")


class SmallMLP(BaseIDSModel):
    """
    Small MLP for windowed IDS inputs.

    Parameters
    ----------
    input_shape : Tuple[int, ...]
        Either (window_len, n_features) or (n_features,). If single-length tuple
        it is treated as flattened input (flatten_input=False by default).
    num_classes : int
        Number of classes (output dim).
    hidden_sizes : Sequence[int]
        Sizes of hidden dense layers (default (128,64,32)).
    dropout_rate : float
    activation : str
        Activation name ('relu','gelu','tanh','leaky_relu').
    use_batchnorm : bool
    use_layernorm : bool
    flatten_input : bool
        If True, the model expects (batch, window_len, n_features) and flattens internally.
        If False and input_shape was a single-dim shape, model expects (batch, n_features).
    seed : Optional[int] reproducible init
    device : Optional[torch.device]
    """
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int = 2,
        hidden_sizes: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.3,
        activation: str = "relu",
        use_batchnorm: bool = False,
        use_layernorm: bool = False,
        flatten_input: bool = True,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        # validate and normalize input shape
        if not (isinstance(input_shape, (tuple, list)) and 1 <= len(input_shape) <= 2):
            raise ValueError("input_shape must be a tuple of length 1 or 2, e.g. (window_len, n_features) or (n_features,)")

        if len(input_shape) == 1:
            # flattened input provided
            window_len, n_features = 1, int(input_shape[0])
            # If flatten_input is True but a 1-D input was provided, treat as already flat.
            flatten_input = False
        else:
            window_len, n_features = int(input_shape[0]), int(input_shape[1])

        flat_input_size = window_len * n_features

        super().__init__(input_shape=(window_len, n_features), num_classes=num_classes, device=device)

        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dropout_rate = float(dropout_rate)
        self.activation_name = activation.lower()
        self.use_batchnorm = bool(use_batchnorm)
        self.use_layernorm = bool(use_layernorm)
        self.flatten_input = bool(flatten_input)
        self.seed = seed

        if self.use_batchnorm and self.use_layernorm:
            raise ValueError("use_batchnorm and use_layernorm are mutually exclusive.")

        # reproducible init if seed provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Build layers
        layers: list[nn.Module] = []
        if self.flatten_input:
            layers.append(nn.Flatten(start_dim=1))

        in_features = flat_input_size
        act_factory = _get_activation(self.activation_name)

        for out_features in self.hidden_sizes:
            layers.append(nn.Linear(in_features, out_features, bias=not self.use_batchnorm))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            elif self.use_layernorm:
                layers.append(nn.LayerNorm(out_features))
            layers.append(act_factory())
            if self.dropout_rate and self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_features = out_features

        # final linear outputs logits (no softmax) — correct for CrossEntropyLoss
        layers.append(nn.Linear(in_features, self.num_classes))

        self.model = nn.Sequential(*layers)

        # initialize weights
        self._init_weights()

        # move the model to device
        self.to(self.device)

        # activation hooks storage
        self._activation_hooks: Dict[str, torch.Tensor] = {}

        # metadata for logging
        self._meta: Dict[str, Any] = {
            "architecture": "SmallMLP",
            "hidden_sizes": self.hidden_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation_name,
            "use_batchnorm": self.use_batchnorm,
            "use_layernorm": self.use_layernorm,
            "flatten_input": self.flatten_input,
            "seed": self.seed,
            "input_flat_size": flat_input_size,
        }

        self.logger.info(
            f"SmallMLP initialized: flat_input={flat_input_size}, hidden={self.hidden_sizes}, "
            f"bn={self.use_batchnorm}, ln={self.use_layernorm}, dropout={self.dropout_rate}, act={self.activation_name}"
        )

    def _init_weights(self) -> None:
        """Weight initialization chosen based on activation family."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in ("relu", "gelu", "leaky_relu"):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                elif self.activation_name == "tanh":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                try:
                    if hasattr(m, "weight") and m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                except Exception:
                    pass  # defensive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits:

        - If flatten_input=True expects x shape (batch, window_len, n_features).
        - If flatten_input=False expects x shape (batch, n_features).
        """
        # move to same device as model
        if x.device != self.device:
            x = x.to(self.device)

        if self.flatten_input:
            if x.ndim != 3:
                raise ValueError(f"Expected 3D input (batch, window_len, n_features) when flatten_input=True; got {tuple(x.shape)}")
        else:
            if x.ndim != 2:
                raise ValueError(f"Expected 2D input (batch, n_features) when flatten_input=False; got {tuple(x.shape)}")

        # Model returns logits; do not apply softmax here.
        return self.model(x)

    # activation hooks (useful for explainability)
    def register_activation_hook(self, layer_index: int, layer_name: Optional[str] = None) -> None:
        """Register a forward hook on module index of self.model (nn.Sequential)."""
        if not (0 <= layer_index < len(self.model)):
            raise IndexError("layer_index out of range")
        module = self.model[layer_index]

        key = layer_name or f"layer_{layer_index}"

        def _hook(module_, inp, outp):
            self._activation_hooks[key] = outp.detach().cpu()

        handle = module.register_forward_hook(_hook)
        self._meta.setdefault("_hook_handles", []).append(handle)
        self.logger.debug(f"Registered activation hook on module {layer_index} as '{key}'")

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Return stored activation for the given layer name."""
        return self._activation_hooks.get(layer_name)



    def remove_all_hooks(self) -> None:
        for h in self._meta.get("_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._meta["_hook_handles"] = []

    # model info & persistence
    def get_model_info(self, input_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        base = super().get_model_info(input_tensor)
        base.update(self._meta)
        return base

    def save_state(self, path: str) -> None:
        """Save state_dict for reproducible experiments."""
        torch.save(self.state_dict(), path)
        self.logger.info(f"Saved model state_dict to {path}")

    def load_state(self, path: str, strict: bool = True) -> None:
        """Load state dict (device-aware)."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, strict=strict)
        self.to(self.device)
        self.logger.info(f"Loaded model state_dict from {path}")

    def __repr__(self) -> str:
        return f"<SmallMLP input={self.input_shape} hidden={self.hidden_sizes} num_classes={self.num_classes}>"

"""
LSTMModel: Sequence model baseline for IDS windowed features.

Design goals:
- Accept (window_len, n_features) inputs directly with batch-first tensors
- Configurable hidden size, layers, bidirectionality, and classifier head
- Careful initialization (orthogonal recurrent weights, forget gate bias boost)
- Device-aware via BaseIDSModel to integrate with Trainer/Evaluator pipeline
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class LSTMModel(BaseIDSModel):
    """Lightweight LSTM classifier for IDS sequences."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        fc_hidden: Optional[int] = 64,
        fc_dropout: float = 0.3,
        use_layernorm: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        if not (isinstance(input_shape, (tuple, list)) and len(input_shape) == 2):
            raise ValueError("input_shape must be (window_length, n_features)")

        window_len, n_features = int(input_shape[0]), int(input_shape[1])
        if window_len <= 0 or n_features <= 0:
            raise ValueError("input_shape values must be positive integers")

        super().__init__(input_shape=(window_len, n_features), num_classes=num_classes, device=device)

        self.window_len = window_len
        self.n_features = n_features
        self.hidden_size = int(hidden_size)
        self.num_layers = max(1, int(num_layers))
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.fc_hidden = None if fc_hidden is None else int(fc_hidden)
        if self.fc_hidden is not None and self.fc_hidden <= 0:
            self.fc_hidden = None
        self.fc_dropout = float(fc_dropout)
        self.use_layernorm = bool(use_layernorm)

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        lstm_out_dim = self.hidden_size * (2 if self.bidirectional else 1)

        self.post_lstm_norm = nn.LayerNorm(lstm_out_dim) if self.use_layernorm else nn.Identity()

        classifier_layers: list[nn.Module] = []
        if self.fc_hidden is not None:
            classifier_layers.append(nn.Linear(lstm_out_dim, self.fc_hidden))
            classifier_layers.append(nn.ReLU(inplace=True))
            if self.fc_dropout > 0:
                classifier_layers.append(nn.Dropout(self.fc_dropout))
            classifier_layers.append(nn.Linear(self.fc_hidden, self.num_classes))
        else:
            classifier_layers.append(nn.Linear(lstm_out_dim, self.num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._init_weights()
        self.to(self.device)

        self._meta: Dict[str, Any] = {
            "architecture": "LSTMModel",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "fc_hidden": self.fc_hidden,
            "fc_dropout": self.fc_dropout,
            "use_layernorm": self.use_layernorm,
        }

        self.logger.info(
            "Initialized LSTMModel window=%d features=%d hidden=%d layers=%d bidir=%s fc_hidden=%s",
            self.window_len,
            self.n_features,
            self.hidden_size,
            self.num_layers,
            self.bidirectional,
            str(self.fc_hidden),
        )

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden = param.shape[0] // 4
                param.data[hidden:2 * hidden] = 1.0

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def register_activation_hook(self) -> None:
        """Capture the LSTM outputs for introspection."""

        def _hook(_: nn.Module, __: Tuple[torch.Tensor, ...], outputs: Any) -> None:
            if isinstance(outputs, tuple):
                self._activation_cache["lstm_output"] = outputs[0].detach().cpu()
            else:
                self._activation_cache["lstm_output"] = outputs.detach().cpu()

        self.lstm.register_forward_hook(_hook)

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        return self._activation_cache.get(layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, window_length, n_features)")
        if x.device != self.device:
            x = x.to(self.device)

        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        last_step = self.post_lstm_norm(last_step)
        logits = self.classifier(last_step)
        return logits

    def get_model_info(self, input_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        info = super().get_model_info(input_tensor)
        info.update(self._meta)
        return info

    def save_state(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        self.logger.info("Saved LSTMModel state_dict to %s", path)

    def load_state(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, strict=strict)
        self.to(self.device)
        self.logger.info("Loaded LSTMModel state_dict from %s", path)

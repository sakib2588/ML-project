"""
BaseIDSModel: Abstract base class for IDS models (MLP, DS-CNN, LSTM).

Provides:
- Forward pass signature enforcement
- Parameter counting
- FLOPs calculation (optional, with thop)
- Model info retrieval
- Device-aware initialization
- Optional activation hooks for explainability
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import logging

import torch
import torch.nn as nn

# FIX: Silence Pylance if thop is not found
try:
    from thop import profile  # type: ignore
    HAS_THOP = True
except ImportError:
    profile = None
    HAS_THOP = False


class BaseIDSModel(nn.Module, ABC):
    """
    Abstract base class for IDS models.
    
    All IDS model implementations should inherit from this class.
    """

    def __init__(
        self, 
        input_shape: Tuple[int, ...], 
        num_classes: int = 2, 
        device: Optional[torch.device] = None
    ):
        """
        Initialize the base model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of input data, e.g., (window_length, n_features).
        num_classes : int, default=2
            Number of output classes.
        device : Optional[torch.device], default=None
            Device to put the model on (CPU/GPU). Defaults to CUDA if available.
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Must be implemented by subclasses.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor (logits or probabilities).
        """
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_flops(self, input_tensor: Optional[torch.Tensor] = None) -> int:
        """
        Calculate FLOPs using thop library.

        Parameters
        ----------
        input_tensor : Optional[torch.Tensor]
            Input tensor for FLOPs calculation. If None, returns 0.

        Returns
        -------
        int
            Estimated FLOPs of the model.
        """
        if profile is None:
            self.logger.debug("thop not installed; skipping FLOPs calculation.")
            return 0

        if input_tensor is None:
            self.logger.warning("No input_tensor provided; cannot compute FLOPs. Returning 0.")
            return 0

        try:
            # Move the dummy input to the same device as the model
            input_tensor = input_tensor.to(self.device)

            # Profile the model
            result = profile(self, inputs=(input_tensor,), verbose=False)

            # Extract MACs (Multiply–Accumulate Operations)
            macs = result[0]  # thop always returns MACs as first element
            # params = result[1]  # optional, currently unused

            # Convert MACs to FLOPs (FLOPs ~ 2 * MACs)
            flops = int(macs * 2)
            return flops

        except Exception as e:
            self.logger.warning(f"Failed to calculate FLOPs: {e}")
            return 0




    def get_model_info(self, input_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Return dictionary with model info: name, input shape, params, classes, FLOPs.

        Parameters
        ----------
        input_tensor : Optional[torch.Tensor]
            Input tensor for FLOPs calculation. If None, dummy tensor is used.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "name": self.__class__.__name__,
            "parameters": self.count_parameters(),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "FLOPs": self.get_flops(input_tensor)
        }

    def summary(self, input_tensor: Optional[torch.Tensor] = None) -> None:
        """Print a summary of the model, including parameters and FLOPs."""
        info = self.get_model_info(input_tensor)
        self.logger.info(f"Model Summary: {info}")
        print(info)

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Placeholder for registering activation hooks on intermediate layers.
        Can be overridden in subclass.

        Parameters
        ----------
        layer_name : str
            Name of the layer to attach hook.

        Returns
        -------
        Optional[torch.Tensor]
            Placeholder activation (None by default).
        """
        self.logger.warning(f"get_activation not implemented in base class. Called layer_name={layer_name}")
        return None

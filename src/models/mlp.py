"""
PyTorch Multi-Layer Perceptron (MLP) for Tabular IoT Intrusion Detection.

This module defines the core neural network backbone (TabularIoTMLP) optimized
for low-latency inference on edge IoT gateways and communication-efficient FL.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 1 & 2: Tabular Backbone & Normalization)
- skills/methodology-audit/SKILL.md (Pillar 2: Mathematical Correctness)
"""

import logging
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None

from src.base.model import BaseFederatedModel

# Backward-compatibility imports from isolated packages
from src.losses.focal_loss import MultiClassFocalLoss
from src.training.trainer import LocalClientTrainer
from src.evaluation.evaluator import evaluate_model

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class TabularIoTMLP(BaseFederatedModel):
        """
        Lightweight Multi-Layer Perceptron for Network Flow Intrusion Detection.

        Architecture:
            Input (input_dim) -> Linear(128) -> ReLU -> Dropout(dropout_rate)
                              -> Linear(64)  -> ReLU -> Dropout(dropout_rate)
                              -> Linear(num_classes)

        Note:
            Batch Normalization is intentionally excluded to prevent severe representation
            corruption caused by aggregating client-specific running statistics under
            Non-IID distributions.
        """
        def __init__(
            self,
            input_dim: int = 39,
            hidden_dims: Tuple[int, ...] = (128, 64),
            num_classes: int = 8,
            dropout_rate: float = 0.2
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.num_classes = num_classes
            self.dropout_rate = dropout_rate

            layers: List[nn.Module] = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)

            # Weight initialization: Kaiming normal for ReLU activations
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

else:
    class TabularIoTMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TabularIoTMLP. Please install torch.")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping model self-test.")
    else:
        print("--- Running Isolated TabularIoTMLP Architecture Test ---")
        model = TabularIoTMLP(input_dim=39, hidden_dims=(128, 64), num_classes=8, dropout_rate=0.2)
        print(f"Model instantiated. Trainable parameters: {model.get_num_parameters():,}")
        print(f"Model payload size: {model.get_model_size_kb():.2f} KB")

        # Forward pass check
        x_dummy = torch.randn(4, 39)
        out = model(x_dummy)
        assert out.shape == (4, 8), f"Expected shape (4, 8), got {out.shape}"

        # Weight getter/setter check
        w = model.get_weights()
        model.set_weights(w)
        print("✅ TabularIoTMLP architecture test passed successfully!")

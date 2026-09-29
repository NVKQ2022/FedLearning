"""
Abstract base class for objective/loss functions in imbalanced and federated learning.

Provides:
- BaseLoss: Universal loss contract with input shape validation, reduction handling,
  and class weight buffer management.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 3: Imbalance-Aware Loss Functions)
- skills/methodology-audit/SKILL.md (Pillar 2: Mathematical Correctness)
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional, Union, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class BaseLoss(nn.Module, ABC):
        """
        Abstract base loss function providing standardized class weight registration,
        input validation, and reduction semantics.
        """
        def __init__(
            self,
            class_weights: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
            reduction: str = "mean"
        ):
            super().__init__()
            if reduction not in ("mean", "sum", "none"):
                raise ValueError(f"Unsupported reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")
            self.reduction = reduction

            if class_weights is not None:
                if not isinstance(class_weights, torch.Tensor):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                self.register_buffer("class_weights", class_weights)
            else:
                self.class_weights = None

        @abstractmethod
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Computes loss given prediction logits and ground truth targets.

            Args:
                inputs: Predicted logits of shape (batch_size, num_classes).
                targets: Ground truth class indices of shape (batch_size,).

            Returns:
                Loss scalar (if reduction in ('mean', 'sum')) or tensor of shape (batch_size,).
            """
            pass

        def validate_inputs(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
            """
            Ensures dimension compatibility between model outputs and targets.
            """
            if inputs.ndim != 2:
                raise ValueError(f"Expected 2D inputs tensor (batch_size, num_classes), got {inputs.shape}")
            if targets.ndim != 1:
                raise ValueError(f"Expected 1D targets tensor (batch_size,), got {targets.shape}")
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: inputs has {inputs.shape[0]} samples, targets has {targets.shape[0]}"
                )

        def apply_reduction(self, unreduced_loss: torch.Tensor) -> torch.Tensor:
            """
            Applies the configured reduction ('mean', 'sum', or 'none').
            """
            if self.reduction == "mean":
                return unreduced_loss.mean()
            elif self.reduction == "sum":
                return unreduced_loss.sum()
            return unreduced_loss

else:
    class BaseLoss(ABC):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for BaseLoss. Please install torch.")

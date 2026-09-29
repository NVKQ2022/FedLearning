"""
Loss functions package for FL-IoT-IDS.

Exports:
- BaseLoss: Universal loss contract with input shape validation, reduction handling, and class weight buffers.
- MultiClassFocalLoss: Down-weights easy well-classified background samples and focuses gradients on hard minority attacks.
- FocalLoss: Alias for MultiClassFocalLoss.
- build_loss_function: Factory function returning standard CE, weighted CE, or Focal Loss.
"""

from .base import BaseLoss
from .focal_loss import MultiClassFocalLoss, FocalLoss, build_loss_function

__all__ = [
    "BaseLoss",
    "MultiClassFocalLoss",
    "FocalLoss",
    "build_loss_function",
]

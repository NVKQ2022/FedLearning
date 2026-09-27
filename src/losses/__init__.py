"""
Loss functions package for FL-IoT-IDS.
"""

from .focal_loss import MultiClassFocalLoss, build_loss_function

__all__ = [
    "MultiClassFocalLoss",
    "build_loss_function",
]

"""
Neural network architectures package for FL-IoT-IDS.
"""

from .mlp import TabularIoTMLP

# Re-exports for backward-compatibility
from src.losses.focal_loss import MultiClassFocalLoss
from src.training.trainer import LocalClientTrainer
from src.evaluation.evaluator import evaluate_model

__all__ = [
    "TabularIoTMLP",
    "MultiClassFocalLoss",
    "LocalClientTrainer",
    "evaluate_model",
]

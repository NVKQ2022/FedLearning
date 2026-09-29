"""
Backward-compatibility proxy for src.models.mlp -> src.models.tabular_mlp.
"""

from .tabular_mlp import TabularIoTMLPModel, TabularIoTMLP

# Re-exports for backward-compatibility
from src.losses.focal_loss import MultiClassFocalLoss
from src.training.trainer import LocalClientTrainer
from src.evaluation.evaluator import evaluate_model

__all__ = [
    "TabularIoTMLPModel",
    "TabularIoTMLP",
    "MultiClassFocalLoss",
    "LocalClientTrainer",
    "evaluate_model",
]

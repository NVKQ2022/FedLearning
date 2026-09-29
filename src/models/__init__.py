"""
Neural network architectures package for FL-IoT-IDS.

Exports:
- BaseModel: Root PyTorch base model with parameter and footprint analytics.
- BaseFederatedModel: Extends BaseModel with parameter list extraction and delta tracking.
- TabularIoTMLPModel: Lightweight 3-layer MLP optimized for network flows and edge IoT gateways.
- TabularIoTMLP: Alias for TabularIoTMLPModel.
"""

from .base import BaseModel, BaseFederatedModel
from .mlp import TabularIoTMLPModel, TabularIoTMLP

# Re-exports for backward-compatibility
from src.losses.focal_loss import MultiClassFocalLoss
from src.training.trainer import LocalClientTrainer
from src.evaluation.evaluator import evaluate_model

__all__ = [
    "BaseModel",
    "BaseFederatedModel",
    "TabularIoTMLPModel",
    "TabularIoTMLP",
    "MultiClassFocalLoss",
    "LocalClientTrainer",
    "evaluate_model",
]

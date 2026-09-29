"""
Base classes package for clean architecture and future extensibility in FL-IoT-IDS.

Exports:
- BaseModel: Root PyTorch model contract with parameter analytics and serialization.
- BaseFederatedModel: Extends BaseModel with parameter list extraction and delta tracking for FL.
- BaseTrainer: Core training lifecycle, evaluation, gradient clipping, and checkpointing.
- BaseLoss: Universal loss function contract with class weight buffers and reduction handling.
- BasePreprocessor: Standardized tabular fitting, transformation, and serialization.
- BasePartitioner: Contract for simulated federated partitioning and heterogeneity verification.
- BaseFederatedStrategy: Server round parameter aggregation (FedAvg, coordinate median, trimmed mean).
"""

from .model import BaseModel, BaseFederatedModel
from .trainer import BaseTrainer
from .loss import BaseLoss
from .preprocessor import BasePreprocessor
from .partitioner import BasePartitioner
from .strategy import BaseFederatedStrategy

__all__ = [
    "BaseModel",
    "BaseFederatedModel",
    "BaseTrainer",
    "BaseLoss",
    "BasePreprocessor",
    "BasePartitioner",
    "BaseFederatedStrategy",
]

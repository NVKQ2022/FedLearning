"""
Training orchestrators package for FL-IoT-IDS.

Exports:
- BaseTrainer: Abstract training lifecycle with evaluate(), clip_gradients(), and checkpointing.
- LocalClientTrainer: Federated local client trainer supporting FedAvg and FedProx (mu).
- CentralizedTrainer: Standalone centralized trainer with early stopping, progress bars, and Top-K checkpointing.
"""

from .base import BaseTrainer
from .trainer import (
    LocalClientTrainer,
    CentralizedTrainer,
)

__all__ = [
    "BaseTrainer",
    "LocalClientTrainer",
    "CentralizedTrainer",
]

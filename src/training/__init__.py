"""
Training orchestrators package for FL-IoT-IDS.

Exports:
- BaseTrainer: Abstract training lifecycle with evaluate(), clip_gradients(), and checkpointing.
- FederatedClientTrainer: Federated local client trainer supporting FedAvg and FedProx (mu).
- LocalClientTrainer: Alias for FederatedClientTrainer for backward-compatibility.
- CentralizedTrainer: Standalone centralized trainer with early stopping, progress bars, and Top-K checkpointing.
"""

from .base import BaseTrainer
from .trainer import (
    FederatedClientTrainer,
    LocalClientTrainer,
    CentralizedTrainer,
)

__all__ = [
    "BaseTrainer",
    "FederatedClientTrainer",
    "LocalClientTrainer",
    "CentralizedTrainer",
]

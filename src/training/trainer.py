"""
Backward-compatibility proxy for src.training.trainer.

Re-exports:
- FederatedClientTrainer (from .federated_trainer)
- LocalClientTrainer (alias from .federated_trainer)
- CentralizedTrainer (from .centralized_trainer)
"""

from .federated_trainer import FederatedClientTrainer, LocalClientTrainer
from .centralized_trainer import CentralizedTrainer

__all__ = [
    "FederatedClientTrainer",
    "LocalClientTrainer",
    "CentralizedTrainer",
]

"""
Training orchestrators package for FL-IoT-IDS.
"""

from .trainer import (
    LocalClientTrainer,
    CentralizedTrainer,
)

__all__ = [
    "LocalClientTrainer",
    "CentralizedTrainer",
]

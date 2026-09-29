"""
Federated learning coordination and aggregation strategies package for FL-IoT-IDS.

Exports:
- BaseFederatedStrategy: Abstract aggregation contract with weighted_average, median, and trimmed_mean.
- FedAvgStrategy: Standard sample-weighted parameter averaging.
- FedProxStrategy: Server coordination with proximal mu tracking.
- FedMedianStrategy: Coordinate-wise median for Byzantine robustness.
- FedTrimmedMeanStrategy: Trimmed mean for adversarial tolerance.
"""

from .base import BaseFederatedStrategy
from .strategies import (
    FedAvgStrategy,
    FedProxStrategy,
    FedMedianStrategy,
    FedTrimmedMeanStrategy,
)

__all__ = [
    "BaseFederatedStrategy",
    "FedAvgStrategy",
    "FedProxStrategy",
    "FedMedianStrategy",
    "FedTrimmedMeanStrategy",
]

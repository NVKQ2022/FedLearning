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
    build_strategy,
)
from .flower_client import FlowerIoTClient, start_flower_client
from .flower_server import (
    FlowerIoTServerStrategy,
    start_flower_server,
    build_flower_server_eval_fn,
)

__all__ = [
    "BaseFederatedStrategy",
    "FedAvgStrategy",
    "FedProxStrategy",
    "FedMedianStrategy",
    "FedTrimmedMeanStrategy",
    "build_strategy",
    "FlowerIoTClient",
    "start_flower_client",
    "FlowerIoTServerStrategy",
    "start_flower_server",
    "build_flower_server_eval_fn",
]

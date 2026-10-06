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
from .scenario import (
    compute_partition_eda,
    plot_class_distribution,
    record_client_round_metric,
    create_federated_scenario,
    create_centralized_scenario,
    load_client_partition,
    load_server_data,
    plot_scenario_convergence,
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
    "compute_partition_eda",
    "plot_class_distribution",
    "record_client_round_metric",
    "create_federated_scenario",
    "create_centralized_scenario",
    "load_client_partition",
    "load_server_data",
    "plot_scenario_convergence",
]

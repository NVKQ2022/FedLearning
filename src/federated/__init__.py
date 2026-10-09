"""
Federated learning coordination and aggregation package for FL-IoT-IDS.

Exports core utilities for setting up clients, servers, and simulation scenarios.
"""

from .flower_client import FlowerIoTClient, start_flower_client
from .flower_server import (
    TrackingStrategyWrapper,
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
    load_server_test_data,
    plot_scenario_convergence,
)
from .grpc_runner import run_flower_grpc

__all__ = [
    "FlowerIoTClient",
    "start_flower_client",
    "TrackingStrategyWrapper",
    "start_flower_server",
    "build_flower_server_eval_fn",
    "run_flower_grpc",
    "compute_partition_eda",
    "plot_class_distribution",
    "record_client_round_metric",
    "create_federated_scenario",
    "create_centralized_scenario",
    "load_client_partition",
    "load_server_data",
    "load_server_test_data",
    "plot_scenario_convergence",
]

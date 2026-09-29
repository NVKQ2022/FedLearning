"""
Abstract base class for federated aggregation and server coordination strategies.

Provides:
- BaseFederatedStrategy: Standard aggregation contract for server rounds,
  including sample-weighted parameter averaging, coordinate-wise median,
  trimmed-mean, and metric aggregation.

Adheres to:
- skills/experiment-orchestration/SKILL.md (Federated Simulation)
- skills/methodology-audit/SKILL.md (Pillar 2: FL Optimization & Mathematical Correctness)
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseFederatedStrategy(ABC):
    """
    Abstract base class for Federated Learning server aggregation strategies.
    
    Subclasses must implement:
    - aggregate_fit(server_round, client_results): Combines client model weights into a new global model.
    - aggregate_evaluate(server_round, client_eval_results): Combines client validation results.

    Provides out-of-the-box:
    - weighted_average(): Mathematically exact sample-weighted parameter averaging (FedAvg).
    - coordinate_wise_median(): Byzantine-resilient aggregation.
    - trimmed_mean(): Robust aggregation filtering out extreme gradient updates.
    """
    def __init__(
        self,
        name: str = "BaseStrategy",
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2
    ):
        self.name = name
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregates client model updates from a completed training round.

        Args:
            server_round: Current communication round index (1-indexed).
            client_results: List of tuples (client_weights, num_samples, client_metrics).

        Returns:
            Tuple of (aggregated_global_weights, server_round_metrics).
        """
        pass

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        client_eval_results: List[Tuple[int, float, Dict[str, float]]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregates client evaluation metrics.

        Args:
            server_round: Current communication round index.
            client_eval_results: List of tuples (num_samples, loss, metrics_dict).

        Returns:
            Tuple of (aggregated_loss, aggregated_metrics_dict).
        """
        pass

    @staticmethod
    def weighted_average(
        weights_list: List[List[np.ndarray]],
        sample_counts: List[int]
    ) -> List[np.ndarray]:
        """
        Computes sample-weighted average:
            w_global = sum_{k=1}^K (n_k / N) * w_k
        """
        if not weights_list:
            raise ValueError("weights_list is empty.")
        if len(weights_list) != len(sample_counts):
            raise ValueError("Length of weights_list must match sample_counts.")

        total_samples = sum(sample_counts)
        if total_samples <= 0:
            raise ValueError(f"Total samples must be positive, got {total_samples}")

        num_layers = len(weights_list[0])
        aggregated_weights: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
            for client_idx, client_weights in enumerate(weights_list):
                weight_factor = sample_counts[client_idx] / total_samples
                layer_sum += client_weights[layer_idx].astype(np.float64) * weight_factor
            aggregated_weights.append(layer_sum.astype(weights_list[0][layer_idx].dtype))

        return aggregated_weights

    @staticmethod
    def coordinate_wise_median(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Computes coordinate-wise median across clients.
        Resistant to Byzantine client poisoning and arbitrary gradient manipulation.
        """
        if not weights_list:
            raise ValueError("weights_list is empty.")

        num_layers = len(weights_list[0])
        aggregated_weights: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            stacked = np.stack([client[layer_idx] for client in weights_list], axis=0)
            median_layer = np.median(stacked, axis=0)
            aggregated_weights.append(median_layer.astype(weights_list[0][layer_idx].dtype))

        return aggregated_weights

    @staticmethod
    def trimmed_mean(
        weights_list: List[List[np.ndarray]],
        trim_fraction: float = 0.1
    ) -> List[np.ndarray]:
        """
        Computes coordinate-wise trimmed mean, excluding top and bottom trim_fraction percentiles.
        """
        if not weights_list:
            raise ValueError("weights_list is empty.")

        num_layers = len(weights_list[0])
        aggregated_weights: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            stacked = np.stack([client[layer_idx] for client in weights_list], axis=0)
            k = int(len(weights_list) * trim_fraction)
            if k > 0 and 2 * k < len(weights_list):
                sorted_stacked = np.sort(stacked, axis=0)
                trimmed = sorted_stacked[k:-k]
                layer_agg = np.mean(trimmed, axis=0)
            else:
                layer_agg = np.mean(stacked, axis=0)
            aggregated_weights.append(layer_agg.astype(weights_list[0][layer_idx].dtype))

        return aggregated_weights

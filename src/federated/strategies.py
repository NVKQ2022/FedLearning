"""
Concrete Federated Learning Server Aggregation Strategies.

Provides:
- FedAvgStrategy: Standard sample-weighted federated averaging (McMahan et al., 2017).
- FedProxStrategy: Server coordination for proximal federated optimization with mu tracking (Li et al., 2020).
- FedMedianStrategy: Coordinate-wise median aggregation for Byzantine poisoning tolerance.
- FedTrimmedMeanStrategy: Coordinate-wise trimmed mean for adversarial and outlier client mitigation.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 4: FedProx Proximal Regularization)
- skills/methodology-audit/SKILL.md (Pillar 2: FL Optimization & Mathematical Correctness)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.base.strategy import BaseFederatedStrategy

logger = logging.getLogger(__name__)


class FedAvgStrategy(BaseFederatedStrategy):
    """
    Standard Federated Averaging (FedAvg) strategy (McMahan et al., 2017).
    Aggregates client parameters using exact sample-weighted averaging:
        w_{t+1} = sum_{k=1}^K (n_k / N) * w_k^{t}
    """
    def __init__(
        self,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2
    ):
        super().__init__(
            name="FedAvg",
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_results:
            raise ValueError(f"Round {server_round}: client_results is empty.")

        weights_list = [res[0] for res in client_results]
        sample_counts = [res[1] for res in client_results]
        metrics_list = [res[2] for res in client_results if len(res) > 2]

        aggregated_weights = self.weighted_average(weights_list, sample_counts)

        # Aggregate training metrics across participating clients
        total_samples = sum(sample_counts)
        round_metrics: Dict[str, Any] = {
            "round": server_round,
            "participating_clients": len(client_results),
            "total_samples": total_samples
        }

        # Weighted loss & accuracy if reported by clients
        if metrics_list and "loss" in metrics_list[0]:
            weighted_loss = sum(m.get("loss", 0.0) * n for m, n in zip(metrics_list, sample_counts)) / total_samples
            round_metrics["train_loss"] = float(weighted_loss)
        if metrics_list and "acc" in metrics_list[0]:
            weighted_acc = sum(m.get("acc", 0.0) * n for m, n in zip(metrics_list, sample_counts)) / total_samples
            round_metrics["train_acc"] = float(weighted_acc)

        logger.debug(f"Round {server_round} FedAvg aggregation completed across {len(client_results)} clients.")
        return aggregated_weights, round_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        client_eval_results: List[Tuple[int, float, Dict[str, float]]]
    ) -> Tuple[float, Dict[str, Any]]:
        if not client_eval_results:
            raise ValueError(f"Round {server_round}: client_eval_results is empty.")

        total_samples = sum(res[0] for res in client_eval_results)
        weighted_loss = sum(res[0] * res[1] for res in client_eval_results) / max(1, total_samples)

        metrics_summary: Dict[str, Any] = {
            "eval_loss": float(weighted_loss),
            "total_eval_samples": total_samples
        }

        # If accuracy is present in client metrics dict
        has_acc = any("accuracy" in res[2] or "acc" in res[2] for res in client_eval_results)
        if has_acc:
            weighted_acc = sum(
                res[0] * res[2].get("accuracy", res[2].get("acc", 0.0))
                for res in client_eval_results
            ) / max(1, total_samples)
            metrics_summary["eval_acc"] = float(weighted_acc)

        return float(weighted_loss), metrics_summary


class FedProxStrategy(FedAvgStrategy):
    """
    FedProx Server Strategy (Li et al., 2020).
    Extends FedAvg with proximal regularization parameter tracking (mu)
    and client drift diagnostics.
    """
    def __init__(
        self,
        mu: float = 0.05,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )
        self.name = "FedProx"
        self.mu = mu

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        aggregated_weights, metrics = super().aggregate_fit(server_round, client_results)
        metrics["mu"] = self.mu
        logger.debug(f"Round {server_round} FedProx (mu={self.mu}) aggregation completed.")
        return aggregated_weights, metrics


class FedMedianStrategy(BaseFederatedStrategy):
    """
    Coordinate-wise Median Federated Aggregation Strategy.
    Resilient against Byzantine clients and malicious poisoning attacks.
    """
    def __init__(
        self,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 3,
        min_available_clients: int = 3
    ):
        super().__init__(
            name="FedMedian",
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_results:
            raise ValueError(f"Round {server_round}: client_results is empty.")

        weights_list = [res[0] for res in client_results]
        aggregated_weights = self.coordinate_wise_median(weights_list)

        round_metrics: Dict[str, Any] = {
            "round": server_round,
            "participating_clients": len(client_results),
            "aggregation": "coordinate_median"
        }
        return aggregated_weights, round_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        client_eval_results: List[Tuple[int, float, Dict[str, float]]]
    ) -> Tuple[float, Dict[str, Any]]:
        losses = [res[1] for res in client_eval_results]
        median_loss = float(np.median(losses))
        return median_loss, {"eval_loss": median_loss, "aggregation": "median"}


class FedTrimmedMeanStrategy(BaseFederatedStrategy):
    """
    Coordinate-wise Trimmed Mean Federated Aggregation Strategy.
    Filters out extreme gradient fractions to protect against adversarial outliers.
    """
    def __init__(
        self,
        trim_fraction: float = 0.1,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 3,
        min_available_clients: int = 3
    ):
        super().__init__(
            name="FedTrimmedMean",
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )
        self.trim_fraction = trim_fraction

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_results:
            raise ValueError(f"Round {server_round}: client_results is empty.")

        weights_list = [res[0] for res in client_results]
        aggregated_weights = self.trimmed_mean(weights_list, trim_fraction=self.trim_fraction)

        round_metrics: Dict[str, Any] = {
            "round": server_round,
            "participating_clients": len(client_results),
            "trim_fraction": self.trim_fraction,
            "aggregation": "trimmed_mean"
        }
        return aggregated_weights, round_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        client_eval_results: List[Tuple[int, float, Dict[str, float]]]
    ) -> Tuple[float, Dict[str, Any]]:
        losses = [res[1] for res in client_eval_results]
        mean_loss = float(np.mean(losses))
        return mean_loss, {"eval_loss": mean_loss, "trim_fraction": self.trim_fraction}


def build_strategy(
    strategy_name: str = "fedavg",
    mu: float = 0.05,
    trim_fraction: float = 0.1,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    **kwargs: Any
) -> BaseFederatedStrategy:
    """
    Factory function to instantiate server aggregation strategies.

    Args:
        strategy_name: Name of strategy ('fedavg', 'fedprox', 'fedmedian', 'fedtrimmedmean').
        mu: Proximal parameter for FedProx (mu=0.0 equivalent to FedAvg).
        trim_fraction: Trimming fraction for FedTrimmedMean.
        fraction_fit: Fraction of available clients selected per round.
        min_fit_clients: Minimum clients participating per round.
        min_available_clients: Minimum total clients required.

    Returns:
        Instance of BaseFederatedStrategy subclass.
    """
    strat = strategy_name.lower().replace("-", "").replace("_", "")
    if strat in ("fedavg", "avg"):
        return FedAvgStrategy(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )
    elif strat in ("fedprox", "prox"):
        return FedProxStrategy(
            mu=mu,
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients
        )
    elif strat in ("fedmedian", "median"):
        return FedMedianStrategy(
            fraction_fit=fraction_fit,
            min_fit_clients=max(min_fit_clients, 3),
            min_available_clients=max(min_available_clients, 3)
        )
    elif strat in ("fedtrimmedmean", "trimmedmean"):
        return FedTrimmedMeanStrategy(
            trim_fraction=trim_fraction,
            fraction_fit=fraction_fit,
            min_fit_clients=max(min_fit_clients, 3),
            min_available_clients=max(min_available_clients, 3)
        )
    else:
        raise ValueError(
            f"Unknown federated strategy: '{strategy_name}'. "
            f"Expected one of: 'fedavg', 'fedprox', 'fedmedian', 'fedtrimmedmean'."
        )

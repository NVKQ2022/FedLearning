"""
Flower Server and Strategy Orchestration for FL-IoT-IDS.

Implements the Federated Server Layer defined in the thesis proposal:
"Xây dựng và đánh giá prototype hệ thống phát hiện xâm nhập IoT sử dụng
Federated Learning trong môi trường dữ liệu non-IID" (Nguyễn Việt Kỳ Quân, 2026).

Responsibilities:
1. Coordinates edge clients over gRPC (client selection, broadcast, aggregation).
2. Supports FedAvg (McMahan et al., 2017) and FedProx (Li et al., 2020) aggregation.
3. Centralized server-side evaluation after every round (Macro-F1, Minority Recall, Loss).
4. Tracks system performance metrics defined in thesis Section 9:
   - Convergence rounds and best model checkpointing.
   - Cumulative communication cost in megabytes: 2 * sum(m_t * B).
   - Round execution wall-clock time and client parameter drift.
5. Operates in standalone gRPC process mode or programmatic simulation runner.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    DataLoader = object

try:
    import flwr as fl
    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        Metrics,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg, FedProx
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    fl = object
    FedAvg = object
    FedProx = object
    Parameters = Any
    NDArrays = Any
    Scalar = Any
    FitIns = Any
    FitRes = Any
    EvaluateIns = Any
    EvaluateRes = Any
    Metrics = Any
    ClientProxy = Any

logger = logging.getLogger(__name__)


def aggregate_weighted_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Computes sample-weighted average across client-reported training metrics.
    Aggregates local loss, accuracy, parameter drift, and client execution time.
    """
    if not metrics:
        return {}

    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated: Dict[str, Scalar] = {}
    keys = metrics[0][1].keys()

    for key in keys:
        if isinstance(metrics[0][1].get(key), (int, float, np.floating, np.integer)):
            weighted_sum = sum(num_samples * float(m.get(key, 0.0)) for num_samples, m in metrics)
            aggregated[key] = float(weighted_sum / total_samples)

    aggregated["total_samples"] = int(total_samples)
    return aggregated


class FlowerIoTServerStrategy(FedAvg if HAS_FLWR else object):
    """
    Publication-grade Flower Strategy tailored for the FL-IoT-IDS thesis.
    Extends FedAvg / FedProx with:
    - Centralized server holdout test set evaluation (Macro-F1 & Minority Recall).
    - Best global model checkpoint preservation.
    - Thesis Section 9 system metrics tracking (communication cost bytes, wall-clock time).
    """
    def __init__(
        self,
        strategy_name: str = "fedavg",
        mu: float = 0.0,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        local_epochs: int = 2,
        model_size_bytes: float = 55584.0,  # ~54.28 KB for TabularIoTMLPModel
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initializes the IoT IDS server strategy.

        Args:
            strategy_name: 'fedavg' or 'fedprox'.
            mu: FedProx proximal coefficient (mu=0.0 reduces to FedAvg).
            fraction_fit: Proportion of available clients sampled per round.
            fraction_evaluate: Proportion of available clients sampled for decentralized evaluation.
            min_fit_clients: Minimum clients participating per round.
            min_evaluate_clients: Minimum clients evaluated per round.
            min_available_clients: Minimum clients connected before starting round.
            local_epochs: Number of local training epochs on each edge client.
            model_size_bytes: Size of one model parameter vector in bytes.
            evaluate_fn: Server-side centralized evaluation callback.
            on_fit_config_fn: Function generating round config dictionary sent to clients for training.
            on_evaluate_config_fn: Function generating round config dictionary sent to clients for evaluation.
            checkpoint_dir: Directory where best model weights are preserved.
        """
        self.strategy_name = strategy_name.lower()
        self.mu = mu if self.strategy_name == "fedprox" else 0.0
        self.local_epochs = local_epochs
        self.model_size_bytes = model_size_bytes
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Track thesis metrics across communication rounds
        self.round_history: Dict[str, Any] = {
            "round": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "round_duration_sec": [],
            "comm_cost_mb": [],
            "drift_l2": [],
            "client_eval": {},
            "client_accuracies": {}
        }
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.cumulative_comm_bytes = 0.0
        self._round_start_time = 0.0

        def default_on_fit_config(server_round: int) -> Dict[str, Scalar]:
            return {
                "server_round": server_round,
                "local_epochs": self.local_epochs,
                "mu": self.mu
            }

        def default_on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
            return {
                "server_round": server_round
            }

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn or default_on_fit_config,
            on_evaluate_config_fn=on_evaluate_config_fn or default_on_evaluate_config,
            fit_metrics_aggregation_fn=aggregate_weighted_metrics,
            evaluate_metrics_aggregation_fn=aggregate_weighted_metrics
        )

        logger.info(
            f"FlowerIoTServerStrategy initialized: Strategy={self.strategy_name.upper()} | "
            f"mu={self.mu} | min_fit_clients={min_fit_clients} | local_epochs={local_epochs}"
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: Any
    ) -> List[Tuple[Any, FitIns]]:
        """
        Records round start time and prepares client fit instructions.
        """
        self._round_start_time = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            logger.warning(f"[Server Round {server_round:02d}] ⚠️ {len(failures)} client(s) failed during fit:")
            for idx, fail in enumerate(failures[:3]):
                logger.warning(f"   Failure {idx + 1}: {fail}")

        if not results:
            logger.error(
                f"[Server Round {server_round:02d}] ❌ No successful client updates received! "
                f"Check client resources or device configurations."
            )
            return None, {}

        # 1. Execute parameter aggregation (standard sample-weighted FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        round_duration = time.perf_counter() - self._round_start_time

        # 2. Update communication cost: 2 * m_t * Model_Size (Upload + Download per client)
        num_participating = len(results)
        round_comm_bytes = 2.0 * num_participating * self.model_size_bytes
        self.cumulative_comm_bytes += round_comm_bytes
        cumulative_comm_mb = self.cumulative_comm_bytes / (1024.0 ** 2)

        # 3. Extract aggregated diagnostics
        train_loss = aggregated_metrics.get("loss", 0.0)
        train_acc = aggregated_metrics.get("accuracy", 0.0)
        drift_l2 = aggregated_metrics.get("drift_l2", 0.0)

        # Append round metrics
        self.round_history["round"].append(server_round)
        self.round_history["train_loss"].append(float(train_loss))
        self.round_history["train_acc"].append(float(train_acc))
        self.round_history["round_duration_sec"].append(float(round_duration))
        self.round_history["comm_cost_mb"].append(float(cumulative_comm_mb))
        self.round_history["drift_l2"].append(float(drift_l2))

        logger.info(
            f"[Server Round {server_round:02d} Aggregated] Clients: {num_participating} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Drift: {drift_l2:.4f} | Comm: {cumulative_comm_mb:.2f} MB | Time: {round_duration:.2f}s"
        )

        return aggregated_parameters, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Runs centralized server evaluation on the global holdout validation set.
        Checkpoints top-performing model weights if Macro-F1 improves.
        """
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is None:
            return None

        loss, metrics = eval_res
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        acc = float(metrics.get("accuracy", 0.0))

        self.round_history["val_loss"].append(float(loss))
        self.round_history["val_acc"].append(float(acc))
        self.round_history["val_macro_f1"].append(float(macro_f1))

        is_best = macro_f1 > self.best_macro_f1
        if is_best:
            self.best_macro_f1 = macro_f1
            self.best_round = server_round

            # Save best global model weights
            best_weights = parameters_to_ndarrays(parameters)
            ckpt_path = os.path.join(self.checkpoint_dir, f"flower_{self.strategy_name}_best_weights.npz")
            np.savez(ckpt_path, *best_weights)
            canonical_path = os.path.join(self.checkpoint_dir, "best_weights.npz")
            np.savez(canonical_path, *best_weights)
            logger.info(f"⭐ New best server model (Macro-F1: {macro_f1*100:.2f}%) saved to {canonical_path}")

        star = "⭐ (Best)" if is_best else ""
        logger.info(
            f"[Server Round {server_round:02d} Central Eval] "
            f"Val Loss: {loss:.4f} | Val Acc: {acc*100:5.2f}% | Val Macro-F1: {macro_f1*100:5.2f}% {star}"
        )

        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates decentralized evaluation results from edge clients.
        Computes sample-weighted loss/accuracy, prints a comparison table across clients,
        and records per-client evaluation metrics into round_history.
        """
        if failures:
            logger.warning(f"[Server Round {server_round:02d}] ⚠️ {len(failures)} client(s) failed during evaluation.")

        if not results:
            return None, {}

        # 1. Standard sample-weighted loss and metrics aggregation
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 2. Extract per-client evaluation metrics
        client_metrics: Dict[Union[int, str], Dict[str, Any]] = {}
        for proxy, eval_res in results:
            cid = eval_res.metrics.get("client_id", proxy.cid)
            try:
                cid_key = int(cid)
            except (ValueError, TypeError):
                cid_key = str(cid)
            c_acc = float(eval_res.metrics.get("accuracy", 0.0))
            c_loss = float(eval_res.loss if eval_res.loss is not None else eval_res.metrics.get("loss", 0.0))
            client_metrics[cid_key] = {
                "accuracy": c_acc,
                "loss": c_loss,
                "samples": int(eval_res.num_examples)
            }

        sorted_clients = sorted(client_metrics.items(), key=lambda x: str(x[0]))

        # 3. Print Clean Comparison Table
        rnd_label = f"ROUND {server_round:02d}" if server_round > 0 else "ROUND 0 (Initial Model)"
        print(f"\n┌────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 📊 {rnd_label:^66} │")
        print(f"│ 🌐 DECENTRALIZED CLIENT LOCAL VALIDATION COMPARISON TABLE              │")
        print(f"├──────────┬──────────────┬──────────────┬───────────────────────────────┤")
        print(f"│ Client   │ Accuracy (%) │ Loss         │ Val Samples                   │")
        print(f"├──────────┼──────────────┼──────────────┼───────────────────────────────┤")
        for cid, m in sorted_clients:
            c_label = f"Client {cid}" if isinstance(cid, int) else f"Client {str(cid)[:5]}"
            print(f"│ {c_label:<8} │ {m['accuracy']*100:10.2f}%  │ {m['loss']:12.4f} │ {m['samples']:29,d} │")
        print(f"├──────────┴──────────────┴──────────────┴───────────────────────────────┤")
        acc_values = [m['accuracy'] for _, m in sorted_clients]
        mean_acc = float(np.mean(acc_values)) if acc_values else 0.0
        min_acc = float(np.min(acc_values)) if acc_values else 0.0
        max_acc = float(np.max(acc_values)) if acc_values else 0.0
        spread = max_acc - min_acc
        print(f"│ Mean Acc: {mean_acc*100:5.2f}% | Min: {min_acc*100:5.2f}% | Max: {max_acc*100:5.2f}% | Spread: {spread*100:5.2f}% │")
        print(f"└────────────────────────────────────────────────────────────────────────┘\n")

        # 4. Record per-round and per-client trajectories
        if "client_eval" not in self.round_history:
            self.round_history["client_eval"] = {}
        self.round_history["client_eval"][str(server_round)] = {
            str(cid): m for cid, m in client_metrics.items()
        }

        if "client_accuracies" not in self.round_history:
            self.round_history["client_accuracies"] = {}
        for cid, m in client_metrics.items():
            ckey = f"client_{cid}"
            if ckey not in self.round_history["client_accuracies"]:
                self.round_history["client_accuracies"][ckey] = []
            self.round_history["client_accuracies"][ckey].append(m["accuracy"])

        logger.info(
            f"[Server Round {server_round:02d} Decentralized Eval] "
            f"Mean Acc: {mean_acc*100:5.2f}% | " +
            " | ".join([f"C{cid}: {m['accuracy']*100:.1f}%" for cid, m in sorted_clients])
        )

        return agg_loss, agg_metrics

    def save_round_history(self, filepath: str) -> None:
        """Exports complete round history dictionary to JSON and renders convergence plot."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.round_history, f, indent=4)
        logger.info(f"Server round history exported to {filepath}")
        try:
            from src.federated.scenario import plot_scenario_convergence
            plot_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), "convergence.png")
            plot_scenario_convergence(self.round_history, plot_path)
            logger.info(f"Server convergence plot saved to {plot_path}")
        except Exception as e:
            logger.debug(f"Convergence plot rendering skipped: {e}")


def build_flower_server_eval_fn(
    model: Any,
    val_loader: Any,
    class_names: List[str],
    criterion: Optional[Any] = None,
    minority_classes: Optional[List[str]] = None,
    device: Union[str, Any] = "cpu"
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Builds the centralized server evaluation function passed to Flower strategies.

    Args:
        model: Global TabularIoTMLPModel instance.
        val_loader: Server-side validation DataLoader.
        class_names: List of all 8 class names.
        criterion: MultiClassFocalLoss criterion.
        minority_classes: Rare attacks to isolate ('Web-based', 'Brute-force').
        device: Target compute device.

    Returns:
        Evaluation callback function adhering to Flower evaluate_fn contract.
    """
    from src.evaluation.evaluator import evaluate_comprehensive

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # 1. Inject server parameters into global model
        model.set_weights(parameters)

        # 2. Run comprehensive publication evaluation
        results = evaluate_comprehensive(
            model=model,
            dataloader=val_loader,
            class_names=class_names,
            criterion=criterion,
            minority_classes=minority_classes,
            device=device
        )

        metrics: Dict[str, Scalar] = {
            "accuracy": float(results["accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "weighted_f1": float(results["weighted_f1"]),
            "macro_precision": float(results["macro_precision"]),
            "macro_recall": float(results["macro_recall"])
        }

        # Include minority attack recalls
        for attack_name, recall_val in results.get("minority_recall", {}).items():
            metrics[f"minority_{attack_name}_recall"] = float(recall_val)

        return float(results["loss"]), metrics

    return evaluate_fn


def start_flower_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    strategy: Optional[Any] = None,
    strategy_name: str = "fedavg",
    mu: float = 0.0,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    local_epochs: int = 2,
    model_size_bytes: float = 55584.0,
    evaluate_fn: Optional[Callable] = None,
    partitions_dir: Optional[str] = None,
    scenario_dir: Optional[str] = None,
    history_save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Configures and starts the Flower server process communicating over gRPC.
    """
    if not HAS_FLWR:
        raise ImportError("Flower (flwr) is required to run start_flower_server. Run: pip install flwr")

    checkpoint_dir = "checkpoints"
    if scenario_dir and os.path.exists(scenario_dir):
        server_dir = os.path.join(scenario_dir, "server") if os.path.exists(os.path.join(scenario_dir, "server")) else scenario_dir
        checkpoint_dir = server_dir
        if partitions_dir is None:
            partitions_dir = server_dir
        if history_save_path is None:
            history_save_path = os.path.join(server_dir, "round_history.json")

    # If partitions_dir is provided and evaluate_fn is not, build evaluate_fn from val_data.npz or server_val.npz
    if evaluate_fn is None and partitions_dir and os.path.exists(partitions_dir):
        meta_path = os.path.join(partitions_dir, "meta.json")
        val_path = os.path.join(partitions_dir, "val_data.npz")
        if not os.path.exists(val_path):
            val_path = os.path.join(partitions_dir, "server_val.npz")

        if os.path.exists(meta_path) and os.path.exists(val_path):
            with open(meta_path) as f:
                meta = json.load(f)
            val_npz = np.load(val_path)
            from src.models.tabular_mlp import TabularIoTMLPModel
            from torch.utils.data import TensorDataset, DataLoader
            eval_model = TabularIoTMLPModel(input_dim=meta["input_dim"], num_classes=meta["num_classes"])
            eval_ds = TensorDataset(torch.from_numpy(val_npz["X"]), torch.from_numpy(val_npz["y"]))
            eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False)
            eval_device = "cuda" if torch.cuda.is_available() else "cpu"
            evaluate_fn = build_flower_server_eval_fn(
                model=eval_model,
                val_loader=eval_loader,
                class_names=meta.get("class_names", []),
                device=eval_device
            )
            logger.info(f"Loaded server holdout validation set ({len(val_npz['X']):,} samples) for evaluation.")

    if strategy is None:
        strategy = FlowerIoTServerStrategy(
            strategy_name=strategy_name,
            mu=mu,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            local_epochs=local_epochs,
            model_size_bytes=model_size_bytes,
            evaluate_fn=evaluate_fn,
            checkpoint_dir=checkpoint_dir
        )

    logger.info(f"Starting Flower gRPC Server on {server_address} for {num_rounds} rounds...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    if history_save_path and hasattr(strategy, "save_round_history"):
        strategy.save_round_history(history_save_path)

    if hasattr(strategy, "round_history"):
        return strategy.round_history
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Flower IoT Server Process.")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="gRPC bind address.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds.")
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"], help="FL Strategy.")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx proximal parameter mu (0.0 for FedAvg).")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients evaluated each round.")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum connected clients.")
    parser.add_argument("--local-epochs", type=int, default=2, help="Number of local epochs per round.")
    parser.add_argument("--partitions-dir", type=str, default=None, help="Directory containing server_val.npz and meta.json.")
    parser.add_argument("--scenario-dir", type=str, default=None, help="Root directory of experimental scenario (e.g. scenarios/E5_fedprox_dirichlet_0.1).")
    parser.add_argument("--history-save-path", type=str, default=None, help="Filepath to export round history JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Flower Server CLI: Running {args.strategy.upper()} on {args.server_address} for {args.rounds} rounds.")
    start_flower_server(
        server_address=args.server_address,
        num_rounds=args.rounds,
        strategy_name=args.strategy,
        mu=args.mu,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        local_epochs=args.local_epochs,
        partitions_dir=args.partitions_dir,
        scenario_dir=args.scenario_dir,
        history_save_path=args.history_save_path
    )

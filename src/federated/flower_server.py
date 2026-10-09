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



class TrackingStrategyWrapper(fl.server.strategy.Strategy if HAS_FLWR else object):
    """
    A Wrapper Strategy that wraps ANY built-in Flower Strategy (FedAvg, FedProx, FedAdam, etc.)
    and injects publication-grade tracking without rewriting aggregation math.
    """
    def __init__(
        self,
        base_strategy: 'fl.server.strategy.Strategy',
        model_size_bytes: float = 55584.0,
        checkpoint_dir: str = "checkpoints"
    ):
        self.base_strategy = base_strategy
        self.model_size_bytes = model_size_bytes
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.round_history: Dict[str, Any] = {
            "round": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "val_macro_f1": [], "round_duration_sec": [], "comm_cost_mb": [], "drift_l2": [],
            "client_eval": {}, "client_accuracies": {}
        }
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.cumulative_comm_bytes = 0.0
        self._round_start_time = 0.0

    def initialize_parameters(self, client_manager: ClientProxy) -> Optional[Parameters]:
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientProxy) -> List[Tuple[ClientProxy, FitIns]]:
        self._round_start_time = time.perf_counter()
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            logger.warning(f"[Server Round {server_round:02d}] ⚠️ {len(failures)} client(s) failed during fit.")

        aggregated_parameters, aggregated_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        round_duration = time.perf_counter() - self._round_start_time
        num_participating = len(results)
        round_comm_bytes = 2.0 * num_participating * self.model_size_bytes
        self.cumulative_comm_bytes += round_comm_bytes
        cumulative_comm_mb = self.cumulative_comm_bytes / (1024.0 ** 2)

        train_loss = aggregated_metrics.get("loss", 0.0)
        train_acc = aggregated_metrics.get("accuracy", 0.0)
        drift_l2 = aggregated_metrics.get("drift_l2", 0.0)

        self.round_history["round"].append(server_round)
        self.round_history["train_loss"].append(float(train_loss))
        self.round_history["train_acc"].append(float(train_acc))
        self.round_history["drift_l2"].append(float(drift_l2))
        self.round_history["comm_cost_mb"].append(float(cumulative_comm_mb))
        self.round_history["round_duration_sec"].append(float(round_duration))

        logger.info(
            f"[Server Round {server_round:02d} Aggregation] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Drift L2: {drift_l2:.4f} | Comm Cost: {cumulative_comm_mb:.2f} MB | Time: {round_duration:.1f}s"
        )
        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientProxy) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        agg_loss, agg_metrics = self.base_strategy.aggregate_evaluate(server_round, results, failures)
        if not results:
            return agg_loss, agg_metrics

        client_metrics = {}
        for proxy, eval_res in results:
            cid = eval_res.metrics.get("client_id", proxy.cid)
            try: cid_key = int(cid)
            except: cid_key = str(cid)
            client_metrics[cid_key] = {
                "accuracy": float(eval_res.metrics.get("accuracy", 0.0)),
                "loss": float(eval_res.loss if eval_res.loss is not None else eval_res.metrics.get("loss", 0.0)),
                "samples": int(eval_res.num_examples)
            }

        sorted_clients = sorted(client_metrics.items(), key=lambda x: str(x[0]))
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
        if acc_values:
            print(f"│ Mean Acc: {float(np.mean(acc_values))*100:5.2f}% | Min: {float(np.min(acc_values))*100:5.2f}% | Max: {float(np.max(acc_values))*100:5.2f}% │")
        print(f"└────────────────────────────────────────────────────────────────────────┘\n")

        if "client_eval" not in self.round_history: self.round_history["client_eval"] = {}
        self.round_history["client_eval"][str(server_round)] = {str(cid): m for cid, m in client_metrics.items()}
        
        if "client_accuracies" not in self.round_history: self.round_history["client_accuracies"] = {}
        for cid, m in client_metrics.items():
            if str(cid) not in self.round_history["client_accuracies"]:
                self.round_history["client_accuracies"][str(cid)] = []
            self.round_history["client_accuracies"][str(cid)].append(m["accuracy"])

        return agg_loss, agg_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        eval_res = self.base_strategy.evaluate(server_round, parameters)
        if eval_res is None:
            return None
            
        loss, metrics = eval_res
        acc = metrics.get("accuracy", 0.0)
        macro_f1 = metrics.get("macro_f1", 0.0)

        self.round_history["val_loss"].append(float(loss))
        self.round_history["val_acc"].append(float(acc))
        self.round_history["val_macro_f1"].append(float(macro_f1))

        is_best = macro_f1 > self.best_macro_f1
        if is_best:
            self.best_macro_f1 = macro_f1
            self.best_round = server_round

            best_weights = parameters_to_ndarrays(parameters)
            ckpt_path = os.path.join(self.checkpoint_dir, f"flower_best_weights.npz")
            np.savez(ckpt_path, *best_weights)
            logger.info(f"⭐ New best server model (Macro-F1: {macro_f1*100:.2f}%) saved to {ckpt_path}")

        star = "⭐ (Best)" if is_best else ""
        logger.info(
            f"[Server Round {server_round:02d} Central Eval] "
            f"Val Loss: {loss:.4f} | Val Acc: {acc*100:5.2f}% | Val Macro-F1: {macro_f1*100:5.2f}% {star}"
        )
        return eval_res

    def save_round_history(self, filepath: str) -> None:
        try:
            with open(filepath, "w") as f:
                json.dump(self.round_history, f, indent=2)
            logger.info(f"Round history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save round history to {filepath}: {e}")

def start_flower_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    strategy: Optional[Any] = None,
    strategy_name: str = "fedavg",
    mu: float = 0.0,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.0,
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
        def on_fit_config(server_round: int) -> Dict[str, Scalar]:
            return {"server_round": server_round, "local_epochs": local_epochs, "mu": mu}
            
        def on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
            return {"server_round": server_round}
            
        if strategy_name.lower() == "fedprox":
            base_strategy = fl.server.strategy.FedProx(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config,
                on_evaluate_config_fn=on_evaluate_config,
                fit_metrics_aggregation_fn=aggregate_weighted_metrics,
                evaluate_metrics_aggregation_fn=aggregate_weighted_metrics,
                proximal_mu=mu
            )
        else:
            base_strategy = fl.server.strategy.FedAvg(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config,
                on_evaluate_config_fn=on_evaluate_config,
                fit_metrics_aggregation_fn=aggregate_weighted_metrics,
                evaluate_metrics_aggregation_fn=aggregate_weighted_metrics
            )
            
        strategy = TrackingStrategyWrapper(
            base_strategy=base_strategy,
            model_size_bytes=model_size_bytes,
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
    parser.add_argument("--fraction-evaluate", type=float, default=0.0, help="Fraction of clients evaluated each round (default: 0.0, local evaluation runs during fit).")
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

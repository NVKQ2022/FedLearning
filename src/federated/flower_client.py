"""
Flower NumPyClient for Decentralized IoT Network Intrusion Detection (FL-IoT-IDS).

Implements the Federated Client Layer defined in the thesis proposal:
"Xây dựng và đánh giá prototype hệ thống phát hiện xâm nhập IoT sử dụng
Federated Learning trong môi trường dữ liệu non-IID" (Nguyễn Việt Kỳ Quân, 2026).

Responsibilities:
1. Receives global model parameters from central Flower server via gRPC.
2. Performs local training on private client partition with optional FedProx proximal penalty (mu).
3. Measures local training wall-clock time and parameter drift (||w_local - w_global||_2).
4. Returns updated weights, sample count, and training diagnostics back to the server.
5. Runs either in Flower simulation mode or as an independent OS process over localhost/network.
"""

import argparse
import copy
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

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
    from flwr.common import Scalar
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    # Mock base class if flwr is not yet imported
    class _MockNumPyClient:
        pass
    fl = object
    Scalar = Any

logger = logging.getLogger(__name__)


class FlowerIoTClient(fl.client.NumPyClient if HAS_FLWR else object):
    """
    Flower NumPyClient wrapping TabularIoTMLPModel and FederatedClientTrainer.
    Adheres strictly to the thesis client architecture:
    - In-place parameter updates via NumPy arrays
    - Local multi-epoch training with gradient clipping
    - FedProx proximal regularization: L_local(w) + (mu / 2) * ||w - w_t||^2
    - Local training duration and parameter drift tracking
    """
    def __init__(
        self,
        client_id: int,
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        trainer: Optional[Any] = None,
        device: Union[str, Any] = "cpu"
    ):
        """
        Initializes the edge Flower IoT client.

        Args:
            client_id: Unique integer identifier for this edge client.
            model: Instance of BaseFederatedModel (e.g. TabularIoTMLPModel).
            train_loader: PyTorch DataLoader containing client's local training split.
            val_loader: Optional PyTorch DataLoader for client-side local validation.
            trainer: Instance of FederatedClientTrainer. If None, instantiates default.
            device: Computing device ('cpu', 'cuda', or torch.device).
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for FlowerIoTClient.")

        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.model.to(self.device)

        if trainer is None:
            from src.optimizers.optimizer import build_optimizer
            from src.losses.focal_loss import build_loss_function
            from src.training.federated_trainer import FederatedClientTrainer

            optimizer = build_optimizer(self.model, "adamw", lr=1e-3, weight_decay=1e-4)
            criterion = build_loss_function("focal_loss", gamma=2.0, device=self.device)
            self.trainer = FederatedClientTrainer(
                model=self.model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                max_grad_norm=5.0
            )
        else:
            self.trainer = trainer

        logger.info(
            f"[Client {self.client_id}] Initialized with "
            f"{len(self.train_loader.dataset)} training samples on device '{self.device}'."
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Returns the current local model parameters as a list of NumPy arrays.
        """
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Executes local client training for the current communication round.

        Args:
            parameters: Global model weights broadcasted by the Flower server.
            config: Server configuration dictionary containing:
                    - 'local_epochs': Number of local training epochs (default: 2)
                    - 'mu': FedProx proximal regularization coefficient (default: 0.0)
                    - 'server_round': Current communication round index

        Returns:
            Tuple of:
            - Updated local model weights (List[np.ndarray])
            - Number of training samples processed (int)
            - Diagnostic metrics dictionary (Dict[str, Scalar])
        """
        # 1. Synchronize local model with received global weights
        self.model.set_weights(parameters)
        initial_weights = copy.deepcopy(parameters)

        # 2. Extract configuration sent by server
        local_epochs = int(config.get("local_epochs", 2))
        mu = float(config.get("mu", 0.0))
        server_round = int(config.get("server_round", 0))

        # Clone global model reference for FedProx proximal penalty calculation: (mu / 2) * ||w - w_t||^2
        global_ref = copy.deepcopy(self.model) if mu > 0.0 else None

        # 3. Execute local training and measure wall-clock time
        start_time = time.perf_counter()
        loss, acc = self.trainer.train_epochs(
            dataloader=self.train_loader,
            num_epochs=local_epochs,
            global_model=global_ref,
            mu=mu
        )
        train_duration = time.perf_counter() - start_time

        # 4. Measure client parameter drift: ||w_local - w_global||_2
        updated_weights = self.model.get_weights()
        drift_norm = float(np.sqrt(sum(
            np.sum((w_loc - w_init) ** 2)
            for w_loc, w_init in zip(updated_weights, initial_weights)
        )))

        num_samples = len(self.train_loader.dataset)

        metrics: Dict[str, Scalar] = {
            "loss": float(loss),
            "accuracy": float(acc),
            "drift_l2": float(drift_norm),
            "train_time_sec": float(train_duration),
            "client_id": int(self.client_id),
            "server_round": int(server_round)
        }

        logger.info(
            f"[Client {self.client_id} | Round {server_round:02d}] "
            f"Trained {local_epochs} epochs | Loss: {loss:.4f} | Acc: {acc*100:5.2f}% | "
            f"Drift: {drift_norm:.4f} | Time: {train_duration:.2f}s"
        )

        return updated_weights, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluates the received global model on local client validation data.
        """
        self.model.set_weights(parameters)

        target_loader = self.val_loader if self.val_loader is not None else self.train_loader
        loss, acc = self.trainer.evaluate(target_loader)
        num_samples = len(target_loader.dataset)

        metrics: Dict[str, Scalar] = {
            "accuracy": float(acc),
            "client_id": int(self.client_id)
        }

        return float(loss), num_samples, metrics


def start_flower_client(
    client_id: int,
    server_address: str = "127.0.0.1:8080",
    model: Optional[Any] = None,
    train_loader: Optional[Any] = None,
    val_loader: Optional[Any] = None,
    trainer: Optional[Any] = None,
    device: str = "cpu"
) -> None:
    """
    Connects and starts a standalone Flower client process communicating over gRPC.

    Args:
        client_id: Client integer index.
        server_address: Host and port of the central Flower server (e.g. '127.0.0.1:8080').
        model: Pre-initialized TabularIoTMLPModel.
        train_loader: Local client PyTorch DataLoader.
        val_loader: Optional validation DataLoader.
        trainer: Optional FederatedClientTrainer.
        device: Computing device.
    """
    if not HAS_FLWR:
        raise ImportError("Flower (flwr) is required to run start_flower_client. Please run: pip install flwr")

    client = FlowerIoTClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer=trainer,
        device=device
    )

    logger.info(f"Connecting Flower IoT Client {client_id} to server at {server_address}...")
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client()
        )
    except Exception:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a standalone Flower IoT Client Process.")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID index.")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Flower server gRPC address.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda').")
    parser.add_argument("--partitions-dir", type=str, default="checkpoints/partitions", help="Directory containing client npz files.")
    parser.add_argument("--batch-size", type=int, default=64, help="Local mini-batch size.")
    parser.add_argument("--strategy", type=str, default="fedprox", choices=["fedavg", "fedprox"], help="Strategy name.")
    parser.add_argument("--epochs", type=int, default=2, help="Local training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Client learning rate.")
    parser.add_argument("--mu", type=float, default=0.05, help="FedProx proximal parameter mu.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Flower Client CLI: Client ID {args.client_id} connecting to {args.server_address}")

    import os
    import json
    meta_path = os.path.join(args.partitions_dir, "meta.json")
    client_file = os.path.join(args.partitions_dir, f"client_{args.client_id}.npz")
    if not os.path.exists(client_file):
        raise FileNotFoundError(f"Client partition file not found at: {client_file}")

    with open(meta_path) as f:
        meta = json.load(f)
    client_npz = np.load(client_file)

    from src.models.tabular_mlp import TabularIoTMLPModel
    from torch.utils.data import TensorDataset, DataLoader

    model = TabularIoTMLPModel(input_dim=meta["input_dim"], num_classes=meta["num_classes"])
    train_ds = TensorDataset(torch.from_numpy(client_npz["X"]), torch.from_numpy(client_npz["y"]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    start_flower_client(
        client_id=args.client_id,
        server_address=args.server_address,
        model=model,
        train_loader=train_loader,
        device=args.device
    )

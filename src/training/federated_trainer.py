"""
Federated client trainer for FL-IoT-IDS.

Contains:
- FederatedClientTrainer: Client local optimization orchestrator supporting
  both FedAvg and FedProx proximal regularization (mu) with gradient clipping.
- LocalClientTrainer: Backward-compatible alias.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 4: FedProx Proximal Regularization)
- skills/methodology-audit/SKILL.md (Pillar 2: FL Optimization & Mathematical Correctness)
"""

import copy
import logging
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
    torch = None
    DataLoader = object

from src.base.trainer import BaseTrainer

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class FederatedClientTrainer(BaseTrainer):
        """
        Federated client training orchestrator for Federated Learning rounds.
        
        Supports:
        - FedAvg: Standard SGD/Adam optimization (mu = 0.0).
        - FedProx: Proximal regularization constraint (mu > 0.0):
            L_prox = L_task + (mu / 2) * ||w - w_t||^2
        - Gradient clipping: Enforces numerical stability on edge devices.
        """
        def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: Union[str, torch.device] = "cpu",
            max_grad_norm: float = 5.0
        ):
            super().__init__(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_grad_norm=max_grad_norm
            )

        def train_epoch(
            self,
            dataloader: DataLoader,
            global_model: Optional[nn.Module] = None,
            mu: float = 0.0
        ) -> Tuple[float, float]:
            """
            Executes 1 local training epoch.

            Args:
                dataloader: Local training DataLoader.
                global_model: Initial global model at start of round (required for FedProx).
                mu: Proximal regularization parameter (mu=0.0 reduces to FedAvg).

            Returns:
                Tuple of (average_loss, accuracy).
            """
            self.model.train()
            total_loss = 0.0
            correct = 0
            total_samples = 0

            # Pre-extract global parameters if FedProx proximal penalty is active
            global_params = None
            if global_model is not None and mu > 0.0:
                global_model.to(self.device)
                global_model.eval()
                global_params = list(global_model.parameters())

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                batch_size = len(y_batch)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                task_loss = self.criterion(outputs, y_batch)

                # Compute proximal regularization for FedProx: (mu / 2) * ||w - w_t||^2
                if global_params is not None:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_params):
                        proximal_term += torch.sum((w - w_t) ** 2)
                    total_batch_loss = task_loss + (mu / 2.0) * proximal_term
                else:
                    total_batch_loss = task_loss

                total_batch_loss.backward()

                # Gradient clipping to maintain stability on edge clients
                self.clip_gradients()

                self.optimizer.step()

                total_loss += total_batch_loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_samples += batch_size

            avg_loss = total_loss / max(total_samples, 1)
            accuracy = correct / max(total_samples, 1)
            return avg_loss, accuracy

        def train_epochs(
            self,
            dataloader: DataLoader,
            num_epochs: int,
            global_model: Optional[nn.Module] = None,
            mu: float = 0.0
        ) -> Tuple[float, float]:
            """
            Trains for multiple local epochs in a single federated communication round.
            """
            epoch_loss = 0.0
            epoch_acc = 0.0
            for _ in range(num_epochs):
                epoch_loss, epoch_acc = self.train_epoch(
                    dataloader, global_model=global_model, mu=mu
                )
            return epoch_loss, epoch_acc

    # Clean alias for backward-compatibility
    LocalClientTrainer = FederatedClientTrainer

else:
    class FederatedClientTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for FederatedClientTrainer. Please install torch.")

    LocalClientTrainer = FederatedClientTrainer

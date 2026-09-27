"""
Training orchestration module for FL-IoT-IDS.

Contains:
- LocalClientTrainer: Federated local client trainer supporting both FedAvg
  and FedProx proximal regularization (mu) with gradient clipping.
- CentralizedTrainer: Standalone centralized baseline trainer with early stopping
  and validation checkpointing for Experiment E1.

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

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class LocalClientTrainer:
        """
        Local client training orchestrator for Federated Learning rounds.
        
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
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = torch.device(device)
            self.max_grad_norm = max_grad_norm
            self.model.to(self.device)

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
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
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
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

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


    class CentralizedTrainer:
        """
        Standalone centralized baseline trainer (Scenario E1).
        Includes epoch loop, validation tracking, early stopping, and history recording.
        """
        def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: Union[str, torch.device] = "cpu",
            max_grad_norm: float = 5.0
        ):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = torch.device(device)
            self.max_grad_norm = max_grad_norm
            self.model.to(self.device)
            self.best_model_weights = None
            self.best_val_loss = float("inf")

        def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 20,
            patience: int = 5
        ) -> Dict[str, List[float]]:
            """
            Executes full centralized training with optional early stopping.

            Returns:
                History dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
            """
            history: Dict[str, List[float]] = {
                "train_loss": [], "train_acc": [],
                "val_loss": [], "val_acc": []
            }
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                # 1. Train epoch
                self.model.train()
                t_loss, t_correct, t_total = 0.0, 0, 0
                for X_b, y_b in train_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(X_b)
                    loss = self.criterion(out, y_b)
                    loss.backward()
                    if self.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    t_loss += loss.item() * len(y_b)
                    t_correct += (out.argmax(dim=1) == y_b).sum().item()
                    t_total += len(y_b)

                train_loss = t_loss / max(t_total, 1)
                train_acc = t_correct / max(t_total, 1)
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)

                # 2. Validation step
                val_loss, val_acc = 0.0, 0.0
                if val_loader is not None:
                    self.model.eval()
                    v_loss, v_correct, v_total = 0.0, 0, 0
                    with torch.no_grad():
                        for X_v, y_v in val_loader:
                            X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                            out_v = self.model(X_v)
                            loss_v = self.criterion(out_v, y_v)
                            v_loss += loss_v.item() * len(y_v)
                            v_correct += (out_v.argmax(dim=1) == y_v).sum().item()
                            v_total += len(y_v)
                    val_loss = v_loss / max(v_total, 1)
                    val_acc = v_correct / max(v_total, 1)
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(val_acc)

                    # Checkpoint best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_weights = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break

                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} - "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
                )

            # Restore best weights if available
            if self.best_model_weights is not None:
                self.model.load_state_dict(self.best_model_weights)
                logger.info(f"Restored model weights from best validation epoch (Loss: {self.best_val_loss:.4f})")

            return history

else:
    class LocalClientTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LocalClientTrainer. Please install torch.")

    class CentralizedTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CentralizedTrainer. Please install torch.")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping trainer verification.")
    else:
        print("--- Running Trainer Module Verification ---")
        from torch.utils.data import TensorDataset

        mock_model = nn.Sequential(nn.Linear(39, 16), nn.ReLU(), nn.Linear(16, 8))
        X_mock = torch.randn(64, 39)
        y_mock = torch.randint(0, 8, (64,))
        loader = DataLoader(TensorDataset(X_mock, y_mock), batch_size=32)

        opt = torch.optim.Adam(mock_model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()

        # Test LocalClientTrainer
        trainer = LocalClientTrainer(mock_model, opt, crit)
        loss, acc = trainer.train_epoch(loader, mu=0.0)
        assert loss > 0.0, "Training loss must be positive!"
        print(f"LocalClientTrainer 1-epoch Loss: {loss:.4f}, Acc: {acc*100:.1f}%")

        # Test FedProx proximal penalty
        global_m = copy.deepcopy(mock_model)
        loss_prox, _ = trainer.train_epoch(loader, global_model=global_m, mu=0.5)
        print(f"FedProx (mu=0.5) Loss: {loss_prox:.4f}")

        print("✅ Trainer module verification passed successfully!")

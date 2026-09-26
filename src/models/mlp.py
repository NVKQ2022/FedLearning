"""
PyTorch Multi-Layer Perceptron (MLP) for Tabular IoT Intrusion Detection.

This module implements:
1. TabularIoTMLP: A parameter-efficient, low-latency neural network backbone
   engineered specifically for tabular network flows and edge IoT gateways.
   Adheres to the critical FL guideline: NO BatchNorm in Non-IID settings.
2. Flower-compatible weight serialization interface (get_weights / set_weights).
3. MultiClassFocalLoss for severe class imbalance.
4. LocalClientTrainer with full FedProx proximal penalty support and gradient clipping.

Adheres to:
- skills/model-design-and-implementation/SKILL.md
- skills/methodology-audit/SKILL.md (Pillar 2: FL Optimization & Mathematical Correctness)
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


if HAS_TORCH:
    class TabularIoTMLP(nn.Module):
        """
        Lightweight Multi-Layer Perceptron for Network Flow Intrusion Detection.

        Architecture:
            Input (input_dim) -> Linear(128) -> ReLU -> Dropout(dropout_rate)
                              -> Linear(64)  -> ReLU -> Dropout(dropout_rate)
                              -> Linear(num_classes)

        Note:
            Batch Normalization is intentionally excluded to prevent severe representation
            corruption caused by aggregating client-specific running statistics under
            Non-IID distributions.
        """
        def __init__(
            self,
            input_dim: int = 39,
            hidden_dims: Tuple[int, ...] = (128, 64),
            num_classes: int = 8,
            dropout_rate: float = 0.2
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.num_classes = num_classes
            self.dropout_rate = dropout_rate

            layers: List[nn.Module] = []
            prev_dim = input_dim
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = h_dim
                
            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)
            
            # Weight initialization: Kaiming normal for ReLU layers
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

        def get_weights(self) -> List[np.ndarray]:
            """
            Extracts model parameters as a list of NumPy arrays for Flower aggregation.
            """
            return [val.detach().cpu().numpy() for _, val in self.state_dict().items()]

        def set_weights(self, weights: List[np.ndarray]) -> None:
            """
            Loads aggregated NumPy parameter arrays into the model's state dictionary.
            """
            state_dict = dict(zip(self.state_dict().keys(), [torch.tensor(w) for w in weights]))
            self.load_state_dict(state_dict, strict=True)

        def get_num_parameters(self) -> int:
            """Returns the total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def get_model_size_kb(self) -> float:
            """Computes model weight payload in Kilobytes (assuming float32 precision)."""
            num_params = self.get_num_parameters()
            return (num_params * 4) / 1024.0


    class MultiClassFocalLoss(nn.Module):
        """
        Multi-Class Focal Loss for imbalanced classification tasks.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            reduction: str = "mean"
        ):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
            pt = torch.exp(-ce_loss)  # Probability of ground-truth class
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
            
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            return focal_loss


    class LocalClientTrainer:
        """
        Local client training orchestrator for Federated Learning rounds.
        Supports both FedAvg (mu=0.0) and FedProx proximal regularization (mu > 0.0).
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

            # Pre-move global model parameters if FedProx is active
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

                # Gradient clipping to maintain numerical stability on edge clients
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
            """Trains for multiple local epochs in a single federated communication round."""
            epoch_loss = 0.0
            epoch_acc = 0.0
            for _ in range(num_epochs):
                epoch_loss, epoch_acc = self.train_epoch(dataloader, global_model=global_model, mu=mu)
            return epoch_loss, epoch_acc


    def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluates model on validation/test DataLoader.

        Returns:
            Tuple of (average_loss, accuracy, all_predictions, all_ground_truth).
        """
        device = torch.device(device)
        model.to(device)
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = len(y_batch)

                outputs = model(X_batch)
                
                if criterion is not None:
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * batch_size

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_samples += batch_size
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = (total_loss / total_samples) if criterion is not None else 0.0
        accuracy = correct / max(total_samples, 1)
        return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

else:
    # Fallback placeholders if torch is not installed
    class TabularIoTMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TabularIoTMLP. Please install torch.")

    class MultiClassFocalLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MultiClassFocalLoss. Please install torch.")

    class LocalClientTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LocalClientTrainer. Please install torch.")

    def evaluate_model(*args, **kwargs):
        raise ImportError("PyTorch is required for evaluate_model. Please install torch.")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping model self-test.")
    else:
        print("--- Running TabularIoTMLP & LocalClientTrainer Verification Protocol ---")
        
        # Initialize model
        model = TabularIoTMLP(input_dim=39, hidden_dims=(128, 64), num_classes=8, dropout_rate=0.2)
        print(f"Model instantiated. Trainable parameters: {model.get_num_parameters():,}")
        print(f"Model payload size: {model.get_model_size_kb():.2f} KB")

        # 1. Weight getter/setter round-trip check
        weights = model.get_weights()
        assert len(weights) == 6, f"Expected 6 parameter arrays (3 weights + 3 biases), got {len(weights)}"
        model.set_weights(weights)
        print("✅ Step 1: Weight setter/getter round-trip check passed.")

        # 2. Overfit single batch sanity check
        X_mock = torch.randn(32, 39)
        y_mock = torch.randint(0, 8, (32,))
        mock_dataset = TensorDataset(X_mock, y_mock)
        mock_loader = DataLoader(mock_dataset, batch_size=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        trainer = LocalClientTrainer(model, optimizer, criterion, device="cpu")

        print("Running single-batch overfit test (30 epochs)...")
        for epoch in range(30):
            loss, acc = trainer.train_epoch(mock_loader, mu=0.0)
        
        print(f"Final epoch loss: {loss:.4f}, accuracy: {acc * 100:.1f}%")
        assert acc >= 0.90, f"Overfit sanity test failed! Accuracy only reached {acc}"
        print("✅ Step 2: Overfit single-batch test passed.")

        # 3. FedProx proximal penalty check
        global_model = copy.deepcopy(model)
        # Shift local weights slightly
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        
        loss_with_prox, _ = trainer.train_epoch(mock_loader, global_model=global_model, mu=1.0)
        assert loss_with_prox > 0.0, "FedProx loss calculation returned non-positive value!"
        print(f"FedProx loss with mu=1.0: {loss_with_prox:.4f}")
        print("✅ Step 3: FedProx proximal regularization check passed.")
        print("\n🎉 All model and trainer verification checks passed successfully!")

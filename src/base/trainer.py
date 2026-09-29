"""
Abstract base class for model trainers in centralized and federated learning workflows.

Provides:
- BaseTrainer: Core training lifecycle, evaluation routines, gradient clipping,
  device dispatching, and checkpoint management.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 4)
- skills/methodology-audit/SKILL.md (Pillar 2: Mathematical Correctness)
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

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

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class BaseTrainer(ABC):
        """
        Abstract base trainer establishing a unified contract for centralized and federated trainers.

        Subclasses must implement:
        - train_epoch(dataloader, **kwargs): Executes a single epoch of forward/backward optimization.

        Provides out-of-the-box:
        - evaluate(dataloader): Standardized zero-grad validation/test evaluation with loss and accuracy.
        - clip_gradients(): Gradient clipping using max_grad_norm to prevent gradient explosion.
        - save_checkpoint() / load_checkpoint(): Comprehensive model and optimizer state serialization.
        - reset_history(): History dictionary clearing.
        """
        def __init__(
            self,
            model: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            criterion: Optional[nn.Module] = None,
            device: Union[str, torch.device] = "cpu",
            max_grad_norm: float = 5.0
        ):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = torch.device(device)
            self.max_grad_norm = max_grad_norm

            # Automatically dispatch model to target device
            self.model.to(self.device)

            self.history: Dict[str, List[float]] = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }

        @abstractmethod
        def train_epoch(self, dataloader: DataLoader, **kwargs: Any) -> Tuple[float, float]:
            """
            Executes one full epoch of model training.

            Args:
                dataloader: DataLoader yielding (features, labels) batches.
                **kwargs: Optional training arguments (e.g. global_model, mu for FedProx).

            Returns:
                Tuple of (average_loss, accuracy_fraction).
            """
            pass

        def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
            """
            Standardized evaluation across a DataLoader in inference mode.

            Args:
                dataloader: Validation or test DataLoader.

            Returns:
                Tuple of (average_loss, accuracy_fraction).
            """
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total_samples = 0

            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    batch_size = len(y_batch)

                    outputs = self.model(X_batch)
                    if self.criterion is not None:
                        loss = self.criterion(outputs, y_batch)
                        total_loss += float(loss.item()) * batch_size

                    preds = outputs.argmax(dim=1)
                    correct += int((preds == y_batch).sum().item())
                    total_samples += batch_size

            avg_loss = total_loss / max(1, total_samples) if self.criterion is not None else 0.0
            avg_acc = correct / max(1, total_samples)
            return avg_loss, avg_acc

        def clip_gradients(self) -> float:
            """
            Enforces gradient clipping up to self.max_grad_norm.
            
            Returns:
                Total norm of the model gradients before clipping.
            """
            if self.max_grad_norm > 0.0:
                return float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item())
            return 0.0

        def save_checkpoint(self, filepath: str, **metadata: Any) -> None:
            """
            Serializes model weights, optimizer state, and arbitrary metadata.
            """
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "history": self.history,
                "metadata": metadata
            }
            torch.save(checkpoint, filepath)
            logger.debug(f"Saved trainer checkpoint to {filepath}")

        def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
            """
            Restores model and optimizer states from a checkpoint file.

            Returns:
                Dictionary of metadata saved in the checkpoint.
            """
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer and checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "history" in checkpoint:
                self.history = checkpoint["history"]
            logger.debug(f"Restored trainer checkpoint from {filepath}")
            return checkpoint.get("metadata", {})

        def reset_history(self) -> None:
            """Resets metric history tracking."""
            self.history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }

else:
    class BaseTrainer(ABC):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for BaseTrainer. Please install torch.")

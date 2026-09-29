"""
Abstract base classes for neural network models in centralized and federated learning.

Provides:
- BaseModel: Universal PyTorch model contract with parameter tracking, serialization,
  and architecture diagnostics.
- BaseFederatedModel: Extends BaseModel with parameter list extraction (get_weights),
  loading (set_weights), parameter delta tracking, and L2 divergence diagnostics
  essential for Flower/Federated Learning workflows.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 1 & 4)
- skills/methodology-audit/SKILL.md (Pillar 2: Mathematical Correctness)
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class BaseModel(nn.Module, ABC):
        """
        Abstract base class for all neural network models in the project.
        
        Subclasses must implement:
        - forward(x, **kwargs): The forward computation graph.
        
        Provides out-of-the-box:
        - get_num_parameters(): Total trainable and non-trainable parameter count.
        - get_model_size_kb(): Weight footprint in Kilobytes.
        - save_weights() / load_weights(): Clean state_dict serialization.
        - freeze() / unfreeze(): Layer parameter gradient toggling.
        - summary(): Formatted architecture inspection string.
        """

        @abstractmethod
        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """
            Executes model forward inference pass.
            
            Args:
                x: Input tensor of shape (batch_size, input_dim).
                **kwargs: Optional additional inputs (e.g. attention masks).

            Returns:
                Output logit tensor of shape (batch_size, num_classes).
            """
            pass

        def get_num_parameters(self, trainable_only: bool = True) -> int:
            """
            Returns the count of model parameters.

            Args:
                trainable_only: If True, counts only parameters with requires_grad=True.
            """
            if trainable_only:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
            return sum(p.numel() for p in self.parameters())

        def get_model_size_kb(self, precision_bytes: int = 4) -> float:
            """
            Computes memory footprint of model parameters in Kilobytes.
            
            Args:
                precision_bytes: Bytes per parameter (4 for float32, 2 for float16).
            """
            num_params = self.get_num_parameters(trainable_only=False)
            return (num_params * precision_bytes) / 1024.0

        def save_weights(self, filepath: str) -> None:
            """
            Serializes model state_dict to disk.
            """
            torch.save(self.state_dict(), filepath)
            logger.debug(f"Saved model weights to {filepath}")

        def load_weights(
            self,
            filepath: str,
            map_location: Optional[Union[str, torch.device]] = None,
            strict: bool = True
        ) -> None:
            """
            Loads model state_dict from disk with safety flags.
            """
            state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
            self.load_state_dict(state_dict, strict=strict)
            logger.debug(f"Loaded model weights from {filepath}")

        def freeze(self) -> None:
            """Freezes all model parameters (disables gradient computation)."""
            for param in self.parameters():
                param.requires_grad = False

        def unfreeze(self) -> None:
            """Unfreezes all model parameters (enables gradient computation)."""
            for param in self.parameters():
                param.requires_grad = True

        def summary(self) -> str:
            """Returns a human-readable summary of the model architecture and payload size."""
            lines = [
                f"Model: {self.__class__.__name__}",
                f"Trainable Parameters:     {self.get_num_parameters(trainable_only=True):,}",
                f"Non-Trainable Parameters: {self.get_num_parameters(trainable_only=False) - self.get_num_parameters(trainable_only=True):,}",
                f"Total Parameters:         {self.get_num_parameters(trainable_only=False):,}",
                f"Estimated Payload Size:   {self.get_model_size_kb():.2f} KB (float32)",
            ]
            return "\n".join(lines)


    class BaseFederatedModel(BaseModel):
        """
        Abstract base class for models designed for Federated Learning.
        
        Extends BaseModel with:
        - NumPy parameter extraction and injection (compatible with Flower / FedAvg).
        - Cloned PyTorch state dictionary export/import.
        - Parameter delta and L2 divergence diagnostics for client drift and proximal regularization.
        """

        def get_weights(self) -> List[np.ndarray]:
            """
            Extracts model parameters as a list of NumPy arrays.
            Used by Flower client get_parameters() and server aggregation routines.
            """
            return [val.detach().cpu().numpy() for _, val in self.state_dict().items()]

        def set_weights(self, weights: List[np.ndarray], strict: bool = True) -> None:
            """
            Injects aggregated NumPy parameter arrays into the model state dictionary.
            
            Args:
                weights: List of NumPy parameter arrays matching self.state_dict() order and shapes.
                strict: Enforces exact key match.
            """
            keys = list(self.state_dict().keys())
            if len(keys) != len(weights):
                raise ValueError(
                    f"Parameter count mismatch: model has {len(keys)} layers, but received {len(weights)} arrays."
                )
            state_dict = dict(zip(keys, [torch.tensor(w) for w in weights]))
            self.load_state_dict(state_dict, strict=strict)

        def get_weights_dict(self) -> Dict[str, np.ndarray]:
            """
            Extracts model parameters as a dictionary mapping layer names to NumPy arrays.
            """
            return {name: param.detach().cpu().numpy() for name, param in self.state_dict().items()}

        def set_weights_dict(self, weights_dict: Dict[str, np.ndarray], strict: bool = True) -> None:
            """
            Loads parameters from a dictionary mapping layer names to NumPy arrays.
            """
            state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
            self.load_state_dict(state_dict, strict=strict)

        def get_weights_tensor(self) -> Dict[str, torch.Tensor]:
            """
            Extracts a deep copy of model state dictionary tensors on CPU.
            """
            return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        def set_weights_tensor(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
            """
            Loads model weights directly from a PyTorch state dictionary.
            """
            self.load_state_dict(state_dict, strict=strict)

        def compute_parameter_delta(self, reference_weights: List[np.ndarray]) -> List[np.ndarray]:
            """
            Computes parameter delta Delta w = w_current - w_reference.
            
            Args:
                reference_weights: List of reference NumPy arrays (e.g. global model weights before round).

            Returns:
                List of parameter deltas.
            """
            current_weights = self.get_weights()
            return [curr - ref for curr, ref in zip(current_weights, reference_weights)]

        def compute_l2_norm_delta(self, reference_model: nn.Module) -> float:
            """
            Computes Euclidean distance ||w_current - w_ref||_2 across all trainable parameters.
            Used to monitor client drift in Non-IID federated optimization.
            """
            diff_sq_sum = 0.0
            ref_params = dict(reference_model.named_parameters())
            for name, param in self.named_parameters():
                if name in ref_params:
                    diff = param.data - ref_params[name].data.to(param.device)
                    diff_sq_sum += float(torch.sum(diff ** 2).item())
            return float(np.sqrt(diff_sq_sum))

else:
    # Graceful fallback when PyTorch is not available
    class BaseModel(ABC):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for BaseModel. Please install torch.")

    class BaseFederatedModel(BaseModel):  # type: ignore[no-redef]
        pass

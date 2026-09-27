"""
Loss functions for imbalanced multi-class intrusion detection.

Includes:
- MultiClassFocalLoss: Down-weights easy well-classified background samples
  and focuses gradients on hard minority attacks (e.g. Web-based, Brute-force).
- build_loss_function: Factory function returning standard CE, weighted CE, or Focal Loss.

Adheres to:
- skills/model-design-and-implementation/SKILL.md (Principle 3: Imbalance-Aware Loss Functions)
- skills/dataset-analysis-and-strategy/SKILL.md (Step 4: Target Class Imbalance Strategy)
"""

import logging
from typing import Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class MultiClassFocalLoss(nn.Module):
        """
        Multi-Class Focal Loss for imbalanced classification tasks.
        
        Formula:
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            
        Args:
            alpha: Optional 1D Tensor of shape (num_classes,) containing per-class weighting factors.
            gamma: Focusing parameter (default: 2.0). Higher gamma increases focus on hard/rare examples.
            reduction: Reduction mode ('mean', 'sum', or 'none'). Default is 'mean'.
        """
        def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            reduction: str = "mean"
        ):
            super().__init__()
            if alpha is not None:
                if not isinstance(alpha, torch.Tensor):
                    alpha = torch.tensor(alpha, dtype=torch.float32)
                self.register_buffer("alpha", alpha)
            else:
                self.alpha = None
                
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                inputs: Predicted logits of shape (batch_size, num_classes).
                targets: Ground truth class indices of shape (batch_size,).
            """
            # Compute true class probabilities via log_softmax
            log_pt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(log_pt)

            # Extract probability and log probability of ground-truth target class
            log_pt_target = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt_target = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Modulating factor (1 - p_t)^gamma focuses loss on hard examples
            focal_modulator = (1.0 - pt_target) ** self.gamma

            # Apply per-class weighting if specified
            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                focal_loss = -alpha_t * focal_modulator * log_pt_target
            else:
                focal_loss = -focal_modulator * log_pt_target

            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            return focal_loss


    def build_loss_function(
        loss_type: str = "cross_entropy",
        class_weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        gamma: float = 2.0,
        device: Union[str, torch.device] = "cpu"
    ) -> nn.Module:
        """
        Factory function to instantiate loss functions with consistent device placement.

        Args:
            loss_type: 'cross_entropy', 'weighted_ce', or 'focal_loss'.
            class_weights: Optional class weight array/tensor for handling imbalance.
            gamma: Focusing parameter for Focal Loss.
            device: Target torch device ('cpu' or 'cuda').

        Returns:
            Configured nn.Module loss criterion.
        """
        device = torch.device(device)
        weights_tensor = None
        if class_weights is not None:
            if isinstance(class_weights, np.ndarray):
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            elif isinstance(class_weights, torch.Tensor):
                weights_tensor = class_weights.to(dtype=torch.float32, device=device)

        loss_type_lower = loss_type.lower()
        if loss_type_lower == "focal_loss":
            logger.info(f"Instantiating MultiClassFocalLoss (gamma={gamma}, weighted={weights_tensor is not None})")
            criterion = MultiClassFocalLoss(alpha=weights_tensor, gamma=gamma, reduction="mean")
        elif loss_type_lower in ["weighted_ce", "weighted_cross_entropy"]:
            logger.info("Instantiating Weighted CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        elif loss_type_lower in ["cross_entropy", "ce"]:
            logger.info("Instantiating Standard CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose from 'cross_entropy', 'weighted_ce', 'focal_loss'.")

        return criterion.to(device)

else:
    class MultiClassFocalLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MultiClassFocalLoss. Please install torch.")

    def build_loss_function(*args, **kwargs):
        raise ImportError("PyTorch is required for build_loss_function. Please install torch.")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping focal loss verification.")
    else:
        print("--- Running MultiClassFocalLoss Verification ---")
        num_classes = 8
        batch_size = 16
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        weights = torch.ones(num_classes)
        weights[7] = 10.0  # Heavy weight on rare attack class 7

        # 1. Unweighted Focal Loss
        fl_unweighted = MultiClassFocalLoss(gamma=2.0)
        loss_val = fl_unweighted(logits, targets)
        assert loss_val.item() > 0, "Focal loss must be strictly positive!"
        print(f"Unweighted Focal Loss: {loss_val.item():.4f}")

        # 2. Weighted Focal Loss
        fl_weighted = MultiClassFocalLoss(alpha=weights, gamma=2.0)
        loss_w_val = fl_weighted(logits, targets)
        assert loss_w_val.item() > 0, "Weighted Focal loss must be strictly positive!"
        print(f"Weighted Focal Loss: {loss_w_val.item():.4f}")

        # 3. Factory function test
        crit = build_loss_function("focal_loss", class_weights=weights.numpy(), gamma=2.0)
        print("✅ Loss module verification passed successfully!")

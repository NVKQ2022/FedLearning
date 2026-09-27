"""
Optimizer and learning rate scheduler factory for FL-IoT-IDS.

Provides clean, standardized instantiation of optimizers and schedulers
for both centralized baselines and local client federated trainers.

Adheres to:
- skills/model-design-and-implementation/SKILL.md
- skills/methodology-audit/SKILL.md (Pillar 2: FL Optimization & Mathematical Correctness)
"""

import logging
from typing import Optional, Union

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam, AdamW, SGD, Optimizer
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, _LRScheduler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Optimizer = object
    _LRScheduler = object

logger = logging.getLogger(__name__)


if HAS_TORCH:
    def build_optimizer(
        model: nn.Module,
        optimizer_type: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9
    ) -> Optimizer:
        """
        Builds a PyTorch optimizer for the given model parameters.

        Args:
            model: Neural network model containing trainable parameters.
            optimizer_type: 'adam', 'adamw', or 'sgd'.
            lr: Learning rate.
            weight_decay: L2 regularization penalty coefficient.
            momentum: Momentum factor for SGD.

        Returns:
            Configured PyTorch Optimizer instance.
        """
        opt_name = optimizer_type.lower()
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if opt_name == "adam":
            optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            optimizer = SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"Unsupported optimizer_type: '{optimizer_type}'. Choose from 'adam', 'adamw', 'sgd'."
            )

        logger.debug(f"Instantiated {opt_name.upper()} optimizer (lr={lr}, weight_decay={weight_decay})")
        return optimizer


    def build_scheduler(
        optimizer: Optimizer,
        scheduler_type: str = "cosine",
        total_steps: int = 30,
        min_lr: float = 1e-5,
        step_size: int = 10,
        gamma: float = 0.5
    ) -> Optional[_LRScheduler]:
        """
        Builds an optional learning rate scheduler.

        Args:
            optimizer: Configured PyTorch optimizer.
            scheduler_type: 'cosine', 'step', 'plateau', or 'none'.
            total_steps: Total communication rounds or training epochs for cosine schedule.
            min_lr: Minimum learning rate floor for cosine annealing.
            step_size: Period of learning rate decay for StepLR.
            gamma: Multiplicative factor of learning rate decay for StepLR.

        Returns:
            Learning rate scheduler instance or None.
        """
        sched_name = scheduler_type.lower()
        if sched_name in ["none", "null", "disabled"]:
            return None
        elif sched_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
        elif sched_name == "step":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sched_name == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=3, min_lr=min_lr)
        else:
            raise ValueError(
                f"Unsupported scheduler_type: '{scheduler_type}'. Choose from 'cosine', 'step', 'plateau', 'none'."
            )

        logger.debug(f"Instantiated {sched_name.upper()} LR scheduler")
        return scheduler


    def get_current_lr(optimizer: Optimizer) -> float:
        """Extracts the current learning rate from the first parameter group."""
        for param_group in optimizer.param_groups:
            return float(param_group["lr"])
        return 0.0

else:
    def build_optimizer(*args, **kwargs):
        raise ImportError("PyTorch is required for build_optimizer. Please install torch.")

    def build_scheduler(*args, **kwargs):
        raise ImportError("PyTorch is required for build_scheduler. Please install torch.")

    def get_current_lr(*args, **kwargs):
        raise ImportError("PyTorch is required for get_current_lr. Please install torch.")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping optimizer verification.")
    else:
        print("--- Running Optimizer Factory Verification ---")
        mock_model = nn.Linear(39, 8)
        
        # Test Adam
        opt_adam = build_optimizer(mock_model, "adam", lr=0.001)
        assert get_current_lr(opt_adam) == 0.001
        print(f"Adam Optimizer LR: {get_current_lr(opt_adam)}")

        # Test SGD with Cosine Scheduler
        opt_sgd = build_optimizer(mock_model, "sgd", lr=0.01, momentum=0.9)
        sched = build_scheduler(opt_sgd, "cosine", total_steps=10, min_lr=0.0001)
        assert sched is not None
        sched.step()
        print(f"SGD LR after step 1: {get_current_lr(opt_sgd):.6f}")

        print("✅ Optimizer module verification passed successfully!")

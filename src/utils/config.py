"""
Centralized Hyperparameter Configuration Module for FL-IoT-IDS.

Provides a unified, type-hinted configuration schema serving as the
single source of truth for experiment reproducibility, dataset preprocessing,
model architectures, training loops, and federated simulation settings.

Adheres to:
- skills/experiment-orchestration/SKILL.md (Step 2: Declarative Configuration Schema)
- skills/model-design-and-implementation/SKILL.md (Principle 1 & 4)
- skills/methodology-audit/SKILL.md (Pillar 2: Reproducibility & Mathematical Correctness)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Unified hyperparameter configuration container.
    """
    # 1. Experiment & System Reproducibility
    experiment_name: str = "centralized_e1_baseline"
    seed: int = 42
    device: str = "auto"  # 'auto', 'cuda', or 'cpu'

    # 2. Dataset & Storage Paths
    gdrive_csv: str = "/content/drive/MyDrive/NguyenVietKyQuanKLTN/datasets/merged_CICIOT2023_data.csv"
    alt_gdrive_csv: str = "/content/drive/MyDrive/FedLearning/merged_CICIOT2023_data.csv"
    target_csv_path: str = "/content/FedLearning/datasets/CICIOT2023/merged_CICIOT2023_data.csv"
    checkpoints_dir: str = "/content/drive/MyDrive/NguyenVietKyQuanKLTN/checkpoints"
    reports_dir: str = "/content/drive/MyDrive/NguyenVietKyQuanKLTN/reports"

    # 3. Data Preprocessing Hyperparameters
    sample_size: Optional[int] = 100000  # Set None for full 5.11M rows, or 50000/100000 for rapid execution
    test_size: float = 0.2               # 20% holdout test set
    val_size: float = 0.1                # 10% validation set
    scaler_type: str = "robust"          # 'robust' or 'standard'
    batch_size: int = 128

    # 4. Neural Network Architecture Hyperparameters
    input_dim: int = 39
    hidden_dims: Tuple[int, ...] = (128, 64)  # (128, 64) or (256, 128, 64)
    num_classes: int = 8
    dropout_rate: float = 0.2

    # 5. Loss Function & Class Imbalance Strategy
    loss_type: str = "focal_loss"            # 'focal_loss', 'cross_entropy', 'weighted_ce'
    class_weight_strategy: str = "balanced"  # 'balanced', 'sqrt_balanced', or 'none'
    focal_gamma: float = 2.0                 # Focusing parameter gamma (e.g. 1.5, 2.0)

    # 6. Optimization & Training Loop Hyperparameters
    optimizer_type: str = "adamw"            # 'adamw', 'adam', or 'sgd'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4
    max_grad_norm: float = 5.0
    monitor_metric: str = "val_loss"         # 'val_loss' (default) or 'val_acc'
    verbose: bool = True

    # 7. Federated Learning Hyperparameters (Scenarios E2, E4, E5)
    num_clients: int = 5
    num_rounds: int = 8
    local_epochs: int = 2
    mu: float = 0.05                         # FedProx proximal coefficient (0.0 for FedAvg)
    dirichlet_alpha: float = 0.1             # Dirichlet heterogeneity parameter (0.1 severe, 0.5 moderate)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration dataclass to a standard Python dictionary."""
        d = asdict(self)
        # Ensure tuples are serializable
        d["hidden_dims"] = list(self.hidden_dims)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiates ExperimentConfig from a dictionary, safely handling tuples."""
        data_copy = dict(data)
        if "hidden_dims" in data_copy and isinstance(data_copy["hidden_dims"], list):
            data_copy["hidden_dims"] = tuple(data_copy["hidden_dims"])
        return cls(**data_copy)

    def save_json(self, filepath: str) -> None:
        """Persists hyperparameter configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Hyperparameter configuration saved to {filepath}")

    def save_yaml(self, filepath: str) -> None:
        """Persists hyperparameter configuration to a YAML file if PyYAML is available."""
        if not HAS_YAML:
            self.save_json(filepath.replace(".yaml", ".json").replace(".yml", ".json"))
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Hyperparameter configuration saved to {filepath}")

    def display(self) -> None:
        """Prints a structured summary table of all active hyperparameters."""
        print("=" * 70)
        print(f"⚙️ HYPERPARAMETER CONFIGURATION: {self.experiment_name.upper()}")
        print("=" * 70)
        groups = {
            "System & Reproducibility": [
                ("Random Seed", self.seed),
                ("Target Device", self.device),
            ],
            "Dataset & Preprocessing": [
                ("Sample Size", f"{self.sample_size:,}" if self.sample_size else "Full Dataset (5.11M)"),
                ("Holdout Test Size", f"{self.test_size * 100:.0f}%"),
                ("Validation Size", f"{self.val_size * 100:.0f}%"),
                ("Scaler Type", self.scaler_type.capitalize()),
                ("Batch Size", self.batch_size),
            ],
            "Model Architecture": [
                ("Input Dimension", self.input_dim),
                ("Hidden Dimensions", " -> ".join(map(str, self.hidden_dims))),
                ("Output Classes", self.num_classes),
                ("Dropout Rate", self.dropout_rate),
            ],
            "Loss & Imbalance Strategy": [
                ("Loss Function", self.loss_type),
                ("Class Weight Strategy", self.class_weight_strategy),
                ("Focal Gamma", self.focal_gamma if self.loss_type == "focal_loss" else "N/A"),
            ],
            "Training & Checkpointing": [
                ("Optimizer", self.optimizer_type.upper()),
                ("Learning Rate", self.learning_rate),
                ("Weight Decay", self.weight_decay),
                ("Max Epochs", self.epochs),
                ("Early Stopping Patience", self.patience),
                ("Monitored Metric", f"{self.monitor_metric} {'(min)' if self.monitor_metric == 'val_loss' else '(max)'}"),
                ("Max Gradient Norm", self.max_grad_norm),
            ],
            "Federated Simulation": [
                ("Number of Clients", self.num_clients),
                ("Communication Rounds", self.num_rounds),
                ("Local Client Epochs", self.local_epochs),
                ("FedProx Mu (μ)", self.mu),
                ("Dirichlet Alpha (α)", self.dirichlet_alpha),
            ],
        }

        for group_name, params in groups.items():
            print(f"\n📂 {group_name}:")
            for name, val in params:
                print(f"   • {name:<26}: {val}")
        print("=" * 70)


if __name__ == "__main__":
    cfg = ExperimentConfig()
    cfg.display()
    print("Configuration module test passed!")

"""
Centralized Hyperparameter Configuration Module for FL-IoT-IDS.

Provides a modular, type-hinted configuration architecture organized into 5 domain sub-configs:
1. Dataset & Preprocessing (DataConfig / DatasetPreprocessingConfig)
2. Architecture & Regularization (ModelConfig / ArchitectureConfig / ArchitectureRegularizationConfig)
3. Loss Function (LossConfig / LossFunctionConfig)
4. Optimization & Training (OptimizerConfig / OptimizationConfig)
5. Federated Parameters (FederatedConfig / FederatedParametersConfig)

Aggregated by the unified Experiment (ExperimentConfig) container with full backward-compatible
attribute delegation for both modular and flat access patterns.

Adheres to:
- skills/experiment-orchestration/SKILL.md (Step 2: Declarative Configuration Schema)
- skills/model-design-and-implementation/SKILL.md (Principle 1 & 4)
- skills/methodology-audit/SKILL.md (Pillar 2: Reproducibility & Mathematical Correctness)
"""

import copy
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
class DataConfig:
    """Dataset & Preprocessing configuration."""
    gdrive_csv: str = "/content/drive/MyDrive/NguyenVietKyQuanKLTN/datasets/merged_CICIOT2023_data.csv"
    alt_gdrive_csv: str = "/content/drive/MyDrive/FedLearning/merged_CICIOT2023_data.csv"
    target_csv_path: str = "/content/FedLearning/datasets/CICIOT2023/merged_CICIOT2023_data.csv"
    checkpoints_dir: str = "/content/drive/MyDrive/NguyenVietKyQuanKLTN/checkpoints"
    reports_dir: str = "/content/drive/MyDrive/NguyenVietKyQuanKLTN/reports"
    sample_size: Optional[int] = 100000  # Set None for full 5.11M rows, or 50000/100000 for rapid execution
    test_size: float = 0.2               # 20% holdout test set
    val_size: float = 0.1                # 10% validation set
    scaler_type: str = "robust"          # 'robust' or 'standard'
    batch_size: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Architecture & Regularization configuration."""
    input_dim: int = 39
    hidden_dims: Union[Tuple[int, ...], List[int]] = (128, 64)  # (128, 64) or (256, 128, 64)
    num_classes: int = 8
    dropout_rate: float = 0.2

    def __post_init__(self):
        if isinstance(self.hidden_dims, list):
            self.hidden_dims = tuple(self.hidden_dims)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d


@dataclass
class LossConfig:
    """Loss Function configuration."""
    loss_type: str = "focal_loss"            # 'focal_loss', 'cross_entropy', 'weighted_ce'
    class_weight_strategy: str = "balanced"  # 'balanced', 'sqrt_balanced', or 'none'
    focal_gamma: float = 2.0                 # Focusing parameter gamma (e.g. 1.5, 2.0)
    use_class_weights: bool = True           # Convenience boolean flag for class weights

    def __post_init__(self):
        if not self.use_class_weights:
            self.class_weight_strategy = "none"
        elif self.class_weight_strategy == "none" and self.use_class_weights:
            self.class_weight_strategy = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimization configuration."""
    optimizer_type: str = "adamw"            # 'adamw', 'adam', or 'sgd'
    learning_rate: float = 1e-3
    lr: Optional[float] = None               # Backward-compatible alias for learning_rate
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4
    max_grad_norm: float = 5.0
    monitor_metric: str = "val_loss"         # 'val_loss' (default) or 'val_acc'
    save_top_k: int = 3                      # Number of best model checkpoints to keep
    verbose: bool = True

    def __post_init__(self):
        if self.lr is not None:
            self.learning_rate = self.lr
        else:
            self.lr = self.learning_rate

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedConfig:
    """Federated Parameters configuration."""
    num_clients: int = 5
    num_rounds: int = 8
    local_epochs: int = 2
    mu: float = 0.0                          # FedProx proximal coefficient (0.0 for FedAvg, 0.05 for FedProx)
    dirichlet_alpha: float = 0.5             # Dirichlet heterogeneity parameter (0.1 severe, 0.5 moderate)
    federated_strategy: str = "fedavg"       # 'fedavg', 'fedprox', 'fedmedian', 'fedtrimmedmean'
    partition_type: str = "iid"              # 'iid' or 'dirichlet'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Aliases for sub-config classes
DatasetPreprocessingConfig = DataConfig
ArchitectureRegularizationConfig = ModelConfig
ArchitectureConfig = ModelConfig
LossFunctionConfig = LossConfig
OptimizationConfig = OptimizerConfig
FederatedParametersConfig = FederatedConfig


class ExperimentConfig:
    """
    Unified Experiment configuration container aggregating modular sub-configs:
    - data / dataset_preprocessing: DataConfig (Dataset & Preprocessing)
    - model / architecture: ModelConfig (Architecture & Regularization)
    - loss / loss_function: LossConfig (Loss Function)
    - optimizer / optimization: OptimizerConfig (Optimization)
    - federated / federated_parameters: FederatedConfig (Federated Parameters)

    Maintains backward compatibility via attribute delegation: accessing `config.batch_size`
    or `config.learning_rate` delegates seamlessly to `config.data.batch_size` or
    `config.optimizer.learning_rate`.
    """
    _BASE_ATTRS = {"experiment_name", "seed", "device"}
    _SECTION_MAP = {
        "data": ("data", "dataset_preprocessing"),
        "dataset_preprocessing": ("data", "dataset_preprocessing"),
        "model": ("model", "architecture"),
        "architecture": ("model", "architecture"),
        "loss": ("loss", "loss_function"),
        "loss_function": ("loss", "loss_function"),
        "optimizer": ("optimizer", "optimization"),
        "optimization": ("optimizer", "optimization"),
        "federated": ("federated", "federated_parameters"),
        "federated_parameters": ("federated", "federated_parameters"),
    }

    def __init__(
        self,
        experiment_name: str = "centralized_e1_baseline",
        seed: int = 42,
        device: str = "auto",
        # Sub-config inputs (accepts instances or dicts)
        data: Optional[Union[DataConfig, Dict[str, Any]]] = None,
        model: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        loss: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        optimizer: Optional[Union[OptimizerConfig, Dict[str, Any]]] = None,
        federated: Optional[Union[FederatedConfig, Dict[str, Any]]] = None,
        # Section name aliases
        dataset_preprocessing: Optional[Union[DataConfig, Dict[str, Any]]] = None,
        architecture: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        loss_function: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        optimization: Optional[Union[OptimizerConfig, Dict[str, Any]]] = None,
        federated_parameters: Optional[Union[FederatedConfig, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        self.experiment_name = experiment_name
        self.seed = seed
        self.device = device

        # 1. Dataset & Preprocessing
        raw_data = data if data is not None else dataset_preprocessing
        if isinstance(raw_data, dict):
            resolved_data = DataConfig(**raw_data)
        elif isinstance(raw_data, DataConfig):
            resolved_data = copy.deepcopy(raw_data)
        else:
            resolved_data = DataConfig()
        super().__setattr__("data", resolved_data)
        super().__setattr__("dataset_preprocessing", resolved_data)

        # 2. Architecture & Regularization
        raw_model = model if model is not None else architecture
        if isinstance(raw_model, dict):
            resolved_model = ModelConfig(**raw_model)
        elif isinstance(raw_model, ModelConfig):
            resolved_model = copy.deepcopy(raw_model)
        else:
            resolved_model = ModelConfig()
        super().__setattr__("model", resolved_model)
        super().__setattr__("architecture", resolved_model)

        # 3. Loss Function
        raw_loss = loss if loss is not None else loss_function
        if isinstance(raw_loss, dict):
            resolved_loss = LossConfig(**raw_loss)
        elif isinstance(raw_loss, LossConfig):
            resolved_loss = copy.deepcopy(raw_loss)
        else:
            resolved_loss = LossConfig()
        super().__setattr__("loss", resolved_loss)
        super().__setattr__("loss_function", resolved_loss)

        # 4. Optimization
        raw_opt = optimizer if optimizer is not None else optimization
        if isinstance(raw_opt, dict):
            resolved_opt = OptimizerConfig(**raw_opt)
        elif isinstance(raw_opt, OptimizerConfig):
            resolved_opt = copy.deepcopy(raw_opt)
        else:
            resolved_opt = OptimizerConfig()
        super().__setattr__("optimizer", resolved_opt)
        super().__setattr__("optimization", resolved_opt)

        # 5. Federated Parameters
        raw_fed = federated if federated is not None else federated_parameters
        if isinstance(raw_fed, dict):
            resolved_fed = FederatedConfig(**raw_fed)
        elif isinstance(raw_fed, FederatedConfig):
            resolved_fed = copy.deepcopy(raw_fed)
        else:
            resolved_fed = FederatedConfig()
        super().__setattr__("federated", resolved_fed)
        super().__setattr__("federated_parameters", resolved_fed)

        # Route remaining kwargs to matching sub-configs or self
        for k, v in kwargs.items():
            routed = False
            for sub in (self.data, self.model, self.loss, self.optimizer, self.federated):
                if hasattr(sub, k):
                    setattr(sub, k, v)
                    routed = True
                    break
            if not routed:
                super().__setattr__(k, v)

    def __getattr__(self, name: str) -> Any:
        for sub_name in ("data", "model", "loss", "optimizer", "federated"):
            sub_obj = self.__dict__.get(sub_name)
            if sub_obj is not None and hasattr(sub_obj, name):
                return getattr(sub_obj, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._BASE_ATTRS:
            super().__setattr__(name, value)
            return

        if name in self._SECTION_MAP:
            for alias in self._SECTION_MAP[name]:
                super().__setattr__(alias, value)
            return

        for sub_name in ("data", "model", "loss", "optimizer", "federated"):
            sub_obj = self.__dict__.get(sub_name)
            if sub_obj is not None and hasattr(sub_obj, name):
                setattr(sub_obj, name, value)
                return

        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a structured dictionary organized by sub-configs."""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "loss": self.loss.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "federated": self.federated.to_dict(),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Converts configuration to a backward-compatible flat dictionary."""
        flat: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
        }
        flat.update(self.data.to_dict())
        flat.update(self.model.to_dict())
        flat.update(self.loss.to_dict())
        flat.update(self.optimizer.to_dict())
        flat.update(self.federated.to_dict())
        return flat

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiates ExperimentConfig from a dictionary (supports flat or nested schemas)."""
        data_copy = dict(data)
        nested_kwargs = {}
        flat_kwargs = {}

        for k, v in data_copy.items():
            if k in ("data", "dataset_preprocessing"):
                nested_kwargs["data"] = v if isinstance(v, DataConfig) else DataConfig(**v)
            elif k in ("model", "architecture"):
                nested_kwargs["model"] = v if isinstance(v, ModelConfig) else ModelConfig(**v)
            elif k in ("loss", "loss_function"):
                nested_kwargs["loss"] = v if isinstance(v, LossConfig) else LossConfig(**v)
            elif k in ("optimizer", "optimization"):
                nested_kwargs["optimizer"] = v if isinstance(v, OptimizerConfig) else OptimizerConfig(**v)
            elif k in ("federated", "federated_parameters"):
                nested_kwargs["federated"] = v if isinstance(v, FederatedConfig) else FederatedConfig(**v)
            else:
                flat_kwargs[k] = v

        return cls(**nested_kwargs, **flat_kwargs)

    def save(self, filepath: str) -> None:
        """Universal save method delegating to save_yaml or save_json based on extension."""
        if filepath.endswith((".yaml", ".yml")):
            self.save_yaml(filepath)
        else:
            self.save_json(filepath)

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
            "System & Reproducibility (Experiment)": [
                ("Random Seed", self.seed),
                ("Target Device", self.device),
            ],
            "Dataset & Preprocessing (DataConfig)": [
                ("Sample Size", f"{self.data.sample_size:,}" if self.data.sample_size else "Full Dataset (5.11M)"),
                ("Holdout Test Size", f"{self.data.test_size * 100:.0f}%"),
                ("Validation Size", f"{self.data.val_size * 100:.0f}%"),
                ("Scaler Type", self.data.scaler_type.capitalize()),
                ("Batch Size", self.data.batch_size),
            ],
            "Architecture & Regularization (ModelConfig)": [
                ("Input Dimension", self.model.input_dim),
                ("Hidden Dimensions", " -> ".join(map(str, self.model.hidden_dims))),
                ("Output Classes", self.model.num_classes),
                ("Dropout Rate", self.model.dropout_rate),
            ],
            "Loss Function (LossConfig)": [
                ("Loss Function", self.loss.loss_type),
                ("Class Weight Strategy", self.loss.class_weight_strategy),
                ("Focal Gamma", self.loss.focal_gamma if self.loss.loss_type == "focal_loss" else "N/A"),
            ],
            "Optimization (OptimizerConfig)": [
                ("Optimizer", self.optimizer.optimizer_type.upper()),
                ("Learning Rate", self.optimizer.learning_rate),
                ("Weight Decay", self.optimizer.weight_decay),
                ("Max Epochs", self.optimizer.epochs),
                ("Early Stopping Patience", self.optimizer.patience),
                ("Monitored Metric", f"{self.optimizer.monitor_metric} {'(min)' if self.optimizer.monitor_metric == 'val_loss' else '(max)'}"),
                ("Save Top-K Checkpoints", f"Top {self.optimizer.save_top_k} models"),
                ("Max Gradient Norm", self.optimizer.max_grad_norm),
            ],
            "Federated Parameters (FederatedConfig)": [
                ("Federated Strategy", self.federated.federated_strategy.upper()),
                ("Partition Type", self.federated.partition_type.upper()),
                ("Number of Clients", self.federated.num_clients),
                ("Communication Rounds", self.federated.num_rounds),
                ("Local Client Epochs", self.federated.local_epochs),
                ("FedProx Mu (μ)", self.federated.mu),
                ("Dirichlet Alpha (α)", self.federated.dirichlet_alpha),
            ],
        }

        for group_name, params in groups.items():
            print(f"\n📂 {group_name}:")
            for name, val in params:
                print(f"   • {name:<26}: {val}")
        print("=" * 70)


# Primary alias: Experiment = ExperimentConfig
Experiment = ExperimentConfig

if __name__ == "__main__":
    cfg = ExperimentConfig()
    cfg.display()
    print("Configuration module test passed!")

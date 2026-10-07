"""
Centralized Hyperparameter Configuration Module for FL-IoT-IDS.

Provides a modular, type-hinted configuration architecture organized into 5 domain sub-configs:
1. Dataset & Preprocessing (DataConfig / DatasetPreprocessingConfig)
2. Architecture & Regularization (ModelConfig / ArchitectureConfig / ArchitectureRegularizationConfig)
3. Loss Function (LossConfig / LossFunctionConfig)
4. Optimization & Training (OptimizerConfig / OptimizationConfig)
5. Federated Parameters (FederatedConfig / FederatedParametersConfig)
   - Extensible FL algorithms: FedAvg (default), FedProx (mu), FedMedian, FedTrimmedMean,
     and arbitrary future algorithms (e.g., SCAFFOLD, FedAdam).

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


# ==============================================================================
# Federated Learning Strategy Configurations
# ==============================================================================

@dataclass
class StrategyConfig:
    """Base configuration for federated aggregation algorithms."""
    name: str = "fedavg"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FedAvgConfig(StrategyConfig):
    """Configuration for standard Federated Averaging (McMahan et al., 2017)."""
    name: str = "fedavg"


@dataclass
class FedProxConfig(StrategyConfig):
    """Configuration for Federated Proximal Optimization (Li et al., 2020)."""
    name: str = "fedprox"
    mu: float = 0.05                         # Proximal regularization coefficient (default: 0.05)


@dataclass
class FedMedianConfig(StrategyConfig):
    """Configuration for Coordinate-wise Median FL (Byzantine-robust)."""
    name: str = "fedmedian"


@dataclass
class FedTrimmedMeanConfig(StrategyConfig):
    """Configuration for Coordinate-wise Trimmed Mean FL (adversarial-robust)."""
    name: str = "fedtrimmedmean"
    trim_fraction: float = 0.1               # Fraction of extreme client updates trimmed


@dataclass
class CustomStrategyConfig(StrategyConfig):
    """Configuration for arbitrary or future federated optimization algorithms."""
    name: str = "custom"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name}
        d.update(self.extra_params)
        return d


class FederatedConfig:
    """
    Federated Parameters configuration container.

    Designed for extensible multi-algorithm support:
    - Native support for FedAvg (McMahan et al., 2017) as the default strategy.
    - Native support for FedProx (Li et al., 2020) proximal regularization (mu).
    - Byzantine-robust strategies (FedMedian, FedTrimmedMean).
    - Extensible parameter routing for future FL algorithms (SCAFFOLD, FedAdam, etc.).

    Can be instantiated via:
    1. Direct kwargs: `FederatedConfig(federated_strategy="fedavg")` or `FederatedConfig(federated_strategy="fedprox", mu=0.05)`
    2. Strategy object: `FederatedConfig(strategy=FedProxConfig(mu=0.05))`
    3. Factory constructors: `FederatedConfig.fedavg()`, `FederatedConfig.fedprox(mu=0.05)`, `FederatedConfig.custom("scaffold", ...)`
    """
    def __init__(
        self,
        num_clients: int = 5,
        num_rounds: int = 8,
        local_epochs: int = 2,
        mu: float = 0.0,
        dirichlet_alpha: float = 0.5,
        federated_strategy: str = "fedavg",
        partition_type: str = "iid",
        trim_fraction: float = 0.1,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        strategy: Optional[Union[str, StrategyConfig]] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.dirichlet_alpha = dirichlet_alpha
        self.partition_type = partition_type
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.trim_fraction = trim_fraction
        self.strategy_params: Dict[str, Any] = dict(strategy_params) if strategy_params else {}

        # Resolve strategy name and strategy-specific parameters
        resolved_strategy = strategy if strategy is not None else federated_strategy

        if isinstance(resolved_strategy, StrategyConfig):
            self.federated_strategy = resolved_strategy.name.lower()
            if hasattr(resolved_strategy, "mu"):
                self.mu = getattr(resolved_strategy, "mu")
            else:
                self.mu = mu
            if hasattr(resolved_strategy, "trim_fraction"):
                self.trim_fraction = getattr(resolved_strategy, "trim_fraction")
            if hasattr(resolved_strategy, "extra_params"):
                self.strategy_params.update(getattr(resolved_strategy, "extra_params"))
        elif isinstance(resolved_strategy, str):
            self.federated_strategy = resolved_strategy.lower()
            self.mu = mu
        else:
            self.federated_strategy = "fedavg"
            self.mu = mu

        # Auto-detect FedProx if mu > 0 is specified without explicit strategy override
        if self.mu > 0.0 and self.federated_strategy == "fedavg" and strategy is None and "federated_strategy" not in kwargs:
            self.federated_strategy = "fedprox"
        elif self.federated_strategy in ("fedprox", "prox") and self.mu == 0.0:
            self.mu = 0.05

        # Store any additional kwargs in strategy_params for future algorithms
        for k, v in kwargs.items():
            self.strategy_params[k] = v

    @property
    def strategy(self) -> str:
        """Alias for federated_strategy."""
        return self.federated_strategy

    @strategy.setter
    def strategy(self, val: Union[str, StrategyConfig]) -> None:
        if isinstance(val, StrategyConfig):
            self.federated_strategy = val.name.lower()
            if hasattr(val, "mu"):
                self.mu = getattr(val, "mu")
            if hasattr(val, "trim_fraction"):
                self.trim_fraction = getattr(val, "trim_fraction")
            if hasattr(val, "extra_params"):
                self.strategy_params.update(getattr(val, "extra_params"))
        elif isinstance(val, str):
            self.federated_strategy = val.lower()

    @property
    def is_proximal(self) -> bool:
        """True if proximal regularization constraint is active (FedProx with mu > 0)."""
        return self.federated_strategy in ("fedprox", "prox") and self.mu > 0.0

    @property
    def is_fedavg(self) -> bool:
        """True if strategy is standard FedAvg."""
        return self.federated_strategy in ("fedavg", "avg")

    @property
    def is_fedprox(self) -> bool:
        """True if strategy is FedProx."""
        return self.federated_strategy in ("fedprox", "prox")

    def get_strategy_kwargs(self) -> Dict[str, Any]:
        """Extracts kwargs specific to the selected aggregation strategy."""
        strat = self.federated_strategy.lower()
        kwargs: Dict[str, Any] = {
            "fraction_fit": self.fraction_fit,
            "min_fit_clients": self.min_fit_clients,
            "min_available_clients": self.min_available_clients,
        }
        if strat in ("fedprox", "prox"):
            kwargs["mu"] = self.mu
        elif strat in ("fedtrimmedmean", "trimmedmean"):
            kwargs["trim_fraction"] = self.trim_fraction
        kwargs.update(self.strategy_params)
        return kwargs

    def build_strategy(self, **override_kwargs: Any) -> Any:
        """
        Instantiates and returns the corresponding server strategy instance
        from src.federated.strategies.
        """
        from src.federated.strategies import build_strategy as _build_strategy
        kwargs = self.get_strategy_kwargs()
        kwargs.update(override_kwargs)
        return _build_strategy(strategy_name=self.federated_strategy, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if "strategy_params" in self.__dict__ and name in self.strategy_params:
            return self.strategy_params[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            "num_clients", "num_rounds", "local_epochs", "mu", "dirichlet_alpha",
            "federated_strategy", "partition_type", "trim_fraction",
            "fraction_fit", "min_fit_clients", "min_available_clients",
            "strategy_params"
        ):
            super().__setattr__(name, value)
            return
        if hasattr(type(self), name) and isinstance(getattr(type(self), name), property):
            super().__setattr__(name, value)
            return
        if "strategy_params" in self.__dict__ and name in self.strategy_params:
            self.strategy_params[name] = value
            return
        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "mu": self.mu,
            "dirichlet_alpha": self.dirichlet_alpha,
            "federated_strategy": self.federated_strategy,
            "partition_type": self.partition_type,
            "trim_fraction": self.trim_fraction,
            "fraction_fit": self.fraction_fit,
            "min_fit_clients": self.min_fit_clients,
            "min_available_clients": self.min_available_clients,
        }
        if self.strategy_params:
            d["strategy_params"] = dict(self.strategy_params)
        return d

    # Factory methods for clean, expressive scenario instantiation:
    @classmethod
    def fedavg(cls, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for FedAvg (McMahan et al., 2017)."""
        kwargs.setdefault("federated_strategy", "fedavg")
        kwargs.setdefault("mu", 0.0)
        return cls(**kwargs)

    @classmethod
    def fedprox(cls, mu: float = 0.05, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for FedProx (Li et al., 2020)."""
        kwargs["federated_strategy"] = "fedprox"
        kwargs["mu"] = mu
        return cls(**kwargs)

    @classmethod
    def fedmedian(cls, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for Byzantine-robust FedMedian aggregation."""
        kwargs["federated_strategy"] = "fedmedian"
        return cls(**kwargs)

    @classmethod
    def fedtrimmedmean(cls, trim_fraction: float = 0.1, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for adversarial-robust FedTrimmedMean aggregation."""
        kwargs["federated_strategy"] = "fedtrimmedmean"
        kwargs["trim_fraction"] = trim_fraction
        return cls(**kwargs)

    @classmethod
    def custom(cls, strategy: str, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for arbitrary or future FL algorithms (e.g. SCAFFOLD, FedAdam)."""
        kwargs["federated_strategy"] = strategy
        return cls(**kwargs)


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
        federated: Optional[Union[FederatedConfig, StrategyConfig, str, Dict[str, Any]]] = None,
        # Section name aliases
        dataset_preprocessing: Optional[Union[DataConfig, Dict[str, Any]]] = None,
        architecture: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        loss_function: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        optimization: Optional[Union[OptimizerConfig, Dict[str, Any]]] = None,
        federated_parameters: Optional[Union[FederatedConfig, StrategyConfig, str, Dict[str, Any]]] = None,
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
        elif isinstance(raw_fed, (StrategyConfig, str)):
            resolved_fed = FederatedConfig(strategy=raw_fed)
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
                if isinstance(v, FederatedConfig):
                    nested_kwargs["federated"] = v
                elif isinstance(v, dict):
                    nested_kwargs["federated"] = FederatedConfig(**v)
                else:
                    nested_kwargs["federated"] = FederatedConfig(strategy=v)
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

        fed_params = [
            ("Federated Strategy", self.federated.federated_strategy.upper()),
            ("Partition Type", self.federated.partition_type.upper()),
            ("Number of Clients", self.federated.num_clients),
            ("Communication Rounds", self.federated.num_rounds),
            ("Local Client Epochs", self.federated.local_epochs),
        ]
        if self.federated.partition_type == "dirichlet":
            fed_params.append(("Dirichlet Alpha (α)", self.federated.dirichlet_alpha))
        if self.federated.is_proximal or self.federated.federated_strategy in ("fedprox", "prox"):
            fed_params.append(("FedProx Mu (μ)", self.federated.mu))
        if self.federated.federated_strategy in ("fedtrimmedmean", "trimmedmean"):
            fed_params.append(("Trim Fraction", self.federated.trim_fraction))
        for k, v in self.federated.strategy_params.items():
            fed_params.append((f"Extra ({k})", v))

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
            "Federated Parameters (FederatedConfig)": fed_params,
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

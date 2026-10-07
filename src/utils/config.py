"""
Centralized Hyperparameter Configuration Module for FL-IoT-IDS.

Provides a modular, type-hinted configuration architecture organized into 5 domain sub-configs:
1. Dataset & Preprocessing (DataConfig / DatasetPreprocessingConfig)
2. Architecture & Regularization (ModelConfig / ArchitectureConfig / ArchitectureRegularizationConfig)
3. Loss Function (LossConfig / LossFunctionConfig)
4. Optimization & Training (OptimizerConfig / OptimizationConfig)
5. Federated Parameters (FederatedConfig / FederatedParametersConfig)
   - Encapsulates common FL parameters (num_clients, num_rounds, local_epochs, partition_type, etc.)
   - Inside contains the Federated Algorithm (FedAlgorithm):
     * FedAvg (McMahan et al., 2017) as default
     * FedProx (Li et al., 2020) with proximal parameter mu
     * FedMedian, FedTrimmedMean (robust aggregations)
     * CustomAlgorithm (extensible container for arbitrary/future FL algorithms like SCAFFOLD, FedAdam)

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


# ==============================================================================
# 1. Dataset & Preprocessing Configuration
# ==============================================================================

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


# ==============================================================================
# 2. Architecture & Regularization Configuration
# ==============================================================================

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


# ==============================================================================
# 3. Loss Function Configuration
# ==============================================================================

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


# ==============================================================================
# 4. Optimization & Training Configuration
# ==============================================================================

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
# 5. Federated Algorithm Classes (Embedded inside FederatedConfig)
# ==============================================================================

@dataclass
class FedAlgorithm:
    """Base class for all Federated Learning aggregation and optimization algorithms."""
    name: str = "fedavg"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FedAvg(FedAlgorithm):
    """
    Standard Federated Averaging algorithm (McMahan et al., 2017).
    Aggregates client model updates using exact sample-weighted averaging.
    """
    name: str = "fedavg"


@dataclass
class FedProx(FedAlgorithm):
    """
    Federated Proximal optimization algorithm (Li et al., 2020).
    Adds a proximal regularization penalty (mu / 2) * ||w - w_t||^2 to client objectives
    to handle system and statistical non-IID heterogeneity.
    """
    name: str = "fedprox"
    mu: float = 0.05                         # Proximal regularization coefficient (default: 0.05)


@dataclass
class FedMedian(FedAlgorithm):
    """
    Coordinate-wise Median Federated Aggregation algorithm.
    Byzantine-tolerant aggregation robust to malicious or corrupted client updates.
    """
    name: str = "fedmedian"


@dataclass
class FedTrimmedMean(FedAlgorithm):
    """
    Coordinate-wise Trimmed Mean Federated Aggregation algorithm.
    Trims extreme client updates to mitigate adversarial poisoning and outliers.
    """
    name: str = "fedtrimmedmean"
    trim_fraction: float = 0.1               # Fraction of extreme client updates trimmed


@dataclass
class CustomAlgorithm(FedAlgorithm):
    """
    Extensible container for arbitrary or future FL algorithms (e.g., SCAFFOLD, FedAdam, FedOpt).
    """
    name: str = "custom"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name}
        d.update(self.params)
        return d


# Algorithm Aliases
FederatedAlgorithm = FedAlgorithm
StrategyConfig = FedAlgorithm

FedAvgAlgorithm = FedAvg
FedAvgConfig = FedAvg

FedProxAlgorithm = FedProx
FedProxConfig = FedProx

FedMedianAlgorithm = FedMedian
FedMedianConfig = FedMedian

FedTrimmedMeanAlgorithm = FedTrimmedMean
FedTrimmedMeanConfig = FedTrimmedMean

CustomStrategyConfig = CustomAlgorithm


# ==============================================================================
# Federated Parameters Configuration (Contains Algorithm + Common FL Attributes)
# ==============================================================================

class FederatedConfig:
    """
    Federated Parameters configuration container.

    Organized into:
    1. Federated Algorithm (self.algorithm: FedAlgorithm):
       - FedAvg (McMahan et al., 2017) [default]
       - FedProx (Li et al., 2020) with proximal coefficient mu
       - FedMedian, FedTrimmedMean (robust aggregations)
       - CustomAlgorithm (for future FL algorithms: SCAFFOLD, FedAdam, etc.)
    2. Common Federated Parameters:
       - num_clients: int = 5
       - num_rounds: int = 8
       - local_epochs: int = 2
       - partition_type: str = "iid"
       - dirichlet_alpha: float = 0.5
       - fraction_fit: float = 1.0
       - min_fit_clients: int = 2
       - min_available_clients: int = 2

    Supports flexible instantiation:
    - Dedicated algorithm: `FederatedConfig(algorithm=FedAvg())` or `FederatedConfig(algorithm=FedProx(mu=0.05))`
    - Factory constructors: `FederatedConfig.fedavg()`, `FederatedConfig.fedprox(mu=0.05)`, `FederatedConfig.custom("scaffold", ...)`
    - Backward-compatible kwargs: `FederatedConfig(federated_strategy="fedavg")` or `FederatedConfig(mu=0.05)`
    """
    _COMMON_FIELDS = {
        "num_clients", "num_rounds", "local_epochs", "partition_type",
        "dirichlet_alpha", "fraction_fit", "min_fit_clients", "min_available_clients",
        "algorithm", "algo", "strategy"
    }

    def __init__(
        self,
        algorithm: Optional[Union[FedAlgorithm, str]] = None,
        # Common FL Parameters
        num_clients: int = 5,
        num_rounds: int = 8,
        local_epochs: int = 2,
        partition_type: str = "iid",
        dirichlet_alpha: float = 0.5,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        # Backward-compatible parameter aliases
        strategy: Optional[Union[FedAlgorithm, str]] = None,
        federated_strategy: Optional[str] = None,
        mu: Optional[float] = None,
        trim_fraction: Optional[float] = None,
        **kwargs: Any,
    ):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.partition_type = partition_type
        self.dirichlet_alpha = dirichlet_alpha
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

        # Resolve algorithm
        raw_algo = algorithm if algorithm is not None else (strategy if strategy is not None else federated_strategy)

        if isinstance(raw_algo, FedAlgorithm):
            self.algorithm = copy.deepcopy(raw_algo)
            if mu is not None and hasattr(self.algorithm, "mu"):
                self.algorithm.mu = mu
            if trim_fraction is not None and hasattr(self.algorithm, "trim_fraction"):
                self.algorithm.trim_fraction = trim_fraction
            if kwargs and isinstance(self.algorithm, CustomAlgorithm):
                self.algorithm.params.update(kwargs)
        elif isinstance(raw_algo, str):
            strat = raw_algo.lower().replace("-", "").replace("_", "")
            if strat in ("fedavg", "avg"):
                self.algorithm = FedAvg()
            elif strat in ("fedprox", "prox"):
                self.algorithm = FedProx(mu=mu if mu is not None else 0.05)
            elif strat in ("fedmedian", "median"):
                self.algorithm = FedMedian()
            elif strat in ("fedtrimmedmean", "trimmedmean"):
                self.algorithm = FedTrimmedMean(trim_fraction=trim_fraction if trim_fraction is not None else 0.1)
            else:
                self.algorithm = CustomAlgorithm(name=raw_algo, params=kwargs)
        elif mu is not None and mu > 0.0:
            self.algorithm = FedProx(mu=mu)
        else:
            self.algorithm = FedAvg()

    @property
    def federated_strategy(self) -> str:
        """Name of the active federated algorithm (e.g. 'fedavg', 'fedprox')."""
        return self.algorithm.name

    @federated_strategy.setter
    def federated_strategy(self, val: Union[FedAlgorithm, str]) -> None:
        if isinstance(val, FedAlgorithm):
            self.algorithm = val
        elif isinstance(val, str):
            strat = val.lower().replace("-", "").replace("_", "")
            if strat in ("fedavg", "avg"):
                self.algorithm = FedAvg()
            elif strat in ("fedprox", "prox"):
                self.algorithm = FedProx(mu=getattr(self.algorithm, "mu", 0.05))
            elif strat in ("fedmedian", "median"):
                self.algorithm = FedMedian()
            elif strat in ("fedtrimmedmean", "trimmedmean"):
                self.algorithm = FedTrimmedMean(trim_fraction=getattr(self.algorithm, "trim_fraction", 0.1))
            else:
                self.algorithm = CustomAlgorithm(name=val)

    @property
    def strategy(self) -> str:
        """Alias for federated_strategy."""
        return self.algorithm.name

    @strategy.setter
    def strategy(self, val: Union[FedAlgorithm, str]) -> None:
        self.federated_strategy = val

    @property
    def mu(self) -> float:
        """FedProx proximal coefficient mu (0.0 for FedAvg)."""
        return getattr(self.algorithm, "mu", 0.0)

    @mu.setter
    def mu(self, val: float) -> None:
        if isinstance(self.algorithm, FedProx):
            self.algorithm.mu = val
        else:
            self.algorithm = FedProx(mu=val)

    @property
    def trim_fraction(self) -> float:
        """Trimmed mean fraction for FedTrimmedMean."""
        return getattr(self.algorithm, "trim_fraction", 0.1)

    @property
    def is_proximal(self) -> bool:
        """True if proximal regularization constraint is active (FedProx with mu > 0)."""
        return isinstance(self.algorithm, FedProx) and self.algorithm.mu > 0.0

    @property
    def is_fedavg(self) -> bool:
        """True if algorithm is standard FedAvg."""
        return self.algorithm.name in ("fedavg", "avg")

    @property
    def is_fedprox(self) -> bool:
        """True if algorithm is FedProx."""
        return self.algorithm.name in ("fedprox", "prox")

    def get_strategy_kwargs(self) -> Dict[str, Any]:
        """Extracts kwargs specific to the selected algorithm for server instantiation."""
        kwargs: Dict[str, Any] = {
            "fraction_fit": self.fraction_fit,
            "min_fit_clients": self.min_fit_clients,
            "min_available_clients": self.min_available_clients,
        }
        if isinstance(self.algorithm, FedProx):
            kwargs["mu"] = self.algorithm.mu
        elif isinstance(self.algorithm, FedTrimmedMean):
            kwargs["trim_fraction"] = self.algorithm.trim_fraction
        elif isinstance(self.algorithm, CustomAlgorithm):
            kwargs.update(self.algorithm.params)
        return kwargs

    def build_strategy(self, **override_kwargs: Any) -> Any:
        """
        Instantiates and returns the concrete server strategy instance
        from src.federated.strategies.
        """
        from src.federated.strategies import build_strategy as _build_strategy
        kwargs = self.get_strategy_kwargs()
        kwargs.update(override_kwargs)
        return _build_strategy(strategy_name=self.algorithm.name, **kwargs)

    def __getattr__(self, name: str) -> Any:
        algo = self.__dict__.get("algorithm")
        if algo is not None:
            if hasattr(algo, name):
                return getattr(algo, name)
            if isinstance(algo, CustomAlgorithm) and name in algo.params:
                return algo.params[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            "num_clients", "num_rounds", "local_epochs", "partition_type",
            "dirichlet_alpha", "fraction_fit", "min_fit_clients", "min_available_clients",
            "algorithm"
        ):
            super().__setattr__(name, value)
            return
        cls_attr = getattr(type(self), name, None)
        if isinstance(cls_attr, property) and cls_attr.fset is not None:
            cls_attr.fset(self, value)
            return
        algo = self.__dict__.get("algorithm")
        if algo is not None:
            if hasattr(algo, name):
                setattr(algo, name, value)
                return
            if isinstance(algo, CustomAlgorithm):
                algo.params[name] = value
                return
        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a dictionary preserving both nested algorithm and flat fields."""
        d: Dict[str, Any] = {
            "algorithm": self.algorithm.to_dict(),
            "federated_strategy": self.algorithm.name,
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "partition_type": self.partition_type,
            "dirichlet_alpha": self.dirichlet_alpha,
            "fraction_fit": self.fraction_fit,
            "min_fit_clients": self.min_fit_clients,
            "min_available_clients": self.min_available_clients,
            "mu": self.mu,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedConfig":
        """Instantiates FederatedConfig from dictionary, handling both nested algorithm and flat schemas."""
        data_copy = dict(data)
        algo_data = data_copy.pop("algorithm", None)
        if isinstance(algo_data, dict):
            algo_name = algo_data.get("name", "fedavg").lower()
            if algo_name in ("fedavg", "avg"):
                algo = FedAvg()
            elif algo_name in ("fedprox", "prox"):
                algo = FedProx(mu=algo_data.get("mu", 0.05))
            elif algo_name in ("fedmedian", "median"):
                algo = FedMedian()
            elif algo_name in ("fedtrimmedmean", "trimmedmean"):
                algo = FedTrimmedMean(trim_fraction=algo_data.get("trim_fraction", 0.1))
            else:
                algo = CustomAlgorithm(name=algo_name, params=algo_data)
            return cls(algorithm=algo, **data_copy)
        elif isinstance(algo_data, FedAlgorithm):
            return cls(algorithm=algo_data, **data_copy)
        return cls(**data_copy)

    # Expressive factory constructors:
    @classmethod
    def fedavg(cls, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for FedAvg (McMahan et al., 2017)."""
        return cls(algorithm=FedAvg(), **kwargs)

    @classmethod
    def fedprox(cls, mu: float = 0.05, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for FedProx (Li et al., 2020)."""
        return cls(algorithm=FedProx(mu=mu), **kwargs)

    @classmethod
    def fedmedian(cls, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for Byzantine-robust FedMedian aggregation."""
        return cls(algorithm=FedMedian(), **kwargs)

    @classmethod
    def fedtrimmedmean(cls, trim_fraction: float = 0.1, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for adversarial-robust FedTrimmedMean aggregation."""
        return cls(algorithm=FedTrimmedMean(trim_fraction=trim_fraction), **kwargs)

    @classmethod
    def custom(cls, name: str, **kwargs: Any) -> "FederatedConfig":
        """Factory constructor for arbitrary or future FL algorithms (e.g. SCAFFOLD, FedAdam)."""
        algo_params = {}
        common_kwargs = {}
        for k, v in kwargs.items():
            if k in cls._COMMON_FIELDS:
                common_kwargs[k] = v
            else:
                algo_params[k] = v
        return cls(algorithm=CustomAlgorithm(name=name, params=algo_params), **common_kwargs)


# Sub-config class aliases
DatasetPreprocessingConfig = DataConfig
ArchitectureRegularizationConfig = ModelConfig
ArchitectureConfig = ModelConfig
LossFunctionConfig = LossConfig
OptimizationConfig = OptimizerConfig
FederatedParametersConfig = FederatedConfig


# ==============================================================================
# Unified Composite Experiment Configuration Container
# ==============================================================================

class ExperimentConfig:
    """
    Unified Experiment configuration container aggregating modular sub-configs:
    - data / dataset_preprocessing: DataConfig (Dataset & Preprocessing)
    - model / architecture: ModelConfig (Architecture & Regularization)
    - loss / loss_function: LossConfig (Loss Function)
    - optimizer / optimization: OptimizerConfig (Optimization)
    - federated / federated_parameters: FederatedConfig (Federated Parameters & Algorithm)

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
        federated: Optional[Union[FederatedConfig, FedAlgorithm, str, Dict[str, Any]]] = None,
        # Section name aliases
        dataset_preprocessing: Optional[Union[DataConfig, Dict[str, Any]]] = None,
        architecture: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        loss_function: Optional[Union[LossConfig, Dict[str, Any]]] = None,
        optimization: Optional[Union[OptimizerConfig, Dict[str, Any]]] = None,
        federated_parameters: Optional[Union[FederatedConfig, FedAlgorithm, str, Dict[str, Any]]] = None,
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

        # 5. Federated Parameters (with internal Algorithm)
        raw_fed = federated if federated is not None else federated_parameters
        if isinstance(raw_fed, dict):
            resolved_fed = FederatedConfig.from_dict(raw_fed)
        elif isinstance(raw_fed, FederatedConfig):
            resolved_fed = copy.deepcopy(raw_fed)
        elif isinstance(raw_fed, (FedAlgorithm, str)):
            resolved_fed = FederatedConfig(algorithm=raw_fed)
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
                    nested_kwargs["federated"] = FederatedConfig.from_dict(v)
                else:
                    nested_kwargs["federated"] = FederatedConfig(algorithm=v)
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

        algo_display = self.federated.algorithm.name.upper()
        if self.federated.is_fedavg:
            algo_display += " (McMahan et al., 2017)"
        elif self.federated.is_fedprox:
            algo_display += " (Li et al., 2020)"

        fed_params = [
            ("Federated Algorithm", algo_display),
            ("Partition Type", self.federated.partition_type.upper()),
            ("Number of Clients", self.federated.num_clients),
            ("Communication Rounds", self.federated.num_rounds),
            ("Local Client Epochs", self.federated.local_epochs),
        ]
        if self.federated.partition_type == "dirichlet":
            fed_params.append(("Dirichlet Alpha (α)", self.federated.dirichlet_alpha))
        if self.federated.is_proximal or hasattr(self.federated.algorithm, "mu"):
            fed_params.append(("Proximal Mu (μ)", self.federated.mu))
        if hasattr(self.federated.algorithm, "trim_fraction"):
            fed_params.append(("Trim Fraction", self.federated.trim_fraction))
        if isinstance(self.federated.algorithm, CustomAlgorithm):
            for k, v in self.federated.algorithm.params.items():
                fed_params.append((f"Algo Param ({k})", v))

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

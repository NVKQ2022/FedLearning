"""
Unit tests for the modular ExperimentConfig / Experiment configuration system,
verifying that FederatedConfig encapsulates common FL attributes while embedding
the Federated Algorithm (FedAvg, FedProx, FedMedian, FedTrimmedMean, CustomAlgorithm).
"""

import os
import tempfile
import pytest
from src.utils.config import (
    ExperimentConfig,
    Experiment,
    DataConfig,
    DatasetPreprocessingConfig,
    ModelConfig,
    ArchitectureConfig,
    ArchitectureRegularizationConfig,
    LossConfig,
    LossFunctionConfig,
    OptimizerConfig,
    OptimizationConfig,
    FederatedConfig,
    FederatedParametersConfig,
    FedAlgorithm,
    FederatedAlgorithm,
    FedAvg,
    FedAvgConfig,
    FedProx,
    FedProxConfig,
    FedMedian,
    FedMedianConfig,
    FedTrimmedMean,
    FedTrimmedMeanConfig,
    CustomAlgorithm,
)


def test_default_initialization():
    cfg = ExperimentConfig()
    assert cfg.experiment_name == "centralized_e1_baseline"
    assert cfg.seed == 42
    assert cfg.device == "auto"

    # Modular references
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.loss, LossConfig)
    assert isinstance(cfg.optimizer, OptimizerConfig)
    assert isinstance(cfg.federated, FederatedConfig)

    # Embedded algorithm
    assert isinstance(cfg.federated.algorithm, FedAvg)
    assert cfg.federated.algorithm.name == "fedavg"

    # Aliases
    assert cfg.dataset_preprocessing is cfg.data
    assert cfg.architecture is cfg.model
    assert cfg.loss_function is cfg.loss
    assert cfg.optimization is cfg.optimizer
    assert cfg.federated_parameters is cfg.federated


def test_modular_initialization_with_instances():
    data_cfg = DataConfig(batch_size=64, sample_size=50000)
    model_cfg = ModelConfig(hidden_dims=(256, 128), dropout_rate=0.3)
    loss_cfg = LossConfig(loss_type="cross_entropy", focal_gamma=1.5)
    opt_cfg = OptimizerConfig(learning_rate=0.005, epochs=20)
    fed_cfg = FederatedConfig(algorithm=FedAvg(), num_clients=10)

    cfg = Experiment(
        experiment_name="test_modular",
        data=data_cfg,
        model=model_cfg,
        loss=loss_cfg,
        optimizer=opt_cfg,
        federated=fed_cfg,
    )

    # Modular access
    assert cfg.data.batch_size == 64
    assert cfg.data.sample_size == 50000
    assert cfg.model.hidden_dims == (256, 128)
    assert cfg.model.dropout_rate == 0.3
    assert cfg.loss.loss_type == "cross_entropy"
    assert cfg.optimizer.learning_rate == 0.005
    assert cfg.optimizer.epochs == 20
    assert cfg.federated.num_clients == 10
    assert isinstance(cfg.federated.algorithm, FedAvg)

    # Flat delegation access
    assert cfg.batch_size == 64
    assert cfg.sample_size == 50000
    assert cfg.hidden_dims == (256, 128)
    assert cfg.dropout_rate == 0.3
    assert cfg.loss_type == "cross_entropy"
    assert cfg.learning_rate == 0.005
    assert cfg.epochs == 20
    assert cfg.num_clients == 10
    assert cfg.federated_strategy == "fedavg"


def test_alias_parameter_initialization():
    cfg = Experiment(
        dataset_preprocessing=DatasetPreprocessingConfig(batch_size=32),
        architecture=ArchitectureRegularizationConfig(hidden_dims=[64, 32]),
        loss_function=LossFunctionConfig(use_class_weights=False),
        optimization=OptimizationConfig(lr=0.01),
        federated_parameters=FederatedParametersConfig(algorithm=FedProx(mu=0.05)),
    )

    assert cfg.batch_size == 32
    assert cfg.data.batch_size == 32
    assert cfg.model.hidden_dims == (64, 32)
    assert cfg.loss.class_weight_strategy == "none"
    assert cfg.optimizer.learning_rate == 0.01
    assert cfg.optimizer.lr == 0.01
    assert isinstance(cfg.federated.algorithm, FedProx)
    assert cfg.federated.mu == 0.05
    assert cfg.mu == 0.05


def test_flat_backward_compatibility_init():
    cfg = ExperimentConfig(
        batch_size=16,
        hidden_dims=(64,),
        learning_rate=0.002,
        num_clients=8,
        federated_strategy="fedavg",
    )

    assert cfg.batch_size == 16
    assert cfg.data.batch_size == 16
    assert cfg.hidden_dims == (64,)
    assert cfg.model.hidden_dims == (64,)
    assert cfg.learning_rate == 0.002
    assert cfg.optimizer.learning_rate == 0.002
    assert cfg.num_clients == 8
    assert cfg.federated.num_clients == 8
    assert cfg.federated_strategy == "fedavg"


def test_flat_attribute_assignment_delegation():
    cfg = ExperimentConfig()

    # Mutating via flat attribute should update sub-config
    cfg.batch_size = 256
    assert cfg.data.batch_size == 256
    assert cfg.dataset_preprocessing.batch_size == 256

    cfg.learning_rate = 5e-4
    assert cfg.optimizer.learning_rate == 5e-4

    cfg.num_clients = 12
    assert cfg.federated.num_clients == 12

    # Mutating sub-config directly should reflect in flat attribute
    cfg.model.dropout_rate = 0.5
    assert cfg.dropout_rate == 0.5


def test_dict_serialization_and_deserialization():
    cfg = ExperimentConfig(
        experiment_name="serialize_test",
        data=DataConfig(batch_size=64),
        optimizer=OptimizerConfig(learning_rate=0.003),
        federated=FederatedConfig(algorithm=FedProx(mu=0.08), num_clients=6),
    )

    nested_dict = cfg.to_dict()
    assert "data" in nested_dict
    assert nested_dict["data"]["batch_size"] == 64
    assert nested_dict["optimizer"]["learning_rate"] == 0.003
    assert nested_dict["federated"]["algorithm"]["name"] == "fedprox"
    assert nested_dict["federated"]["algorithm"]["mu"] == 0.08
    assert nested_dict["federated"]["num_clients"] == 6

    restored = ExperimentConfig.from_dict(nested_dict)
    assert restored.batch_size == 64
    assert restored.optimizer.learning_rate == 0.003
    assert isinstance(restored.federated.algorithm, FedProx)
    assert restored.federated.mu == 0.08
    assert restored.federated.num_clients == 6

    flat_dict = cfg.to_flat_dict()
    assert flat_dict["batch_size"] == 64
    assert flat_dict["learning_rate"] == 0.003
    assert flat_dict["num_clients"] == 6

    restored_flat = ExperimentConfig.from_dict(flat_dict)
    assert restored_flat.batch_size == 64
    assert restored_flat.optimizer.learning_rate == 0.003
    assert restored_flat.num_clients == 6


def test_file_save_and_display():
    cfg = ExperimentConfig(experiment_name="save_display_test")

    # Verify display runs without error
    cfg.display()

    # Verify save to json
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "config.json")
        cfg.save(json_path)
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            content = f.read()
            assert "save_display_test" in content


# ==============================================================================
# Embedded Algorithm & FederatedConfig Architecture Tests
# ==============================================================================

def test_federated_config_embedded_algorithm_fedavg():
    # Common attributes on FederatedConfig, algorithm inside
    fed = FederatedConfig(
        algorithm=FedAvg(),
        num_clients=5,
        num_rounds=10,
        local_epochs=2,
        partition_type="iid",
    )
    assert isinstance(fed.algorithm, FedAvg)
    assert fed.algorithm.name == "fedavg"
    # Common attributes preserved at config level
    assert fed.num_clients == 5
    assert fed.num_rounds == 10
    assert fed.local_epochs == 2
    assert fed.partition_type == "iid"
    # Convenience properties
    assert fed.is_fedavg is True
    assert fed.is_fedprox is False
    assert fed.is_proximal is False
    assert fed.mu == 0.0


def test_federated_config_embedded_algorithm_fedprox():
    # Common attributes on FederatedConfig, algorithm FedProx(mu) inside
    fed = FederatedConfig(
        algorithm=FedProx(mu=0.05),
        num_clients=7,
        num_rounds=12,
        partition_type="dirichlet",
        dirichlet_alpha=0.1,
    )
    assert isinstance(fed.algorithm, FedProx)
    assert fed.algorithm.mu == 0.05
    # Common attributes preserved at config level
    assert fed.num_clients == 7
    assert fed.num_rounds == 12
    assert fed.partition_type == "dirichlet"
    assert fed.dirichlet_alpha == 0.1
    # Proximal properties
    assert fed.is_fedprox is True
    assert fed.is_proximal is True
    assert fed.mu == 0.05


def test_federated_config_robust_and_custom_algorithms():
    # FedMedian
    fed_med = FederatedConfig(algorithm=FedMedian(), num_clients=5)
    assert isinstance(fed_med.algorithm, FedMedian)
    assert fed_med.algorithm.name == "fedmedian"
    assert fed_med.num_clients == 5

    # FedTrimmedMean
    fed_trim = FederatedConfig(algorithm=FedTrimmedMean(trim_fraction=0.15), num_clients=5)
    assert isinstance(fed_trim.algorithm, FedTrimmedMean)
    assert fed_trim.algorithm.trim_fraction == 0.15
    assert fed_trim.trim_fraction == 0.15

    # CustomAlgorithm (e.g., future SCAFFOLD, FedAdam)
    fed_custom = FederatedConfig(
        algorithm=CustomAlgorithm(name="scaffold", params={"server_lr": 0.01}),
        num_clients=6,
    )
    assert isinstance(fed_custom.algorithm, CustomAlgorithm)
    assert fed_custom.algorithm.name == "scaffold"
    assert fed_custom.server_lr == 0.01
    assert fed_custom.num_clients == 6


def test_federated_config_factory_constructors():
    # FedAvg factory
    f_avg = FederatedConfig.fedavg(num_clients=8, num_rounds=15)
    assert isinstance(f_avg.algorithm, FedAvg)
    assert f_avg.num_clients == 8
    assert f_avg.num_rounds == 15

    # FedProx factory
    f_prox = FederatedConfig.fedprox(mu=0.04, num_clients=9)
    assert isinstance(f_prox.algorithm, FedProx)
    assert f_prox.algorithm.mu == 0.04
    assert f_prox.num_clients == 9

    # Custom factory
    f_cust = FederatedConfig.custom(name="fedadam", num_clients=4, server_lr=0.02)
    assert isinstance(f_cust.algorithm, CustomAlgorithm)
    assert f_cust.num_clients == 4
    assert f_cust.server_lr == 0.02


def test_federated_config_build_strategy():
    strat_avg = FederatedConfig(algorithm=FedAvg()).build_strategy()
    assert strat_avg.name == "FedAvg"

    strat_prox = FederatedConfig(algorithm=FedProx(mu=0.07)).build_strategy()
    assert strat_prox.name == "FedProx"
    assert strat_prox.mu == 0.07

    strat_med = FederatedConfig(algorithm=FedMedian()).build_strategy()
    assert strat_med.name == "FedMedian"

    strat_trim = FederatedConfig(algorithm=FedTrimmedMean(trim_fraction=0.2)).build_strategy()
    assert strat_trim.name == "FedTrimmedMean"
    assert strat_trim.trim_fraction == 0.2


def test_experiment_with_federated_algorithm():
    cfg = Experiment(
        experiment_name="test_fedprox_embedded",
        federated=FederatedConfig(
            algorithm=FedProx(mu=0.06),
            num_clients=5,
            num_rounds=10,
        ),
    )
    assert isinstance(cfg.federated.algorithm, FedProx)
    assert cfg.federated.algorithm.mu == 0.06
    assert cfg.num_clients == 5
    assert cfg.num_rounds == 10
    assert cfg.is_proximal is True
    assert cfg.mu == 0.06

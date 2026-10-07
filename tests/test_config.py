"""
Unit tests for the modular ExperimentConfig / Experiment configuration system.
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
    fed_cfg = FederatedConfig(num_clients=10, federated_strategy="fedavg")

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
    assert cfg.federated.federated_strategy == "fedavg"

    # Flat delegation access
    assert cfg.batch_size == 64
    assert cfg.sample_size == 50000
    assert cfg.hidden_dims == (256, 128)
    assert cfg.dropout_rate == 0.3
    assert cfg.loss_type == "cross_entropy"
    assert cfg.learning_rate == 0.005
    assert cfg.epochs == 20
    assert cfg.num_clients == 10


def test_alias_parameter_initialization():
    cfg = Experiment(
        dataset_preprocessing=DatasetPreprocessingConfig(batch_size=32),
        architecture=ArchitectureRegularizationConfig(hidden_dims=[64, 32]),
        loss_function=LossFunctionConfig(use_class_weights=False),
        optimization=OptimizationConfig(lr=0.01),
        federated_parameters=FederatedParametersConfig(mu=0.05),
    )

    assert cfg.batch_size == 32
    assert cfg.data.batch_size == 32
    assert cfg.model.hidden_dims == (64, 32)
    assert cfg.loss.class_weight_strategy == "none"
    assert cfg.optimizer.learning_rate == 0.01
    assert cfg.optimizer.lr == 0.01
    assert cfg.federated.mu == 0.05


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
    )

    nested_dict = cfg.to_dict()
    assert "data" in nested_dict
    assert nested_dict["data"]["batch_size"] == 64
    assert nested_dict["optimizer"]["learning_rate"] == 0.003

    restored = ExperimentConfig.from_dict(nested_dict)
    assert restored.batch_size == 64
    assert restored.optimizer.learning_rate == 0.003

    flat_dict = cfg.to_flat_dict()
    assert flat_dict["batch_size"] == 64
    assert flat_dict["learning_rate"] == 0.003

    restored_flat = ExperimentConfig.from_dict(flat_dict)
    assert restored_flat.batch_size == 64
    assert restored_flat.optimizer.learning_rate == 0.003


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

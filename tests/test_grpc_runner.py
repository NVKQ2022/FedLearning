"""
Unit tests for gRPC runner utilities and port management.
"""

import socket
from unittest.mock import MagicMock, patch
from src.federated.grpc_runner import is_port_in_use, find_available_port, run_flower_grpc
from src.utils.config import Experiment, FederatedConfig, FedAvg, FedProx


def test_is_port_in_use_and_find_available_port():
    # Bind a temporary socket to check detection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    bound_port = sock.getsockname()[1]

    try:
        assert is_port_in_use(bound_port) is True
        # find_available_port starting at bound_port should return a different free port
        free_port = find_available_port(start_port=bound_port)
        assert free_port != bound_port
        assert is_port_in_use(free_port) is False
    finally:
        sock.close()


def test_run_flower_grpc_with_federated_config_directly():
    """Verify run_flower_grpc works using only FederatedConfig without entire ExperimentConfig."""
    fed_cfg = FederatedConfig(
        algorithm=FedProx(mu=0.05),
        num_clients=3,
        num_rounds=5,
        local_epochs=4,
        scenario_name="test_direct_fed_scenario",
    )

    with patch("os.path.exists", return_value=True), \
         patch("subprocess.Popen") as mock_popen, \
         patch("src.federated.grpc_runner.find_available_port", return_value=8080), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"val_macro_f1": [0.88]}):

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = None
        mock_popen.return_value = mock_proc

        # Pass federated_config directly with scenario_name
        result = run_flower_grpc(
            scenario_name="test_direct_fed_scenario",
            federated_config=fed_cfg,
            stream_logs=False
        )

        assert mock_popen.call_count == 4  # 1 server + 3 clients
        server_call = mock_popen.call_args_list[0][0][0]
        assert "--rounds" in server_call
        assert server_call[server_call.index("--rounds") + 1] == "5"
        assert "--strategy" in server_call
        assert server_call[server_call.index("--strategy") + 1] == "fedprox"
        assert "--mu" in server_call
        assert server_call[server_call.index("--mu") + 1] == "0.05"
        assert "--local-epochs" in server_call
        assert server_call[server_call.index("--local-epochs") + 1] == "4"


def test_run_flower_grpc_with_experiment_config():
    """Verify run_flower_grpc correctly extracts all parameters from Experiment config."""
    cfg = Experiment(
        experiment_name="test_experiment_e2",
        federated=FederatedConfig(
            algorithm=FedAvg(),
            num_clients=3,
            num_rounds=4,
            local_epochs=3,
        )
    )

    with patch("os.path.exists", return_value=True), \
         patch("subprocess.Popen") as mock_popen, \
         patch("src.federated.grpc_runner.find_available_port", return_value=8080), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"val_macro_f1": [0.85]}):

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = None
        mock_popen.return_value = mock_proc

        result = run_flower_grpc(config=cfg, stream_logs=False)

        assert mock_popen.call_count == 4  # 1 server + 3 clients
        # Verify server command
        server_call = mock_popen.call_args_list[0][0][0]
        assert "--rounds" in server_call
        assert server_call[server_call.index("--rounds") + 1] == "4"
        assert "--strategy" in server_call
        assert server_call[server_call.index("--strategy") + 1] == "fedavg"
        assert "--local-epochs" in server_call
        assert server_call[server_call.index("--local-epochs") + 1] == "3"

        # Verify client command
        client_call = mock_popen.call_args_list[1][0][0]
        assert "--epochs" in client_call
        assert client_call[client_call.index("--epochs") + 1] == "3"


def test_run_flower_grpc_positional_config_with_overrides():
    """Verify positional Experiment config and argument overrides."""
    cfg = Experiment(
        experiment_name="test_fedprox_scenario",
        federated=FederatedConfig(
            algorithm=FedProx(mu=0.08),
            num_clients=4,
            num_rounds=6,
        )
    )

    with patch("os.path.exists", return_value=True), \
         patch("subprocess.Popen") as mock_popen, \
         patch("src.federated.grpc_runner.find_available_port", return_value=8081), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"val_macro_f1": [0.9]}):

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = None
        mock_popen.return_value = mock_proc

        # Override rounds to 2
        run_flower_grpc(cfg, rounds=2, stream_logs=False)

        server_call = mock_popen.call_args_list[0][0][0]
        assert server_call[server_call.index("--rounds") + 1] == "2"
        assert server_call[server_call.index("--strategy") + 1] == "fedprox"
        assert server_call[server_call.index("--mu") + 1] == "0.08"


def test_experiment_for_scenario_helper():
    """Verify Experiment.for_scenario generates correct configs for thesis scenarios."""
    e2 = Experiment.for_scenario("E2")
    assert e2.experiment_name == "flower_e2_fedavg_iid"
    assert e2.federated_strategy == "fedavg"
    assert e2.partition_type == "iid"

    e5 = Experiment.for_scenario("E5")
    assert e5.experiment_name == "flower_e5_fedprox_dirichlet01"
    assert e5.federated_strategy == "fedprox"
    assert e5.dirichlet_alpha == 0.1
    assert e5.mu == 0.05


def test_run_flower_grpc_with_fraction_evaluate():
    """Verify run_flower_grpc propagates fraction_evaluate to flower_server."""
    fed_cfg = FederatedConfig(
        algorithm=FedAvg(),
        num_clients=2,
        num_rounds=3,
        fraction_evaluate=0.75,
        scenario_name="test_fraction_eval_scenario"
    )

    with patch("os.path.exists", return_value=True), \
         patch("subprocess.Popen") as mock_popen, \
         patch("src.federated.grpc_runner.find_available_port", return_value=8082), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"val_macro_f1": [0.80]}):

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = None
        mock_popen.return_value = mock_proc

        run_flower_grpc(
            scenario_name="test_fraction_eval_scenario",
            federated_config=fed_cfg,
            stream_logs=False
        )

        server_call = mock_popen.call_args_list[0][0][0]
        assert "--fraction-evaluate" in server_call
        assert server_call[server_call.index("--fraction-evaluate") + 1] == "0.75"


def test_create_federated_scenario_client_val_split():
    """Verify create_federated_scenario splits client partition into 80% train and 20% val."""
    import tempfile
    import os
    import numpy as np
    from src.federated.scenario import create_federated_scenario, load_client_partition

    with tempfile.TemporaryDirectory() as tmp_dir:
        num_samples = 200
        input_dim = 10
        num_classes = 4

        X = np.random.randn(num_samples, input_dim).astype(np.float32)
        y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)

        # 2 clients each with 100 samples
        client_partitions = {
            0: np.arange(0, 100),
            1: np.arange(100, 200)
        }

        scenario_dir = create_federated_scenario(
            scenario_name="test_scenario_split",
            client_partitions=client_partitions,
            X_train=X,
            y_train=y,
            class_names=[f"C{i}" for i in range(num_classes)],
            base_dir=tmp_dir,
            generate_plots=False,
            client_val_ratio=0.2
        )

        # Check client 0
        c0_dir = os.path.join(scenario_dir, "client_0")
        assert os.path.exists(os.path.join(c0_dir, "partition.npz"))
        assert os.path.exists(os.path.join(c0_dir, "val_partition.npz"))

        c0_data = np.load(os.path.join(c0_dir, "partition.npz"))
        assert "X_train" in c0_data
        assert "y_train" in c0_data
        assert "X_val" in c0_data
        assert "y_val" in c0_data
        assert "X" in c0_data
        assert "y" in c0_data

        # 80% train = 80 samples, 20% val = 20 samples
        assert len(c0_data["X_train"]) == 80
        assert len(c0_data["y_train"]) == 80
        assert len(c0_data["X_val"]) == 20
        assert len(c0_data["y_val"]) == 20
        # Backward compatibility aliases
        assert len(c0_data["X"]) == 80
        assert len(c0_data["y"]) == 80

        val_data = np.load(os.path.join(c0_dir, "val_partition.npz"))
        assert len(val_data["X_val"]) == 20
        assert len(val_data["y_val"]) == 20

        # Test load_client_partition
        X_tr, y_tr, meta = load_client_partition(scenario_dir, 0, split="train")
        assert len(X_tr) == 80
        assert meta["client_val_ratio"] == 0.2

        X_va, y_va, _ = load_client_partition(scenario_dir, 0, split="val")
        assert len(X_va) == 20

        X_all, y_all, _ = load_client_partition(scenario_dir, 0, split="all")
        assert len(X_all) == 100


def test_create_federated_scenario_server_test_split():
    """Verify create_federated_scenario serializes and loads server holdout test data (30% upcoming data)."""
    import tempfile
    import os
    import numpy as np
    from src.federated.scenario import (
        create_federated_scenario,
        load_server_data,
        load_server_test_data
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        num_train = 70
        num_val = 15
        num_test = 30
        input_dim = 8
        num_classes = 3

        X_train = np.random.randn(num_train, input_dim).astype(np.float32)
        y_train = np.random.randint(0, num_classes, size=num_train).astype(np.int64)

        X_val = np.random.randn(num_val, input_dim).astype(np.float32)
        y_val = np.random.randint(0, num_classes, size=num_val).astype(np.int64)

        X_test = np.random.randn(num_test, input_dim).astype(np.float32)
        y_test = np.random.randint(0, num_classes, size=num_test).astype(np.int64)

        client_partitions = {0: np.arange(0, 35), 1: np.arange(35, 70)}

        scenario_dir = create_federated_scenario(
            scenario_name="test_server_test_scenario",
            client_partitions=client_partitions,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=[f"C{i}" for i in range(num_classes)],
            base_dir=tmp_dir,
            generate_plots=False,
            client_val_ratio=0.2
        )

        server_dir = os.path.join(scenario_dir, "server")
        assert os.path.exists(os.path.join(server_dir, "test_data.npz"))
        assert os.path.exists(os.path.join(server_dir, "global_test.npz"))

        # Verify load_server_data with split="val"
        X_v, y_v, meta = load_server_data(scenario_dir, split="val")
        assert len(X_v) == num_val
        assert meta["total_test_samples"] == num_test

        # Verify load_server_data with split="test"
        X_t, y_t, meta_t = load_server_data(scenario_dir, split="test")
        assert len(X_t) == num_test
        assert len(y_t) == num_test

        # Verify load_server_test_data helper
        X_t2, y_t2, _ = load_server_test_data(scenario_dir)
        assert len(X_t2) == num_test
        np.testing.assert_array_equal(X_t, X_t2)
        np.testing.assert_array_equal(y_t, y_t2)

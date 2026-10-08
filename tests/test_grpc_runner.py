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

"""
Flower Multi-Process gRPC Execution Runner for FL-IoT-IDS.

Coordinates Flower Central Server and Edge Client processes communicating over real
gRPC network sockets over TCP/IP (default port 8080).
- Pure distributed execution via standard gRPC network sockets
- Standalone OS processes for Server and Clients (realistic edge IoT deployment)
- Reads from and persists directly into scenarios/<scenario_name>/
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("grpc_runner")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is currently open and bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int = 8080, max_attempts: int = 20) -> int:
    """Finds the first available TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port


def run_flower_grpc(
    scenario_name: Optional[str] = None,
    federated_config: Optional[Union[Any, Any]] = None,
    num_clients: Optional[int] = None,
    rounds: Optional[int] = None,
    strategy: Optional[str] = None,
    mu: Optional[float] = None,
    local_epochs: Optional[int] = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    port: int = 8080,
    scenarios_dir: str = "scenarios",
    device: str = "cpu",
    stream_logs: bool = True,
    auto_find_port: bool = True,
    fraction_evaluate: Optional[float] = None,
    # Backward compatibility aliases
    scenario_name_or_config: Optional[Union[str, Any]] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes a complete multi-process Flower gRPC training session using FederatedConfig.

    Args:
        scenario_name: Folder name of the scenario under scenarios_dir (e.g. CONFIG.experiment_name).
        federated_config: FederatedConfig instance containing algorithm, rounds, local epochs, etc.
                          (e.g. CONFIG.federated).
        num_clients: Number of edge clients K to launch (overrides federated_config if provided).
        rounds: Number of federated communication rounds (overrides federated_config if provided).
        strategy: 'fedavg' or 'fedprox' (overrides federated_config if provided).
        mu: FedProx proximal coefficient (overrides federated_config if provided).
        local_epochs: Local training epochs per client per round (overrides federated_config if provided).
        batch_size: Client local mini-batch size (default: 64).
        learning_rate: Client local learning rate (default: 1e-3).
        port: Preferred gRPC TCP port (default: 8080).
        scenarios_dir: Root directory containing scenario partitions (default: 'scenarios').
        device: Compute device for clients ('cpu' or 'cuda').
        stream_logs: Whether to stream server output live to stdout.
        auto_find_port: If True, automatically find next free port if preferred is busy.
        fraction_evaluate: Fraction of clients to evaluate each round (default: 1.0).
        scenario_name_or_config: Backward-compatible positional argument.
        config: Backward-compatible alias for federated_config or Experiment.

    Returns:
        Dictionary containing round_history from the server.
    """
    target_fed = federated_config

    # Support legacy positional scenario_name_or_config
    if scenario_name is None and scenario_name_or_config is not None:
        if isinstance(scenario_name_or_config, str):
            scenario_name = scenario_name_or_config
        else:
            if hasattr(scenario_name_or_config, "federated"):
                target_fed = scenario_name_or_config.federated
                scenario_name = getattr(scenario_name_or_config, "experiment_name", None)
            else:
                target_fed = scenario_name_or_config

    # If first positional argument is an object (FederatedConfig or Experiment) rather than a string
    if scenario_name is not None and not isinstance(scenario_name, str):
        if hasattr(scenario_name, "federated"):
            target_fed = scenario_name.federated
            scenario_name = getattr(scenario_name, "experiment_name", None)
        else:
            target_fed = scenario_name
            scenario_name = getattr(target_fed, "scenario_name", None)

    # Support config keyword alias
    if target_fed is None and config is not None:
        if hasattr(config, "federated"):
            target_fed = config.federated
            if scenario_name is None:
                scenario_name = getattr(config, "experiment_name", None)
        else:
            target_fed = config
            if scenario_name is None:
                scenario_name = getattr(target_fed, "scenario_name", None)

    # If target_fed is an Experiment, extract federated sub-config
    if target_fed is not None and hasattr(target_fed, "federated"):
        if scenario_name is None:
            scenario_name = getattr(target_fed, "experiment_name", None)
        target_fed = target_fed.federated

    # Extract all parameters directly from FederatedConfig
    if target_fed is not None:
        if scenario_name is None and hasattr(target_fed, "scenario_name"):
            scenario_name = target_fed.scenario_name
        if num_clients is None and hasattr(target_fed, "num_clients"):
            num_clients = target_fed.num_clients
        if rounds is None and hasattr(target_fed, "num_rounds"):
            rounds = target_fed.num_rounds
        if strategy is None:
            if hasattr(target_fed, "federated_strategy"):
                strategy = target_fed.federated_strategy
            elif hasattr(target_fed, "strategy"):
                strategy = target_fed.strategy
            elif hasattr(target_fed, "algorithm"):
                strategy = target_fed.algorithm.name
        if mu is None:
            strat_name = str(strategy or getattr(target_fed, "strategy", "")).lower()
            if "prox" in strat_name:
                mu = float(getattr(target_fed, "mu", 0.05))
            else:
                mu = 0.0
        if local_epochs is None and hasattr(target_fed, "local_epochs"):
            local_epochs = target_fed.local_epochs
        if fraction_evaluate is None and hasattr(target_fed, "fraction_evaluate"):
            fraction_evaluate = target_fed.fraction_evaluate

    # Canonical defaults
    scenario_name = scenario_name or "federated_experiment"
    num_clients = int(num_clients) if num_clients is not None else 5
    rounds = int(rounds) if rounds is not None else 10
    if strategy is None and federated_config is not None:
        strategy = getattr(federated_config, "strategy", "fedavg").lower()
    else:
        strategy = str(strategy or "fedavg").lower()
        
    mu = float(mu) if mu is not None else (0.05 if "prox" in strategy else 0.0)
    local_epochs = int(local_epochs) if local_epochs is not None else 2
    fraction_evaluate = float(fraction_evaluate) if fraction_evaluate is not None else 0.0
    batch_size = int(batch_size) if batch_size is not None else 64
    learning_rate = float(learning_rate) if learning_rate is not None else 1e-3
    device = str(device or "cpu")

    scenario_dir = os.path.join(scenarios_dir, scenario_name)
    server_meta = os.path.join(scenario_dir, "server", "meta.json")
    if not os.path.exists(server_meta):
        raise FileNotFoundError(
            f"Scenario structure not found at '{scenario_dir}'. "
            f"Please run setup_scenario_hierarchy or create_federated_scenario first."
        )

    target_port = find_available_port(port) if auto_find_port else port
    server_address = f"127.0.0.1:{target_port}"

    print("=" * 80)
    print("🌐 FL-IoT-IDS: Flower Distributed gRPC Multi-Process Execution")
    print(f"Scenario Name:   {scenario_name}")
    print(f"Scenario Dir:    {scenario_dir}")
    print(f"Server Address:  {server_address}")
    print(f"Clients (K):     {num_clients}")
    print(f"Rounds (T):      {rounds}")
    print(f"Local Epochs:    {local_epochs}")
    print(f"Decentralized Eval: {fraction_evaluate * 100:.0f}% of clients")
    print(f"Batch Size:      {batch_size}")
    print(f"Learning Rate:   {learning_rate}")
    print(f"Strategy:        {strategy.upper()} (mu={mu})")
    print(f"Transport:       TCP/IP gRPC sockets (port {target_port})")
    print("=" * 80)

    server_proc: Optional[subprocess.Popen] = None
    client_procs: List[subprocess.Popen] = []

    try:
        # 1. Launch Flower Central Server process
        server_cmd = [
            sys.executable, "-m", "src.federated.flower_server",
            "--server-address", f"0.0.0.0:{target_port}",
            "--rounds", str(rounds),
            "--strategy", strategy,
            "--mu", str(mu),
            "--local-epochs", str(local_epochs),
            "--min-clients", str(num_clients),
            "--fraction-evaluate", str(fraction_evaluate),
            "--scenario-dir", scenario_dir
        ]
        logger.info(f"Starting Server process on port {target_port}...")
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 2. Wait for server socket initialization
        time.sleep(2.0)

        # 3. Launch K Client processes
        for client_id in range(num_clients):
            client_cmd = [
                sys.executable, "-m", "src.federated.flower_client",
                "--client-id", str(client_id),
                "--server-address", server_address,
                "--strategy", strategy,
                "--mu", str(mu),
                "--epochs", str(local_epochs),
                "--batch-size", str(batch_size),
                "--lr", str(learning_rate),
                "--scenario-dir", scenario_dir,
                "--device", device
            ]
            logger.info(f"Starting Client {client_id} process...")
            p = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            client_procs.append(p)
            time.sleep(0.3)

        logger.info(f"All {num_clients} edge clients connected over gRPC. Training in progress...\n")

        # 4. Stream Server logs in real time
        if stream_logs and server_proc.stdout is not None:
            for line in iter(server_proc.stdout.readline, ""):
                print(line, end="", flush=True)

        server_proc.wait()
        logger.info(f"Server process completed with exit code: {server_proc.returncode}")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Terminating all gRPC processes...")
    finally:
        # Ensure all client processes are cleanly terminated
        for p in client_procs:
            if p.poll() is None:
                p.terminate()
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()

    # 5. Load and return round history
    history_path = os.path.join(scenario_dir, "server", "round_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            round_history = json.load(f)
        logger.info(f"Successfully loaded round history from: {history_path}")
        return round_history

    logger.warning("round_history.json was not generated. Check process logs above.")
    return {}

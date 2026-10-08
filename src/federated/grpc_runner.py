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
    scenario_name_or_config: Optional[Union[str, Any]] = None,
    num_clients: Optional[int] = None,
    rounds: Optional[int] = None,
    strategy: Optional[str] = None,
    mu: Optional[float] = None,
    local_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    port: int = 8080,
    scenarios_dir: str = "scenarios",
    device: Optional[str] = None,
    stream_logs: bool = True,
    auto_find_port: bool = True,
    config: Optional[Any] = None,
    scenario_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a complete multi-process Flower gRPC training session.

    Accepts either an Experiment / FederatedConfig instance directly (via `config=CONFIG`
    or as first argument `run_flower_grpc(CONFIG)`), or individual parameters for
    backward compatibility.

    Args:
        scenario_name_or_config: Scenario folder name or Experiment / FederatedConfig instance.
        num_clients: Number of edge clients K to launch (overrides config if provided).
        rounds: Number of federated communication rounds (overrides config if provided).
        strategy: 'fedavg' or 'fedprox' (overrides config if provided).
        mu: FedProx proximal coefficient (0.0 for FedAvg, 0.05 for FedProx).
        local_epochs: Local training epochs per client per round (overrides config if provided).
        batch_size: Client local mini-batch size (overrides config if provided).
        learning_rate: Client local learning rate (overrides config if provided).
        port: Preferred gRPC TCP port (default: 8080).
        scenarios_dir: Root directory containing scenario partitions (default: 'scenarios').
        device: Compute device for clients ('cpu' or 'cuda').
        stream_logs: Whether to stream server output live to stdout.
        auto_find_port: If True, automatically find next free port if preferred is busy.
        config: Optional Experiment or FederatedConfig instance.
        scenario_name: Optional explicit scenario folder name.

    Returns:
        Dictionary containing round_history from the server.
    """
    # 1. Resolve config object if passed positionally or via keyword
    resolved_config = config
    if resolved_config is None and scenario_name_or_config is not None:
        if not isinstance(scenario_name_or_config, str):
            resolved_config = scenario_name_or_config
        else:
            scenario_name = scenario_name or scenario_name_or_config

    # 2. Extract hyperparameters from config
    if resolved_config is not None:
        if scenario_name is None:
            if hasattr(resolved_config, "experiment_name"):
                scenario_name = resolved_config.experiment_name
            elif isinstance(resolved_config, dict):
                scenario_name = resolved_config.get("experiment_name", resolved_config.get("scenario_name"))
        if num_clients is None:
            if hasattr(resolved_config, "num_clients"):
                num_clients = resolved_config.num_clients
            elif hasattr(resolved_config, "federated") and hasattr(resolved_config.federated, "num_clients"):
                num_clients = resolved_config.federated.num_clients
            elif isinstance(resolved_config, dict):
                num_clients = resolved_config.get("num_clients", resolved_config.get("federated", {}).get("num_clients"))
        if rounds is None:
            if hasattr(resolved_config, "num_rounds"):
                rounds = resolved_config.num_rounds
            elif hasattr(resolved_config, "federated") and hasattr(resolved_config.federated, "num_rounds"):
                rounds = resolved_config.federated.num_rounds
            elif isinstance(resolved_config, dict):
                rounds = resolved_config.get("num_rounds", resolved_config.get("federated", {}).get("num_rounds"))
        if strategy is None:
            if hasattr(resolved_config, "federated_strategy"):
                strategy = resolved_config.federated_strategy
            elif hasattr(resolved_config, "strategy"):
                strategy = resolved_config.strategy
            elif hasattr(resolved_config, "federated") and hasattr(resolved_config.federated, "algorithm"):
                strategy = resolved_config.federated.algorithm.name
            elif isinstance(resolved_config, dict):
                strategy = resolved_config.get(
                    "federated_strategy",
                    resolved_config.get("strategy", resolved_config.get("federated", {}).get("federated_strategy"))
                )
        if mu is None:
            strat_name = str(strategy or "").lower()
            if "prox" in strat_name:
                if hasattr(resolved_config, "mu"):
                    mu = float(resolved_config.mu)
                elif hasattr(resolved_config, "federated") and hasattr(resolved_config.federated, "mu"):
                    mu = float(resolved_config.federated.mu)
                elif isinstance(resolved_config, dict):
                    mu = float(resolved_config.get("mu", resolved_config.get("federated", {}).get("mu", 0.05)))
                else:
                    mu = 0.05
            else:
                mu = 0.0
        if local_epochs is None:
            if hasattr(resolved_config, "local_epochs"):
                local_epochs = resolved_config.local_epochs
            elif hasattr(resolved_config, "federated") and hasattr(resolved_config.federated, "local_epochs"):
                local_epochs = resolved_config.federated.local_epochs
            elif isinstance(resolved_config, dict):
                local_epochs = resolved_config.get("local_epochs", resolved_config.get("federated", {}).get("local_epochs"))
        if batch_size is None:
            if hasattr(resolved_config, "batch_size"):
                batch_size = resolved_config.batch_size
            elif hasattr(resolved_config, "data") and hasattr(resolved_config.data, "batch_size"):
                batch_size = resolved_config.data.batch_size
            elif isinstance(resolved_config, dict):
                batch_size = resolved_config.get("batch_size", resolved_config.get("data", {}).get("batch_size"))
        if learning_rate is None:
            if hasattr(resolved_config, "learning_rate"):
                learning_rate = resolved_config.learning_rate
            elif hasattr(resolved_config, "optimizer") and hasattr(resolved_config.optimizer, "learning_rate"):
                learning_rate = resolved_config.optimizer.learning_rate
            elif isinstance(resolved_config, dict):
                learning_rate = resolved_config.get("learning_rate", resolved_config.get("optimizer", {}).get("learning_rate"))
        if device is None:
            cfg_device = getattr(resolved_config, "device", None)
            if isinstance(resolved_config, dict):
                cfg_device = resolved_config.get("device", None)
            if cfg_device and str(cfg_device).lower() not in ("auto", "none"):
                device = str(cfg_device)

    # 3. Apply canonical defaults for any unresolved fields
    scenario_name = scenario_name or "federated_experiment"
    num_clients = int(num_clients) if num_clients is not None else 5
    rounds = int(rounds) if rounds is not None else 10
    strategy = str(strategy or "fedavg").lower()
    mu = float(mu) if mu is not None else (0.05 if "prox" in strategy else 0.0)
    local_epochs = int(local_epochs) if local_epochs is not None else 2
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

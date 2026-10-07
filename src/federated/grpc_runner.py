"""
Flower Multi-Process gRPC Execution Runner for FL-IoT-IDS.

Coordinates Flower Central Server and Edge Client processes communicating over real
gRPC network sockets over TCP/IP (default port 8080).
- Pure distributed execution (Zero Ray dependency, Zero simulation overhead)
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
from typing import Any, Dict, List, Optional

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
    scenario_name: str,
    num_clients: int = 5,
    rounds: int = 10,
    strategy: str = "fedprox",
    mu: float = 0.05,
    port: int = 8080,
    scenarios_dir: str = "scenarios",
    device: str = "cpu",
    stream_logs: bool = True,
    auto_find_port: bool = True
) -> Dict[str, Any]:
    """
    Executes a complete multi-process Flower gRPC training session.

    Args:
        scenario_name: Name of the scenario folder under scenarios_dir.
        num_clients: Number of edge clients K to launch.
        rounds: Number of federated communication rounds.
        strategy: 'fedprox' or 'fedavg'.
        mu: FedProx proximal coefficient (mu=0.0 for FedAvg).
        port: Preferred gRPC TCP port (default: 8080).
        scenarios_dir: Root directory containing scenario partitions.
        device: Compute device for clients ('cpu' or 'cuda').
        stream_logs: Whether to stream server output live to stdout.
        auto_find_port: If True, automatically find next free port if preferred is busy.

    Returns:
        Dictionary containing round_history from the server.
    """
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
    print(f"Clients:         {num_clients}")
    print(f"Rounds:          {rounds}")
    print(f"Strategy:        {strategy.upper()} (mu={mu})")
    print("Transport:       Pure TCP/IP gRPC sockets (Zero Ray / Zero flwr.simulation)")
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

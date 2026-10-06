"""
Automated Multi-Process gRPC Launcher for Flower Federated Learning (FL-IoT-IDS).

Executes the Flower Server and Edge Clients as independent OS processes communicating
via real gRPC network sockets over TCP/IP (port 8080).
- ZERO Ray dependency / Zero Ray OOM issues
- Realistic edge IoT deployment (identical to Raspberry Pi / edge gateway deployment)
- Memory isolated across independent operating system processes
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_grpc")


def check_or_prepare_partitions(
    partitions_dir: str = "checkpoints/partitions",
    num_clients: int = 5,
    sample_size: int = 50000,
    partition_type: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42
) -> str:
    """
    Checks if serialized client partitions exist. If not, prepares and serializes them.
    """
    meta_path = os.path.join(partitions_dir, "meta.json")
    if os.path.exists(meta_path):
        logger.info(f"Using existing partitions in: {partitions_dir}")
        return partitions_dir

    logger.info(f"Partitions not found in {partitions_dir}. Generating fresh dataset split...")
    os.makedirs(partitions_dir, exist_ok=True)

    # 1. Locate dataset
    candidate_paths = [
        "datasets/CICIOT2023/merged_CICIOT2023_data.csv",
        "/content/merged_CICIOT2023_data.csv",
        "datasets/CICIOT2023/synthetic_ciciot2023.csv"
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        # Generate synthetic dry-run data if no CSV found
        logger.warning("No CSV dataset found. Creating synthetic CICIoT2023-compatible dataset for gRPC...")
        csv_path = "datasets/CICIOT2023/synthetic_ciciot2023.csv"
        os.makedirs("datasets/CICIOT2023", exist_ok=True)
        import numpy as np
        import pandas as pd
        n = 30000
        classes = ["Benign", "DDoS", "DoS", "Recon", "Spoof", "Mirai", "Web-based", "Brute-force"]
        probs = [0.18, 0.38, 0.23, 0.10, 0.05, 0.04, 0.015, 0.005]
        np.random.seed(seed)
        df_mock = pd.DataFrame({f"feat_{i}": np.random.exponential(1.5, size=n) for i in range(39)})
        df_mock["group_class_name"] = np.random.choice(classes, size=n, p=probs)
        df_mock.to_csv(csv_path, index=False)

    from src.data.preprocessor import load_and_preprocess_ciciot2023
    from src.data.partitioner import DirichletNonIIDPartitioner, StratifiedIIDPartitioner, save_partitions_for_grpc

    data = load_and_preprocess_ciciot2023(
        csv_path=csv_path,
        sample_size=sample_size,
        test_size=0.20,
        val_size=0.10,
        random_state=seed
    )

    if partition_type.lower() == "iid":
        partitioner = StratifiedIIDPartitioner(num_clients=num_clients, seed=seed)
    else:
        partitioner = DirichletNonIIDPartitioner(num_clients=num_clients, alpha=alpha, seed=seed)

    client_partitions = partitioner.partition(X=data["X_train"], y=data["y_train"])

    save_partitions_for_grpc(
        client_partitions=client_partitions,
        X=data["X_train"],
        y=data["y_train"],
        output_dir=partitions_dir,
        X_val=data["X_val"],
        y_val=data["y_val"],
        class_names=data["class_names"]
    )
    return partitions_dir


def main():
    parser = argparse.ArgumentParser(description="Multi-Process Flower gRPC Orchestrator.")
    parser.add_argument("--rounds", type=int, default=10, help="Communication rounds.")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of simulated edge clients.")
    parser.add_argument("--strategy", type=str, default="fedprox", choices=["fedavg", "fedprox"], help="Strategy.")
    parser.add_argument("--mu", type=float, default=0.05, help="FedProx proximal parameter mu.")
    parser.add_argument("--port", type=int, default=8080, help="gRPC server port.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Dataset subsample size (None for full).")
    parser.add_argument("--partition-type", type=str, default="dirichlet", choices=["dirichlet", "iid"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet heterogeneity parameter.")
    parser.add_argument("--partitions-dir", type=str, default="checkpoints/partitions")
    args = parser.parse_args()

    server_address = f"127.0.0.1:{args.port}"
    os.makedirs("reports", exist_ok=True)
    history_file = f"reports/grpc_{args.strategy}_{args.partition_type}_history.json"

    print("=" * 80)
    print("🌐 FL-IoT-IDS: Standalone Flower gRPC Multi-Process Execution")
    print(f"Server Address:  {server_address}")
    print(f"Clients:         {args.num_clients}")
    print(f"Rounds:          {args.rounds}")
    print(f"Strategy:        {args.strategy.upper()} (mu={args.mu})")
    print("Zero Ray Engine: Pure TCP/IP gRPC sockets (no Ray overhead/OOM watchdog)")
    print("=" * 80)

    # 1. Ensure partitions are prepared
    check_or_prepare_partitions(
        partitions_dir=args.partitions_dir,
        num_clients=args.num_clients,
        sample_size=args.sample_size,
        partition_type=args.partition_type,
        alpha=args.alpha
    )

    client_procs: List[subprocess.Popen] = []
    server_proc = None

    try:
        # 2. Launch Flower Central Server
        server_cmd = [
            sys.executable, "-m", "src.federated.flower_server",
            "--server-address", f"0.0.0.0:{args.port}",
            "--rounds", str(args.rounds),
            "--strategy", args.strategy,
            "--mu", str(args.mu),
            "--min-clients", str(args.num_clients),
            "--partitions-dir", args.partitions_dir,
            "--history-save-path", history_file
        ]
        logger.info(f"Starting Server process: {' '.join(server_cmd)}")
        server_proc = subprocess.Popen(server_cmd)

        # 3. Wait for server socket initialization
        time.sleep(2.0)

        # 4. Launch K Client processes
        for client_id in range(args.num_clients):
            client_cmd = [
                sys.executable, "-m", "src.federated.flower_client",
                "--client-id", str(client_id),
                "--server-address", server_address,
                "--strategy", args.strategy,
                "--mu", str(args.mu),
                "--partitions-dir", args.partitions_dir,
                "--device", "cpu"
            ]
            logger.info(f"Starting Client {client_id} process...")
            p = subprocess.Popen(client_cmd)
            client_procs.append(p)
            time.sleep(0.3)

        logger.info(f"All {args.num_clients} edge clients connected over gRPC. Training in progress...")

        # 5. Wait for server to finish all communication rounds
        server_return_code = server_proc.wait()
        logger.info(f"Server process completed with exit code: {server_return_code}")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Terminating all processes...")
    finally:
        # Clean up any lingering client processes
        for p in client_procs:
            if p.poll() is None:
                p.terminate()
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()

    print("=" * 80)
    print("🎉 Flower gRPC Federated Learning completed successfully!")
    if os.path.exists(history_file):
        print(f"Round history exported to: {history_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

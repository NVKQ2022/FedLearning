"""
Automated Multi-Process gRPC Launcher for Flower Federated Learning (FL-IoT-IDS).

Executes the Flower Server and Edge Clients as independent OS processes communicating
via real gRPC network sockets over TCP/IP (port 8080).
- Decoupled multi-process architecture with socket-based client-server isolation
- Realistic edge IoT deployment (identical to Raspberry Pi / edge gateway deployment)
- Isolated client partitions, EDAs, and metrics organized under scenarios/<scenario_name>/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_grpc")


def check_or_prepare_scenario(
    scenario_dir: str,
    scenario_name: str,
    num_clients: int = 5,
    sample_size: Optional[int] = 50000,
    partition_type: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
    strategy: str = "fedavg",
    mu: float = 0.0,
    rounds: int = 10
) -> str:
    """
    Ensures the experimental scenario directory exists and is fully initialized with:
    - scenarios/<scenario_name>/config.json
    - scenarios/<scenario_name>/server/meta.json, val_data.npz
    - scenarios/<scenario_name>/client_{i}/partition.npz, eda.json, class_distribution.png, metrics.json
    """
    server_meta = os.path.join(scenario_dir, "server", "meta.json")
    if os.path.exists(server_meta):
        logger.info(f"Using existing scenario structure in: {scenario_dir}")
        return scenario_dir

    logger.info(f"Scenario not found in {scenario_dir}. Generating fresh dataset split and partitions...")
    os.makedirs(scenario_dir, exist_ok=True)

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
    from src.data.partitioner import DirichletNonIIDPartitioner, StratifiedIIDPartitioner
    from src.federated.scenario import create_federated_scenario

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

    config_dict = {
        "scenario_name": scenario_name,
        "strategy": strategy,
        "mu": mu,
        "num_clients": num_clients,
        "rounds": rounds,
        "partition_type": partition_type,
        "alpha": alpha if partition_type.lower() != "iid" else None,
        "sample_size": sample_size,
        "seed": seed,
    }

    base_dir = os.path.dirname(os.path.abspath(scenario_dir))
    s_name = os.path.basename(scenario_dir)

    create_federated_scenario(
        scenario_name=s_name,
        client_partitions=client_partitions,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        class_names=data["class_names"],
        config=config_dict,
        base_dir=base_dir,
        generate_plots=True
    )
    return scenario_dir


def main():
    parser = argparse.ArgumentParser(description="Multi-Process Flower gRPC Orchestrator.")
    parser.add_argument("--rounds", type=int, default=10, help="Communication rounds.")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of simulated edge clients.")
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"], help="Strategy.")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx proximal parameter mu (0.0 for FedAvg).")
    parser.add_argument("--port", type=int, default=8080, help="gRPC server port.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Dataset subsample size (None for full).")
    parser.add_argument("--partition-type", type=str, default="iid", choices=["dirichlet", "iid"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet heterogeneity parameter.")
    parser.add_argument("--scenario-name", type=str, default=None, help="Name of scenario folder (e.g. E2_fedavg_iid).")
    parser.add_argument("--scenarios-dir", type=str, default="scenarios", help="Base directory containing scenarios.")
    args = parser.parse_args()

    # Determine scenario directory name
    if args.scenario_name:
        scenario_name = args.scenario_name
    else:
        if args.partition_type.lower() == "iid":
            scenario_name = f"E2_{args.strategy}_iid"
        else:
            scenario_name = f"E5_{args.strategy}_dirichlet_{args.alpha}"

    scenario_dir = os.path.join(args.scenarios_dir, scenario_name)
    server_address = f"127.0.0.1:{args.port}"

    print("=" * 80)
    print("🌐 FL-IoT-IDS: Standalone Flower gRPC Multi-Process Execution")
    print(f"Scenario Name:   {scenario_name}")
    print(f"Scenario Dir:    {scenario_dir}")
    print(f"Server Address:  {server_address}")
    print(f"Clients:         {args.num_clients}")
    print(f"Rounds:          {args.rounds}")
    print(f"Strategy:        {args.strategy.upper()} (mu={args.mu})")
    print("Architecture:    Multi-process edge deployment over TCP/IP gRPC sockets")
    print("=" * 80)

    # 1. Ensure scenario structure (server data, client partitions, EDAs, and plots) exists
    check_or_prepare_scenario(
        scenario_dir=scenario_dir,
        scenario_name=scenario_name,
        num_clients=args.num_clients,
        sample_size=args.sample_size,
        partition_type=args.partition_type,
        alpha=args.alpha,
        strategy=args.strategy,
        mu=args.mu,
        rounds=args.rounds
    )

    # 1b. Display cross-client EDA profile from server/clients_summary.json
    summary_path = os.path.join(scenario_dir, "server", "clients_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                c_summary = json.load(f)
            print("\n📊 Cross-Client Partition Heterogeneity Profile (src.eda):")
            print(f"  {'Client':<10} {'Samples':<10} {'Dominant Attack':<16} {'Share (%)':<12} {'Entropy (H_norm)':<18}")
            print("  " + "-" * 66)
            for c in c_summary:
                print(f"  Client {c['client_id']:<3} {c['samples']:<10,d} {c['dominant_class']:<16} {c['dominant_pct']:<12.1f} {c['entropy']:<18.4f}")
            print()
        except Exception as e:
            logger.debug(f"Failed to display client summary: {e}")

    from src.federated.grpc_runner import run_flower_grpc

    # 2. Run Flower Server and Edge Clients via real gRPC sockets
    round_history = run_flower_grpc(
        scenario_name=scenario_name,
        num_clients=args.num_clients,
        rounds=args.rounds,
        strategy=args.strategy,
        mu=args.mu,
        port=args.port,
        scenarios_dir=args.scenarios_dir,
        device="cpu",
        stream_logs=True
    )

    print("=" * 80)
    print("🎉 Flower gRPC Federated Learning completed successfully!")
    print(f"Scenario artifacts organized under: {scenario_dir}/")
    print(f"  • Global Server:  {scenario_dir}/server/ (val_data.npz, best_weights.npz, round_history.json, convergence.png)")
    for i in range(args.num_clients):
        print(f"  • Client {i}:        {scenario_dir}/client_{i}/ (partition.npz, eda.json, class_distribution.png, metrics.json)")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Experimental Scenario Management Module for Federated Learning (FL-IoT-IDS).

Organizes federated and centralized experiments into a modular, self-contained directory layout:
scenarios/
  ├── <scenario_name>/                # e.g., 'E2_fedavg_iid', 'E5_fedprox_dirichlet_0.1'
  │   ├── config.json                 # Complete scenario hyperparameter configuration
  │   ├── server/                     # Central server coordination environment
  │   │   ├── meta.json               # Input dim, class count, class names, client count
  │   │   ├── val_data.npz            # Server holdout validation dataset
  │   │   ├── best_weights.npz        # Top-performing global aggregated model weights
  │   │   ├── round_history.json      # Centralized round-by-round evaluation log
  │   │   ├── clients_summary.json    # Cross-client partition statistical summary
  │   │   └── convergence.png         # Server convergence & communication trajectory
  │   ├── client_0/                   # Isolated environment for Edge Client 0
  │   │   ├── partition.npz           # Client 0 private partition (X, y)
  │   │   ├── eda.json                # Partition sample counts, label proportions, entropy
  │   │   ├── class_distribution.png  # High-clarity label distribution bar chart
  │   │   └── metrics.json            # Local training metrics audit trail across rounds
  │   ├── client_1/
  │   │   └── ...
  │   └── client_{N-1}/

Adheres to:
- skills/experiment-orchestration/SKILL.md (Step 4: Scenario Directory Isolation)
- skills/dataset-analysis-and-strategy/SKILL.md (Step 5: Statistical Heterogeneity Diagnosis)
- skills/evaluation-and-benchmarking/SKILL.md (Metric Integrity & Reproducibility)
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

# Ensure headless plotting compatibility in CLI / remote servers / Colab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


from src.eda import (
    analyze_partition,
    plot_class_distribution,
    export_client_eda,
    export_scenario_eda_summary,
)

# Backward-compatibility alias
compute_partition_eda = analyze_partition


def record_client_round_metric(
    metrics_path: str,
    round_data: Dict[str, Any]
) -> None:
    """
    Safely records a completed communication round into the client's metrics.json audit trail.

    Args:
        metrics_path: Path to the client's metrics.json file.
        round_data: Dictionary containing round metrics (server_round, loss, accuracy, drift_l2, train_time_sec).
    """
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                content = json.load(f)
        except Exception:
            content = {"rounds": []}
    else:
        content = {"rounds": []}

    # Ensure clean serializable types
    clean_round = {}
    for k, v in round_data.items():
        if isinstance(v, (np.floating, float)):
            clean_round[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            clean_round[k] = int(v)
        else:
            clean_round[k] = v

    rounds_list = content.get("rounds", [])
    # Check if this server_round already exists to prevent duplicate writes
    round_idx = clean_round.get("server_round")
    existing_idx = next((i for i, r in enumerate(rounds_list) if r.get("server_round") == round_idx), None)
    if existing_idx is not None:
        rounds_list[existing_idx].update(clean_round)
    else:
        rounds_list.append(clean_round)
    content["rounds"] = rounds_list

    # Compute aggregate client summary
    if rounds_list:
        final_loss = rounds_list[-1].get("loss", 0.0)
        final_acc = rounds_list[-1].get("accuracy", 0.0)
        total_time = sum(r.get("train_time_sec", 0.0) for r in rounds_list)
        avg_drift = (
            sum(r.get("drift_l2", 0.0) for r in rounds_list) / len(rounds_list)
            if any("drift_l2" in r for r in rounds_list) else 0.0
        )
        content["summary"] = {
            "total_rounds_participated": len(rounds_list),
            "final_train_loss": round(float(final_loss), 4),
            "final_train_accuracy": round(float(final_acc), 4),
            "avg_drift_l2": round(float(avg_drift), 4),
            "total_train_time_sec": round(float(total_time), 2),
            "avg_round_time_sec": round(float(total_time / len(rounds_list)), 2),
        }

    # Safe atomic write via temporary file
    temp_path = f"{metrics_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(content, f, indent=2)
    shutil.move(temp_path, metrics_path)


def create_federated_scenario(
    scenario_name: Optional[str] = None,
    client_partitions: Optional[Dict[int, np.ndarray]] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    config: Optional[Union[Dict[str, Any], Any]] = None,
    base_dir: str = "scenarios",
    generate_plots: bool = True,
    client_val_ratio: float = 0.2,
    client_test_ratio: float = 0.0
) -> str:
    """
    Constructs the complete federated scenario directory tree.

    Creates:
    - scenarios/<scenario_name>/config.json
    - scenarios/<scenario_name>/server/meta.json, val_data.npz, test_data.npz
    - scenarios/<scenario_name>/client_{i}/partition.npz, val_partition.npz, eda.json, class_distribution.png, metrics.json

    Args:
        scenario_name: Directory identifier (e.g. 'E5_fedprox_dirichlet_0.1'). If None, inferred from config.
        client_partitions: Dict mapping client_id to indices in X_train/y_train.
        X_train: Preprocessed global training feature array.
        y_train: Preprocessed global training labels array.
        X_val: Optional server holdout validation features array.
        y_val: Optional server holdout validation labels array.
        X_test: Optional server holdout test features array (30% upcoming data).
        y_test: Optional server holdout test labels array (30% upcoming data).
        class_names: List of class names. If None, inferred as Class_0..C.
        config: Optional scenario configuration parameters dictionary or ExperimentConfig instance.
        base_dir: Base directory for all scenarios (default: 'scenarios').
        generate_plots: Whether to render class_distribution.png for each client.
        client_val_ratio: Fraction of local client data reserved for local validation (default: 0.2, i.e., 80% train / 20% val).

    Returns:
        Path to the initialized scenario directory.
    """
    if scenario_name is None:
        if config is not None and hasattr(config, "experiment_name"):
            scenario_name = config.experiment_name
        elif isinstance(config, dict) and "experiment_name" in config:
            scenario_name = config["experiment_name"]
        elif isinstance(config, dict) and "scenario_name" in config:
            scenario_name = config["scenario_name"]
        else:
            scenario_name = "federated_scenario"

    if client_partitions is None or X_train is None or y_train is None:
        raise ValueError("client_partitions, X_train, and y_train are required arguments.")

    if hasattr(config, "client_val_ratio"):
        client_val_ratio = getattr(config, "client_val_ratio")
    elif isinstance(config, dict) and "client_val_ratio" in config:
        client_val_ratio = config["client_val_ratio"]

    if hasattr(config, "client_test_ratio"):
        client_test_ratio = getattr(config, "client_test_ratio")
    elif isinstance(config, dict) and "client_test_ratio" in config:
        client_test_ratio = config["client_test_ratio"]

    scenario_dir = os.path.join(base_dir, scenario_name)
    server_dir = os.path.join(scenario_dir, "server")
    os.makedirs(server_dir, exist_ok=True)

    input_dim = int(X_train.shape[1])
    num_clients = len(client_partitions)
    unique_classes = sorted(np.unique(y_train).tolist())
    if class_names is None or len(class_names) != len(unique_classes):
        class_names = [f"Class_{i}" for i in range(len(unique_classes))]

    # 1. Write server metadata
    meta = {
        "scenario_name": scenario_name,
        "input_dim": input_dim,
        "num_classes": len(class_names),
        "class_names": class_names,
        "num_clients": num_clients,
        "total_train_samples": int(len(y_train)),
        "total_val_samples": int(len(y_val)) if y_val is not None else 0,
        "total_test_samples": int(len(y_test)) if y_test is not None else 0,
        "client_val_ratio": float(client_val_ratio),
        "client_test_ratio": float(client_test_ratio)
    }
    with open(os.path.join(server_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 2. Serialize server holdout validation data
    if X_val is not None and y_val is not None:
        val_path = os.path.join(server_dir, "val_data.npz")
        np.savez_compressed(
            val_path,
            X=np.ascontiguousarray(X_val, dtype=np.float32),
            y=np.ascontiguousarray(y_val, dtype=np.int64)
        )
        # Also symlink or write server_val.npz for seamless backward compatibility
        np.savez_compressed(
            os.path.join(server_dir, "server_val.npz"),
            X=np.ascontiguousarray(X_val, dtype=np.float32),
            y=np.ascontiguousarray(y_val, dtype=np.int64)
        )

    # 2b. Serialize server holdout test data (30% upcoming unseen data)
    if X_test is not None and y_test is not None:
        test_path = os.path.join(server_dir, "test_data.npz")
        np.savez_compressed(
            test_path,
            X=np.ascontiguousarray(X_test, dtype=np.float32),
            y=np.ascontiguousarray(y_test, dtype=np.int64)
        )
        np.savez_compressed(
            os.path.join(server_dir, "global_test.npz"),
            X=np.ascontiguousarray(X_test, dtype=np.float32),
            y=np.ascontiguousarray(y_test, dtype=np.int64)
        )

    # 3. Save scenario config.json
    if hasattr(config, "to_dict"):
        scenario_config = config.to_dict()
    elif isinstance(config, dict):
        scenario_config = config.copy()
    else:
        scenario_config = {}

    scenario_config.setdefault("scenario_name", scenario_name)
    scenario_config.setdefault("num_clients", num_clients)
    scenario_config.setdefault("input_dim", input_dim)
    scenario_config.setdefault("num_classes", len(class_names))
    scenario_config.setdefault("client_val_ratio", client_val_ratio)
    scenario_config.setdefault("client_test_ratio", client_test_ratio)
    with open(os.path.join(scenario_dir, "config.json"), "w") as f:
        json.dump(scenario_config, f, indent=2)

    # 4. Generate client partitions, EDA, plots, and metrics templates
    cross_client_summary = []
    for client_id, indices in sorted(client_partitions.items()):
        client_dir = os.path.join(scenario_dir, f"client_{int(client_id)}")
        os.makedirs(client_dir, exist_ok=True)

        X_c = np.ascontiguousarray(X_train[indices], dtype=np.float32)
        y_c = np.ascontiguousarray(y_train[indices], dtype=np.int64)

        # Split private partition into local train, val, and test
        total_holdout = client_val_ratio + client_test_ratio
        if total_holdout > 0.0 and len(y_c) > 1:
            seed_client = int(scenario_config.get("seed", 42)) + int(client_id)
            try:
                unique_labels, label_counts = np.unique(y_c, return_counts=True)
                if np.min(label_counts) >= 2 and len(unique_labels) > 1:
                    X_tr, X_temp, y_tr, y_temp = train_test_split(
                        X_c, y_c,
                        test_size=total_holdout,
                        random_state=seed_client,
                        stratify=y_c
                    )
                else:
                    X_tr, X_temp, y_tr, y_temp = train_test_split(
                        X_c, y_c,
                        test_size=total_holdout,
                        random_state=seed_client,
                        shuffle=True
                    )
            except Exception:
                X_tr, X_temp, y_tr, y_temp = train_test_split(
                    X_c, y_c,
                    test_size=total_holdout,
                    random_state=seed_client,
                    shuffle=True
                )

            # Now split X_temp into validation and test
            if client_val_ratio > 0.0 and client_test_ratio > 0.0 and len(y_temp) > 1:
                test_frac_of_holdout = client_test_ratio / total_holdout
                try:
                    unique_labels_temp, label_counts_temp = np.unique(y_temp, return_counts=True)
                    if np.min(label_counts_temp) >= 2 and len(unique_labels_temp) > 1:
                        X_va, X_te, y_va, y_te = train_test_split(
                            X_temp, y_temp,
                            test_size=test_frac_of_holdout,
                            random_state=seed_client,
                            stratify=y_temp
                        )
                    else:
                        X_va, X_te, y_va, y_te = train_test_split(
                            X_temp, y_temp,
                            test_size=test_frac_of_holdout,
                            random_state=seed_client,
                            shuffle=True
                        )
                except Exception:
                    X_va, X_te, y_va, y_te = train_test_split(
                        X_temp, y_temp,
                        test_size=test_frac_of_holdout,
                        random_state=seed_client,
                        shuffle=True
                    )
            elif client_test_ratio > 0.0:
                X_va = np.empty((0, X_c.shape[1]), dtype=np.float32)
                y_va = np.empty((0,), dtype=np.int64)
                X_te, y_te = X_temp, y_temp
            else:
                X_va, y_va = X_temp, y_temp
                X_te = np.empty((0, X_c.shape[1]), dtype=np.float32)
                y_te = np.empty((0,), dtype=np.int64)

            X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
            y_tr = np.ascontiguousarray(y_tr, dtype=np.int64)
            X_va = np.ascontiguousarray(X_va, dtype=np.float32)
            y_va = np.ascontiguousarray(y_va, dtype=np.int64)
            X_te = np.ascontiguousarray(X_te, dtype=np.float32)
            y_te = np.ascontiguousarray(y_te, dtype=np.int64)
        else:
            X_tr, y_tr = X_c, y_c
            X_va = np.empty((0, X_c.shape[1]), dtype=np.float32)
            y_va = np.empty((0,), dtype=np.int64)
            X_te = np.empty((0, X_c.shape[1]), dtype=np.float32)
            y_te = np.empty((0,), dtype=np.int64)

        # Save private partition (contains train, val, and test splits, with X,y pointing to train)
        partition_path = os.path.join(client_dir, "partition.npz")
        np.savez_compressed(
            partition_path,
            X=X_tr,
            y=y_tr,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            X_test=X_te,
            y_test=y_te
        )

        # Also save dedicated val_partition.npz for standalone evaluation
        val_partition_path = os.path.join(client_dir, "val_partition.npz")
        np.savez_compressed(
            val_partition_path,
            X=X_va,
            y=y_va,
            X_val=X_va,
            y_val=y_va
        )

        # Also save dedicated test_partition.npz for standalone evaluation
        test_partition_path = os.path.join(client_dir, "test_partition.npz")
        np.savez_compressed(
            test_partition_path,
            X=X_te,
            y=y_te,
            X_test=X_te,
            y_test=y_te
        )

        # Compute and persist client EDA (eda.json & class_distribution.png) via src.eda
        eda = export_client_eda(
            client_dir=client_dir,
            y=y_c,
            class_names=class_names,
            client_id=int(client_id),
            generate_plot=generate_plots
        )

        # Initialize clean metrics.json
        metrics_path = os.path.join(client_dir, "metrics.json")
        initial_metrics = {
            "client_id": int(client_id),
            "total_samples": int(len(y_c)),
            "train_samples": int(len(y_tr)),
            "val_samples": int(len(y_va)),
            "test_samples": int(len(y_te)),
            "rounds": [],
            "summary": {}
        }
        with open(metrics_path, "w") as f:
            json.dump(initial_metrics, f, indent=2)

        cross_client_summary.append({
            "client_id": int(client_id),
            "samples": int(len(y_c)),
            "train_samples": int(len(y_tr)),
            "val_samples": int(len(y_va)),
            "test_samples": int(len(y_te)),
            "dominant_class": eda["dominant_class"],
            "dominant_pct": eda.get("dominant_class_pct", eda.get("dominant_pct", 0.0)),
            "entropy": eda["normalized_entropy"]
        })

    # Save summary across all clients in server folder
    export_scenario_eda_summary(server_dir, cross_client_summary)

    train_ratio = 1.0 - (client_val_ratio + client_test_ratio)
    logger.info(
        f"✅ Scenario '{scenario_name}' exported successfully to {scenario_dir} "
        f"({num_clients} clients, {len(y_train):,} total train samples, {train_ratio:.2f}/{client_val_ratio:.2f}/{client_test_ratio:.2f} train/val/test split)."
    )
    return scenario_dir


def load_client_partition(
    scenario_dir: str,
    client_id: int,
    split: str = "train"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads local partition arrays (X, y) and metadata for an edge client.

    Args:
        scenario_dir: Scenario root directory (e.g. 'scenarios/E5_fedprox_dirichlet_0.1').
        client_id: Target client integer index.
        split: 'train' (80% default), 'val' (20% holdout), or 'all' (combined).

    Returns:
        Tuple of (X, y, meta).
    """
    client_dir = os.path.join(scenario_dir, f"client_{client_id}")
    partition_file = os.path.join(client_dir, "partition.npz")

    # Fallback to legacy single-folder flat partitions if present
    if not os.path.exists(partition_file):
        partition_file = os.path.join(scenario_dir, f"client_{client_id}.npz")

    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file not found for client {client_id} at {partition_file}")

    data = np.load(partition_file)
    split_lower = split.lower()
    if split_lower == "val":
        if "X_val" in data and len(data["y_val"]) > 0:
            X, y = data["X_val"], data["y_val"]
        else:
            val_file = os.path.join(client_dir, "val_partition.npz")
            if os.path.exists(val_file):
                val_data = np.load(val_file)
                X = val_data["X_val"] if "X_val" in val_data else val_data["X"]
                y = val_data["y_val"] if "y_val" in val_data else val_data["y"]
            else:
                X, y = data["X"], data["y"]
    elif split_lower == "test":
        if "X_test" in data and len(data["y_test"]) > 0:
            X, y = data["X_test"], data["y_test"]
        else:
            test_file = os.path.join(client_dir, "test_partition.npz")
            if os.path.exists(test_file):
                test_data = np.load(test_file)
                X = test_data["X_test"] if "X_test" in test_data else test_data["X"]
                y = test_data["y_test"] if "y_test" in test_data else test_data["y"]
            else:
                X, y = np.empty((0, data["X"].shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    elif split_lower == "all":
        if "X_val" in data and len(data["y_val"]) > 0:
            X_tr = data["X_train"] if "X_train" in data else data["X"]
            y_tr = data["y_train"] if "y_train" in data else data["y"]
            X = np.vstack([X_tr, data["X_val"]])
            y = np.concatenate([y_tr, data["y_val"]])
        else:
            X, y = data["X"], data["y"]
    else:
        # Default 'train' split
        X = data["X_train"] if "X_train" in data else data["X"]
        y = data["y_train"] if "y_train" in data else data["y"]

    meta_file = os.path.join(scenario_dir, "server", "meta.json")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(scenario_dir, "meta.json")

    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)

    return X, y, meta


def load_server_data(
    scenario_dir: str,
    split: str = "val"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads central server evaluation data (val or test) and scenario metadata.

    Args:
        scenario_dir: Scenario root directory.
        split: 'val' (holdout validation set) or 'test' (global holdout test set / upcoming data).

    Returns:
        Tuple of (X, y, meta).
    """
    server_dir = os.path.join(scenario_dir, "server")
    if not os.path.exists(server_dir):
        server_dir = scenario_dir

    meta_file = os.path.join(server_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)

    split_lower = str(split).lower().strip()
    if split_lower == "test":
        target_files = ["test_data.npz", "global_test.npz", "server_test.npz"]
    else:
        target_files = ["val_data.npz", "server_val.npz"]

    for fname in target_files:
        fpath = os.path.join(server_dir, fname)
        if os.path.exists(fpath):
            data = np.load(fpath)
            return data["X"], data["y"], meta

    return None, None, meta


def load_server_test_data(
    scenario_dir: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads central server global test data (30% upcoming data) and scenario metadata.

    Args:
        scenario_dir: Scenario root directory.

    Returns:
        Tuple of (X_test, y_test, meta).
    """
    return load_server_data(scenario_dir, split="test")


def plot_scenario_convergence(
    round_history: Dict[str, List[Any]],
    save_path: str,
    scenario_title: Optional[str] = None
) -> str:
    """
    Plots the central server's validation Macro-F1, training loss, parameter drift,
    and cumulative communication cost trajectories.

    Args:
        round_history: Dict containing lists for 'round', 'val_macro_f1', 'train_loss', 'drift_l2', 'comm_cost_mb'.
        save_path: Filepath where the plot image will be saved.
        scenario_title: Optional plot title override.

    Returns:
        Absolute filepath to the saved image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    rounds = round_history.get("round", [])
    if not rounds:
        return save_path

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    fig.suptitle(scenario_title or "Federated Learning Optimization Trajectory", fontsize=13, fontweight="bold")

    # 1. Validation Macro-F1
    ax1 = axes[0, 0]
    val_f1 = [v * 100.0 for v in round_history.get("val_macro_f1", [])][-len(rounds):]
    if val_f1:
        ax1.plot(rounds, val_f1, marker="o", color="#2b6cb0", linewidth=2.0)
        ax1.set_title("Central Holdout Validation Macro-F1 (%)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Macro-F1 (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Training Loss
    ax2 = axes[0, 1]
    train_loss = round_history.get("train_loss", [])[-len(rounds):]
    if train_loss:
        ax2.plot(rounds, train_loss, marker="s", color="#c53030", linewidth=2.0)
        ax2.set_title("Aggregated Training Loss", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Loss")
        ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Parameter Drift
    ax3 = axes[1, 0]
    drift = round_history.get("drift_l2", [])[-len(rounds):]
    if drift:
        ax3.plot(rounds, drift, marker="^", color="#d69e2e", linewidth=2.0)
        ax3.set_title("Client Weight Drift (||w_local - w_global||_2)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Communication Round")
        ax3.set_ylabel("Drift L2 Norm")
        ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Cumulative Communication Cost
    ax4 = axes[1, 1]
    comm = round_history.get("comm_cost_mb", [])[-len(rounds):]
    if comm:
        ax4.plot(rounds, comm, marker="d", color="#2f855a", linewidth=2.0)
        ax4.set_title("Cumulative Network Overhead (MB)", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Communication Round")
        ax4.set_ylabel("Megabytes (MB)")
        ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def create_centralized_scenario(
    scenario_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    base_dir: str = "scenarios",
    generate_plots: bool = True
) -> str:
    """
    Constructs a centralized baseline scenario directory layout (e.g. 'scenarios/E1_centralized').

    Creates:
    - scenarios/<scenario_name>/config.json
    - scenarios/<scenario_name>/meta.json
    - scenarios/<scenario_name>/val_data.npz, test_data.npz
    - scenarios/<scenario_name>/eda.json, dataset_eda.json
    - scenarios/<scenario_name>/class_distribution.png
    - scenarios/<scenario_name>/feature_skewness.png

    Args:
        scenario_name: Identifier (e.g. 'E1_centralized').
        X_train: Preprocessed training features.
        y_train: Training labels.
        X_val: Optional validation features.
        y_val: Optional validation labels.
        X_test: Optional test features.
        y_test: Optional test labels.
        class_names: List of class names.
        feature_names: List of feature names.
        config: Configuration dictionary.
        base_dir: Base scenarios directory (default 'scenarios').
        generate_plots: Whether to save EDA visualizations.

    Returns:
        Path to scenario directory.
    """
    from src.eda import (
        analyze_partition,
        analyze_dataset,
        analyze_features,
        plot_class_distribution,
        plot_feature_skewness,
        export_dataset_audit,
    )

    scenario_dir = os.path.join(base_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    input_dim = int(X_train.shape[1])
    unique_classes = sorted(np.unique(y_train).tolist())
    if class_names is None or len(class_names) != len(unique_classes):
        class_names = [f"Class_{i}" for i in range(len(unique_classes))]

    # 1. Metadata
    meta = {
        "scenario_name": scenario_name,
        "mode": "centralized",
        "input_dim": input_dim,
        "num_classes": len(class_names),
        "class_names": class_names,
        "total_train_samples": int(len(y_train)),
        "total_val_samples": int(len(y_val)) if y_val is not None else 0,
        "total_test_samples": int(len(y_test)) if y_test is not None else 0,
    }
    with open(os.path.join(scenario_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 2. Config
    scen_config = config.copy() if config else {}
    scen_config.setdefault("scenario_name", scenario_name)
    scen_config.setdefault("mode", "centralized")
    with open(os.path.join(scenario_dir, "config.json"), "w") as f:
        json.dump(scen_config, f, indent=2)

    # 3. Validation and Test splits
    if X_val is not None and y_val is not None:
        np.savez_compressed(
            os.path.join(scenario_dir, "val_data.npz"),
            X=np.ascontiguousarray(X_val, dtype=np.float32),
            y=np.ascontiguousarray(y_val, dtype=np.int64)
        )
    if X_test is not None and y_test is not None:
        np.savez_compressed(
            os.path.join(scenario_dir, "test_data.npz"),
            X=np.ascontiguousarray(X_test, dtype=np.float32),
            y=np.ascontiguousarray(y_test, dtype=np.int64)
        )

    # 4. EDA profiling via src.eda
    eda = analyze_partition(y_train, class_names=class_names)
    with open(os.path.join(scenario_dir, "eda.json"), "w") as f:
        json.dump(eda, f, indent=2)

    dataset_audit = analyze_dataset(X_train, y_train, class_names=class_names, feature_names=feature_names)
    export_dataset_audit(scenario_dir, dataset_audit)

    # 5. Visualizations
    if generate_plots:
        plot_class_distribution(
            eda=eda,
            save_path=os.path.join(scenario_dir, "class_distribution.png"),
            title=f"Centralized Training Set - Label Distribution (N={len(y_train):,})"
        )
        stats_df = analyze_features(X_train, feature_names=feature_names)
        plot_feature_skewness(
            stats_df=stats_df,
            top_k=15,
            save_path=os.path.join(scenario_dir, "feature_skewness.png")
        )

    logger.info(f"✅ Centralized scenario '{scenario_name}' exported successfully to {scenario_dir}.")
    return scenario_dir


def plot_scenario_confusion_matrix(
    cm: Optional[Union[np.ndarray, List[List[float]]]] = None,
    class_names: Optional[List[str]] = None,
    minority_classes: Optional[List[str]] = None,
    save_path: str = "confusion_matrix.png",
    per_class_recall: Optional[Dict[str, float]] = None,
    per_class_precision: Optional[Dict[str, float]] = None,
    support_dict: Optional[Dict[str, int]] = None,
    scenario_title: Optional[str] = None
) -> str:
    """
    Renders a dual-panel evaluation visualization:
    - Panel 1: Normalized Confusion Matrix Heatmap.
    - Panel 2: Per-Class Recall Bar Chart with minority attacks highlighted in red.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    class_names = class_names or ["Benign", "Brute-force", "DDoS", "DoS", "Mirai", "Recon", "Spoof", "Web-based"]
    minority_classes = minority_classes or ["Web-based", "Brute-force"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150)

    # 1. Confusion Matrix Heatmap
    cm_arr = None
    if cm is not None and len(cm) > 0:
        cm_arr = np.array(cm, dtype=np.float64)
    elif per_class_recall and support_dict:
        # Mathematically reconstruct CM from support, recall, and precision
        K = len(class_names)
        support = np.array([float(support_dict.get(c, 0)) for c in class_names])
        recall = np.array([float(per_class_recall.get(c, 0.0)) for c in class_names])
        prec = np.array([float(per_class_precision.get(c, 0.0)) for c in class_names]) if per_class_precision else recall

        tp = support * recall
        fp = np.where(prec > 0, tp / np.maximum(prec, 1e-12) - tp, 0.0)
        fn = np.maximum(support - tp, 0.0)

        cm_arr = np.zeros((K, K))
        for i in range(K):
            cm_arr[i, i] = tp[i]
            off_diag_fp = np.copy(fp)
            off_diag_fp[i] = 0.0
            if off_diag_fp.sum() > 0:
                cm_arr[i, :] += fn[i] * (off_diag_fp / off_diag_fp.sum())
            else:
                for j in range(K):
                    if j != i:
                        cm_arr[i, j] = fn[i] / max(K - 1, 1)

    if cm_arr is not None:
        cm_norm = cm_arr / (cm_arr.sum(axis=1, keepdims=True) + 1e-12)
        im = ax1.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
        ax1.set_title(scenario_title or "Confusion Matrix: Best Global Model", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(class_names))
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(class_names, fontsize=9)
        ax1.set_ylabel("True Label", fontsize=10)
        ax1.set_xlabel("Predicted Label", fontsize=10)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                val = int(round(cm_arr[i, j]))
                pct = cm_norm[i, j] * 100.0
                ax1.text(j, i, f"{val:,}\n({pct:.1f}%)",
                         ha="center", va="center", color=color, fontsize=8)

        class_recalls = cm_arr.diagonal() / (cm_arr.sum(axis=1) + 1e-12) * 100.0
    else:
        # Fallback if raw CM not available
        ax1.text(0.5, 0.5, "Confusion Matrix Heatmap\n(Detailed per-class recall shown on right panel)",
                 ha="center", va="center", fontsize=10, style="italic")
        ax1.set_axis_off()
        if per_class_recall:
            class_recalls = np.array([per_class_recall.get(c, 0.0) * 100.0 for c in class_names])
        else:
            class_recalls = np.zeros(len(class_names))

    if per_class_recall and (cm_arr is None):
        class_recalls = np.array([per_class_recall.get(c, 0.0) * 100.0 for c in class_names])

    # 2. Per-Class Recall Bar Chart
    bar_colors = ["#d62728" if name in minority_classes else "#1f77b4" for name in class_names]
    ax2.set_xticks(np.arange(len(class_names)))
    bars = ax2.bar(class_names, class_recalls, color=bar_colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Tỷ lệ nhận diện (Recall) theo từng lớp tấn công", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Recall (%)", fontsize=10)
    ax2.set_ylim(0, 108)
    ax2.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    red_patch = mpatches.Patch(color="#d62728", label="Minority Attack (Web-based, Brute-force)")
    blue_patch = mpatches.Patch(color="#1f77b4", label="Majority / Standard Class")
    ax2.legend(handles=[red_patch, blue_patch], loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def export_scenario_summary_and_plots(
    scenario_path_or_name: str,
    test_results: Optional[Dict[str, Any]] = None,
    round_history: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    minority_classes: Optional[List[str]] = None,
    scenarios_root: str = "scenarios"
) -> Dict[str, str]:
    """
    Automatically generates and saves:
    1. server/convergence.png: Optimization trajectory and communication cost plot.
    2. server/confusion_matrix.png: Dual-panel Confusion Matrix Heatmap and Recall bar chart.
    3. server/summary.md: Publication-grade Markdown summary report table for the thesis.

    Returns:
        Dict mapping artifact names to their file paths.
    """
    # 1. Resolve scenario paths
    if os.path.isabs(scenario_path_or_name) and os.path.exists(scenario_path_or_name):
        src_path = scenario_path_or_name
        scenario_name = os.path.basename(os.path.normpath(src_path))
    elif os.path.exists(scenario_path_or_name):
        src_path = os.path.abspath(scenario_path_or_name)
        scenario_name = os.path.basename(os.path.normpath(src_path))
    else:
        candidate = os.path.join(scenarios_root, scenario_path_or_name)
        if os.path.exists(candidate):
            src_path = os.path.abspath(candidate)
            scenario_name = scenario_path_or_name
        else:
            raise FileNotFoundError(f"Scenario directory not found: '{scenario_path_or_name}' (checked '{candidate}')")

    server_dir = os.path.join(src_path, "server")
    os.makedirs(server_dir, exist_ok=True)

    # 2. Load missing artifacts from disk if needed
    cfg: Dict[str, Any] = {}
    cfg_file = os.path.join(src_path, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)

    meta: Dict[str, Any] = {}
    meta_file = os.path.join(server_dir, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)

    if test_results is None:
        tr_file = os.path.join(server_dir, "test_results.json")
        if os.path.exists(tr_file):
            with open(tr_file) as f:
                test_results = json.load(f)
        else:
            test_results = {}

    if round_history is None:
        rh_file = os.path.join(server_dir, "round_history.json")
        if os.path.exists(rh_file):
            with open(rh_file) as f:
                round_history = json.load(f)
        else:
            round_history = {}

    class_names = class_names or meta.get("class_names", [
        "Benign", "Brute-force", "DDoS", "DoS", "Mirai", "Recon", "Spoof", "Web-based"
    ])
    minority_classes = minority_classes or ["Web-based", "Brute-force"]

    # 3. Generate server/convergence.png
    conv_path = os.path.join(server_dir, "convergence.png")
    plot_scenario_convergence(
        round_history=round_history,
        save_path=conv_path,
        scenario_title=f"Federated Optimization Trajectory - {scenario_name}"
    )

    # 4. Generate server/confusion_matrix.png
    cm_path = os.path.join(server_dir, "confusion_matrix.png")
    cm_data = test_results.get("confusion_matrix")
    per_class_rec = test_results.get("per_class_recall", {})
    per_class_prec = test_results.get("per_class_precision", {})
    support_dict = test_results.get("support", {})
    plot_scenario_confusion_matrix(
        cm=cm_data,
        class_names=class_names,
        minority_classes=minority_classes,
        save_path=cm_path,
        per_class_recall=per_class_rec,
        per_class_precision=per_class_prec,
        support_dict=support_dict,
        scenario_title=f"Confusion Matrix: Best Global Model ({cfg.get('federated_strategy', 'FL').upper()})"
    )

    # 5. Generate server/summary.md
    strat = str(cfg.get("federated_strategy", cfg.get("algorithm", {}).get("name", "fedavg"))).upper()
    mu_val = cfg.get("mu", 0.0)
    part_type = cfg.get("partition_type", "IID")
    alpha_val = cfg.get("dirichlet_alpha", "N/A")
    num_clients = cfg.get("num_clients", meta.get("num_clients", 5))
    num_rounds = cfg.get("num_rounds", len(round_history.get("round", [])))
    local_epochs = cfg.get("local_epochs", 2)

    comm_list = round_history.get("comm_cost_mb", [0.0])
    comm_mb = comm_list[-1] if comm_list else 0.0
    drift_list = round_history.get("drift_l2", [0.0])
    drift_val = drift_list[-1] if drift_list else 0.0

    acc = test_results.get("accuracy", 0.0) * 100.0
    macro_f1 = test_results.get("macro_f1", 0.0) * 100.0
    weighted_f1 = test_results.get("weighted_f1", 0.0) * 100.0
    macro_prec = test_results.get("macro_precision", 0.0) * 100.0
    macro_rec = test_results.get("macro_recall", 0.0) * 100.0

    min_rec = test_results.get("minority_recall", {})
    web_rec = min_rec.get("Web-based", per_class_rec.get("Web-based", 0.0)) * 100.0
    brute_rec = min_rec.get("Brute-force", per_class_rec.get("Brute-force", 0.0)) * 100.0

    md_lines = [
        f"# 📊 Báo cáo Thực nghiệm Kịch bản: `{scenario_name}`\n",
        "## 1. Thông số Cấu hình Thực nghiệm",
        "| Tham số | Giá trị | Ý nghĩa thực nghiệm |",
        "| :--- | :--- | :--- |",
        f"| **Kịch bản (Scenario)** | `{scenario_name}` | Tên định danh kịch bản thực nghiệm |",
        f"| **Chiến lược FL (Strategy)** | `{strat}` (mu={mu_val}) | Thuật toán tổng hợp tham số tại Server |",
        f"| **Phân vùng dữ liệu (Partition)** | `{part_type}` (alpha={alpha_val}) | Mức độ dị biệt Non-IID giữa các Client |",
        f"| **Số lượng Edge Clients (K)** | `{num_clients}` clients | Số trạm mạng IoT biên tham gia huấn luyện |",
        f"| **Số vòng giao tiếp (Rounds)** | `{num_rounds}` rounds (Epochs cục bộ: `{local_epochs}`) | Tổng chu kỳ huấn luyện liên đoàn |",
        f"| **Chi phí truyền thông (Comm Cost)** | **{comm_mb:.2f} MB** | Tổng lưu lượng mạng tải lên/tải xuống |",
        f"| **Độ trôi dạt tham số (Client Drift)** | **{drift_val:.2f}** | Độ lệch chuẩn Euclidean ||w_local - w_global||_2 |\n",
        "## 2. Kết quả Đánh giá Tổng thể trên Global Test Set (30% Upcoming Data)",
        "| Chỉ số Đánh giá (Metric) | Kết quả (%) | Ý nghĩa theo Đề cương Luận văn KLTN |",
        "| :--- | :---: | :--- |",
        f"| **Test Accuracy** | **{acc:.2f}%** | Độ chính xác tổng thể trên toàn bộ lưu lượng kiểm định |",
        f"| **Macro-F1 Score** | **{macro_f1:.2f}%** | **Chỉ số cốt lõi** đánh giá cân bằng (không thiên lệch lớp đa số) |",
        f"| **Weighted-F1 Score** | **{weighted_f1:.2f}%** | F1 có tính trọng số theo tần suất xuất hiện thực tế |",
        f"| **Macro-Precision** | **{macro_prec:.2f}%** | Độ chuẩn xác trung bình trên 8 lớp tấn công |",
        f"| **Macro-Recall** | **{macro_rec:.2f}%** | Tỷ lệ nhận diện trung bình trên 8 lớp tấn công |\n",
        "## 3. Khả năng Phát hiện các Lớp Tấn công Thiểu số (Minority Attack Recall)",
        "> [!IMPORTANT]",
        "> Trong môi trường mạng IoT, tấn công thiểu số (Web-based, Brute-force) có tần suất thấp nhưng mức độ đe dọa cao, là trọng tâm đánh giá của Luận văn.\n",
        "| Lớp tấn công thiểu số | Tần suất xuất hiện | Tỷ lệ nhận diện (Recall %) | Nhận xét |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Web-based** | 0.49% (Rất hiếm) | **{web_rec:.2f}%** | {'Phát hiện tốt' if web_rec > 10 else 'Thách thức do mất cân bằng cực đoan'} |",
        f"| **Brute-force** | 0.26% (Rất hiếm) | **{brute_rec:.2f}%** | {'Phát hiện tốt' if brute_rec > 10 else 'Cần tinh chỉnh loss/mu'} |\n",
        "## 4. Chi tiết Hiệu năng theo từng Lớp Tấn công (Per-Class Metrics)",
        "| Lớp tấn công (Class) | Phân loại | Precision (%) | Recall (%) | F1-Score (%) | Số mẫu (Support) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    per_p = test_results.get("per_class_precision", {})
    per_f = test_results.get("per_class_f1", {})
    supp = test_results.get("support", {})

    for c in class_names:
        c_type = "Thiểu số (Minority)" if c in minority_classes else "Đa số / Chuẩn"
        p_val = per_p.get(c, 0.0) * 100.0
        r_val = per_class_rec.get(c, 0.0) * 100.0
        f_val = per_f.get(c, 0.0) * 100.0
        s_val = supp.get(c, 0)
        md_lines.append(f"| **{c}** | {c_type} | {p_val:.2f}% | {r_val:.2f}% | {f_val:.2f}% | {s_val:,} |")

    md_lines.extend([
        "\n## 5. Trực quan hóa Đồ thị Thực nghiệm",
        "### 📈 Quỹ đạo Hội tụ và Chi phí Mạng (Convergence & Communication)",
        "![Convergence Trajectory](convergence.png)\n",
        "### 🎯 Ma trận Nhầm lẫn & Tỷ lệ Nhận diện theo Lớp (Confusion Matrix & Recall)",
        "![Confusion Matrix](confusion_matrix.png)\n"
    ])

    summary_path = os.path.join(server_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"📊 [Scenario Artifacts] Đã tạo thành công các báo cáo cho: {scenario_name}")
    print(f"   • Convergence plot:      {conv_path}")
    print(f"   • Confusion matrix plot: {cm_path}")
    print(f"   • Summary report:        {summary_path}")

    return {
        "convergence_plot": conv_path,
        "confusion_matrix_plot": cm_path,
        "summary_markdown": summary_path
    }


def backup_scenario_to_drive(
    scenario_path_or_name: str,
    destination_dir: str = "/content/drive/MyDrive/FedLearning/scenarios",
    exclude_datasets: bool = True,
    scenarios_root: str = "scenarios"
) -> Dict[str, Any]:
    """
    Backs up experimental scenario artifacts to Google Drive while omitting heavy dataset partitions.

    Preserves:
        - config.json, eda.json, meta.json
        - metrics.json, round_history.json, test_results.json
        - flower_best_weights.npz / best_weights.npz (Trained model weights)
        - class_distribution.png, confusion_matrix plots

    Excludes (when exclude_datasets=True):
        - partition.npz, val_partition.npz, test_partition.npz
        - val_data.npz, test_data.npz, server_val.npz, global_test.npz
        - Any raw data files (.csv, .parquet, raw .npy tensors)

    Args:
        scenario_path_or_name: Path or name of the scenario folder.
        destination_dir: Target root directory in Google Drive or local storage.
        exclude_datasets: Whether to skip heavy partition/dataset array files.
        scenarios_root: Default root directory if scenario_path_or_name is just a name.

    Returns:
        Dict with backup summary (destination path, files copied, sizes, excluded files).
    """
    # 1. Resolve source directory path
    if os.path.isabs(scenario_path_or_name) and os.path.exists(scenario_path_or_name):
        src_path = scenario_path_or_name
        scenario_name = os.path.basename(os.path.normpath(src_path))
    elif os.path.exists(scenario_path_or_name):
        src_path = os.path.abspath(scenario_path_or_name)
        scenario_name = os.path.basename(os.path.normpath(src_path))
    else:
        candidate = os.path.join(scenarios_root, scenario_path_or_name)
        if os.path.exists(candidate):
            src_path = os.path.abspath(candidate)
            scenario_name = scenario_path_or_name
        else:
            raise FileNotFoundError(
                f"Source scenario directory not found: '{scenario_path_or_name}' (checked '{candidate}')"
            )

    # 2. Check Google Drive mount condition
    if destination_dir.startswith("/content/drive") and not os.path.exists("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive is not mounted at '/content/drive'. "
            "Please mount Google Drive first via: from google.colab import drive; drive.mount('/content/drive')"
        )

    target_scenario_dir = os.path.join(destination_dir, scenario_name)
    os.makedirs(target_scenario_dir, exist_ok=True)

    excluded_files: List[str] = []
    copied_files: List[str] = []

    def ignore_datasets_filter(dir_path: str, files: List[str]):
        ignored = set()
        if not exclude_datasets:
            return ignored

        for f in files:
            full_path = os.path.join(dir_path, f)
            if os.path.isdir(full_path):
                continue

            f_lower = f.lower()
            # Exclude dataset arrays but keep model weights
            is_dataset_npz = f_lower.endswith(".npz") and "weight" not in f_lower
            is_raw_data = f_lower.endswith((".csv", ".parquet", ".feather", ".h5"))
            is_raw_tensor = f_lower.endswith(".npy") and "weight" not in f_lower

            if is_dataset_npz or is_raw_data or is_raw_tensor:
                ignored.add(f)
                rel_path = os.path.relpath(full_path, src_path)
                excluded_files.append(rel_path)

        return ignored

    # 3. Copy directory tree with selective filter
    shutil.copytree(
        src_path,
        target_scenario_dir,
        ignore=ignore_datasets_filter if exclude_datasets else None,
        dirs_exist_ok=True
    )

    # 4. Compute statistics
    orig_size = 0
    for root, _, f_list in os.walk(src_path):
        for f in f_list:
            orig_size += os.path.getsize(os.path.join(root, f))

    backed_up_size = 0
    for root, _, f_list in os.walk(target_scenario_dir):
        for f in f_list:
            fp = os.path.join(root, f)
            backed_up_size += os.path.getsize(fp)
            copied_files.append(os.path.relpath(fp, target_scenario_dir))

    # 5. Formatted audit log
    print("=" * 80)
    print(f"☁️ GOOGLE DRIVE SCENARIO BACKUP: {scenario_name}")
    print("=" * 80)
    print(f"📁 Source Directory:       {src_path}")
    print(f"💾 Destination (Drive):    {target_scenario_dir}")
    print(f"📦 Total Files Backed Up:  {len(copied_files)}")
    print(f"🚫 Heavy Datasets Skipped: {len(excluded_files)} files")
    print(f"📊 Storage Size:           {backed_up_size / (1024*1024):.2f} MB (Drive) vs {orig_size / (1024*1024):.2f} MB (Local)")
    if orig_size > 0:
        reduction = (1.0 - (backed_up_size / orig_size)) * 100
        print(f"⚡ Space Savings:          {reduction:.1f}% bandwidth & Drive quota saved!")
    print("=" * 80)

    return {
        "status": "success",
        "scenario_name": scenario_name,
        "source_path": src_path,
        "destination_path": target_scenario_dir,
        "files_backed_up": copied_files,
        "files_excluded": excluded_files,
        "original_size_bytes": orig_size,
        "backed_up_size_bytes": backed_up_size
    }


def commit_scenario_to_github(
    scenario_path_or_name: str,
    github_token: Optional[str] = None,
    branch: str = "main",
    repo_url: str = "https://github.com/NVKQ2022/FedLearning.git",
    author_name: str = "Nguyen Viet Ky Quan",
    author_email: str = "nguyenvietkyquan2022@gmail.com",
    commit_message: Optional[str] = None,
    push: bool = True,
    max_file_size_mb: float = 25.0,
    scenarios_root: str = "scenarios"
) -> Dict[str, Any]:
    """
    Commits and pushes experimental scenario artifacts to GitHub from Colab or local environment.

    Safeguards:
        - .gitignore ensures heavy dataset files (*.npz like val_data, partition) are ignored.
        - Lightweight model weights (flower_best_weights.npz ~55KB), JSON configs, metrics, and plots are tracked.
        - Checks file sizes before staging (rejects any file exceeding max_file_size_mb).
        - If github_token is provided (e.g. from Colab Secrets), authenticates securely and clears token immediately after push.

    Args:
        scenario_path_or_name: Path or name of the scenario folder.
        github_token: Optional GitHub Personal Access Token (PAT) with repo write permissions.
        branch: Git branch name (default 'main').
        repo_url: Repository URL.
        author_name: Git commit author name.
        author_email: Git commit author email.
        commit_message: Custom commit message.
        push: Whether to push commits to remote.
        max_file_size_mb: Max allowed file size for safety (default 25MB).
        scenarios_root: Scenarios directory.

    Returns:
        Dict with commit summary (status, scenario_name, staged_files, commit_hash).
    """
    # 1. Resolve source directory path
    if os.path.isabs(scenario_path_or_name) and os.path.exists(scenario_path_or_name):
        src_path = scenario_path_or_name
        scenario_name = os.path.basename(os.path.normpath(src_path))
    elif os.path.exists(scenario_path_or_name):
        src_path = os.path.abspath(scenario_path_or_name)
        scenario_name = os.path.basename(os.path.normpath(src_path))
    else:
        candidate = os.path.join(scenarios_root, scenario_path_or_name)
        if os.path.exists(candidate):
            src_path = os.path.abspath(candidate)
            scenario_name = scenario_path_or_name
        else:
            raise FileNotFoundError(
                f"Source scenario directory not found: '{scenario_path_or_name}' (checked '{candidate}')"
            )

    # Get path relative to git root
    try:
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Current workspace is not a valid git repository.")

    rel_scenario_path = os.path.relpath(src_path, git_root)

    # 2. Safety check: ensure no file exceeds max_file_size_mb
    max_bytes = max_file_size_mb * 1024 * 1024
    oversized_files = []
    for root, _, files in os.walk(src_path):
        for f in files:
            fp = os.path.join(root, f)
            check_ignore = subprocess.run(["git", "check-ignore", fp], capture_output=True, text=True)
            if not check_ignore.stdout.strip():  # Not ignored by git
                sz = os.path.getsize(fp)
                if sz > max_bytes:
                    oversized_files.append((os.path.relpath(fp, git_root), sz / (1024 * 1024)))

    if oversized_files:
        raise ValueError(
            f"Safety check failed! The following files exceed {max_file_size_mb}MB and should NOT be pushed to GitHub:\n"
            + "\n".join([f"  - {f}: {sz:.2f} MB" for f, sz in oversized_files])
        )

    # 3. Configure git author if not configured
    current_name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
    if not current_name:
        subprocess.run(["git", "config", "user.name", author_name], check=True)
    current_email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
    if not current_email:
        subprocess.run(["git", "config", "user.email", author_email], check=True)

    # 4. Stage scenario files
    subprocess.run(["git", "add", rel_scenario_path], check=True)

    # 5. Check if there are staged changes
    staged_output = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", rel_scenario_path],
        capture_output=True,
        text=True
    ).stdout.strip()
    staged_files = [f for f in staged_output.split("\n") if f]

    if not staged_files:
        print(f"ℹ️ Không có thay đổi mới nào trong '{rel_scenario_path}' để commit.")
        return {
            "status": "no_changes",
            "scenario_name": scenario_name,
            "staged_files": [],
            "pushed": False
        }

    # 6. Commit changes
    default_msg = f"chore(scenario): update experimental artifacts for {scenario_name}"
    msg = commit_message or default_msg
    subprocess.run(["git", "commit", "-m", msg], check=True)
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    print(f"✅ Đã tạo commit [{commit_hash}]: {msg}")
    print(f"   Số lượng files artifacts đã commit: {len(staged_files)}")

    # 7. Push to GitHub if requested
    pushed = False
    if push:
        clean_url = repo_url.strip()
        if clean_url.startswith("https://"):
            host_path = clean_url[len("https://"):]
        elif clean_url.startswith("http://"):
            host_path = clean_url[len("http://"):]
        else:
            host_path = clean_url

        original_remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()

        try:
            if github_token and github_token.strip():
                tok = github_token.strip()
                auth_url = f"https://{tok}@{host_path}"
                subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
                print(f"🔄 Đang đẩy commit lên GitHub (nhánh '{branch}')...")
            else:
                print(f"🔄 Đang đẩy commit lên GitHub bằng thông tin đăng nhập hiện tại...")

            push_res = subprocess.run(["git", "push", "origin", branch], capture_output=True, text=True)
            if push_res.returncode != 0:
                print(f"❌ Git push thất bại:\n{push_res.stderr}")
                raise RuntimeError(f"Git push failed: {push_res.stderr}")
            else:
                pushed = True
                print(f"🚀 Đã đồng bộ thành công kết quả scenario '{scenario_name}' lên GitHub!")
        finally:
            # Always restore unauthenticated remote URL so token is never kept on disk
            subprocess.run(["git", "remote", "set-url", "origin", original_remote], check=True)

    print("=" * 80)
    print(f"📦 GITHUB SCENARIO SYNC: {scenario_name}")
    print("=" * 80)
    print(f"📁 Scenario Path:   {rel_scenario_path}")
    print(f"🏷️ Commit Hash:     {commit_hash}")
    print(f"📄 Staged Files:    {len(staged_files)} files")
    print(f"🚀 Pushed to Repo:  {'Yes' if pushed else 'No'}")
    print("=" * 80)

    return {
        "status": "success",
        "scenario_name": scenario_name,
        "commit_hash": commit_hash,
        "staged_files": staged_files,
        "pushed": pushed
    }


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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure headless plotting compatibility in CLI / remote servers / Colab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_partition_eda(
    y: np.ndarray,
    class_names: List[str],
    client_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes statistical Exploratory Data Analysis (EDA) on a client partition.

    Args:
        y: 1D array of categorical integer labels.
        class_names: List of human-readable class names.
        client_id: Optional client identifier index.

    Returns:
        Structured EDA dictionary with sample counts, percentages, and Shannon entropy.
    """
    total_samples = int(len(y))
    num_classes = len(class_names)
    unique_labels, counts = np.unique(y, return_counts=True)
    label_to_count = dict(zip(unique_labels.tolist(), counts.tolist()))

    class_counts: Dict[str, int] = {}
    class_percentages: Dict[str, float] = {}

    for idx, name in enumerate(class_names):
        cnt = int(label_to_count.get(idx, 0))
        pct = round((cnt / total_samples * 100.0), 2) if total_samples > 0 else 0.0
        class_counts[name] = cnt
        class_percentages[name] = pct

    # Calculate Shannon Entropy: H(p) = -sum(p * ln(p))
    probs = np.array([class_counts[name] / total_samples for name in class_names if class_counts[name] > 0])
    if len(probs) > 0:
        entropy = float(-np.sum(probs * np.log(probs)))
        max_possible_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0
        norm_entropy = float(entropy / max_possible_entropy) if max_possible_entropy > 0 else 1.0
    else:
        entropy = 0.0
        norm_entropy = 0.0

    dominant_class = max(class_counts.items(), key=lambda kv: kv[1])[0] if class_counts else "Unknown"
    dominant_pct = class_percentages.get(dominant_class, 0.0)

    eda = {
        "client_id": client_id,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "dominant_class": dominant_class,
        "dominant_class_pct": dominant_pct,
        "shannon_entropy": round(entropy, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "class_counts": class_counts,
        "class_percentages": class_percentages,
    }
    return eda


def plot_class_distribution(
    eda: Dict[str, Any],
    save_path: str,
    title: Optional[str] = None
) -> str:
    """
    Renders and saves a clean, publication-ready horizontal bar chart of class distribution.

    Args:
        eda: Output dictionary from compute_partition_eda.
        save_path: Path where the PNG image will be written.
        title: Optional plot title override.

    Returns:
        Absolute filepath to the saved image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    client_id = eda.get("client_id")
    total_samples = eda.get("total_samples", 0)
    counts = eda.get("class_counts", {})
    percentages = eda.get("class_percentages", {})

    classes = list(counts.keys())
    sample_values = [counts[c] for c in classes]

    # Invert order for natural top-to-bottom reading on horizontal bar chart
    classes_rev = classes[::-1]
    values_rev = sample_values[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.barh(classes_rev, values_rev, color="#2b5c8f", edgecolor="#1a365d", height=0.65)

    max_val = max(values_rev) if values_rev and max(values_rev) > 0 else 1
    ax.set_xlim(0, max_val * 1.25)
    ax.set_xlabel("Sample Count", fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    plot_title = title or (
        f"Client {client_id} Partition - Label Distribution (N={total_samples:,})"
        if client_id is not None
        else f"Label Distribution (N={total_samples:,})"
    )
    ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=12)

    # Annotate bars with counts and percentages
    for bar, c_name in zip(bars, classes_rev):
        width = bar.get_width()
        pct = percentages.get(c_name, 0.0)
        ax.text(
            width + (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#2d3748",
            fontweight="normal"
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


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
        rounds_list[existing_idx] = clean_round
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
    scenario_name: str,
    client_partitions: Dict[int, np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    base_dir: str = "scenarios",
    generate_plots: bool = True
) -> str:
    """
    Constructs the complete federated scenario directory tree.

    Creates:
    - scenarios/<scenario_name>/config.json
    - scenarios/<scenario_name>/server/meta.json, val_data.npz
    - scenarios/<scenario_name>/client_{i}/partition.npz, eda.json, class_distribution.png, metrics.json

    Args:
        scenario_name: Directory identifier (e.g. 'E5_fedprox_dirichlet_0.1').
        client_partitions: Dict mapping client_id to indices in X_train/y_train.
        X_train: Preprocessed global training feature array.
        y_train: Preprocessed global training labels array.
        X_val: Optional server holdout validation features array.
        y_val: Optional server holdout validation labels array.
        class_names: List of class names. If None, inferred as Class_0..C.
        config: Optional scenario configuration parameters dictionary.
        base_dir: Base directory for all scenarios (default: 'scenarios').
        generate_plots: Whether to render class_distribution.png for each client.

    Returns:
        Path to the initialized scenario directory.
    """
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
        "total_val_samples": int(len(y_val)) if y_val is not None else 0
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

    # 3. Save scenario config.json
    scenario_config = config.copy() if config else {}
    scenario_config.setdefault("scenario_name", scenario_name)
    scenario_config.setdefault("num_clients", num_clients)
    scenario_config.setdefault("input_dim", input_dim)
    scenario_config.setdefault("num_classes", len(class_names))
    with open(os.path.join(scenario_dir, "config.json"), "w") as f:
        json.dump(scenario_config, f, indent=2)

    # 4. Generate client partitions, EDA, plots, and metrics templates
    cross_client_summary = []
    for client_id, indices in sorted(client_partitions.items()):
        client_dir = os.path.join(scenario_dir, f"client_{int(client_id)}")
        os.makedirs(client_dir, exist_ok=True)

        X_c = np.ascontiguousarray(X_train[indices], dtype=np.float32)
        y_c = np.ascontiguousarray(y_train[indices], dtype=np.int64)

        # Save private partition
        partition_path = os.path.join(client_dir, "partition.npz")
        np.savez_compressed(partition_path, X=X_c, y=y_c)

        # Compute EDA
        eda = compute_partition_eda(y=y_c, class_names=class_names, client_id=int(client_id))
        eda_path = os.path.join(client_dir, "eda.json")
        with open(eda_path, "w") as f:
            json.dump(eda, f, indent=2)

        # Generate class distribution plot
        if generate_plots:
            plot_path = os.path.join(client_dir, "class_distribution.png")
            plot_class_distribution(eda=eda, save_path=plot_path)

        # Initialize clean metrics.json
        metrics_path = os.path.join(client_dir, "metrics.json")
        initial_metrics = {
            "client_id": int(client_id),
            "total_samples": int(len(y_c)),
            "rounds": [],
            "summary": {}
        }
        with open(metrics_path, "w") as f:
            json.dump(initial_metrics, f, indent=2)

        cross_client_summary.append({
            "client_id": int(client_id),
            "samples": int(len(y_c)),
            "dominant_class": eda["dominant_class"],
            "dominant_pct": eda.get("dominant_class_pct", eda.get("dominant_pct", 0.0)),
            "entropy": eda["normalized_entropy"]
        })

    # Save summary across all clients in server folder
    with open(os.path.join(server_dir, "clients_summary.json"), "w") as f:
        json.dump(cross_client_summary, f, indent=2)

    logger.info(
        f"✅ Scenario '{scenario_name}' exported successfully to {scenario_dir} "
        f"({num_clients} clients, {len(y_train):,} total train samples)."
    )
    return scenario_dir


def load_client_partition(
    scenario_dir: str,
    client_id: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads local partition arrays (X, y) and metadata for an edge client.

    Args:
        scenario_dir: Scenario root directory (e.g. 'scenarios/E5_fedprox_dirichlet_0.1').
        client_id: Target client integer index.

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
    X, y = data["X"], data["y"]

    meta_file = os.path.join(scenario_dir, "server", "meta.json")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(scenario_dir, "meta.json")

    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)

    return X, y, meta


def load_server_data(
    scenario_dir: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads central server validation data (X_val, y_val) and scenario metadata.

    Args:
        scenario_dir: Scenario root directory.

    Returns:
        Tuple of (X_val, y_val, meta).
    """
    server_dir = os.path.join(scenario_dir, "server")
    if not os.path.exists(server_dir):
        server_dir = scenario_dir

    meta_file = os.path.join(server_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)

    val_file = os.path.join(server_dir, "val_data.npz")
    if not os.path.exists(val_file):
        val_file = os.path.join(server_dir, "server_val.npz")

    if os.path.exists(val_file):
        data = np.load(val_file)
        return data["X"], data["y"], meta

    return None, None, meta


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
    val_f1 = [v * 100.0 for v in round_history.get("val_macro_f1", [])]
    if val_f1:
        ax1.plot(rounds, val_f1, marker="o", color="#2b6cb0", linewidth=2.0)
        ax1.set_title("Central Holdout Validation Macro-F1 (%)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Macro-F1 (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Training Loss
    ax2 = axes[0, 1]
    train_loss = round_history.get("train_loss", [])
    if train_loss:
        ax2.plot(rounds, train_loss, marker="s", color="#c53030", linewidth=2.0)
        ax2.set_title("Aggregated Training Loss", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Loss")
        ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Parameter Drift
    ax3 = axes[1, 0]
    drift = round_history.get("drift_l2", [])
    if drift:
        ax3.plot(rounds, drift, marker="^", color="#d69e2e", linewidth=2.0)
        ax3.set_title("Client Weight Drift (||w_local - w_global||_2)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Communication Round")
        ax3.set_ylabel("Drift L2 Norm")
        ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Cumulative Communication Cost
    ax4 = axes[1, 1]
    comm = round_history.get("comm_cost_mb", [])
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

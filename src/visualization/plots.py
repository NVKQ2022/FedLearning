"""
Publication-Grade Visualization Module for FL-IoT-IDS.

Provides plug-and-play, publication-ready plotting functions:
- set_publication_style: Configures aesthetic fonts, grids, and high-DPI rendering.
- plot_learning_curves: Dual-panel training and validation loss & accuracy trajectories.
- plot_confusion_matrix: Row-normalized or raw confusion matrix heatmaps.
- plot_scenario_comparison: Cross-scenario benchmark bar charts (e.g. E1 vs E2 vs E5).
- plot_client_distribution: Stacked bar chart of client class partition heterogeneity (Dirichlet vs IID).
- plot_federated_convergence: Multi-round FL metric convergence trajectories (FedAvg vs FedProx).
- plot_minority_recall: Isolated recall comparison for stealthy attacks (Web-based, Brute-force).
- plot_per_class_metrics: Multi-metric grouped bar chart (Precision, Recall, F1) across attack classes.

Designed for turnkey, one-line usage:
    from src.visualization import plot_learning_curves, plot_confusion_matrix
    plot_learning_curves(history, save_path="reports/figures/curves.png")
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Default publication palette for scenarios and classes
DEFAULT_SCENARIO_COLORS = [
    "#2ecc71",  # Emerald Green (Upper bound / Centralized)
    "#3498db",  # Sky Blue (FedAvg IID)
    "#f39c12",  # Amber / Orange (FedAvg Non-IID)
    "#e74c3c",  # Crimson Red (FedProx Non-IID)
    "#9b59b6",  # Amethyst Purple (FedMedian / Robust)
    "#1abc9c",  # Turquoise
    "#34495e"   # Slate Gray
]


def set_publication_style(
    font_scale: float = 1.1,
    style: str = "whitegrid",
    font_family: str = "sans-serif"
) -> None:
    """
    Configures publication-grade visual styling across Matplotlib and Seaborn.

    Args:
        font_scale: Relative font scaling factor (default 1.1).
        style: Seaborn background theme ('whitegrid', 'white', 'ticks').
        font_family: Font family family for clean typography.
    """
    sns.set_theme(style=style, font_scale=font_scale)
    plt.rcParams.update({
        "font.family": font_family,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.autolayout": False
    })


def _ensure_dir(path: Optional[str]) -> None:
    """Safely creates parent directories for a target save path."""
    if path:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)


def _safe_convert_tensor(data: Any) -> np.ndarray:
    """Safely converts PyTorch tensors, Pandas objects, or nested lists to NumPy."""
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "numpy"):
        return data.numpy()
    if isinstance(data, (list, tuple)):
        return np.asarray(data)
    return np.asarray(data)


def plot_learning_curves(
    history: Union[Dict[str, List[float]], Any],
    save_path: Optional[str] = None,
    title: str = "Centralized Training & Validation Convergence",
    loss_name: str = "Loss",
    figsize: Tuple[int, int] = (14, 5),
    show: bool = False
) -> plt.Figure:
    """
    Generates a publication-grade dual-panel learning curve plot.
    - Left panel: Training and Validation Loss vs Epoch.
    - Right panel: Training and Validation Accuracy (%) vs Epoch.

    Args:
        history: Dictionary containing 'train_loss' and 'train_acc' (and optionally 'val_loss', 'val_acc'),
                 or an object with a .history dictionary attribute.
        save_path: Optional filepath to save the figure (e.g. 'reports/figures/learning_curves.png').
        title: Main super-title for the figure.
        loss_name: Label for the loss function (e.g. 'Cross-Entropy Loss', 'Focal Loss').
        figsize: Figure dimensions (width, height).
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    hist_dict = getattr(history, "history", history)
    if not isinstance(hist_dict, dict):
        raise TypeError("history must be a dictionary or have a .history attribute.")

    train_loss = hist_dict.get("train_loss", [])
    train_acc = hist_dict.get("train_acc", [])
    val_loss = hist_dict.get("val_loss", [])
    val_acc = hist_dict.get("val_acc", [])

    if len(train_loss) == 0:
        raise ValueError("history dictionary contains empty 'train_loss'.")

    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Loss Panel
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#e74c3c", linewidth=2.2, markersize=5)
    if len(val_loss) == len(train_loss):
        axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="#2980b9", linewidth=2.2, markersize=5)
        # Highlight best validation epoch
        best_val_idx = int(np.argmin(val_loss))
        best_loss = val_loss[best_val_idx]
        axes[0].scatter(
            [best_val_idx + 1], [best_loss],
            color="#f39c12", s=130, zorder=5, edgecolors="black", linewidths=1.5,
            label=f"Min Val Loss ({best_loss:.4f})"
        )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel(loss_name, fontsize=12)
    axes[0].set_title(f"Convergence Trajectory ({loss_name})", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)

    # 2. Accuracy Panel
    train_acc_pct = [a * 100 if a <= 1.0 else a for a in train_acc]
    axes[1].plot(epochs, train_acc_pct, "o-", label="Train Acc", color="#27ae60", linewidth=2.2, markersize=5)
    if len(val_acc) == len(train_acc):
        val_acc_pct = [a * 100 if a <= 1.0 else a for a in val_acc]
        axes[1].plot(epochs, val_acc_pct, "s--", label="Val Acc", color="#8e44ad", linewidth=2.2, markersize=5)
        best_acc_idx = int(np.argmax(val_acc_pct))
        best_acc = val_acc_pct[best_acc_idx]
        axes[1].scatter(
            [best_acc_idx + 1], [best_acc],
            color="#f1c40f", s=130, zorder=5, edgecolors="black", linewidths=1.5,
            label=f"Max Val Acc ({best_acc:.2f}%)"
        )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Accuracy Trajectory", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower right", frameon=True)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Learning curves saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, Any],
    class_names: Optional[List[str]] = None,
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
    cmap: str = "Blues",
    annot: bool = True,
    fmt: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    cbar: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Renders an annotated, publication-grade confusion matrix heatmap.

    Args:
        cm: Confusion matrix 2D array (ground truth rows, prediction columns).
        class_names: List of human-readable class names.
        title: Title of the heatmap.
        save_path: Optional file path to save figure.
        normalize: Whether to normalize by true rows (so rows sum to 1.00 / 100%).
        cmap: Matplotlib colormap string.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations (default: '.2f' for normalized, 'd' for raw).
        figsize: Figure dimensions (width, height).
        cbar: Whether to draw colorbar.
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    matrix = _safe_convert_tensor(cm).astype(float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square 2D confusion matrix, got shape {matrix.shape}")

    num_classes = matrix.shape[0]
    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Class {i}" for i in range(num_classes)]

    if normalize:
        # Check if already normalized (row sums ~ 1.0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-2):
            matrix = matrix / row_sums
        if fmt is None:
            fmt = ".2f"
    else:
        if fmt is None:
            fmt = ".0f"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=cbar,
        ax=ax,
        linewidths=0.5,
        linecolor="#dddddd",
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_scenario_comparison(
    results: Optional[Dict[str, Union[float, Dict[str, Any]]]] = None,
    metric: str = "macro_f1",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    palette: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
    bar_width: float = 0.45,
    show: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Renders a comparative bar chart across experimental scenarios (e.g. E1, E2, E5).
    Automatically extracts the chosen metric, scales to percentages, and prints bold labels over each bar.

    Args:
        results: Dictionary mapping scenario names to either scalar float metrics
                 or evaluation result dictionaries containing the metric key.
                 Example:
                 {
                     "E1: Centralized": 0.884,
                     "E2: FedAvg (IID)": 0.852,
                     "E5: FedProx (Non-IID)": 0.826
                 }
                 Or:
                 {"E1": e1_results_dict, "E2": e2_results_dict, ...}
        metric: Metric key to extract if dictionary values are passed (e.g. 'macro_f1', 'accuracy').
        title: Title of the chart.
        ylabel: Y-axis label.
        save_path: Optional filepath to save the figure.
        palette: Optional list of hex color codes.
        figsize: Figure dimensions (width, height).
        bar_width: Width of individual bars.
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    if results is None:
        results = kwargs.get("scenarios", kwargs.get("comparison_dict", {}))

    scenario_names = []
    scores = []

    for name, val in results.items():
        scenario_names.append(name)
        if isinstance(val, dict):
            raw_score = val.get(metric, 0.0)
        else:
            raw_score = float(val)
        # Convert 0-1 scale to percentage
        scores.append(raw_score * 100.0 if raw_score <= 1.0 else raw_score)

    if palette is None:
        palette = DEFAULT_SCENARIO_COLORS[:len(scenario_names)]
        if len(palette) < len(scenario_names):
            palette = sns.color_palette("tab10", len(scenario_names))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        scenario_names,
        scores,
        color=palette,
        edgecolor="#2c3e50",
        linewidth=1.2,
        width=bar_width
    )

    metric_title = metric.replace("_", " ").title()
    if ylabel is None:
        ylabel = f"{metric_title} (%)"
    if title is None:
        title = f"Scenario Performance Comparison ({metric_title})"

    ax.set_ylabel(ylabel, fontsize=12)
    max_score = max(scores) if scores else 100.0
    ax.set_ylim(0, min(105, max(max_score + 10, 100)))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Annotate bar heights
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    plt.setp(ax.get_xticklabels(), rotation=15 if len(scenario_names) > 3 else 0, ha="right" if len(scenario_names) > 3 else "center")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Scenario comparison saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_client_distribution(
    distribution: Union[pd.DataFrame, np.ndarray, Dict[int, Any]],
    y: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    title: str = "Client Label Distribution (Data Skew / Non-IID Dirichlet)",
    save_path: Optional[str] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "tab10",
    show: bool = False
) -> plt.Figure:
    """
    Renders a publication-grade stacked bar chart displaying label distributions across federated clients.
    Exposes statistical skew between IID (balanced) vs Non-IID Dirichlet partitioning.

    Accepts:
    1. A pandas DataFrame from `get_client_distribution(client_partitions, y, class_names)`.
    2. A partition dictionary `Dict[client_id, sample_indices]` combined with `y` ground truth labels.
    3. A 2D contingency table / array of shape `(num_clients, num_classes)`.

    Args:
        distribution: Distribution DataFrame, 2D array, or partition indices dictionary.
        y: Ground truth labels (required only if distribution is a partition dictionary).
        class_names: Human-readable class names.
        title: Figure title.
        save_path: Optional filepath to save the figure.
        normalize: If True, normalizes bars to 100% (proportional distribution).
        figsize: Figure dimensions (width, height).
        cmap: Colormap name for class segments.
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    # Case 1: Partition dictionary passed
    if isinstance(distribution, dict) and y is not None:
        unique_classes = sorted(np.unique(y))
        num_classes = len(unique_classes)
        if class_names is None or len(class_names) != num_classes:
            class_names = [f"Class {c}" for c in unique_classes]
        
        matrix_rows = []
        client_labels = []
        for client_id, idxs in sorted(distribution.items()):
            client_labels.append(f"Client {client_id}")
            c_y = y[idxs]
            counts = [(c_y == c).sum() for c in unique_classes]
            matrix_rows.append(counts)
        counts_df = pd.DataFrame(matrix_rows, index=client_labels, columns=class_names)

    # Case 2: DataFrame passed
    elif isinstance(distribution, pd.DataFrame):
        df = distribution.copy()
        # Check if output is from partitioner.get_client_distribution
        count_cols = [c for c in df.columns if c.endswith("_count")]
        if count_cols:
            extracted_classes = [c.replace("_count", "") for c in count_cols]
            client_ids = [f"Client {int(cid)}" for cid in df["client_id"]] if "client_id" in df.columns else df.index
            counts_df = pd.DataFrame(df[count_cols].values, index=client_ids, columns=extracted_classes)
        else:
            # Assume table where rows are clients and columns are classes
            counts_df = df
    
    # Case 3: 2D numpy array
    elif isinstance(distribution, np.ndarray):
        arr = distribution
        num_clients, num_classes = arr.shape
        if class_names is None or len(class_names) != num_classes:
            class_names = [f"Class {i}" for i in range(num_classes)]
        client_labels = [f"Client {i}" for i in range(num_clients)]
        counts_df = pd.DataFrame(arr, index=client_labels, columns=class_names)
    else:
        raise ValueError("Invalid distribution format. Pass a DataFrame, partition dict with y, or 2D array.")

    # Normalize if requested
    plot_df = counts_df.copy()
    if normalize:
        plot_df = plot_df.div(plot_df.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap=cmap,
        edgecolor="#333333",
        linewidth=0.6,
        width=0.7
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Federated Client Node", fontsize=12)
    ax.set_ylabel("Proportion (%)" if normalize else "Sample Count", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0 if len(plot_df) <= 10 else 45, ha="right" if len(plot_df) > 10 else "center")
    
    ax.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        borderaxespad=0.
    )
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Client distribution saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_federated_convergence(
    rounds_data: Union[Dict[str, List[float]], List[float]],
    metric_name: str = "Macro-F1",
    title: str = "Federated Learning Convergence Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = False
) -> plt.Figure:
    """
    Renders metric convergence trajectories across federated communication rounds.
    Compares multiple strategies (e.g. FedAvg IID vs FedAvg Non-IID vs FedProx).

    Args:
        rounds_data: Either a single list of round scores, or a dict mapping strategy
                     names to lists of scores across rounds.
        metric_name: Label of the metric (e.g. 'Macro-F1', 'Accuracy', 'Loss').
        title: Figure title.
        save_path: Optional filepath to save the figure.
        figsize: Figure dimensions (width, height).
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    if isinstance(rounds_data, (list, tuple)):
        series_dict = {"Global Model": list(rounds_data)}
    elif isinstance(rounds_data, dict):
        series_dict = rounds_data
    else:
        raise TypeError("rounds_data must be a list of floats or a dictionary mapping names to lists of floats.")

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    for idx, (label, vals) in enumerate(series_dict.items()):
        # Convert to percentage if <= 1.0 and not loss
        is_loss = "loss" in metric_name.lower()
        processed_vals = vals if is_loss else [v * 100.0 if v <= 1.0 else v for v in vals]
        rounds = range(1, len(processed_vals) + 1)
        marker = markers[idx % len(markers)]
        color = DEFAULT_SCENARIO_COLORS[idx % len(DEFAULT_SCENARIO_COLORS)]
        
        ax.plot(
            rounds,
            processed_vals,
            f"{marker}-",
            label=label,
            color=color,
            linewidth=2.2,
            markersize=6
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Communication Round (t)", fontsize=12)
    y_unit = "" if "loss" in metric_name.lower() else " (%)"
    ax.set_ylabel(f"{metric_name}{y_unit}", fontsize=12)
    ax.legend(loc="lower right" if "loss" not in metric_name.lower() else "upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Federated convergence saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_minority_recall(
    minority_data: Optional[Union[Dict[str, float], Dict[str, Dict[str, float]]]] = None,
    title: str = "Minority Attack Recall Isolation (Stealth Intrusions)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
    show: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Renders targeted detection recall for ultra-rare, stealthy attack categories
    (<1% prevalence such as Web-based and Brute-force attacks).

    Accepts:
    1. Single evaluation minority dict: {'Web-based': 0.74, 'Brute-force': 0.68}
    2. Multi-scenario comparative dict:
       {
           'E1: Centralized': {'Web-based': 0.82, 'Brute-force': 0.79},
           'E5: FedProx': {'Web-based': 0.65, 'Brute-force': 0.61}
       }

    Args:
        minority_data: Dictionary of minority recalls or nested scenario comparisons.
        title: Title of figure.
        save_path: Optional save filepath.
        figsize: Figure dimensions (width, height).
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    if minority_data is None:
        minority_data = kwargs.get("minority_recall", kwargs.get("data", {}))

    # Case 1: Nested scenario comparisons
    if any(isinstance(v, dict) for v in minority_data.values()):
        df = pd.DataFrame(minority_data).T
        # Scale to percentage
        df = df.map(lambda x: x * 100.0 if x <= 1.0 else x)
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(
            kind="bar",
            ax=ax,
            edgecolor="#2c3e50",
            linewidth=1.0,
            colormap="Set2",
            width=0.6
        )
        ax.set_ylabel("Recall Rate (%)", fontsize=12)
        ax.set_ylim(0, 105)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.legend(title="Minority Attack", frameon=True)
    # Case 2: Single scenario classes
    else:
        classes = list(minority_data.keys())
        recalls = [v * 100.0 if v <= 1.0 else v for v in minority_data.values()]
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(classes, recalls, color=["#e67e22", "#9b59b6", "#e74c3c"][:len(classes)], edgecolor="#2c3e50", width=0.45)
        ax.set_ylabel("Recall Rate (%)", fontsize=12)
        ax.set_ylim(0, 105)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Minority recall plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_per_class_metrics(
    metrics_dict: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    title: str = "Per-Class Performance Breakdown (Precision, Recall, F1)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Renders grouped bar charts comparing Precision, Recall, and F1 across all evaluated attack classes.
    Directly consumes output dictionary from `compute_comprehensive_metrics`.

    Args:
        metrics_dict: Dictionary containing 'per_class_precision', 'per_class_recall', and 'per_class_f1'.
        class_names: Optional explicit list of class names.
        title: Figure title.
        save_path: Optional save filepath.
        figsize: Figure dimensions (width, height).
        show: If True, calls plt.show().

    Returns:
        matplotlib.figure.Figure instance.
    """
    set_publication_style()

    if metrics_dict is None:
        if "metrics" in kwargs:
            metrics_dict = kwargs["metrics"]
        else:
            metrics_dict = {
                "per_class_precision": kwargs.get("per_class_precision", {}),
                "per_class_recall": kwargs.get("per_class_recall", {}),
                "per_class_f1": kwargs.get("per_class_f1", {}),
            }

    prec = metrics_dict.get("per_class_precision", {})
    rec = metrics_dict.get("per_class_recall", {})
    f1 = metrics_dict.get("per_class_f1", {})

    if class_names is None:
        class_names = list(prec.keys()) if prec else list(rec.keys())

    prec_vals = [prec.get(c, 0.0) * 100.0 if prec.get(c, 0.0) <= 1.0 else prec.get(c, 0.0) for c in class_names]
    rec_vals = [rec.get(c, 0.0) * 100.0 if rec.get(c, 0.0) <= 1.0 else rec.get(c, 0.0) for c in class_names]
    f1_vals = [f1.get(c, 0.0) * 100.0 if f1.get(c, 0.0) <= 1.0 else f1.get(c, 0.0) for c in class_names]

    df = pd.DataFrame({
        "Precision": prec_vals,
        "Recall": rec_vals,
        "F1-Score": f1_vals
    }, index=class_names)

    fig, ax = plt.subplots(figsize=figsize)
    df.plot(
        kind="bar",
        ax=ax,
        color=["#3498db", "#2ecc71", "#e74c3c"],
        edgecolor="#2c3e50",
        linewidth=0.8,
        width=0.75
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(frameon=True)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Per-class metrics plot saved to {save_path}")

    if show:
        plt.show()

    return fig

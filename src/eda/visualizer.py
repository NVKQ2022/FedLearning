"""
Visualizations for Exploratory Data Analysis (EDA) in FL-IoT-IDS.

Provides publication-grade visualizations for:
- Partition class distributions (horizontal bar charts with counts & percentages)
- Cross-client statistical heterogeneity (stacked bar charts & heatmaps)
- Feature skewness profiles (heavy-tailed feature detection)

Adheres to:
- skills/dataset-analysis-and-strategy/SKILL.md (Step 4 & 5)
- skills/evaluation-and-benchmarking/SKILL.md (Publication Plotting)
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
# Headless safety: use Agg if not in interactive environment
if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(
    eda: Dict[str, Any],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4.5),
    bar_color: str = "#2b5c8f",
    show: bool = False
) -> plt.Figure:
    """
    Renders a clean, publication-ready horizontal bar chart of a partition's class distribution.

    Args:
        eda: Dictionary returned by analyze_partition (or compute_partition_eda).
        save_path: Optional path to save the generated image.
        title: Optional plot title override.
        figsize: Figure dimensions (width, height).
        bar_color: Hex color for the bars.
        show: Whether to call plt.show() (useful in interactive notebooks).

    Returns:
        The matplotlib Figure object.
    """
    client_id = eda.get("client_id")
    total_samples = eda.get("total_samples", 0)
    counts = eda.get("class_counts", {})
    percentages = eda.get("class_percentages", {})

    classes = list(counts.keys())
    sample_values = [counts[c] for c in classes]

    # Invert order for natural top-to-bottom reading
    classes_rev = classes[::-1]
    values_rev = sample_values[::-1]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    bars = ax.barh(classes_rev, values_rev, color=bar_color, edgecolor="#1a365d", height=0.65)

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
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return fig


def plot_partition_heterogeneity(
    summary_df: pd.DataFrame,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = False
) -> plt.Figure:
    """
    Renders a 100% stacked horizontal bar chart showing class composition across edge clients.
    Immediately illustrates statistical heterogeneity (Dirichlet label skew vs IID uniformity).

    Args:
        summary_df: DataFrame returned by summarize_partitions.
        class_names: List of class names.
        save_path: Optional path to save image.
        title: Plot title override.
        figsize: Figure dimensions.
        show: Whether to display figure in notebook.

    Returns:
        The matplotlib Figure object.
    """
    clients = [f"Client {int(cid)}" for cid in summary_df["client_id"]]
    num_clients = len(clients)

    # Extract percentage columns
    pct_cols = [f"{c}_pct" for c in class_names if f"{c}_pct" in summary_df.columns]
    pct_matrix = summary_df[pct_cols].values  # shape: (num_clients, num_classes)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    cmap = plt.get_cmap("tab10")
    left_accum = np.zeros(num_clients)

    for i, c_name in enumerate(class_names):
        col_name = f"{c_name}_pct"
        if col_name in summary_df.columns:
            widths = summary_df[col_name].values
            ax.barh(clients, widths, left=left_accum, label=c_name, color=cmap(i % 10), height=0.65)
            left_accum += widths

    ax.set_xlim(0, 100)
    ax.set_xlabel("Class Share (%)", fontsize=10, fontweight="bold")
    ax.set_title(title or "Cross-Client Label Distribution Heterogeneity", fontsize=11, fontweight="bold", pad=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return fig


def plot_feature_skewness(
    stats_df: pd.DataFrame,
    top_k: int = 15,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
    show: bool = False
) -> plt.Figure:
    """
    Visualizes the top-k most skewed features with a reference threshold line (|γ1| = 2).

    Args:
        stats_df: DataFrame returned by analyze_features.
        top_k: Number of highest skew features to display.
        save_path: Optional save path.
        title: Title override.
        figsize: Figure dimensions.
        show: Whether to display figure in notebook.

    Returns:
        The matplotlib Figure object.
    """
    if "skewness" not in stats_df.columns or "feature" not in stats_df.columns:
        raise ValueError("stats_df must contain 'feature' and 'skewness' columns.")

    top_df = stats_df.sort_values(by="skewness", key=abs, ascending=False).head(top_k)
    features = top_df["feature"].tolist()[::-1]
    skews = top_df["skewness"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    colors = ["#c53030" if abs(s) > 2.0 else "#2b6cb0" for s in skews]
    ax.barh(features, skews, color=colors, height=0.65)

    # Reference lines at +/- 2.0
    ax.axvline(2.0, color="#e53e3e", linestyle="--", linewidth=1.2, label="Heavy-tail threshold (|γ| = 2)")
    ax.axvline(-2.0, color="#e53e3e", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Fisher-Pearson Skewness (γ₁)", fontsize=10, fontweight="bold")
    ax.set_title(title or f"Top {top_k} Features by Absolute Skewness", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return fig

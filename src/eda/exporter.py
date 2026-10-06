"""
EDA Serialization and Export Module for FL-IoT-IDS.

Handles standardized persistence of EDA diagnostics, client partition profiles,
and cross-client heterogeneity summaries into JSON files and publication figures.
"""

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

from .analyzer import analyze_partition
from .visualizer import plot_class_distribution

logger = logging.getLogger(__name__)


def export_client_eda(
    client_dir: str,
    y: np.ndarray,
    class_names: List[str],
    client_id: int,
    generate_plot: bool = True
) -> Dict[str, Any]:
    """
    Computes statistical EDA for an edge client partition, persists 'eda.json',
    and optionally renders 'class_distribution.png'.

    Args:
        client_dir: Target directory for the client (e.g. 'scenarios/E5/client_0').
        y: 1D array of client labels.
        class_names: List of class names.
        client_id: Integer index of the client.
        generate_plot: Whether to save class_distribution.png.

    Returns:
        Structured EDA dictionary.
    """
    os.makedirs(client_dir, exist_ok=True)

    # 1. Compute statistical EDA
    eda = analyze_partition(y=y, class_names=class_names, client_id=client_id)

    # 2. Persist eda.json atomically
    eda_json_path = os.path.join(client_dir, "eda.json")
    temp_json = f"{eda_json_path}.tmp"
    with open(temp_json, "w") as f:
        json.dump(eda, f, indent=2)
    shutil.move(temp_json, eda_json_path)

    # 3. Render class_distribution.png
    if generate_plot:
        plot_path = os.path.join(client_dir, "class_distribution.png")
        plot_class_distribution(eda=eda, save_path=plot_path)

    return eda


def export_scenario_eda_summary(
    server_dir: str,
    cross_client_summary: List[Dict[str, Any]]
) -> str:
    """
    Persists cross-client partition statistical summary to server directory.

    Args:
        server_dir: Path to scenario server directory (e.g. 'scenarios/E5/server').
        cross_client_summary: List of client summaries.

    Returns:
        Path to written clients_summary.json.
    """
    os.makedirs(server_dir, exist_ok=True)
    summary_path = os.path.join(server_dir, "clients_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cross_client_summary, f, indent=2)
    return summary_path


def export_dataset_audit(
    output_dir: str,
    audit_data: Dict[str, Any],
    filename: str = "dataset_eda.json"
) -> str:
    """
    Persists comprehensive dataset quality audit and statistical profile to disk.

    Args:
        output_dir: Target output folder.
        audit_data: Output dictionary from analyze_dataset.
        filename: Target filename.

    Returns:
        Path to written audit file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(audit_data, f, indent=2)
    return filepath

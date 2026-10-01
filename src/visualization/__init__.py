"""
Publication-ready visualization package for FL-IoT-IDS experiments.
"""

from .plots import (
    set_publication_style,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_scenario_comparison,
    plot_client_distribution,
    plot_federated_convergence,
    plot_minority_recall,
    plot_per_class_metrics,
)

__all__ = [
    "set_publication_style",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "plot_scenario_comparison",
    "plot_client_distribution",
    "plot_federated_convergence",
    "plot_minority_recall",
    "plot_per_class_metrics",
]

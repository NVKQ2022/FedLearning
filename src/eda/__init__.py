"""
Exploratory Data Analysis (EDA) Package for FL-IoT-IDS.

Provides a unified, publication-grade analytical suite for:
- Statistical profiling of IoT network intrusion traffic
- Data quality, anomaly, and skewness auditing
- Non-IID Dirichlet partition heterogeneity diagnostics (Shannon & normalized entropy)
- Publication-ready visualizations and structured JSON exports

Exports:
- compute_shannon_entropy: Shannon and normalized entropy calculation.
- analyze_partition: Statistical profile of a single partition / client.
- summarize_partitions: Cross-client comparative summary DataFrame.
- diagnose_data_quality: Audit for NaNs, Infs, zero-variance, and negative values.
- analyze_features: Summary moments and Fisher-Pearson skewness detection.
- analyze_dataset: Complete end-to-end dataset profiling.
- plot_class_distribution: Single partition horizontal bar chart.
- plot_partition_heterogeneity: 100% stacked bar chart of cross-client class share.
- plot_feature_skewness: Bar chart of top-k skewed features with |γ1| = 2 threshold.
- export_client_eda: Persists client eda.json and class_distribution.png.
- export_scenario_eda_summary: Persists server clients_summary.json.
- export_dataset_audit: Persists dataset_eda.json.
"""

from .analyzer import (
    compute_shannon_entropy,
    analyze_partition,
    summarize_partitions,
    diagnose_data_quality,
    analyze_features,
    analyze_dataset,
)
from .visualizer import (
    plot_class_distribution,
    plot_partition_heterogeneity,
    plot_feature_skewness,
)
from .exporter import (
    export_client_eda,
    export_scenario_eda_summary,
    export_dataset_audit,
)

__all__ = [
    "compute_shannon_entropy",
    "analyze_partition",
    "summarize_partitions",
    "diagnose_data_quality",
    "analyze_features",
    "analyze_dataset",
    "plot_class_distribution",
    "plot_partition_heterogeneity",
    "plot_feature_skewness",
    "export_client_eda",
    "export_scenario_eda_summary",
    "export_dataset_audit",
]

"""
Data preprocessing, transformation, and partitioning modules for FL-IoT-IDS.

Exports:
- BasePreprocessor: Abstract preprocessor contract ensuring leak-free tabular transformations.
- BasePartitioner: Abstract client data partitioning contract with data leakage verification.
- TabularDataPreprocessor: Robust tabular scaling, median imputation, and label encoding pipeline.
- CICIoT2023Preprocessor: Alias for TabularDataPreprocessor.
- StratifiedIIDPartitioner: Uniform IID client partitioner.
- DirichletNonIIDPartitioner: Dirichlet label distribution skew partitioner (alpha).
- DirichletPartitioner: Alias for DirichletNonIIDPartitioner.
- partition_iid / partition_dirichlet: Convenience functional wrappers.
- TabularFlowDataset: PyTorch Dataset wrapper.
- load_and_preprocess_ciciot2023: End-to-end data pipeline factory.
- compute_balanced_class_weights: Inverse-frequency class weight calculator.
- summarize_client_partitions: Dataframe distribution summary.
- create_client_dataloaders: Multi-client DataLoader factory.
"""

from .base import BasePreprocessor, BasePartitioner
from .preprocess import (
    TabularDataPreprocessor,
    CICIoT2023Preprocessor,
    TabularFlowDataset,
    load_and_preprocess_ciciot2023,
    compute_balanced_class_weights,
)
from .partition import (
    StratifiedIIDPartitioner,
    DirichletNonIIDPartitioner,
    DirichletPartitioner,
    partition_iid,
    partition_dirichlet,
    summarize_client_partitions,
    create_client_dataloaders,
)

__all__ = [
    "BasePreprocessor",
    "BasePartitioner",
    "TabularDataPreprocessor",
    "CICIoT2023Preprocessor",
    "TabularFlowDataset",
    "StratifiedIIDPartitioner",
    "DirichletNonIIDPartitioner",
    "DirichletPartitioner",
    "load_and_preprocess_ciciot2023",
    "compute_balanced_class_weights",
    "partition_iid",
    "partition_dirichlet",
    "summarize_client_partitions",
    "create_client_dataloaders",
]

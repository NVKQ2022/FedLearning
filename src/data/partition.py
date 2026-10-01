"""
Backward-compatibility proxy for src.data.partition -> src.data.partitioner.
"""

from .partitioner import (
    StratifiedIIDPartitioner,
    DirichletNonIIDPartitioner,
    DirichletPartitioner,
    partition_iid,
    partition_dirichlet,
    summarize_client_partitions,
    get_client_distribution,
    create_client_dataloaders,
)

__all__ = [
    "StratifiedIIDPartitioner",
    "DirichletNonIIDPartitioner",
    "DirichletPartitioner",
    "partition_iid",
    "partition_dirichlet",
    "summarize_client_partitions",
    "get_client_distribution",
    "create_client_dataloaders",
]

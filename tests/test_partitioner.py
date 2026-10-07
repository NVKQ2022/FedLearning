"""
Unit tests for Federated Data Partitioning (IID and Dirichlet Non-IID).
Ensures flexible signatures partition(y), partition(X, y), and partition(X=X, y=y) work seamlessly.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.partitioner import (
    StratifiedIIDPartitioner,
    DirichletNonIIDPartitioner,
    partition_iid,
    partition_dirichlet,
    summarize_client_partitions,
)


class TestPartitionerModule:
    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        cls.num_samples = 1200
        cls.X = np.random.randn(cls.num_samples, 8)
        # Class distribution: 70% 0, 20% 1, 8% 2, 2% 3
        cls.y = np.random.choice([0, 1, 2, 3], size=cls.num_samples, p=[0.70, 0.20, 0.08, 0.02])
        cls.class_names = ["DDoS", "DoS", "Recon", "Web"]

    def test_stratified_iid_single_arg(self):
        partitioner = StratifiedIIDPartitioner(num_clients=4, seed=42)
        partitions = partitioner.partition(self.y)
        assert len(partitions) == 4
        partitioner.verify_partition(partitions, total_samples=self.num_samples, min_samples_per_client=50)

    def test_stratified_iid_two_args(self):
        partitioner = StratifiedIIDPartitioner(num_clients=4, seed=42)
        partitions = partitioner.partition(self.X, self.y)
        assert len(partitions) == 4
        partitioner.verify_partition(partitions, total_samples=self.num_samples, min_samples_per_client=50)

    def test_stratified_iid_kwargs(self):
        partitioner = StratifiedIIDPartitioner(num_clients=4, seed=42)
        partitions = partitioner.partition(X=self.X, y=self.y)
        assert len(partitions) == 4
        partitioner.verify_partition(partitions, total_samples=self.num_samples, min_samples_per_client=50)

    def test_dirichlet_non_iid_single_arg(self):
        partitioner = DirichletNonIIDPartitioner(num_clients=3, alpha=0.5, min_samples_per_client=50, seed=42)
        partitions = partitioner.partition(self.y)
        assert len(partitions) == 3
        partitioner.verify_partition(partitions, total_samples=self.num_samples, min_samples_per_client=50)

    def test_dirichlet_non_iid_two_args(self):
        # Critical regression test for notebook caller: partition(data['X_train'], data['y_train'])
        partitioner = DirichletNonIIDPartitioner(num_clients=3, alpha=0.5, min_samples_per_client=50, seed=42)
        partitions = partitioner.partition(self.X, self.y)
        assert len(partitions) == 3
        partitioner.verify_partition(partitions, total_samples=self.num_samples, min_samples_per_client=50)

    def test_dirichlet_non_iid_kwargs(self):
        partitioner = DirichletNonIIDPartitioner(num_clients=3, alpha=0.5, min_samples_per_client=50, seed=42)
        partitions = partitioner.partition(X=self.X, y=self.y)
        assert len(partitions) == 3
        partitioner.verify_partition(partitions, total_samples=self.num_samples, min_samples_per_client=50)

    def test_pandas_series_labels(self):
        partitioner = DirichletNonIIDPartitioner(num_clients=3, alpha=0.5, min_samples_per_client=50, seed=42)
        y_series = pd.Series(self.y)
        partitions = partitioner.partition(self.X, y_series)
        assert len(partitions) == 3

    def test_sample_count_mismatch_raises(self):
        partitioner = StratifiedIIDPartitioner(num_clients=3, seed=42)
        with pytest.raises(ValueError, match="Sample count mismatch"):
            partitioner.partition(self.X[:100], self.y)

    def test_functional_wrappers(self):
        iid_parts = partition_iid(self.X, self.y, num_clients=3)
        assert len(iid_parts) == 3

        noniid_parts = partition_dirichlet(self.X, self.y, num_clients=3, alpha=0.5, min_samples_per_client=50)
        assert len(noniid_parts) == 3

    def test_distribution_summary_and_heterogeneity(self):
        partitioner = DirichletNonIIDPartitioner(num_clients=3, alpha=0.5, min_samples_per_client=50, seed=42)
        partitions = partitioner.partition(self.X, self.y)
        
        # Test distribution dataframe
        df = partitioner.get_client_distribution(partitions, self.y, class_names=self.class_names)
        assert len(df) == 3
        
        # Test heterogeneity score
        het = partitioner.compute_heterogeneity_score(self.y, partitions)
        assert 0.0 <= het["mean_total_variation"] <= 1.0

        # Test summarize_client_partitions helper
        summary_df = summarize_client_partitions(partitions, self.y, class_names=self.class_names)
        assert len(summary_df) == 3

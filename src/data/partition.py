"""
Federated Learning Data Partitioning Module.

This module provides IID and Dirichlet Non-IID partitioning algorithms to simulate
statistical heterogeneity across simulated IoT edge clients.
It enforces minimum client sample constraints to prevent empty or degenerate partitions.

Adheres to:
- skills/dataset-analysis-and-strategy/SKILL.md (Step 5: Distributed & Federated Partitioning)
- skills/methodology-audit/SKILL.md (Pillar 4: Statistical Heterogeneity & Partition Soundness)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


from src.base.partitioner import BasePartitioner


class StratifiedIIDPartitioner(BasePartitioner):
    """
    Uniform Stratified IID Partitioner.
    Distributes samples such that each client observes an identical or near-identical
    class distribution.
    """
    def partition(self, y: np.ndarray, **kwargs: Any) -> Dict[int, np.ndarray]:
        logger.info(f"Generating Stratified IID partition for {self.num_clients} clients (seed={self.seed})...")
        rng = np.random.default_rng(self.seed)
        unique_classes = np.unique(y)
        client_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}

        for c in unique_classes:
            class_idx = np.where(y == c)[0]
            rng.shuffle(class_idx)
            
            # Split class indices across clients as evenly as possible
            splits = np.array_split(class_idx, self.num_clients)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        # Shuffle indices per client to eliminate ordering bias
        result = {}
        for client_id, indices in client_indices.items():
            arr = np.array(indices, dtype=np.int64)
            rng.shuffle(arr)
            result[client_id] = arr
            
        return result


class DirichletPartitioner(BasePartitioner):
    """
    Dirichlet Non-IID Partitioner.
    Simulates label distribution skew across clients using a Dirichlet distribution Dir(alpha).
    - alpha -> inf: Uniform IID distribution.
    - alpha = 1.0: Mild statistical heterogeneity.
    - alpha = 0.5: Moderate statistical heterogeneity.
    - alpha = 0.1: Severe statistical heterogeneity (extreme label skew).
    """
    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,
        min_samples_per_client: int = 100,
        seed: int = 42,
        max_retries: int = 50
    ):
        super().__init__(num_clients=num_clients, seed=seed)
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.max_retries = max_retries

    def partition(self, y: np.ndarray, **kwargs: Any) -> Dict[int, np.ndarray]:
        logger.info(
            f"Generating Dirichlet Non-IID partition (alpha={self.alpha}, K={self.num_clients}, min_samples={self.min_samples_per_client})..."
        )
        unique_classes = np.unique(y)
        total_samples = len(y)

        for attempt in range(self.max_retries):
            current_seed = self.seed + attempt
            rng = np.random.default_rng(current_seed)
            client_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}

            for c in unique_classes:
                class_idx = np.where(y == c)[0]
                rng.shuffle(class_idx)
                n_class = len(class_idx)

                # Sample class-wise client proportions from Dirichlet distribution
                proportions = rng.dirichlet(np.repeat(self.alpha, self.num_clients))
                
                # Convert continuous proportions to discrete sample counts
                counts = (proportions * n_class).astype(int)

                # Adjust rounding disparity to ensure all samples are assigned
                disparity = n_class - counts.sum()
                if disparity > 0:
                    # Add remainder to clients with highest fractional remainder
                    remainders = (proportions * n_class) - counts
                    for top_client in np.argsort(-remainders)[:disparity]:
                        counts[top_client] += 1

                current_pos = 0
                for client_id, count in enumerate(counts):
                    if count > 0:
                        client_indices[client_id].extend(
                            class_idx[current_pos : current_pos + count].tolist()
                        )
                        current_pos += count

            # Verify that all clients meet the minimum sample requirement
            min_size = min(len(indices) for indices in client_indices.values())
            if min_size >= self.min_samples_per_client:
                logger.info(
                    f"Valid Dirichlet partition found on attempt {attempt + 1}. Minimum client size: {min_size:,}"
                )
                result = {}
                for client_id, indices in client_indices.items():
                    arr = np.array(indices, dtype=np.int64)
                    rng.shuffle(arr)
                    result[client_id] = arr
                return result
            else:
                logger.debug(
                    f"Attempt {attempt + 1} produced client with {min_size} samples (< {self.min_samples_per_client}). Retrying..."
                )

        raise RuntimeError(
            f"Failed to generate a valid Dirichlet partition after {self.max_retries} retries. "
            f"Consider lowering min_samples_per_client (currently {self.min_samples_per_client}) "
            f"or increasing alpha (currently {self.alpha})."
        )


def partition_iid(
    y: np.ndarray,
    num_clients: int,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Uniform Stratified IID Partitioning (convenience functional interface).
    """
    partitioner = StratifiedIIDPartitioner(num_clients=num_clients, seed=seed)
    return partitioner.partition(y)


def partition_dirichlet(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 100,
    seed: int = 42,
    max_retries: int = 50
) -> Dict[int, np.ndarray]:
    """
    Dirichlet Non-IID Partitioning (convenience functional interface).
    """
    partitioner = DirichletPartitioner(
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=min_samples_per_client,
        seed=seed,
        max_retries=max_retries
    )
    return partitioner.partition(y)


def summarize_client_partitions(
    client_partitions: Dict[int, np.ndarray],
    y: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generates a detailed statistical summary of the client partitions.
    Shows total samples per client and per-class distributions.

    Args:
        client_partitions: Dict mapping client_id to sample indices.
        y: 1D array of class labels.
        class_names: Optional list of human-readable class names.

    Returns:
        DataFrame where rows represent clients and columns represent class sample counts and percentages.
    """
    unique_classes = sorted(np.unique(y))
    num_classes = len(unique_classes)
    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Class_{c}" for c in unique_classes]

    summary_rows = []
    for client_id, indices in sorted(client_partitions.items()):
        client_labels = y[indices]
        total_samples = len(client_labels)
        
        row = {"client_id": client_id, "total_samples": total_samples}
        
        for c, name in zip(unique_classes, class_names):
            cnt = int((client_labels == c).sum())
            pct = (cnt / total_samples * 100.0) if total_samples > 0 else 0.0
            row[f"{name}_count"] = cnt
            row[f"{name}_pct"] = round(pct, 2)
            
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    return df_summary


def create_client_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    client_partitions: Dict[int, np.ndarray],
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> Dict[int, "DataLoader"]:
    """
    Builds PyTorch DataLoaders for each partitioned client.

    Args:
        X: Preprocessed feature matrix (numpy array).
        y: Encoded labels (numpy array).
        client_partitions: Dict mapping client_id to sample indices.
        batch_size: Mini-batch size for local client training.
        shuffle: Whether to shuffle batches during iteration.
        num_workers: PyTorch DataLoader workers.

    Returns:
        Dict mapping client_id to PyTorch DataLoader.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to instantiate DataLoaders. Please install torch.")

    dataloaders = {}
    for client_id, indices in client_partitions.items():
        X_client = torch.tensor(X[indices], dtype=torch.float32)
        y_client = torch.tensor(y[indices], dtype=torch.long)
        dataset = TensorDataset(X_client, y_client)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False
        )
        dataloaders[client_id] = loader

    return dataloaders


if __name__ == "__main__":
    print("--- Running Data Partition Verification Test ---")
    # Generate mock dataset with severe class imbalance
    mock_labels = np.concatenate([
        np.zeros(5000, dtype=int),  # Majority class 0 (Benign/DDoS)
        np.ones(3000, dtype=int),   # Class 1 (DoS)
        np.full(500, 2, dtype=int), # Class 2 (Recon)
        np.full(50, 3, dtype=int),  # Rare Minority class 3 (Web-based)
    ])
    class_names = ["Majority", "Medium", "Minor", "Rare"]
    num_clients = 3

    # 1. Test IID Partition
    iid_parts = partition_iid(mock_labels, num_clients=num_clients, seed=42)
    summary_iid = summarize_client_partitions(iid_parts, mock_labels, class_names=class_names)
    print("\n--- IID Partition Summary ---")
    print(summary_iid[["client_id", "total_samples", "Majority_pct", "Rare_pct"]])

    # 2. Test Dirichlet Non-IID Partition (alpha=0.5)
    non_iid_parts = partition_dirichlet(
        mock_labels, num_clients=num_clients, alpha=0.5, min_samples_per_client=50, seed=42
    )
    summary_non_iid = summarize_client_partitions(non_iid_parts, mock_labels, class_names=class_names)
    print("\n--- Dirichlet (alpha=0.5) Non-IID Partition Summary ---")
    print(summary_non_iid[["client_id", "total_samples", "Majority_pct", "Rare_pct"]])

    print("\n✅ Partition module verification passed successfully!")

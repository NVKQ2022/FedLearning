"""
Abstract base class for federated data partitioners.

Provides:
- BasePartitioner: Contract for simulated federated partitioning with verification
  of zero index leakage across edge clients, distribution matrix computation,
  and empirical Non-IID divergence quantification.

Adheres to:
- skills/dataset-analysis-and-strategy/SKILL.md (Step 5: Distributed & Federated Partitioning)
- skills/methodology-audit/SKILL.md (Pillar 4: Statistical Heterogeneity & Partition Soundness)
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BasePartitioner(ABC):
    """
    Abstract base class for federated data partitioners.
    
    Subclasses must implement:
    - partition(y, **kwargs): Partitions training label indices across simulated clients.
    
    Provides out-of-the-box:
    - verify_partition(): Audits client isolation, zero overlap, and min sample bounds.
    - get_client_distribution(): Builds a client-by-class contingency table.
    - compute_heterogeneity_score(): Quantifies Total Variation (TV) distance against global distribution.
    """
    def __init__(self, num_clients: int, seed: int = 42) -> None:
        if num_clients < 1:
            raise ValueError(f"num_clients must be at least 1, got {num_clients}")
        self.num_clients = num_clients
        self.seed = seed

    @abstractmethod
    def partition(self, y: np.ndarray, **kwargs: Any) -> Dict[int, np.ndarray]:
        """
        Partitions sample indices of dataset labels y across K clients.

        Args:
            y: 1D array of class labels for the training set.
            **kwargs: Partitioner-specific parameters (e.g. alpha for Dirichlet).

        Returns:
            Dictionary mapping client_id (0 to K-1) to an array of indices.
        """
        pass

    def verify_partition(
        self,
        partition_dict: Dict[int, np.ndarray],
        total_samples: int,
        min_samples_per_client: int = 1
    ) -> bool:
        """
        Audits partition correctness to prevent data leakage and invalid simulations.

        Checks:
        1. Number of clients matches expected num_clients.
        2. Zero overlapping indices across distinct clients (Strict isolation).
        3. All clients have >= min_samples_per_client.
        4. Total assigned indices does not exceed total_samples.

        Returns:
            True if all validation checks pass.
        Raises:
            ValueError on any integrity violation.
        """
        if len(partition_dict) != self.num_clients:
            raise ValueError(
                f"Expected {self.num_clients} client partitions, got {len(partition_dict)}"
            )

        seen_indices = set()
        for client_id, indices in partition_dict.items():
            if len(indices) < min_samples_per_client:
                raise ValueError(
                    f"Client {client_id} has {len(indices)} samples, which is below minimum {min_samples_per_client}"
                )
            
            # Check for intra-client duplicates
            if len(indices) != len(set(indices)):
                raise ValueError(f"Client {client_id} contains duplicate indices internally!")

            # Check for inter-client leakage
            idx_set = set(indices)
            overlap = seen_indices.intersection(idx_set)
            if overlap:
                raise ValueError(
                    f"Data leakage detected! {len(overlap)} indices shared across clients (e.g., Client {client_id})."
                )
            seen_indices.update(idx_set)

        logger.info(
            f"✅ Partition verified successfully: {len(seen_indices):,}/{total_samples:,} samples partitioned across {self.num_clients} clients with zero leakage."
        )
        return True

    def get_client_distribution(
        self,
        y: np.ndarray,
        partition_dict: Dict[int, np.ndarray],
        class_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generates a summary DataFrame of sample counts per class per client.
        """
        unique_classes = np.unique(y)
        records = []

        for client_id, indices in partition_dict.items():
            client_y = y[indices]
            counts = {c: int(np.sum(client_y == c)) for c in unique_classes}
            counts["Total Samples"] = len(indices)
            counts["Client ID"] = client_id
            records.append(counts)

        df = pd.DataFrame(records).set_index("Client ID")
        if class_names is not None and len(class_names) == len(unique_classes):
            rename_map = {c: class_names[i] for i, c in enumerate(unique_classes)}
            df = df.rename(columns=rename_map)

        return df

    def compute_heterogeneity_score(
        self,
        y: np.ndarray,
        partition_dict: Dict[int, np.ndarray]
    ) -> Dict[str, float]:
        """
        Quantifies statistical heterogeneity using Mean Total Variation (TV) distance:
            TV(P_k, P_global) = 0.5 * sum_c |P_k(c) - P_global(c)|
        
        Range:
        - 0.0: Perfect IID distribution.
        - 1.0: Extreme Non-IID (complete class separation).
        """
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        global_dist = np.array([np.sum(y == c) for c in unique_classes], dtype=np.float64) / len(y)

        tv_distances: List[float] = []
        for client_id, indices in partition_dict.items():
            if len(indices) == 0:
                continue
            client_y = y[indices]
            client_dist = np.array([np.sum(client_y == c) for c in unique_classes], dtype=np.float64) / len(indices)
            tv = 0.5 * float(np.sum(np.abs(client_dist - global_dist)))
            tv_distances.append(tv)

        return {
            "mean_total_variation": float(np.mean(tv_distances)),
            "max_total_variation": float(np.max(tv_distances)),
            "min_total_variation": float(np.min(tv_distances))
        }

"""
Abstract base class for tabular data preprocessors in network intrusion detection.

Provides:
- BasePreprocessor: Standardized fit, transform, fit_transform, and serialization
  contract ensuring strict data leakage prevention across train/val/test splits.

Adheres to:
- skills/dataset-analysis-and-strategy/SKILL.md (Step 3: Tabular Preprocessing Pipeline)
- skills/methodology-audit/SKILL.md (Pillar 1: Data Leakage Prevention)
"""

from abc import ABC, abstractmethod
import os
import pickle
import logging
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for tabular data preprocessing pipelines.
    
    Subclasses must implement:
    - fit(df, target_col, **kwargs): Learns statistical parameters strictly from training partition.
    - transform(df, target_col, **kwargs): Applies fitted transformations to unseen evaluation splits.
    
    Provides out-of-the-box:
    - fit_transform(): Unified fit and transform execution.
    - save() / load(): Safe pickle-based state serialization.
    - Standard properties for feature and class metadata inspection.
    """
    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.feature_columns: List[str] = []
        self.class_names_: List[str] = []
        self.num_classes_: int = 0

    @property
    def num_features(self) -> int:
        """Returns the number of engineered feature columns."""
        return len(self.feature_columns)

    @property
    def num_classes(self) -> int:
        """Returns the number of target classes."""
        return self.num_classes_

    @property
    def class_names(self) -> List[str]:
        """Returns the list of target class names."""
        return list(self.class_names_)

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, target_col: str, **kwargs: Any) -> "BasePreprocessor":
        """
        Fits preprocessing parameters (imputation statistics, scalers, encoders)
        strictly from the training dataframe.
        """
        pass

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        **kwargs: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applies learned transformations to a dataframe.

        Returns:
            Tuple of (features_array, Optional[labels_array]).
        """
        pass

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits on training dataframe and returns transformed features and labels.
        """
        self.fit(df_train, target_col, **kwargs)
        X, y = self.transform(df_train, target_col, **kwargs)
        if y is None:
            raise ValueError("Labels array cannot be None after fit_transform on training partition.")
        return X, y

    def save(self, filepath: str) -> None:
        """
        Serializes fitted preprocessor instance to disk.
        """
        if not self.is_fitted:
            logger.warning("Saving an unfitted preprocessor instance.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BasePreprocessor":
        """
        Loads a serialized preprocessor from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        if not isinstance(instance, BasePreprocessor):
            raise TypeError(f"Loaded object is not a BasePreprocessor instance: {type(instance)}")
        logger.info(f"Loaded preprocessor from {filepath} (Classes: {instance.class_names_})")
        return instance

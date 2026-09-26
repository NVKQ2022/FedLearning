"""
Data Preprocessing Pipeline for CICIoT2023 Tabular Network Traffic.

This module implements a leak-free, production-grade tabular preprocessor designed
specifically for network intrusion detection and federated learning workloads.
It handles numerical anomalies (infinities, missing values, clock-drift negatives),
heavy-tailed skewness, and severe class imbalance.

Adheres to:
- skills/dataset-analysis-and-strategy/SKILL.md
- skills/methodology-audit/SKILL.md (Pillar 1: Data Leakage Prevention)
"""

import os
import pickle
import logging
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical columns for CICIoT2023 (39 features + 1 target)
TARGET_COLUMN = "group_class_name"

HEAVY_TAIL_COLUMNS = ["IAT", "Rate", "Variance", "Std", "Tot sum", "Tot size", "AVG", "Max"]
NON_NEGATIVE_COLUMNS = ["IAT", "Rate", "Time_To_Live", "Header_Length", "Min", "Max", "AVG", "Std", "Tot sum", "Tot size"]


class TabularFlowDataset(Dataset):
    """
    PyTorch Dataset wrapper for tabular network flow features and labels.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TabularFlowDataset. Please install torch.")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class TabularDataPreprocessor:
    """
    Leak-free preprocessor for tabular network flow traffic.
    Guarantees all statistics (imputation medians, scalers, label encodings)
    are fitted strictly on the training set and applied without leakage.
    """
    def __init__(self, scaler_type: str = "robust", apply_log1p: bool = True):
        self.scaler_type = scaler_type.lower()
        self.apply_log1p = apply_log1p
        self.scaler = RobustScaler() if self.scaler_type == "robust" else StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.feature_columns: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.is_fitted: bool = False
        self.class_names_: List[str] = []
        self.num_classes_: int = 0

    def fit(self, df_train: pd.DataFrame, target_col: str = TARGET_COLUMN) -> "TabularDataPreprocessor":
        """
        Learns statistical parameters strictly from the training dataframe.
        """
        logger.info("Fitting TabularDataPreprocessor on training partition...")
        
        # 1. Determine feature columns
        self.feature_columns = [col for col in df_train.columns if col != target_col]
        X = df_train[self.feature_columns].copy()
        y = df_train[target_col].copy()

        # 2. Learn label encoding
        self.label_encoder.fit(y)
        self.class_names_ = list(self.label_encoder.classes_)
        self.num_classes_ = len(self.class_names_)
        logger.info(f"Target classes detected ({self.num_classes_}): {self.class_names_}")

        # 3. Clean and compute imputation medians strictly on train
        for col in self.feature_columns:
            # Replace infinities with NaN
            series = X[col].replace([np.inf, -np.inf], np.nan)
            median_val = float(series.median())
            if np.isnan(median_val):
                median_val = 0.0
            self.medians_[col] = median_val

        # 4. Clean train features using learned medians
        X_clean = self._clean_features(X)

        # 5. Fit the scaler strictly on cleaned training features
        self.scaler.fit(X_clean)
        self.is_fitted = True
        logger.info("TabularDataPreprocessor successfully fitted.")
        return self

    def transform(
        self, df: pd.DataFrame, target_col: Optional[str] = TARGET_COLUMN
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms a dataframe using previously learned parameters.
        Can transform validation/test sets or unlabelled inference data.
        """
        if not self.is_fitted:
            raise RuntimeError("TabularDataPreprocessor must be fitted before calling transform()!")

        X = df[self.feature_columns].copy()
        X_clean = self._clean_features(X)
        X_scaled = self.scaler.transform(X_clean).astype(np.float32)

        y_encoded = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            y_encoded = self.label_encoder.transform(y).astype(np.int64)

        return X_scaled, y_encoded

    def fit_transform(
        self, df_train: pd.DataFrame, target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits on df_train and transforms it in a single step.
        """
        self.fit(df_train, target_col=target_col)
        return self.transform(df_train, target_col=target_col)

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning routines (inf replacement, imputation, clipping, log1p)
        using statistics learned during fit().
        """
        X_clean = X.copy()
        
        for col in self.feature_columns:
            # Replace inf with nan
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Impute using train median
            median_val = self.medians_.get(col, 0.0)
            X_clean[col] = X_clean[col].fillna(median_val)

            # Non-negative clip for physical flow properties
            if col in NON_NEGATIVE_COLUMNS or "iat" in col.lower() or "time" in col.lower():
                X_clean[col] = X_clean[col].clip(lower=0.0)

            # Apply log1p on heavy-tailed columns to compress dynamic range
            if self.apply_log1p and (col in HEAVY_TAIL_COLUMNS or "tot" in col.lower() or "iat" in col.lower()):
                X_clean[col] = np.log1p(X_clean[col])

        return X_clean

    def save(self, filepath: str) -> None:
        """Saves fitted preprocessor state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor state successfully saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "TabularDataPreprocessor":
        """Loads fitted preprocessor state from disk."""
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor state successfully loaded from {filepath}")
        return preprocessor


def compute_balanced_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Computes inverse frequency class weights to mitigate severe class imbalance:
    w_c = N / (C * N_c)
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return weights.astype(np.float32)


def load_and_preprocess_ciciot2023(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    sample_size: Optional[int] = None,
    scaler_type: str = "robust",
    random_state: int = 42,
    save_preprocessor_path: Optional[str] = None
) -> Dict[str, Union[np.ndarray, List[str], int, TabularDataPreprocessor]]:
    """
    High-level end-to-end data loading and preprocessing pipeline for CICIoT2023.

    Args:
        csv_path: Path to merged_CICIOT2023_data.csv.
        test_size: Fraction of dataset reserved for the holdout global test set.
        val_size: Fraction of the training set reserved for validation.
        sample_size: Optional stratified downsampling (e.g. 500,000) for rapid development.
        scaler_type: 'robust' (default) or 'standard'.
        random_state: Deterministic random seed.
        save_preprocessor_path: Optional path to persist fitted preprocessor.

    Returns:
        Dictionary containing:
        - X_train, y_train
        - X_val, y_val
        - X_test, y_test
        - class_weights
        - class_names
        - feature_names
        - preprocessor
    """
    logger.info(f"Loading CICIoT2023 dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # 1. Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Raw dataset shape: {df.shape}")

    # 2. Stratified downsampling for rapid prototyping if requested
    if sample_size is not None and sample_size < len(df):
        logger.info(f"Applying stratified sampling to {sample_size:,} records...")
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[TARGET_COLUMN],
            random_state=random_state
        )
        logger.info(f"Subsampled dataset shape: {df.shape}")

    # 3. Leak-free Train/Test split BEFORE any transformation
    df_train_full, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state
    )

    # 4. Optional Train/Val split
    if val_size > 0.0:
        val_relative_size = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(
            df_train_full,
            test_size=val_relative_size,
            stratify=df_train_full[TARGET_COLUMN],
            random_state=random_state
        )
    else:
        df_train = df_train_full
        df_val = pd.DataFrame()

    logger.info(f"Partition sizes: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

    # 5. Fit preprocessor strictly on df_train
    preprocessor = TabularDataPreprocessor(scaler_type=scaler_type, apply_log1p=True)
    preprocessor.fit(df_train, target_col=TARGET_COLUMN)

    # 6. Transform partitions
    X_train, y_train = preprocessor.transform(df_train, target_col=TARGET_COLUMN)
    X_test, y_test = preprocessor.transform(df_test, target_col=TARGET_COLUMN)
    
    X_val, y_val = None, None
    if len(df_val) > 0:
        X_val, y_val = preprocessor.transform(df_val, target_col=TARGET_COLUMN)

    # 7. Compute balanced class weights on training labels
    class_weights = compute_balanced_class_weights(y_train, num_classes=preprocessor.num_classes_)

    if save_preprocessor_path:
        preprocessor.save(save_preprocessor_path)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_weights": class_weights,
        "class_names": preprocessor.class_names_,
        "feature_names": preprocessor.feature_columns,
        "input_dim": len(preprocessor.feature_columns),
        "num_classes": preprocessor.num_classes_,
        "preprocessor": preprocessor
    }


if __name__ == "__main__":
    # Self-test using a small synthetic batch to verify numerical stability and leak-free logic
    print("--- Running TabularDataPreprocessor Verification Test ---")
    mock_data = {
        "IAT": [0.001, -0.005, np.nan, 120.5, 0.05],
        "Rate": [100.0, np.inf, 25.0, 0.0, 500.0],
        "Variance": [12.0, 30.0, np.nan, 400.0, 15.0],
        "Protocol Type": [6, 17, 6, 6, 1],
        "group_class_name": ["Benign", "DDoS", "Benign", "DoS", "Brute-force"]
    }
    df_mock = pd.DataFrame(mock_data)
    preprocessor = TabularDataPreprocessor(scaler_type="robust")
    X_mock, y_mock = preprocessor.fit_transform(df_mock)
    
    print(f"Cleaned and scaled feature shape: {X_mock.shape}")
    print(f"Encoded labels: {y_mock}")
    print(f"Class names: {preprocessor.class_names_}")
    assert not np.isnan(X_mock).any(), "Sanity check failed: NaNs detected in transformed features!"
    assert not np.isinf(X_mock).any(), "Sanity check failed: Infs detected in transformed features!"
    print("✅ Preprocessor verification passed successfully!")

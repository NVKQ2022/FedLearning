"""
Core Exploratory Data Analysis (EDA) Module for FL-IoT-IDS.

Provides publication-grade data auditing, statistical profiling, and non-IID
heterogeneity diagnosis across tabular IoT traffic datasets and partitioned edge clients.

Adheres to:
- skills/dataset-analysis-and-strategy/SKILL.md (Steps 1 to 5)
- skills/evaluation-and-benchmarking/SKILL.md (Statistical Validity)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_shannon_entropy(counts: Union[np.ndarray, List[int]]) -> Tuple[float, float]:
    """
    Computes Shannon Entropy and Normalized Entropy for a class distribution.

    Normalized Entropy H_norm in [0, 1]:
    - H_norm = 1.0: Perfectly uniform (IID)
    - H_norm -> 0.0: Extreme non-IID concentration (all samples belong to single class)

    Args:
        counts: Class sample frequencies.

    Returns:
        Tuple of (shannon_entropy, normalized_entropy).
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = np.sum(counts_arr)
    if total <= 0:
        return 0.0, 0.0

    probs = counts_arr[counts_arr > 0] / total
    entropy = float(-np.sum(probs * np.log(probs)))
    num_classes = len(counts_arr)
    max_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0

    return round(entropy, 4), round(norm_entropy, 4)


def analyze_partition(
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    client_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes detailed statistical Exploratory Data Analysis (EDA) on a dataset partition.

    Args:
        y: 1D array of categorical integer labels.
        class_names: Optional human-readable class names.
        client_id: Optional client identifier integer.

    Returns:
        Structured EDA dictionary with sample counts, proportions, dominant class,
        minority representation, and normalized entropy.
    """
    total_samples = int(len(y))
    unique_labels, counts = np.unique(y, return_counts=True)
    label_to_count = dict(zip(unique_labels.tolist(), counts.tolist()))

    if class_names is None:
        num_classes = int(np.max(y) + 1) if len(y) > 0 else len(unique_labels)
        class_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        num_classes = len(class_names)

    class_counts: Dict[str, int] = {}
    class_percentages: Dict[str, float] = {}
    missing_classes: List[str] = []
    present_classes: List[str] = []

    for idx, name in enumerate(class_names):
        cnt = int(label_to_count.get(idx, 0))
        pct = round((cnt / total_samples * 100.0), 2) if total_samples > 0 else 0.0
        class_counts[name] = cnt
        class_percentages[name] = pct
        if cnt == 0:
            missing_classes.append(name)
        else:
            present_classes.append(name)

    # Calculate entropy
    entropy, norm_entropy = compute_shannon_entropy(list(class_counts.values()))

    # Dominant class
    dominant_class = max(class_counts.items(), key=lambda kv: kv[1])[0] if class_counts else "Unknown"
    dominant_pct = class_percentages.get(dominant_class, 0.0)

    # Imbalance ratio: max count / min non-zero count
    non_zero_counts = [c for c in class_counts.values() if c > 0]
    imbalance_ratio = float(max(non_zero_counts) / min(non_zero_counts)) if non_zero_counts else 1.0

    eda = {
        "client_id": client_id,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "dominant_class": dominant_class,
        "dominant_class_pct": dominant_pct,
        "shannon_entropy": entropy,
        "normalized_entropy": norm_entropy,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "present_classes_count": len(present_classes),
        "missing_classes": missing_classes,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
    }
    return eda


def summarize_partitions(
    client_partitions: Dict[int, np.ndarray],
    y: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generates a structured comparison DataFrame summarizing all client partitions.

    Args:
        client_partitions: Dict mapping client_id to sample indices.
        y: 1D array of class labels.
        class_names: Optional list of human-readable class names.

    Returns:
        DataFrame containing sample counts, class breakdown percentages, and entropy per client.
    """
    if class_names is None:
        unique_classes = sorted(np.unique(y).tolist())
        class_names = [f"Class_{c}" for c in unique_classes]

    summary_rows = []
    for client_id, indices in sorted(client_partitions.items()):
        client_labels = y[indices]
        eda = analyze_partition(client_labels, class_names=class_names, client_id=client_id)

        row = {
            "client_id": int(client_id),
            "total_samples": eda["total_samples"],
            "dominant_class": eda["dominant_class"],
            "dominant_pct": eda["dominant_class_pct"],
            "entropy": eda["normalized_entropy"],
        }
        for name in class_names:
            row[f"{name}_count"] = eda["class_counts"][name]
            row[f"{name}_pct"] = eda["class_percentages"][name]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def diagnose_data_quality(
    df_or_X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Audits tabular data quality to prevent silent numerical failures in neural networks.
    Checks:
    - Missing values (NaN / Null)
    - Infinite values (+inf / -inf)
    - Zero-variance / constant features
    - Negative values in non-negative metrics

    Args:
        df_or_X: Feature matrix as DataFrame or 2D NumPy array.
        feature_names: Optional feature column names.

    Returns:
        Quality diagnostic report dictionary.
    """
    if isinstance(df_or_X, np.ndarray):
        num_features = df_or_X.shape[1] if df_or_X.ndim > 1 else 1
        cols = feature_names or [f"feat_{i}" for i in range(num_features)]
        df = pd.DataFrame(df_or_X, columns=cols)
    else:
        df = df_or_X.copy()
        cols = list(df.columns)

    total_rows = len(df)
    nan_counts = df.isna().sum().to_dict()
    cols_with_nan = {c: int(v) for c, v in nan_counts.items() if v > 0}

    # Inf check (numeric columns only)
    numeric_df = df.select_dtypes(include=[np.number])
    cols_with_inf: Dict[str, int] = {}
    for c in numeric_df.columns:
        inf_count = int(np.isinf(numeric_df[c]).sum())
        if inf_count > 0:
            cols_with_inf[c] = inf_count

    # Constant / zero-variance columns
    zero_var_cols = [c for c in numeric_df.columns if numeric_df[c].std() == 0.0]

    # Negative value check
    negative_cols = [c for c in numeric_df.columns if (numeric_df[c] < 0).any()]

    report = {
        "total_records": total_rows,
        "total_features": len(cols),
        "has_missing_values": len(cols_with_nan) > 0,
        "columns_with_nans": cols_with_nan,
        "has_infinite_values": len(cols_with_inf) > 0,
        "columns_with_infs": cols_with_inf,
        "zero_variance_columns": zero_var_cols,
        "columns_with_negatives": negative_cols,
        "is_training_safe": len(cols_with_nan) == 0 and len(cols_with_inf) == 0 and len(zero_var_cols) == 0,
    }
    return report


def analyze_features(
    df_or_X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Computes statistical moments and Fisher-Pearson skewness across numerical features.
    Identifies heavy-tailed features (|skewness| > 2) requiring log1p or RobustScaler.

    Args:
        df_or_X: Feature matrix.
        feature_names: Optional column names.

    Returns:
        DataFrame with feature statistics: mean, std, median, min, max, skewness, is_heavy_tailed.
    """
    if isinstance(df_or_X, np.ndarray):
        num_features = df_or_X.shape[1] if df_or_X.ndim > 1 else 1
        cols = feature_names or [f"feat_{i}" for i in range(num_features)]
        df = pd.DataFrame(df_or_X, columns=cols)
    else:
        df = df_or_X.select_dtypes(include=[np.number])

    stats_rows = []
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        skew = float(vals.skew()) if len(vals) > 2 else 0.0
        stats_rows.append({
            "feature": col,
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "median": round(float(vals.median()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "skewness": round(skew, 4),
            "is_heavy_tailed": abs(skew) > 2.0,
        })

    return pd.DataFrame(stats_rows)


def analyze_dataset(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive, all-in-one exploratory data analysis for an intrusion detection dataset.

    Args:
        X: Feature matrix.
        y: Class labels.
        class_names: Optional class names.
        feature_names: Optional feature names.

    Returns:
        Combined dictionary with quality audit, target label distribution, and top skewed features.
    """
    quality_report = diagnose_data_quality(X, feature_names=feature_names)
    partition_report = analyze_partition(y, class_names=class_names)
    feature_stats = analyze_features(X, feature_names=feature_names)

    heavy_tailed = feature_stats[feature_stats["is_heavy_tailed"]]["feature"].tolist()

    return {
        "data_quality": quality_report,
        "target_distribution": partition_report,
        "num_heavy_tailed_features": len(heavy_tailed),
        "heavy_tailed_features": heavy_tailed[:10],
    }

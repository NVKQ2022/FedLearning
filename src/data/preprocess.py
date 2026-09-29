"""
Backward-compatibility proxy for src.data.preprocess -> src.data.preprocessor.
"""

from .preprocessor import (
    TabularFlowDataset,
    TabularDataPreprocessor,
    CICIoT2023Preprocessor,
    compute_balanced_class_weights,
    load_and_preprocess_ciciot2023,
    TARGET_COLUMN,
    HEAVY_TAIL_COLUMNS,
    NON_NEGATIVE_COLUMNS,
)

__all__ = [
    "TabularFlowDataset",
    "TabularDataPreprocessor",
    "CICIoT2023Preprocessor",
    "compute_balanced_class_weights",
    "load_and_preprocess_ciciot2023",
    "TARGET_COLUMN",
    "HEAVY_TAIL_COLUMNS",
    "NON_NEGATIVE_COLUMNS",
]

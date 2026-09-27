"""
Evaluation and metrics package for FL-IoT-IDS.
"""

from .evaluator import (
    evaluate_model,
    compute_comprehensive_metrics,
    evaluate_comprehensive,
)

__all__ = [
    "evaluate_model",
    "compute_comprehensive_metrics",
    "evaluate_comprehensive",
]

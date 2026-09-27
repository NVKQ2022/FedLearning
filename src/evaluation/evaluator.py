"""
Evaluation and benchmarking module for FL-IoT-IDS.

Provides publication-grade multi-metric evaluation:
- Overall Accuracy
- Macro-averaged Precision, Recall, and F1-Score (essential for severe class imbalance)
- Minority Recall explicitly tracking stealthy attacks (<1% prevalence: Web-based, Brute-force)
- Normalized Confusion Matrix

Adheres to:
- skills/evaluation-and-benchmarking/SKILL.md (Dimension 1: Predictive Classification Metrics)
- skills/methodology-audit/SKILL.md (Pillar 5: Metric Alignment & Scientific Integrity)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None

logger = logging.getLogger(__name__)


def compute_comprehensive_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    class_names: List[str],
    minority_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes rigorous multi-class evaluation metrics on ground truth vs predictions.

    Args:
        all_preds: 1D array of predicted class labels.
        all_targets: 1D array of ground truth class labels.
        class_names: List of class names ordered by integer index.
        minority_classes: Optional list of rare class names (e.g. ['Web-based', 'Brute-force']).

    Returns:
        Dictionary containing overall, macro, per-class, and minority metrics.
    """
    # 1. Overall accuracy
    acc = float(accuracy_score(all_targets, all_preds))

    # 2. Macro & Weighted averages (zero_division=0 prevents crashes on unpredicted classes)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    # 3. Per-class metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    # 4. Minority class recall isolation
    minority_recall_dict = {}
    if minority_classes:
        for cls_name in minority_classes:
            if cls_name in class_names:
                cls_idx = class_names.index(cls_name)
                minority_recall_dict[cls_name] = float(r_class[cls_idx])

    # 5. Normalized Confusion Matrix (by true rows)
    cm = confusion_matrix(all_targets, all_preds, normalize="true")

    return {
        "accuracy": acc,
        "macro_f1": float(f1_macro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "weighted_f1": float(f1_weighted),
        "per_class_f1": {name: float(f1) for name, f1 in zip(class_names, f1_class)},
        "per_class_recall": {name: float(r) for name, r in zip(class_names, r_class)},
        "per_class_precision": {name: float(p) for name, p in zip(class_names, p_class)},
        "support": {name: int(s) for name, s in zip(class_names, support)},
        "minority_recall": minority_recall_dict,
        "confusion_matrix": cm
    }


if HAS_TORCH:
    def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Executes inference pass over a DataLoader and computes loss & accuracy.

        Returns:
            Tuple of (average_loss, accuracy, all_predictions, all_targets).
        """
        device = torch.device(device)
        model.to(device)
        model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = len(y_batch)

                outputs = model(X_batch)

                if criterion is not None:
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * batch_size

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_samples += batch_size

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y_batch.cpu().numpy().tolist())

        avg_loss = (total_loss / total_samples) if criterion is not None else 0.0
        accuracy = correct / max(total_samples, 1)
        return avg_loss, accuracy, np.array(all_preds, dtype=np.int64), np.array(all_targets, dtype=np.int64)


    def evaluate_comprehensive(
        model: nn.Module,
        dataloader: DataLoader,
        class_names: List[str],
        criterion: Optional[nn.Module] = None,
        minority_classes: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, Any]:
        """
        End-to-end evaluation computing loss, accuracy, and full publication metrics.
        """
        avg_loss, accuracy, all_preds, all_targets = evaluate_model(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device
        )
        metrics = compute_comprehensive_metrics(
            all_preds=all_preds,
            all_targets=all_targets,
            class_names=class_names,
            minority_classes=minority_classes
        )
        metrics["loss"] = avg_loss
        return metrics

else:
    def evaluate_model(*args, **kwargs):
        raise ImportError("PyTorch is required for evaluate_model. Please install torch.")

    def evaluate_comprehensive(*args, **kwargs):
        raise ImportError("PyTorch is required for evaluate_comprehensive. Please install torch.")


if __name__ == "__main__":
    print("--- Running Evaluation Module Verification ---")
    mock_classes = ["Benign", "DDoS", "DoS", "Recon", "Spoof", "Mirai", "Web-based", "Brute-force"]
    y_true = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7])
    y_pred = np.array([0, 0, 1, 0, 2, 3, 4, 5, 6, 7, 0])  # One minority attack misclassified

    res = compute_comprehensive_metrics(
        y_pred, y_true, class_names=mock_classes, minority_classes=["Web-based", "Brute-force"]
    )
    print(f"Accuracy: {res['accuracy']*100:.2f}%")
    print(f"Macro-F1: {res['macro_f1']*100:.2f}%")
    print(f"Minority Recall: {res['minority_recall']}")
    assert "accuracy" in res and "macro_f1" in res
    print("✅ Evaluation module verification passed successfully!")

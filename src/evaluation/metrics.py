"""Classification evaluation metrics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(
    y_true: Any, y_pred: Any, zero_division: int = 0
) -> dict[str, float]:
    """Return summary metrics: accuracy, macro-F1, weighted-F1."""
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=zero_division
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def full_report(y_true: Any, y_pred: Any, zero_division: int = 0) -> dict[str, Any]:
    """Return the full per-class + aggregate ``classification_report`` dict."""
    return classification_report(
        y_true, y_pred, output_dict=True, zero_division=zero_division
    )


def confusion_matrix_list(
    y_true: Any, y_pred: Any, labels: Optional[Any] = None
) -> list[list[int]]:
    """Return the confusion matrix as a nested list."""
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Return accuracy as a float."""
    return float(accuracy_score(y_true, y_pred))

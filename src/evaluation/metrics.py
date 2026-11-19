"""Classification evaluation metrics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
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


def confusion_matrix_dataframe(y_true: Any, y_pred: Any) -> pd.DataFrame:
    """Return a labeled confusion matrix (rows = true, columns = predicted)."""
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def report_to_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten a ``classification_report`` dict into a tidy DataFrame.

    Rows are classes, averages, and ``accuracy``; columns are ``precision``,
    ``recall``, ``f1_score`` and ``support`` (plus ``accuracy`` for the scalar
    accuracy row).
    """
    records: list[dict[str, Any]] = []
    for label, values in report.items():
        if isinstance(values, dict):
            records.append(
                {
                    "class": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1_score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
        else:
            records.append({"class": label, "accuracy": values})
    return pd.DataFrame(records)

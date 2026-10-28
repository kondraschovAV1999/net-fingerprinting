"""Classification metrics comparing predicted labels against ground truth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_metrics(
    y_true: Sequence | np.ndarray,
    y_pred: Sequence | np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall and F1 from true vs. predicted labels.

    Args:
        y_true: Ground-truth labels. Pass ``y_true``/``y_pred`` by keyword --
            swapping them silently swaps precision and recall.
        y_pred: Predicted labels, same length as ``y_true``.
        class_names: Optional list where ``class_names[i]`` names integer class ``i``.
            If given, every class is reported even if it is absent from both arrays.
            If omitted, classes are the sorted unique labels seen in either array.

    Returns:
        A dict with:
            ``accuracy``;
            ``precision_macro``, ``recall_macro``, ``f1_macro`` (each class counts equally);
            ``precision_weighted``, ``recall_weighted``, ``f1_weighted`` (weighted by support);
            ``per_class``: DataFrame of precision, recall, f1, support indexed by class name;
            ``confusion_matrix``: DataFrame with rows = true class, columns = predicted class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(label) for label in labels]
    else:
        labels = np.arange(len(class_names))

    metrics: dict[str, Any] = {"accuracy": accuracy_score(y_true, y_pred)}

    for avg in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = precision
        metrics[f"recall_{avg}"] = recall
        metrics[f"f1_{avg}"] = f1

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    metrics["per_class"] = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=class_names,
    )
    metrics["confusion_matrix"] = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )
    return metrics

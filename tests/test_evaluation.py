import numpy as np

from evaluation.metrics import (
    accuracy,
    compute_metrics,
    confusion_matrix_dataframe,
    confusion_matrix_list,
    full_report,
    report_to_dataframe,
)


def test_perfect_metrics():
    y_true = np.array(["a", "b", "a", "b"])
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0


def test_zero_accuracy():
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["b", "b", "a", "a"])
    assert compute_metrics(y_true, y_pred)["accuracy"] == 0.0


def test_partial_accuracy():
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    assert accuracy(y_true, y_pred) == 0.75


def test_full_report_has_classes_and_aggregates():
    report = full_report(np.array(["a", "a", "b"]), np.array(["a", "b", "b"]))
    assert "a" in report and "b" in report
    assert "accuracy" in report
    assert "macro avg" in report and "weighted avg" in report


def test_confusion_matrix_shape():
    y_true = np.array(["a", "b", "a"])
    y_pred = np.array(["a", "b", "b"])
    cm = confusion_matrix_list(y_true, y_pred)
    assert cm == [[1, 1], [0, 1]]


def test_confusion_matrix_dataframe_labels():
    y_true = np.array(["a", "b", "a", "b", "a"])
    y_pred = np.array(["a", "b", "b", "b", "a"])
    df = confusion_matrix_dataframe(y_true, y_pred)
    assert list(df.index) == ["a", "b"]
    assert list(df.columns) == ["a", "b"]
    assert df.loc["a", "a"] == 2
    assert df.loc["a", "b"] == 1
    assert df.loc["b", "b"] == 2


def test_report_to_dataframe():
    report = full_report(np.array(["a", "a", "b"]), np.array(["a", "b", "b"]))
    df = report_to_dataframe(report)
    classes = set(df["class"])
    assert {"a", "b", "accuracy", "macro avg", "weighted avg"} <= classes
    a_row = df[df["class"] == "a"].iloc[0]
    assert a_row["precision"] == 1.0
    assert a_row["recall"] == 0.5
    acc_row = df[df["class"] == "accuracy"].iloc[0]
    assert acc_row["accuracy"] is not None

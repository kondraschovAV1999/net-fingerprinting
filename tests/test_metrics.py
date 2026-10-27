import pytest

from evaluation.metrics import evaluate_metrics

# Toy example with hand-computed values:
#   class A: TP=1 FP=1 FN=1 -> P=1/2 R=1/2 F1=1/2
#   class B: TP=2 FP=1 FN=0 -> P=2/3 R=1   F1=4/5
#   class C: TP=1 FP=0 FN=1 -> P=1   R=1/2 F1=2/3
Y_TRUE = [0, 0, 1, 1, 2, 2]
Y_PRED = [0, 1, 1, 1, 2, 0]
CLASS_NAMES = ["A", "B", "C"]


@pytest.fixture
def toy_metrics():
    return evaluate_metrics(y_true=Y_TRUE, y_pred=Y_PRED, class_names=CLASS_NAMES)


def test_scalar_metrics_match_hand_computed_values(toy_metrics):
    assert toy_metrics["accuracy"] == pytest.approx(4 / 6)
    assert toy_metrics["precision_macro"] == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert toy_metrics["recall_macro"] == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert toy_metrics["f1_macro"] == pytest.approx((1 / 2 + 4 / 5 + 2 / 3) / 3)


def test_weighted_equals_macro_when_supports_are_equal(toy_metrics):
    for name in ("precision", "recall", "f1"):
        assert toy_metrics[f"{name}_weighted"] == pytest.approx(toy_metrics[f"{name}_macro"])


def test_per_class_table(toy_metrics):
    per_class = toy_metrics["per_class"]
    assert list(per_class.index) == CLASS_NAMES
    assert per_class["precision"].tolist() == pytest.approx([1 / 2, 2 / 3, 1])
    assert per_class["recall"].tolist() == pytest.approx([1 / 2, 1, 1 / 2])
    assert per_class["f1"].tolist() == pytest.approx([1 / 2, 4 / 5, 2 / 3])
    assert per_class["support"].tolist() == [2, 2, 2]


def test_confusion_matrix_rows_are_true_columns_are_predicted(toy_metrics):
    cm = toy_metrics["confusion_matrix"]
    assert list(cm.index) == ["true_A", "true_B", "true_C"]
    assert list(cm.columns) == ["pred_A", "pred_B", "pred_C"]
    assert cm.to_numpy().tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_class_absent_from_both_arrays_is_still_reported():
    metrics = evaluate_metrics(y_true=[0, 0, 1], y_pred=[0, 1, 1], class_names=["A", "B", "C"])
    assert metrics["per_class"].loc["C"].tolist() == [0, 0, 0, 0]
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_string_labels_without_class_names():
    metrics = evaluate_metrics(y_true=["cat", "dog", "dog"], y_pred=["cat", "cat", "dog"])
    assert list(metrics["per_class"].index) == ["cat", "dog"]
    assert metrics["accuracy"] == pytest.approx(2 / 3)

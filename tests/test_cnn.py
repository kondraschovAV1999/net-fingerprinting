import numpy as np
import pytest
import torch

from evaluation.metrics import evaluate_metrics
from models.cnn import FlowCNN, class_weights, predict, train_cnn

CLASS_NAMES = ["A", "B", "C"]


def _separable_flows(n_per_class=30, seq_len=32, seed=0):
    """Three easily separable classes: mostly outbound, mostly inbound, and alternating packets."""
    rng = np.random.default_rng(seed)
    patterns = [np.ones(seq_len), -np.ones(seq_len), np.resize([1.0, -1.0], seq_len)]
    X, y = [], []
    for label, pattern in enumerate(patterns):
        sizes = pattern * rng.uniform(0.3, 1.0, size=(n_per_class, seq_len))
        timing = rng.uniform(0, 3, size=(n_per_class, seq_len))
        X.append(np.stack([sizes, timing], axis=1))
        y += [label] * n_per_class
    return np.concatenate(X).astype(np.float32), np.array(y)


def test_forward_shape_for_any_sequence_length():
    model = FlowCNN(in_channels=2, n_classes=3).eval()
    with torch.no_grad():
        assert model(torch.zeros(4, 2, 32)).shape == (4, 3)
        assert model(torch.zeros(5, 2, 20)).shape == (5, 3)


def test_parameter_count_matches_the_paper():
    model = FlowCNN(in_channels=2, n_classes=3)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 36_195  # "about 36K"


def test_class_weights_are_inverse_frequency():
    assert class_weights(np.array([0] * 6 + [1] * 2 + [2] * 2), 3) == pytest.approx([10 / 18, 10 / 6, 10 / 6])


def test_predict_returns_one_class_index_per_flow_regardless_of_batch_size():
    X, _ = _separable_flows(n_per_class=10)
    torch.manual_seed(0)
    model = FlowCNN()
    full = predict(model, X)
    assert full.shape == (30,) and set(full) <= {0, 1, 2}
    assert np.array_equal(predict(model, X, batch_size=7), full)
    assert predict(model, X[:0]).shape == (0,)


@pytest.fixture(scope="module")
def trained():
    X_train, y_train = _separable_flows(seed=1)
    X_val, y_val = _separable_flows(n_per_class=10, seed=2)
    model, history = train_cnn(X_train, y_train, X_val, y_val, CLASS_NAMES, epochs=8, verbose=False)
    return model, history, X_val, y_val


def test_training_learns_separable_classes(trained):
    model, history, X_val, y_val = trained
    assert len(history) == 8
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert max(h["val_f1_macro"] for h in history) >= 0.9


def test_best_epoch_weights_are_restored(trained):
    model, history, X_val, y_val = trained
    val_f1 = evaluate_metrics(y_true=y_val, y_pred=predict(model, X_val), class_names=CLASS_NAMES)["f1_macro"]
    assert val_f1 == pytest.approx(max(h["val_f1_macro"] for h in history))


def test_same_seed_gives_same_model():
    X, y = _separable_flows(n_per_class=10, seed=3)
    run = lambda: train_cnn(X, y, X, y, CLASS_NAMES, epochs=2, verbose=False)[1]
    assert run() == run()


def test_empty_validation_set_raises():
    X, y = _separable_flows(n_per_class=5)
    with pytest.raises(ValueError, match="validation set is empty"):
        train_cnn(X, y, X[:0], y[:0], CLASS_NAMES, epochs=1, verbose=False)

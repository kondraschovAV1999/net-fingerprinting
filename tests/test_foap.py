import numpy as np
import pandas as pd
import pytest

from models.foap import (FOAP, N_FEATURES, flow_feature_matrix, foap_features, segment,
                         segment_similarity, window_mean)

# Toy flow: out 100B @0.0s, in 1500B @0.5s, in 1500B @1.5s, out 60B @2.0s
T = [0.0, 0.5, 1.5, 2.0]
SIZE = [100, 1500, 1500, 60]
INBOUND = [False, True, True, False]


@pytest.fixture
def feats():
    return foap_features(T, SIZE, INBOUND)


def test_feature_count_and_finite(feats):
    assert feats.shape == (N_FEATURES,)
    assert np.isfinite(feats).all()


def test_general_features(feats):
    # n_in, n_out, n, in%, out%, in bytes, out bytes, duration
    assert feats[:8].tolist() == pytest.approx([2, 2, 4, 0.5, 0.5, 3000, 160, 2.0])


def test_interactive_pattern(feats):
    # running direction sum: 0, -1, 0, 1, 0; first bin centre x=0.1 -> -0.1, last x=3.9 -> 0.1
    interactive = feats[8:28]
    assert interactive[0] == pytest.approx(-0.1)
    assert interactive[-1] == pytest.approx(0.1)


def test_packet_rate(feats):
    # 1s windows [0,1), [1,2), [2,3) hold 2, 1, 1 packets
    assert feats[28:33].tolist() == pytest.approx([4 / 3, 2, 1, 1, np.std([2, 1, 1])])


def test_single_packet_flow_is_finite():
    feats = foap_features([3.0], [60], [True])
    assert feats.shape == (N_FEATURES,)
    assert np.isfinite(feats).all()


def test_flow_feature_matrix_one_row_per_flow():
    pkt_df = pd.DataFrame({
        "flow_id": ["a"] * 4 + ["b"] * 2,
        "time": T + [5.0, 5.1],
        "len": SIZE + [80, 90],
        "direction": [1, -1, -1, 1, 1, -1],  # +1 = sent by device
    })
    matrix = flow_feature_matrix(pkt_df)
    assert matrix.shape == (2, N_FEATURES)
    assert np.allclose(matrix.loc["a"].to_numpy(), foap_features(T, SIZE, INBOUND))


def test_window_mean_stays_within_trace_and_window():
    q = np.array([1.0, 0.0, 1.0, 0.0])
    t = np.array([0.0, 5.0, 30.0, 0.0])
    trace = np.array(["a", "a", "a", "b"])
    # flow 0 sees flows 0,1; flow 1 sees 0,1; flow 2 only itself; flow 3 is alone in trace b
    assert window_mean(q, t, trace, half_width=10).tolist() == pytest.approx([0.5, 0.5, 1.0, 0.0])


def test_segment_splits_two_clear_blocks():
    q = np.r_[np.full(15, 0.1), np.full(15, 0.9)]
    t = np.arange(30) * 2.0  # each block spans 28s >= tau_min
    assert segment(q, t, lam=1.0).tolist() == [0] * 15 + [1] * 15


def test_segment_keeps_constant_trace_whole():
    assert segment(np.full(30, 0.5), np.arange(30) * 2.0, lam=1.0).tolist() == [0] * 30


def test_segment_respects_minimum_duration():
    q = np.r_[np.full(15, 0.1), np.full(15, 0.9)]
    t = np.arange(30) * 0.5  # whole trace spans 14.5s < tau_min=20: cannot split
    assert segment(q, t, lam=1.0).tolist() == [0] * 30


def test_segment_similarity_matches_hand_computation():
    # segment 0: patterns 0 (max q 0.9) and 1 (max q 0.7) -> (0.4 + 0.2) / (2 * 0.5) = 0.6
    # segment 1: pattern 0 (max q 0.2) -> 0 / (1 * 0.5) = 0
    q = np.array([0.9, 0.6, 0.7, 0.2, 0.0])
    pattern = np.array([0, 0, 1, 0, 0])
    seg = np.array([0, 0, 0, 1, 1])
    assert segment_similarity(q, pattern, seg).tolist() == pytest.approx([0.6, 0.6, 0.6, 0.0, 0.0])


def _synthetic_flows(n_per_app=60, seed=0):
    """Three apps with separable features; each app's flows form one trace over time."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n_per_app)
    F = rng.normal(size=(len(y), N_FEATURES)) + y[:, None] * 3.0
    t = np.tile(np.arange(n_per_app) * 3.0, 3)
    return F, y, t, y.astype(str)


def test_foap_fit_predict_on_separable_apps():
    F, y, t, trace = _synthetic_flows()
    model = FOAP().fit(F, y, t, trace)
    for filtering in (False, True):
        scores = model.predict_scores(F, t, trace, filtering=filtering)
        assert scores.shape == (len(y), 3)
        assert ((scores >= 0) & (scores <= 1)).all()
        assert (model.predict(F, t, trace, filtering=filtering) == y).mean() > 0.9


def test_foap_returns_unknown_when_no_model_accepts():
    F, y, t, trace = _synthetic_flows()
    model = FOAP().fit(F, y, t, trace)
    labels = model.predict(F, t, trace, threshold=1.0)  # no probability exceeds 1.0
    assert (labels == -1).all()

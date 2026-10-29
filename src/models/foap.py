"""FOAP open-world app fingerprinting (Li et al., USENIX Security 2022), rebuilt from the paper.

Stage 1: the 123 per-flow features (Section 4).
Stage 2: per-app local similarity (random forest) + bilevel recognition (Section 5.3).
Stage 3: traffic segmentation and filtering by structural similarity (Sections 5.1-5.2).

Deviations from the paper, because our dataset has only PCAPs:
- No background-flow classifier (the paper's r_i needs on-device logs), so the high-level
  model uses (q_i, mean q of neighbouring flows) instead of (q_i, mean q, q_i - r_i).
- Flow patterns are clustered in a standardised 10-component PCA space instead of the paper's
  learned metric space. The paper's delta was tuned in that metric space, so here the
  clustering cut is delta times the median pairwise distance between the app's flows.
  An app with no distinct sub-behaviours can still split into many patterns; check
  ``n_patterns`` in each fitted model.
- Segmentation (Eq. 1) is solved exactly by dynamic programming instead of the paper's
  greedy divisive-agglomerative tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, median_abs_deviation, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PERCENTILES = np.arange(10, 100, 10)  # 10%, 20%, ..., 90%
N_FEATURES = 123
MAX_CLUSTER_FLOWS = 5000  # agglomerative clustering is O(n^2) in memory


def _temporal(t: np.ndarray, t0: float) -> np.ndarray:
    """13 features: inter-arrival mean/max/min/std + 9 percentiles of time since the flow's first packet."""
    if len(t) == 0:
        return np.zeros(13)
    iat = np.diff(t) if len(t) > 1 else np.zeros(1)
    return np.r_[iat.mean(), iat.max(), iat.min(), iat.std(), np.percentile(t - t0, PERCENTILES)]


def _size(s: np.ndarray) -> np.ndarray:
    """17 features: mean/max/min/MAD/std/var/skew/kurtosis + 9 percentiles of packet size."""
    if len(s) == 0:
        return np.zeros(17)
    varies = np.ptp(s) > 0  # skew/kurtosis are undefined for constant sizes; use 0
    return np.r_[s.mean(), s.max(), s.min(), median_abs_deviation(s), s.std(), s.var(),
                 skew(s) if varies else 0.0, kurtosis(s) if varies else 0.0, np.percentile(s, PERCENTILES)]


def foap_features(t, size, inbound) -> np.ndarray:
    """Compute FOAP's 123 features for one flow.

    Args:
        t: packet arrival times in seconds, sorted ascending.
        size: packet sizes in bytes.
        inbound: True for packets received by the device, False for packets it sent.

    Returns:
        Array of 123 features in the paper's order: general (8), interactive pattern (20),
        packet rate (5), temporal (39: bidirectional, inbound, outbound) and
        packet size (51: inbound, outbound, bidirectional).
    """
    t = np.asarray(t, dtype=float)
    size = np.asarray(size, dtype=float)
    inbound = np.asarray(inbound, dtype=bool)
    n, t0 = len(t), t[0]
    duration = t[-1] - t0
    n_in = int(inbound.sum())

    general = [n_in, n - n_in, n, n_in / n, (n - n_in) / n,
               size[inbound].sum(), size[~inbound].sum(), duration]

    # f(x) linearly interpolates the running sum of directions (+1 inbound, -1 outbound),
    # sampled at the centres of 20 equal bins over (0, n)
    direction = np.where(inbound, 1, -1)
    interactive = np.interp((np.arange(20) + 0.5) * n / 20, np.arange(n + 1), np.r_[0, np.cumsum(direction)])

    counts = np.histogram(t - t0, bins=np.arange(0, np.floor(duration) + 2))[0]  # 1-second windows
    rate = [counts.mean(), counts.max(), counts.min(), np.median(counts), counts.std()]

    temporal = np.r_[_temporal(t, t0), _temporal(t[inbound], t0), _temporal(t[~inbound], t0)]
    sizes = np.r_[_size(size[inbound]), _size(size[~inbound]), _size(size)]

    return np.r_[general, interactive, rate, temporal, sizes]


def flow_feature_matrix(pkt_df: pd.DataFrame) -> pd.DataFrame:
    """FOAP features for every flow in a packet table.

    ``pkt_df`` needs the columns built in the notebook's flow cell: ``flow_id``, ``time`` (s),
    ``len`` (bytes) and ``direction`` (+1 sent by the device, -1 received).

    Returns a DataFrame indexed by ``flow_id`` with one column per feature.
    """
    pkt_df = pkt_df.sort_values(["flow_id", "time"])
    features = {
        flow_id: foap_features(g["time"].to_numpy(), g["len"].to_numpy(), g["direction"].to_numpy() < 0)
        for flow_id, g in pkt_df.groupby("flow_id", sort=False)
    }
    return pd.DataFrame.from_dict(features, orient="index", columns=[f"f{i}" for i in range(N_FEATURES)])


def _trace_groups(t: np.ndarray, trace: np.ndarray) -> list[np.ndarray]:
    """Indices of each trace's flows, sorted by start time."""
    trace = pd.factorize(np.asarray(trace))[0]
    order = np.lexsort((t, trace))
    bounds = np.flatnonzero(np.diff(trace[order])) + 1
    return np.split(order, bounds)


def window_mean(q: np.ndarray, t: np.ndarray, trace: np.ndarray, half_width: float) -> np.ndarray:
    """Mean of q over flows in the same trace starting within +-half_width seconds (the flow included)."""
    out = np.empty(len(q))
    for idx in _trace_groups(t, trace):
        ts, cum = t[idx], np.r_[0, np.cumsum(q[idx])]
        lo = np.searchsorted(ts, ts - half_width, side="left")
        hi = np.searchsorted(ts, ts + half_width, side="right")
        out[idx] = (cum[hi] - cum[lo]) / (hi - lo)
    return out


def segment(q: np.ndarray, t: np.ndarray, lam: float, n_min: int = 10, tau_min: float = 20.0) -> np.ndarray:
    """Segment id (0, 1, ... in time order) for each flow of one trace, by the paper's Eq. 1.

    Minimises sum over segments of |Q_c| * Var(Q_c), plus lam per segment, where every segment
    holds at least n_min flows spanning at least tau_min seconds. q and t must be sorted by t.
    """
    n = len(q)
    if n < n_min:
        return np.zeros(n, dtype=int)
    c1, c2 = np.r_[0, np.cumsum(q)], np.r_[0, np.cumsum(np.square(q))]
    best = np.r_[0.0, np.full(n, np.inf)]  # best[j] = min cost of segmenting q[:j]
    prev = np.zeros(n + 1, dtype=int)
    for j in range(n_min, n + 1):
        i = np.arange(j - n_min + 1)
        i = i[t[j - 1] - t[i] >= tau_min]
        if len(i) == 0:
            continue
        cost = best[i] + (c2[j] - c2[i]) - (c1[j] - c1[i]) ** 2 / (j - i) + lam  # |Q| * Var(Q) of q[i:j]
        k = np.argmin(cost)
        best[j], prev[j] = cost[k], i[k]
    if not np.isfinite(best[n]):  # trace too short to satisfy the constraints: one segment
        return np.zeros(n, dtype=int)
    starts, j = [], n
    while j > 0:
        j = prev[j]
        starts.append(j)
    return np.searchsorted(starts[::-1], np.arange(n), side="right") - 1
    # ponytail: O(n^2) per trace; fine for thousands of flows, switch to PELT if traces get much longer


def segment_similarity(q: np.ndarray, pattern: np.ndarray, seg: np.ndarray, q_min: float = 0.5) -> np.ndarray:
    """Structural similarity (the paper's Eq. 2) of each flow's segment.

    Closed form of the integral: for every app pattern present in a segment take the highest q
    among the segment's flows of that pattern (m_p); S = sum(max(m_p - q_min, 0)) / (#{m_p > 0} * (1 - q_min)).
    """
    m = pd.DataFrame({"seg": seg, "pattern": pattern, "q": q}).groupby(["seg", "pattern"])["q"].max()
    num = (m - q_min).clip(lower=0).groupby(level="seg").sum()
    n_matched = (m > 0).groupby(level="seg").sum()
    s = (num / (n_matched * (1 - q_min))).where(n_matched > 0, 0.0)
    return s.reindex(seg).to_numpy()


class FOAP:
    """Open-world app traffic recognition from FOAP, one one-vs-rest model per app.

    Defaults are the paper's tuned values (Table 10), except ``lam``, which the paper does not
    report (placeholder). ``window`` is the paper's T: neighbours lie within +-window/2 seconds.
    """

    def __init__(self, n_min=10, tau_min=20.0, lam=1.0, delta=0.5, s_min=0.1, s_max=0.25,
                 window=20.0, q_min=0.5, random_state=42):
        self.n_min, self.tau_min, self.lam, self.delta = n_min, tau_min, lam, delta
        self.s_min, self.s_max, self.window, self.q_min = s_min, s_max, window, q_min
        self.random_state = random_state

    def _segment_ids(self, q, t, trace):
        seg, offset = np.empty(len(q), dtype=int), 0
        for idx in _trace_groups(t, trace):
            z = segment(q[idx], t[idx], self.lam, self.n_min, self.tau_min)
            seg[idx] = z + offset
            offset += z.max() + 1
        return seg

    def _similarity(self, model, F, q, t, trace):
        pattern = model["knn"].predict(model["embed"].transform(F))
        return segment_similarity(q, pattern, self._segment_ids(q, t, trace), self.q_min)

    def fit(self, F, y, t, trace):
        """F: flow features, y: app label, t: flow start time (s), trace: trace id the flow belongs to."""
        F, y, t, trace = np.asarray(F, float), np.asarray(y), np.asarray(t, float), np.asarray(trace)
        rng = np.random.default_rng(self.random_state)
        self.apps_ = np.unique(y)
        self.models_ = {}
        for app in self.apps_:
            is_app = y == app
            rf = RandomForestClassifier(oob_score=True, class_weight="balanced",
                                        random_state=self.random_state, n_jobs=-1).fit(F, is_app)
            # Out-of-bag q decouples the high-level model from the forest (the paper uses bootstrap resampling)
            q = np.nan_to_num(rf.oob_decision_function_[:, 1])
            q_bar = window_mean(q, t, trace, self.window / 2)
            lr = LogisticRegression().fit(np.c_[q, q_bar], is_app)

            # Flow patterns of this app: hierarchical clustering + kNN to assign new flows
            F_app = F[is_app]
            if len(F_app) > MAX_CLUSTER_FLOWS:
                F_app = F_app[rng.choice(len(F_app), MAX_CLUSTER_FLOWS, replace=False)]
            embed = make_pipeline(StandardScaler(), PCA(n_components=min(10, len(F_app)))).fit(F_app)
            Z = embed.transform(F_app)
            cut = self.delta * np.median(pdist(Z)) if len(Z) > 1 else 0.0
            pattern = (AgglomerativeClustering(n_clusters=None, distance_threshold=cut,
                                               linkage="average").fit_predict(Z)
                       if cut > 0 else np.zeros(len(Z), dtype=int))
            knn = KNeighborsClassifier(n_neighbors=min(5, len(Z))).fit(Z, pattern)

            model = {"rf": rf, "lr": lr, "embed": embed, "knn": knn, "n_patterns": int(pattern.max()) + 1}
            # App-specific threshold: keep 95% of the app's training flows, bounded by [s_min, s_max]
            s = self._similarity(model, F, q, t, trace)
            model["threshold"] = float(np.clip(np.percentile(s[is_app], 5), self.s_min, self.s_max))
            self.models_[app] = model
        return self

    def predict_scores(self, F, t, trace, filtering=True):
        """Probability each flow belongs to each app (columns follow ``apps_``).

        filtering=False skips segmentation and filtering (the paper's FOAP-F ablation).
        """
        F, t, trace = np.asarray(F, float), np.asarray(t, float), np.asarray(trace)
        scores = np.zeros((len(F), len(self.apps_)))
        for k, app in enumerate(self.apps_):
            model = self.models_[app]
            q = model["rf"].predict_proba(F)[:, 1]
            score = model["lr"].predict_proba(np.c_[q, window_mean(q, t, trace, self.window / 2)])[:, 1]
            if filtering:
                score[self._similarity(model, F, q, t, trace) < model["threshold"]] = 0.0
            scores[:, k] = score
        return scores

    def predict(self, F, t, trace, filtering=True, threshold=0.5):
        """Most likely app per flow, or -1 ("unknown") if no app's model scores it above threshold."""
        scores = self.predict_scores(F, t, trace, filtering)
        return np.where(scores.max(axis=1) > threshold, self.apps_[scores.argmax(axis=1)], -1)

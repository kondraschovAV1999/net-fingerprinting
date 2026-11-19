"""FOAP-style flow feature extraction for network fingerprinting.

Ports the 123-dimensional flow feature set from FOAP (USENIX Security 2022,
github.com/jflixjtu/FOAP) into a configurable extractor.

The extractor consumes a packet DataFrame with a precomputed flow identifier
and direction (see :func:`prepare_packet_dataframe` to derive both from the raw
columns produced by ``data_loading.parsing.parse_pcap``):

    flow_id    : str, canonical bidirectional flow key (5-tuple)
    time       : float, packet arrival time in seconds
    len        : int/float, packet size in bytes
    direction  : int, +1 outbound, -1 inbound

Feature groups (``FeatureGroups`` toggles each):

    general      : 8  — packet counts, byte counts, ratios, duration
    interactive  : 20 — interaction pattern sampled into bins
    packet_rate  : 5  — mean/max/min/median/std packets per second window
    temporal     : 39 — 3 directions x (IAT stats + 9 percentiles)
    packet_size  : 51 — 3 directions x (size stats + 9 percentiles)
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, median_abs_deviation, skew

PERCENTILES = np.arange(10, 100, 10)

#: Metadata columns carried through to ``FlowFeatureExtractor.flow_meta`` so
#: the splitter can assign flows to pcaps/applications after feature extraction.
META_COLUMNS = ("file_source", "application", "capture_timestamp", "dataset_name")


@dataclass(frozen=True)
class FeatureGroups:
    general: bool = True
    interactive: bool = True
    packet_rate: bool = True
    temporal: bool = True
    packet_size: bool = True


#: Named feature-group presets.
FEATURE_CONFIGS = {
    "foap": FeatureGroups(),
    "basic": FeatureGroups(
        interactive=False, packet_rate=False, temporal=False, packet_size=False
    ),
    "statistical": FeatureGroups(interactive=False),
}


def make_extractor(
    feature_config: str = "statistical",
    rate_interval: float = 1.0,
    interactive_bins: int = 20,
) -> "FlowFeatureExtractor":
    """Build a :class:`FlowFeatureExtractor` from a named preset."""
    if feature_config not in FEATURE_CONFIGS:
        raise ValueError(
            f"Unknown feature_config {feature_config!r}; "
            f"expected one of {sorted(FEATURE_CONFIGS)}"
        )
    return FlowFeatureExtractor(
        groups=FEATURE_CONFIGS[feature_config],
        rate_interval=rate_interval,
        interactive_bins=interactive_bins,
    )


def _is_private(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def detect_client_ip(pkt_df: pd.DataFrame) -> str:
    """Return the capture host's IP (prefers a private/local address)."""
    ips = pd.concat([pkt_df["src_ip"].astype(str), pkt_df["dst_ip"].astype(str)])
    private = ips[ips.map(_is_private)]
    if not private.empty:
        return str(private.mode().iloc[0])
    return str(ips.mode().iloc[0])


def derive_direction(
    pkt_df: pd.DataFrame, client_ip: Optional[str] = None
) -> np.ndarray:
    """Return +1 for outbound (src == client) and -1 for inbound packets."""
    src = pkt_df["src_ip"].astype(str)
    if client_ip is None:
        client_ip = detect_client_ip(pkt_df)
    return np.where(src == client_ip, 1, -1).astype(int)


def derive_flow_id(pkt_df: pd.DataFrame) -> pd.Series:
    """Return a canonical bidirectional flow key (order-invariant 5-tuple)."""
    src = pkt_df["src_ip"].astype(str)
    dst = pkt_df["dst_ip"].astype(str)
    sp = pkt_df["src_port"].fillna(0).astype(int).astype(str)
    dp = pkt_df["dst_port"].fillna(0).astype(int).astype(str)
    proto = pkt_df["protocol"].astype(str)

    swap = (src + ":" + sp) > (dst + ":" + dp)
    a = np.where(swap, dst, src)
    ap = np.where(swap, dp, sp)
    b = np.where(swap, src, dst)
    bp = np.where(swap, sp, dp)
    return a + ":" + ap + "<->" + b + ":" + bp + ":" + proto


def prepare_packet_dataframe(
    pkt_df: pd.DataFrame, client_ip: Optional[str] = None
) -> pd.DataFrame:
    """Derive ``flow_id`` and ``direction`` from raw packet columns.

    Drops rows without both ``src_ip`` and ``dst_ip`` (no 5-tuple to group on).
    """
    required = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol"}
    missing = required - set(pkt_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = pkt_df.copy()
    df = df[df["src_ip"].notna() & df["dst_ip"].notna()].copy()
    df["direction"] = derive_direction(df, client_ip=client_ip)
    df["flow_id"] = derive_flow_id(df)
    return df


class FlowFeatureExtractor:
    """Configurable network flow feature extractor."""

    def __init__(
        self,
        groups: FeatureGroups = FeatureGroups(),
        rate_interval: float = 1.0,
        interactive_bins: int = 20,
    ):
        self.groups = groups
        self.rate_interval = rate_interval
        self.interactive_bins = interactive_bins
        self.features_: Optional[pd.DataFrame] = None
        self.flow_meta: Optional[pd.DataFrame] = None

    @staticmethod
    def _validate(t, size, inbound):
        t = np.asarray(t, dtype=float)
        size = np.asarray(size, dtype=float)
        inbound = np.asarray(inbound, dtype=bool)

        if not (len(t) == len(size) == len(inbound)):
            raise ValueError("Packet arrays must have equal lengths.")

        if len(t) == 0:
            raise ValueError("Cannot extract features from an empty flow.")

        if not np.all(np.isfinite(t)):
            raise ValueError("Timestamps must be finite.")

        if not np.all(np.isfinite(size)):
            raise ValueError("Packet sizes must be finite.")

        if np.any(size < 0):
            raise ValueError("Packet sizes cannot be negative.")

        if np.any(np.diff(t) < 0):
            raise ValueError("Timestamps must be sorted.")

        return t, size, inbound

    @staticmethod
    def _general(t, size, inbound):
        n = len(t)
        n_in = int(inbound.sum())
        n_out = n - n_in

        return {
            "inbound_packets": n_in,
            "outbound_packets": n_out,
            "total_packets": n,
            "inbound_ratio": n_in / n,
            "outbound_ratio": n_out / n,
            "inbound_bytes": size[inbound].sum(),
            "outbound_bytes": size[~inbound].sum(),
            "duration": t[-1] - t[0],
        }

    def _interactive(self, inbound):
        direction = np.where(inbound, 1, -1)
        n = len(direction)

        values = np.interp(
            (np.arange(self.interactive_bins) + 0.5) * n / self.interactive_bins,
            np.arange(n + 1),
            np.r_[0, np.cumsum(direction)],
        )

        return {f"interactive_{i}": value for i, value in enumerate(values)}

    def _packet_rate(self, t):
        duration = t[-1] - t[0]

        # Preserve FOAP's one-second interval definition.
        counts = np.histogram(
            t - t[0],
            bins=np.arange(0, np.floor(duration) + 2),
        )[0]

        return {
            "rate_mean": counts.mean(),
            "rate_max": counts.max(),
            "rate_min": counts.min(),
            "rate_median": np.median(counts),
            "rate_std": counts.std(),
        }

    @staticmethod
    def _temporal(t, t0, prefix):
        names = [
            "iat_mean",
            "iat_max",
            "iat_min",
            "iat_std",
        ] + [f"relative_time_p{p}" for p in PERCENTILES]

        t = np.asarray(t, dtype=float)
        if len(t) == 0:
            return {f"{prefix}_{name}": 0.0 for name in names}

        iat = np.diff(t) if len(t) > 1 else np.zeros(1)
        values = np.r_[
            iat.mean(),
            iat.max(),
            iat.min(),
            iat.std(),
            np.percentile(t - t0, PERCENTILES),
        ]

        return {f"{prefix}_{name}": value for name, value in zip(names, values)}

    @staticmethod
    def _size(size, prefix):
        names = [
            "mean",
            "max",
            "min",
            "mad",
            "std",
            "var",
            "skew",
            "kurtosis",
        ] + [f"p{p}" for p in PERCENTILES]

        size = np.asarray(size, dtype=float)
        if len(size) == 0:
            return {f"{prefix}_{name}": 0.0 for name in names}

        varies = np.ptp(size) > 0
        values = np.r_[
            size.mean(),
            size.max(),
            size.min(),
            median_abs_deviation(size),
            size.std(),
            size.var(),
            skew(size) if varies else 0.0,
            kurtosis(size) if varies else 0.0,
            np.percentile(size, PERCENTILES),
        ]

        return {f"{prefix}_{name}": value for name, value in zip(names, values)}

    def extract_flow(self, t, size, inbound):
        t, size, inbound = self._validate(t, size, inbound)

        features = {}

        if self.groups.general:
            features.update(self._general(t, size, inbound))

        if self.groups.interactive:
            features.update(self._interactive(inbound))

        if self.groups.packet_rate:
            features.update(self._packet_rate(t))

        if self.groups.temporal:
            features.update(self._temporal(t, t[0], "bidir"))
            features.update(self._temporal(t[inbound], t[0], "inbound"))
            features.update(self._temporal(t[~inbound], t[0], "outbound"))

        if self.groups.packet_size:
            features.update(self._size(size[inbound], "inbound_size"))
            features.update(self._size(size[~inbound], "outbound_size"))
            features.update(self._size(size, "bidir_size"))

        return features

    def transform(self, pkt_df: pd.DataFrame) -> pd.DataFrame:
        """Extract one feature row per flow.

        Returns a DataFrame of features indexed by ``flow_id``. Also stores
        ``self.flow_meta`` (one row per flow) from any of
        :data:`META_COLUMNS` present in ``pkt_df``.
        """
        required = {"flow_id", "time", "len", "direction"}
        missing = required - set(pkt_df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if pkt_df.empty:
            raise ValueError("Packet DataFrame is empty.")

        if not pkt_df["direction"].isin([-1, 1]).all():
            raise ValueError("Direction must contain only -1 and +1.")

        pkt_df = pkt_df.sort_values(
            ["flow_id", "time"],
            kind="stable",
        )

        meta_cols = [c for c in META_COLUMNS if c in pkt_df.columns]

        features = {}
        meta = {}

        for flow_id, group in pkt_df.groupby("flow_id", sort=False):
            features[flow_id] = self.extract_flow(
                t=group["time"].to_numpy(),
                size=group["len"].to_numpy(),
                inbound=group["direction"].to_numpy() < 0,
            )
            if meta_cols:
                meta[flow_id] = {c: group[c].iloc[0] for c in meta_cols}

        self.features_ = pd.DataFrame.from_dict(
            features,
            orient="index",
        ).rename_axis("flow_id")

        self.flow_meta = (
            pd.DataFrame.from_dict(meta, orient="index").rename_axis("flow_id")
            if meta_cols
            else None
        )

        return self.features_

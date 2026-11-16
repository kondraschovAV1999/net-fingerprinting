"""Chronological, per-pcap, per-application train/validation/test splitting.

The split is performed at the PCAP level: all flows from a single ``file_source``
share one split. Within each application (and optionally dataset), PCAPs are
ordered by ``capture_timestamp`` and the oldest ``ratios[0]`` go to train, the
next ``ratios[1]`` to validation, and the newest ``ratios[2]`` to test. The
per-application sets are then combined so every split contains every app.

``sklearn.model_selection.train_test_split`` with ``shuffle=False`` is used to
carve the chronological boundaries (it preserves input order).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _split_pcaps(
    pcaps: np.ndarray, ratios: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split of an already time-sorted array of pcap ids."""
    val_test = float(ratios[1] + ratios[2])
    train, rest = train_test_split(pcaps, test_size=val_test, shuffle=False)
    val, test = train_test_split(
        rest, test_size=float(ratios[2] / val_test), shuffle=False
    )
    return np.asarray(train), np.asarray(val), np.asarray(test)


def split_by_pcap(
    flow_meta: pd.DataFrame,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    dataset_scope: str = "combined",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx, test_idx)`` positional indices into ``flow_meta``.

    ``flow_meta`` must have columns ``file_source``, ``application`` and
    ``capture_timestamp``. With ``dataset_scope="per_dataset"``, a
    ``dataset_name`` column is also required and splitting is performed
    independently within each ``(dataset_name, application)`` group.
    """
    required = {"file_source", "application", "capture_timestamp"}
    missing = required - set(flow_meta.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if dataset_scope not in {"combined", "per_dataset"}:
        raise ValueError(f"Unknown dataset_scope {dataset_scope!r}")

    if dataset_scope == "per_dataset" and "dataset_name" not in flow_meta.columns:
        raise ValueError("dataset_scope='per_dataset' requires a 'dataset_name' column")

    ratios_arr = np.asarray(ratios, dtype=float)
    if (
        ratios_arr.shape != (3,)
        or np.any(ratios_arr < 0)
        or not np.isclose(ratios_arr.sum(), 1.0)
    ):
        raise ValueError("ratios must be 3 non-negative values summing to 1")

    group_cols = ["application"]
    pcap_cols = ["file_source", "application", "capture_timestamp"]
    if dataset_scope == "per_dataset":
        group_cols = ["dataset_name", "application"]
        pcap_cols = ["dataset_name"] + pcap_cols

    pcap_df = flow_meta[pcap_cols].drop_duplicates(subset="file_source")

    split_of: dict[str, str] = {}
    for _, sub in pcap_df.groupby(group_cols, sort=False):
        sub = sub.sort_values("capture_timestamp", kind="stable")
        train, val, test = _split_pcaps(sub["file_source"].to_numpy(), ratios_arr)
        for pcap in train:
            split_of[pcap] = "train"
        for pcap in val:
            split_of[pcap] = "val"
        for pcap in test:
            split_of[pcap] = "test"

    split = flow_meta["file_source"].map(split_of)
    if split.isna().any():
        raise RuntimeError("Some flows have no pcap split assignment.")

    split_arr = split.to_numpy()
    return (
        np.where(split_arr == "train")[0],
        np.where(split_arr == "val")[0],
        np.where(split_arr == "test")[0],
    )

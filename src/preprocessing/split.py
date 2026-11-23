"""Chronological train/val/test split by pcap, done separately for each application."""

from __future__ import annotations

import pandas as pd

PCAP_COLUMNS = ["application", "file_source", "capture_timestamp"]


def assign_pcap_splits(
    pcaps: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_pcaps: int = 3,
) -> pd.DataFrame:
    """Split each application's pcaps in time order: oldest to train, then val, newest to test.

    Val and test each get ``max(1, round(frac * n))`` of an application's ``n`` pcaps and train
    gets the rest, so every set contains every application.

    Args:
        pcaps: One row per pcap with columns ``application``, ``file_source`` and
            ``capture_timestamp``.
        val_frac: Fraction of each application's pcaps used for validation.
        test_frac: Fraction of each application's pcaps used for testing.
        min_pcaps: Minimum pcaps an application needs; at least 3 (one per set).

    Returns:
        ``pcaps`` sorted by application and timestamp, with a ``split`` column.

    Raises:
        ValueError: If a column is missing, an application has fewer than ``min_pcaps`` pcaps,
            or the fractions leave an application with no training pcaps.
    """
    missing = set(PCAP_COLUMNS) - set(pcaps.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    pcaps = pcaps.sort_values(["application", "capture_timestamp", "file_source"], kind="stable").copy()
    splits = []
    for app, app_pcaps in pcaps.groupby("application", sort=False):
        n = len(app_pcaps)
        if n < max(min_pcaps, 3):
            raise ValueError(f"{app!r} has {n} pcap(s); the split needs at least {max(min_pcaps, 3)}.")
        n_val = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"{app!r}: val_frac/test_frac leave no training pcaps out of {n}.")
        splits += ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    pcaps["split"] = splits
    return pcaps


def chronological_pcap_split(
    flow_info: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_pcaps: int = 3,
) -> pd.Series:
    """Assign every flow the split of the pcap it came from (see ``assign_pcap_splits``).

    Args:
        flow_info: One row per flow with columns ``application``, ``file_source`` and
            ``capture_timestamp``, e.g. the flow table from ``preprocessing.flows.build_flows``.

    Returns:
        A Series aligned with ``flow_info.index`` holding ``'train'``, ``'val'`` or ``'test'``.
    """
    pcaps = assign_pcap_splits(
        flow_info[PCAP_COLUMNS].drop_duplicates("file_source"), val_frac, test_frac, min_pcaps
    )
    return flow_info["file_source"].map(pcaps.set_index("file_source")["split"]).rename("split")

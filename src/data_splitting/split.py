from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


def _split_pcaps(pcaps: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split of an already time-sorted array of pcap ids.

    Groups with fewer than 2 PCAPs cannot be split into both train and test
    and are dropped (returns two empty arrays).
    """
    n = len(pcaps)
    if n < 2:
        return pcaps[:0], pcaps[:0]
    n_test = min(n - 1, max(1, math.ceil(n * test_size)))
    return pcaps[:-n_test], pcaps[-n_test:]


def split_by_pcap(
    flow_meta: pd.DataFrame,
    test_size: float = 0.15,
    dataset_scope: str = "combined",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` positional indices into ``flow_meta``.

    ``flow_meta`` must have columns ``file_source``, ``application`` and
    ``capture_timestamp``. With ``dataset_scope="per_dataset"``, a
    ``dataset_name`` column is also required and splitting is performed
    independently within each ``(dataset_name, application)`` group.

    Groups with fewer than 2 PCAPs are dropped from both splits.
    """
    required = {"file_source", "application", "capture_timestamp"}
    missing = required - set(flow_meta.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if dataset_scope not in {"combined", "per_dataset"}:
        raise ValueError(f"Unknown dataset_scope {dataset_scope!r}")

    if dataset_scope == "per_dataset" and "dataset_name" not in flow_meta.columns:
        raise ValueError("dataset_scope='per_dataset' requires a 'dataset_name' column")

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a fraction in (0, 1)")

    group_cols = ["application"]
    pcap_cols = ["file_source", "application", "capture_timestamp"]
    if dataset_scope == "per_dataset":
        group_cols = ["dataset_name", "application"]
        pcap_cols = ["dataset_name"] + pcap_cols

    pcap_df = flow_meta[pcap_cols].drop_duplicates(subset="file_source")

    split_of: dict[str, str] = {}
    for group_key, sub in pcap_df.groupby(group_cols, sort=False):
        sub = sub.sort_values(["capture_timestamp", "file_source"], kind="stable")
        pcaps = sub["file_source"].to_numpy()
        train, test = _split_pcaps(pcaps, test_size)

        if len(train) == 0 and len(test) == 0:
            logger.warning(
                "Dropping group %s: only %d PCAP(s) available", group_key, len(pcaps)
            )
            for pcap in pcaps:
                split_of[pcap] = "drop"
            continue

        for pcap in train:
            split_of[pcap] = "train"
        for pcap in test:
            split_of[pcap] = "test"

    split = flow_meta["file_source"].map(split_of)
    if split.isna().any():
        raise RuntimeError("Some flows have no pcap split assignment.")

    split_arr = split.to_numpy()
    return (
        np.where(split_arr == "train")[0],
        np.where(split_arr == "test")[0],
    )


def min_pcaps_per_app(flow_meta: pd.DataFrame, dataset_scope: str = "combined") -> int:
    """Return the smallest number of pcaps across applications.

    With ``dataset_scope="per_dataset"``, the minimum is computed over
    ``(dataset_name, application)`` groups instead.
    """
    required = {"file_source", "application"}
    missing = required - set(flow_meta.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    group_cols = ["application"]
    pcap_cols = ["file_source", "application"]
    if dataset_scope == "per_dataset":
        if "dataset_name" not in flow_meta.columns:
            raise ValueError(
                "dataset_scope='per_dataset' requires a 'dataset_name' column"
            )
        group_cols = ["dataset_name", "application"]
        pcap_cols = ["file_source", "dataset_name", "application"]
    elif dataset_scope != "combined":
        raise ValueError(f"Unknown dataset_scope {dataset_scope!r}")

    counts = (
        flow_meta[pcap_cols]
        .drop_duplicates(subset="file_source")
        .groupby(group_cols, sort=False)["file_source"]
        .nunique()
    )
    if counts.empty:
        raise ValueError("No applications present in flow_meta")
    return int(counts.min())


def cross_validate_by_pcap(
    flow_meta: pd.DataFrame,
    n_folds: int,
    dataset_scope: str = "combined",
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return one ``(train_idx, test_idx)`` pair per CV fold.

    Uses ``StratifiedGroupKFold`` with ``file_source`` as the group (so whole
    pcaps stay together) and ``application`` as the stratification target.
    ``n_folds`` must not exceed the smallest per-application pcap count.
    """
    required = {"file_source", "application"}
    missing = required - set(flow_meta.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if dataset_scope not in {"combined", "per_dataset"}:
        raise ValueError(f"Unknown dataset_scope {dataset_scope!r}")

    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")

    smallest = min_pcaps_per_app(flow_meta, dataset_scope)
    if n_folds > smallest:
        raise ValueError(
            f"n_folds={n_folds} exceeds the smallest per-application pcap count "
            f"({smallest})"
        )

    pcap_cols = ["file_source", "application"]
    stratify_cols = ["application"]
    if dataset_scope == "per_dataset":
        if "dataset_name" not in flow_meta.columns:
            raise ValueError(
                "dataset_scope='per_dataset' requires a 'dataset_name' column"
            )
        pcap_cols = ["file_source", "dataset_name", "application"]
        stratify_cols = ["dataset_name", "application"]

    pcap_df = flow_meta[pcap_cols].drop_duplicates(subset="file_source")
    pcap_ids = pcap_df["file_source"].to_numpy()
    stratify = pcap_df[stratify_cols].agg("::".join, axis=1).to_numpy()

    flow_src = flow_meta["file_source"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _train_pos, test_pos in sgkf.split(pcap_ids, stratify, groups=pcap_ids):
        test_pcaps = set(pcap_ids[test_pos])
        is_test = np.isin(flow_src, list(test_pcaps))
        folds.append((np.where(~is_test)[0], np.where(is_test)[0]))
    return folds

"""Filtering helpers to reduce the data entering a classification run."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def filter_applications(
    pkt_df: pd.DataFrame, applications: Optional[list[str]]
) -> pd.DataFrame:
    """Keep only rows whose ``application`` is in ``applications``.

    Matching is case-insensitive and ignores surrounding whitespace. ``None`` or
    an empty list keeps all rows.
    """
    if not applications:
        return pkt_df
    wanted = {a.strip().lower() for a in applications}
    return pkt_df[
        pkt_df["application"].astype(str).str.strip().str.lower().isin(wanted)
    ].copy()


def cap_pcaps_per_app(
    pkt_df: pd.DataFrame, max_pcaps_per_app: Optional[int]
) -> pd.DataFrame:
    """Keep only the oldest ``max_pcaps_per_app`` pcaps per application.

    PCAPs are ordered by ``capture_timestamp`` within each application. ``None``
    keeps all pcaps.
    """
    if max_pcaps_per_app is None:
        return pkt_df

    pcap_info = (
        pkt_df[["application", "file_source", "capture_timestamp"]]
        .drop_duplicates(subset="file_source")
        .sort_values("capture_timestamp", kind="stable")
    )
    kept = pcap_info.groupby("application", sort=False).head(max_pcaps_per_app)
    return pkt_df[pkt_df["file_source"].isin(kept["file_source"])].copy()

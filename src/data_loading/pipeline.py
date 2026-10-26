"""Orchestrate the data loading pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data_loading import download, extract, parse_filename, parse_pcap
from utils.schema.config import DataLoadingConfig
from utils.schema.data_download import DatasetConfig

logger = logging.getLogger(__name__)


def normalize_selection(
    selected: Any,
    datasets: list[DatasetConfig],
) -> list[DatasetConfig]:
    """Resolve ``selected`` (None | str | list[str]) to registered datasets."""
    if selected is None:
        return list(datasets)

    ids = [selected] if isinstance(selected, str) else list(selected)
    lookup = {d.dataset_id: d for d in datasets}
    result: list[DatasetConfig] = []
    for dataset_id in ids:
        if dataset_id in lookup:
            result.append(lookup[dataset_id])
        else:
            logger.warning(
                "Unknown dataset_id %s (not in registry); skipping", dataset_id
            )
    return result


def _process_dataset(
    dataset: DatasetConfig,
    config: DataLoadingConfig,
) -> list[pd.DataFrame]:
    raw_dir = Path(config.paths.raw_dir) / dataset.name
    interim_dir = Path(config.paths.interim_dir) / dataset.name
    processed_dir = Path(config.paths.processed_dir) / dataset.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    archives = download.download_dataset(dataset, config.download, raw_dir)

    pcap_paths: list[Path] = []
    for archive in archives:
        pcap_paths.extend(extract.extract_archive(archive, interim_dir, config.extract))

    frames: list[pd.DataFrame] = []
    for pcap_path in pcap_paths:
        try:
            metadata = parse_filename.parse_filename(
                pcap_path.name, config.filename_pattern
            )
        except ValueError as exc:
            logger.warning("Skipping %s: %s", pcap_path.name, exc)
            continue
        frame = parse_pcap.parse_pcap_to_dataframe(
            pcap_path, config.parsing, metadata, dataset.name
        )
        if not frame.empty:
            frames.append(frame)

    if frames:
        dataset_df = pd.concat(frames, ignore_index=True)
        out = processed_dir / "packets.csv"
        dataset_df.to_csv(out, index=False)
        logger.info("Saved %s -> %s (%d rows)", dataset.name, out, len(dataset_df))
    return frames


def run(config: DataLoadingConfig) -> None:
    selected = normalize_selection(config.selected_dataset_ids, config.datasets)
    if not selected:
        logger.warning("No datasets selected; nothing to do.")
        return

    frames: list[pd.DataFrame] = []
    for dataset in selected:
        frames.extend(_process_dataset(dataset, config))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out = Path(config.paths.combined_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        logger.info("Saved combined dataset -> %s (%d rows)", out, len(combined))

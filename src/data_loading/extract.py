"""Extract downloaded archives (.rar / .zip) into a target directory."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import rarfile

from utils.schema.extraction import ExtractConfig

logger = logging.getLogger(__name__)


def _pcap_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pcap"))


def extract_rar(archive_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with rarfile.RarFile(archive_path) as rf:
        rf.extractall(path=str(dest_dir))
    return _pcap_files(dest_dir)


def extract_zip(archive_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(path=dest_dir)
    return _pcap_files(dest_dir)


def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    cfg: ExtractConfig,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.overwrite and _pcap_files(dest_dir):
        logger.info(
            "Skipping extraction of %s (destination already has PCAPs)",
            archive_path.name,
        )
        return _pcap_files(dest_dir)

    suffix = archive_path.suffix.lower()
    if suffix == ".rar":
        return extract_rar(archive_path, dest_dir)
    if suffix == ".zip":
        return extract_zip(archive_path, dest_dir)

    logger.warning("Unsupported archive type: %s", suffix)
    return []

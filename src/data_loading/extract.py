"""Extract downloaded archives (.rar / .zip) into a target directory."""

from __future__ import annotations

import logging
import subprocess
import zipfile
from pathlib import Path

from utils.schema.extraction import ExtractConfig

logger = logging.getLogger(__name__)


def _pcap_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pcap"))


def extract_rar(archive_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a .rar archive using bsdtar (libarchive)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["bsdtar", "-xf", str(archive_path), "-C", str(dest_dir)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "bsdtar (libarchive) is required to extract .rar files but was not found on PATH"
        ) from None
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"bsdtar failed to extract {archive_path.name}"
        )
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
    existing = _pcap_files(dest_dir)
    if not cfg.overwrite and any(p.stat().st_size > 0 for p in existing):
        logger.info(
            "Skipping extraction of %s (destination already has PCAPs)",
            archive_path.name,
        )
        return existing

    suffix = archive_path.suffix.lower()
    try:
        if suffix == ".rar":
            return extract_rar(archive_path, dest_dir)
        if suffix == ".zip":
            return extract_zip(archive_path, dest_dir)
    except (RuntimeError, zipfile.BadZipFile) as exc:
        logger.warning("Failed to extract %s: %s", archive_path.name, exc)
        return []

    logger.warning("Unsupported archive type: %s", suffix)
    return []

"""Download dataset archives from the Mendeley Data public API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from utils.schema.data_download import DatasetConfig, DownloadConfig

logger = logging.getLogger(__name__)


def list_dataset_files(
    dataset_id: str,
    version: int,
    base_url: str,
    user_agent: str,
) -> list[dict[str, Any]]:
    url = f"{base_url}/{dataset_id}/versions/{version}"
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    return response.json().get("files", [])


def filter_archives(
    files: list[dict[str, Any]],
    archive_extensions: list[str],
) -> list[dict[str, Any]]:
    extensions = {ext.lower() for ext in archive_extensions}
    return [
        f for f in files if str(f.get("name", "")).lower().endswith(tuple(extensions))
    ]


def _resolve_download_url(file_info: dict[str, Any]) -> str | None:
    url = file_info.get("content_details", {}).get("download_url")
    if not url:
        url = file_info.get("download_url")
    return url


def download_file(
    download_url: str,
    dest_path: Path,
    user_agent: str,
    chunk_size: int,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        download_url,
        headers={"User-Agent": user_agent},
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as out:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    out.write(chunk)
    return dest_path


def download_dataset(
    dataset: DatasetConfig,
    download: DownloadConfig,
    dest_dir: Path,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = list_dataset_files(
        dataset.dataset_id,
        dataset.version,
        download.base_url,
        download.user_agent,
    )
    archives = filter_archives(files, download.archive_extensions)
    if not archives:
        logger.warning("No archive files found for dataset %s", dataset.dataset_id)
        return []

    downloaded: list[Path] = []
    for file_info in archives:
        name = file_info.get("name") or "archive.bin"
        url = _resolve_download_url(file_info)
        if not url:
            logger.warning("Skipping %s: no download URL", name)
            continue
        dest_path = dest_dir / name
        try:
            downloaded.append(
                download_file(url, dest_path, download.user_agent, download.chunk_size)
            )
            logger.info("Downloaded %s -> %s", name, dest_path)
        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", name, exc)
    return downloaded

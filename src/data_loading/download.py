"""Download dataset archives from the Mendeley Data public API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from utils.schema.data_download import DatasetConfig, DownloadConfig

logger = logging.getLogger(__name__)


def list_dataset_files(
    dataset_id: str,
    version: int,
    base_url: str,
    user_agent: str,
) -> list[dict[str, Any]]:
    url = f"{base_url}/{dataset_id}/files?folder_id=root&version={version}"
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data
    return data.get("files", [])


def filter_archives(
    files: list[dict[str, Any]],
    archive_extensions: list[str],
) -> list[dict[str, Any]]:
    extensions = {ext.lower() for ext in archive_extensions}
    return [
        f
        for f in files
        if str(f.get("filename", "")).lower().endswith(tuple(extensions))
    ]


def _resolve_download_url(file_info: dict[str, Any]) -> str | None:
    content_details = file_info.get("content_details") or {}
    url = content_details.get("download_url")
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
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with requests.get(
            download_url,
            headers={"User-Agent": user_agent},
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0) or 0)
            with open(part_path, "wb") as out:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest_path.name,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            out.write(chunk)
                            pbar.update(len(chunk))
        os.replace(part_path, dest_path)
    except requests.RequestException:
        part_path.unlink(missing_ok=True)
        raise
    return dest_path


def download_dataset(
    dataset: DatasetConfig,
    download: DownloadConfig,
    dest_dir: Path,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = list_dataset_files(
            dataset.dataset_id,
            dataset.version,
            download.base_url,
            download.user_agent,
        )
    except requests.RequestException as exc:
        logger.warning(
            "Failed to list files for dataset %s: %s", dataset.dataset_id, exc
        )
        return []

    archives = filter_archives(files, download.archive_extensions)
    if download.max_files is not None:
        archives = archives[: download.max_files]
    if not archives:
        logger.warning("No archive files found for dataset %s", dataset.dataset_id)
        return []

    downloaded: list[Path] = []
    for file_info in archives:
        name = Path(file_info.get("filename") or "archive.bin").name
        url = _resolve_download_url(file_info)
        if not url:
            logger.warning("Skipping %s: no download URL", name)
            continue
        dest_path = dest_dir / name
        if dest_path.exists() and dest_path.stat().st_size > 0:
            logger.info("Skipping download (already exists): %s", dest_path)
            downloaded.append(dest_path)
            continue
        try:
            downloaded.append(
                download_file(url, dest_path, download.user_agent, download.chunk_size)
            )
            logger.info("Downloaded %s -> %s", name, dest_path)
        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", name, exc)
    return downloaded

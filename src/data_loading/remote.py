"""Remote seekable file access and selective RAR extraction."""

from __future__ import annotations

import io
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import rarfile
import requests

from data_loading import download, extract
from data_loading.parse_filename import parse_filename
from utils.schema.data_download import DatasetConfig, DownloadConfig
from utils.schema.extraction import ExtractConfig

logger = logging.getLogger(__name__)


class RemoteRangeFile:
    """Seekable read-only file object backed by HTTP Range requests.

    Reads are served from a bounded block cache (default 1 MiB blocks) so that
    sequential streaming does not issue one HTTP request per small read.
    """

    def __init__(
        self,
        url: str,
        user_agent: str,
        block_size: int = 1 << 20,
        timeout: int = 60,
    ) -> None:
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        )
        self.size = self._determine_size()
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def _determine_size(self) -> int:
        response = self.session.head(
            self.url, allow_redirects=True, timeout=self.timeout
        )
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length is not None:
            return int(length)
        response = self.session.get(
            self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout
        )
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        raise RuntimeError(f"Unable to determine file size for {self.url}")

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self.position = max(0, min(self.position, self.size))
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""

        result = bytearray()
        end = self.position + size
        while self.position < end:
            block_id = self.position // self.block_size
            block = self._fetch_block(block_id)
            offset = self.position % self.block_size
            count = min(end - self.position, len(block) - offset)
            if count <= 0:
                raise RuntimeError(
                    f"Short read from {self.url} at offset {self.position}"
                )
            result.extend(block[offset : offset + count])
            self.position += count
            self._evict_cache()
        return bytes(result)

    def _fetch_block(self, block_id: int) -> bytes:
        if block_id in self.cache:
            return self.cache[block_id]
        start = block_id * self.block_size
        stop = min(start + self.block_size, self.size) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{stop}"},
            timeout=self.timeout,
        )
        if response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"Server does not support HTTP Range requests "
                f"(status {response.status_code})"
            )
        self.cache[block_id] = response.content
        response.close()
        return self.cache[block_id]

    def _evict_cache(self) -> None:
        current_block = self.position // self.block_size
        for block_id in [b for b in self.cache if b < current_block]:
            del self.cache[block_id]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.session.close()


def select_application_members(
    members: list[Any],
    applications: Optional[list[str]],
    filename_pattern: str,
) -> list[str]:
    """Return member names whose parsed application is in ``applications``.

    ``None`` or empty ``applications`` selects all non-directory members.
    Matching is case-insensitive; members whose names do not match
    ``filename_pattern`` are skipped.
    """
    if isinstance(applications, str):
        applications = [applications]
    selected: Optional[set[str]] = None
    if applications:
        selected = {a.strip().lower() for a in applications}

    result: list[str] = []
    for member in members:
        if hasattr(member, "is_dir") and member.is_dir():
            continue
        name = getattr(member, "filename", member)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        name = str(name)
        if selected is None:
            result.append(name)
            continue
        try:
            metadata = parse_filename(name, filename_pattern)
        except ValueError:
            logger.warning("Skipping %s: filename does not match convention", name)
            continue
        if metadata.application.strip().lower() in selected:
            result.append(name)
    return result


def download_application_pcaps(
    dataset: DatasetConfig,
    download_cfg: DownloadConfig,
    extract_cfg: ExtractConfig,
    applications: Optional[list[str]],
    filename_pattern: str,
    raw_dir: Path,
    interim_dir: Path,
) -> list[Path]:
    """Download only the PCAPs matching ``applications`` from ``dataset``.

    Non-solid RARs are read remotely via HTTP Range and only matching members
    are extracted. Solid (or non-RAR) archives fall back to full download.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = download.list_dataset_files(
            dataset.dataset_id,
            dataset.version,
            download_cfg.base_url,
            download_cfg.user_agent,
        )
    except requests.RequestException as exc:
        logger.warning(
            "Failed to list files for dataset %s: %s", dataset.dataset_id, exc
        )
        return []

    archives = download.filter_archives(files, download_cfg.archive_extensions)
    if download_cfg.max_files is not None:
        archives = archives[: download_cfg.max_files]

    seen: set[str] = set()
    downloaded: list[Path] = []

    def collect(paths: list[Path]) -> None:
        for p in paths:
            key = str(p)
            if key not in seen:
                seen.add(key)
                downloaded.append(p)

    for file_info in archives:
        name = Path(file_info.get("filename") or "archive.bin").name
        url = download.resolve_download_url(file_info)
        if not url:
            logger.warning("Skipping %s: no download URL", name)
            continue
        try:
            if Path(name).suffix.lower() == ".rar":
                try:
                    collect(
                        _selective_rar(
                            name,
                            url,
                            download_cfg,
                            extract_cfg,
                            applications,
                            filename_pattern,
                            raw_dir,
                            interim_dir,
                        )
                    )
                except rarfile.RarCannotExec:
                    if download_cfg.solid_fallback:
                        logger.info(
                            "No RAR decompressor for %s; falling back to full download",
                            name,
                        )
                        collect(
                            _full_download(
                                name,
                                url,
                                download_cfg,
                                extract_cfg,
                                applications,
                                filename_pattern,
                                raw_dir,
                                interim_dir,
                            )
                        )
                    else:
                        logger.warning(
                            "No RAR decompressor available for %s; skipping", name
                        )
            else:
                collect(
                    _full_download(
                        name,
                        url,
                        download_cfg,
                        extract_cfg,
                        applications,
                        filename_pattern,
                        raw_dir,
                        interim_dir,
                    )
                )
        except (
            requests.RequestException,
            rarfile.Error,
            RuntimeError,
            OSError,
        ) as exc:
            logger.warning("Failed to process %s: %s", name, exc)

    return downloaded


def _full_download(
    name: str,
    url: str,
    download_cfg: DownloadConfig,
    extract_cfg: ExtractConfig,
    applications: Optional[list[str]],
    filename_pattern: str,
    raw_dir: Path,
    interim_dir: Path,
) -> list[Path]:
    dest = raw_dir / name
    if not (dest.exists() and dest.stat().st_size > 0):
        download.download_file(
            url, dest, download_cfg.user_agent, download_cfg.chunk_size
        )
    pcaps = extract.extract_archive(dest, interim_dir, extract_cfg)
    if isinstance(applications, str):
        applications = [applications]
    if not applications:
        return pcaps
    wanted = {a.strip().lower() for a in applications}
    kept: list[Path] = []
    for p in pcaps:
        try:
            meta = parse_filename(p.name, filename_pattern)
        except ValueError:
            continue
        if meta.application.strip().lower() in wanted:
            kept.append(p)
    return kept


def _selective_rar(
    name: str,
    url: str,
    download_cfg: DownloadConfig,
    extract_cfg: ExtractConfig,
    applications: Optional[list[str]],
    filename_pattern: str,
    raw_dir: Path,
    interim_dir: Path,
) -> list[Path]:
    remote_file = RemoteRangeFile(url, download_cfg.user_agent)
    try:
        archive = rarfile.RarFile(remote_file)
        try:
            if archive.is_solid():
                if download_cfg.solid_fallback:
                    logger.info(
                        "Archive %s is solid; falling back to full download", name
                    )
                    return _full_download(
                        name,
                        url,
                        download_cfg,
                        extract_cfg,
                        applications,
                        filename_pattern,
                        raw_dir,
                        interim_dir,
                    )
                logger.warning(
                    "Archive %s is solid; skipping (solid_fallback=False)", name
                )
                return []

            member_names = select_application_members(
                archive.infolist(), applications, filename_pattern
            )
            result: list[Path] = []
            for member_name in member_names:
                dest = interim_dir / Path(member_name).name
                if (
                    dest.exists()
                    and dest.stat().st_size > 0
                    and not extract_cfg.overwrite
                ):
                    logger.info("Skipping %s (already exists)", dest)
                    result.append(dest)
                    continue
                with archive.open(member_name) as src:
                    try:
                        with open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except OSError as exc:
                        logger.warning("Failed to extract %s: %s", member_name, exc)
                        continue
                logger.info(
                    "Extracted %s -> %s (%d bytes)",
                    member_name,
                    dest,
                    dest.stat().st_size,
                )
                result.append(dest)
            return result
        finally:
            archive.close()
    finally:
        remote_file.close()

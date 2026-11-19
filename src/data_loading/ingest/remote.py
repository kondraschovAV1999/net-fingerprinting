"""Application-selective download and extraction of dataset archives."""

from __future__ import annotations

import logging
import queue
import re
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import rarfile
import requests

from data_loading.ingest import download, extract
from data_loading.ingest.rar import (
    extract_member_7z,
    resolve_sevenzip,
    write_sparse_archive,
)
from data_loading.ingest.remote_file import RemoteRangeFile
from data_loading.parsing.parse_filename import parse_filename
from utils.schema.data_download import DatasetConfig, DownloadConfig
from utils.schema.extraction import ExtractConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ExtractionContext:
    """Shared settings passed through archive download/extraction helpers."""

    download_cfg: DownloadConfig
    extract_cfg: ExtractConfig
    applications: Optional[set[str]]
    filename_pattern: str
    raw_dir: Path
    interim_dir: Path


@dataclass(frozen=True)
class _PendingMember:
    name: str
    info: Any
    dest: Path
    data_offset: int


@dataclass(frozen=True)
class _QueuedArchive:
    """Item passed from download workers to extraction workers."""

    name: str
    path: Path
    pending: Optional[list[_PendingMember]] = None  # None => full archive extraction
    delete_after: bool = True


def _normalize_applications(
    applications: Optional[list[str] | str],
) -> Optional[set[str]]:
    if not applications:
        return None
    if isinstance(applications, str):
        applications = [applications]
    return {app.strip().lower() for app in applications}


def _sanitize(app: str) -> str:
    """Turn an application name into a filesystem-safe folder name."""
    return re.sub(r"[^a-z0-9]+", "_", app.lower()).strip("_")


def _app_dest(interim_dir: Path, name: str, filename_pattern: str) -> Path:
    """Return ``interim_dir / <app_slug> / <name>``, creating the app folder."""
    try:
        app = parse_filename(name, filename_pattern).application
    except ValueError:
        app = "_unclassified"
    dest_dir = interim_dir / _sanitize(app)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / name


def _matches_application(
    filename: str,
    selected: Optional[set[str]],
    filename_pattern: str,
    *,
    warn_on_mismatch: bool = False,
) -> bool:
    if selected is None:
        return True
    try:
        metadata = parse_filename(filename, filename_pattern)
    except ValueError:
        if warn_on_mismatch:
            logger.warning("Skipping %s: filename does not match convention", filename)
        return False
    return metadata.application.strip().lower() in selected


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
    selected = _normalize_applications(applications)
    result: list[str] = []

    for member in members:
        if hasattr(member, "is_dir") and member.is_dir():
            continue
        name = getattr(member, "filename", member)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        name = str(name)
        if _matches_application(
            name, selected, filename_pattern, warn_on_mismatch=True
        ):
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
    download_workers: int = 3,
    extract_workers: int = 4,
) -> list[Path]:
    """Download only the PCAPs matching ``applications`` from ``dataset``.

    Download workers fetch archives and enqueue them onto a bounded
    ``queue.Queue``. Extraction workers pull from the queue, extract matching
    PCAPs into ``interim_dir/<app>/``, and delete temporary archives
    immediately afterward. Full archives are cached in ``raw_dir``.
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

    ctx = _ExtractionContext(
        download_cfg=download_cfg,
        extract_cfg=extract_cfg,
        applications=_normalize_applications(applications),
        filename_pattern=filename_pattern,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
    )

    has_rar = any(
        Path(f.get("filename") or "").suffix.lower() == ".rar" for f in archives
    )
    try:
        sevenzip = resolve_sevenzip(download_cfg.sevenzip_path) if has_rar else None
    except RuntimeError as exc:
        logger.warning(
            "7-Zip unavailable; selective RAR extraction will be skipped: %s", exc
        )
        sevenzip = None

    seen: set[str] = set()
    downloaded: list[Path] = []

    def collect(paths: list[Path]) -> None:
        for path in paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                downloaded.append(path)

    arrived_queue: queue.Queue[Optional[_QueuedArchive]] = queue.Queue(
        maxsize=max(download_workers, extract_workers) * 2
    )

    with (
        ThreadPoolExecutor(max_workers=download_workers) as dl_pool,
        ThreadPoolExecutor(max_workers=extract_workers) as ex_pool,
    ):
        # Start extraction consumers
        ex_futures = [
            ex_pool.submit(_extract_worker, arrived_queue, ctx, sevenzip)
            for _ in range(extract_workers)
        ]

        # Submit download producers
        dl_futures: dict[Future[list[Path]], str] = {}
        for file_info in archives:
            name = Path(file_info.get("filename") or "archive.bin").name
            url = download.resolve_download_url(file_info)
            if not url:
                logger.warning("Skipping %s: no download URL", name)
                continue
            fut = dl_pool.submit(
                _download_stage, name, url, ctx, arrived_queue, sevenzip
            )
            dl_futures[fut] = name

        # Wait for all downloads to finish
        for fut in as_completed(dl_futures):
            name = dl_futures[fut]
            try:
                collect(fut.result())
            except (
                requests.RequestException,
                rarfile.Error,
                RuntimeError,
                OSError,
            ) as exc:
                logger.warning("Failed to download %s: %s", name, exc)

        # Signal extraction workers to exit once the queue is drained
        for _ in range(extract_workers):
            arrived_queue.put(None)

        # Collect extracted paths from all extraction workers
        for ex_fut in as_completed(ex_futures):
            collect(ex_fut.result())

    return downloaded


def _download_stage(
    name: str,
    url: str,
    ctx: _ExtractionContext,
    arrived_queue: queue.Queue[Optional[_QueuedArchive]],
    sevenzip: Optional[str],
) -> list[Path]:
    """Download archive (or RAR ranges) and enqueue it for extraction.

    Returns any PCAP paths that were already extracted on disk.
    """
    if Path(name).suffix.lower() != ".rar":
        dest = _download_full(name, url, ctx)
        arrived_queue.put(_QueuedArchive(name=name, path=dest, delete_after=False))
        return []

    with (
        RemoteRangeFile(url, ctx.download_cfg.user_agent, evict=False) as remote_file,
        rarfile.RarFile(remote_file) as archive,
    ):
        if archive.is_solid():
            if not ctx.download_cfg.solid_fallback:
                logger.warning(
                    "Archive %s is solid; skipping (solid_fallback=False)", name
                )
                return []
            logger.info("Archive %s is solid; falling back to full download", name)
            dest = _download_full(name, url, ctx)
            arrived_queue.put(_QueuedArchive(name=name, path=dest, delete_after=False))
            return []

        if sevenzip is None:
            logger.warning(
                "Skipping %s: 7-Zip unavailable for selective RAR extraction", name
            )
            return []

        member_names = select_application_members(
            archive.infolist(),
            list(ctx.applications) if ctx.applications is not None else None,
            ctx.filename_pattern,
        )
        already_extracted, pending = _partition_members(
            archive,
            member_names,
            ctx.interim_dir,
            ctx.extract_cfg.overwrite,
            ctx.filename_pattern,
        )
        if not pending:
            return already_extracted

        fetched = _fetch_member_ranges(remote_file, pending)
        if not fetched:
            return already_extracted

        sparse_archive = write_sparse_archive(remote_file, name)
        arrived_queue.put(
            _QueuedArchive(name=name, path=sparse_archive, pending=fetched)
        )
        return already_extracted


def _download_full(name: str, url: str, ctx: _ExtractionContext) -> Path:
    """Download a full archive to ``raw_dir`` (cached), returning its path."""
    dest = ctx.raw_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        download.download_file(
            url, dest, ctx.download_cfg.user_agent, ctx.download_cfg.chunk_size
        )
    return dest


def _extract_worker(
    arrived_queue: queue.Queue[Optional[_QueuedArchive]],
    ctx: _ExtractionContext,
    sevenzip: Optional[str],
) -> list[Path]:
    """Consume arrived archives from ``arrived_queue`` until a ``None`` sentinel."""
    extracted: list[Path] = []
    while True:
        item = arrived_queue.get()
        try:
            if item is None:
                return extracted
            extracted.extend(_extract_archive(item, ctx, sevenzip))
        except (OSError, RuntimeError) as exc:
            if item is not None:
                logger.warning("Failed to extract %s: %s", item.name, exc)
        finally:
            arrived_queue.task_done()


def _extract_archive(
    item: _QueuedArchive, ctx: _ExtractionContext, sevenzip: Optional[str]
) -> list[Path]:
    """Extract a single queued archive into ``interim_dir/<app>/``."""
    extracted: list[Path] = []
    try:
        if item.pending is None:
            with tempfile.TemporaryDirectory() as flat_str:
                flat_dir = Path(flat_str)
                pcaps = extract.extract_archive(item.path, flat_dir, ctx.extract_cfg)
                extracted = _organize_pcaps(pcaps, ctx)
        else:
            assert sevenzip is not None
            for member in item.pending:
                try:
                    extract_member_7z(sevenzip, item.path, member.name, member.dest)
                except (OSError, RuntimeError) as exc:
                    logger.warning("Failed to extract %s: %s", member.name, exc)
                    continue
                logger.info(
                    "Extracted %s -> %s (%d bytes)",
                    member.name,
                    member.dest,
                    member.dest.stat().st_size,
                )
                extracted.append(member.dest)
        return extracted
    finally:
        if item.delete_after:
            item.path.unlink(missing_ok=True)


def _organize_pcaps(pcaps: list[Path], ctx: _ExtractionContext) -> list[Path]:
    """Move extracted PCAPs into per-application subfolders under ``interim_dir``."""
    result: list[Path] = []
    for pcap in pcaps:
        if not _matches_application(pcap.name, ctx.applications, ctx.filename_pattern):
            continue
        dest = _app_dest(ctx.interim_dir, pcap.name, ctx.filename_pattern)
        if dest.exists() and dest.stat().st_size > 0 and not ctx.extract_cfg.overwrite:
            result.append(dest)
            continue
        shutil.move(str(pcap), str(dest))
        result.append(dest)
    return result


def _partition_members(
    archive: rarfile.RarFile,
    member_names: list[str],
    interim_dir: Path,
    overwrite: bool,
    filename_pattern: str,
) -> tuple[list[Path], list[_PendingMember]]:
    already_extracted: list[Path] = []
    pending: list[_PendingMember] = []

    for member_name in member_names:
        dest = _app_dest(interim_dir, member_name, filename_pattern)
        if dest.exists() and dest.stat().st_size > 0 and not overwrite:
            logger.info("Skipping %s (already exists)", dest)
            already_extracted.append(dest)
            continue

        info = archive.getinfo(member_name)
        data_offset = getattr(info, "data_offset", None)
        if data_offset is None:
            logger.warning("Skipping %s: no data offset available", member_name)
            continue

        pending.append(_PendingMember(member_name, info, dest, data_offset))

    return already_extracted, pending


def _fetch_member_ranges(
    remote_file: RemoteRangeFile,
    pending: list[_PendingMember],
) -> list[_PendingMember]:
    fetched: list[_PendingMember] = []
    for member in pending:
        available = remote_file.prefetch(member.data_offset, member.info.compress_size)
        if available != member.info.compress_size:
            logger.warning(
                "Skipping %s: incomplete compressed read (%d of %d bytes)",
                member.name,
                available,
                member.info.compress_size,
            )
            continue
        fetched.append(member)
    return fetched

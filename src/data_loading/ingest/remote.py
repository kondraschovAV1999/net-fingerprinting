"""Application-selective download and extraction of dataset archives."""

from __future__ import annotations

import logging
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


def _normalize_applications(
    applications: Optional[list[str] | str],
) -> Optional[set[str]]:
    if not applications:
        return None
    if isinstance(applications, str):
        applications = [applications]
    return {app.strip().lower() for app in applications}


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

    ctx = _ExtractionContext(
        download_cfg=download_cfg,
        extract_cfg=extract_cfg,
        applications=_normalize_applications(applications),
        filename_pattern=filename_pattern,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
    )

    seen: set[str] = set()
    downloaded: list[Path] = []

    for file_info in archives:
        name = Path(file_info.get("filename") or "archive.bin").name
        url = download.resolve_download_url(file_info)
        if not url:
            logger.warning("Skipping %s: no download URL", name)
            continue

        handler = (
            _selective_rar if Path(name).suffix.lower() == ".rar" else _full_download
        )
        try:
            paths = handler(name, url, ctx)
        except (requests.RequestException, rarfile.Error, RuntimeError, OSError) as exc:
            logger.warning("Failed to process %s: %s", name, exc)
            continue

        for path in paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                downloaded.append(path)

    return downloaded


def _full_download(name: str, url: str, ctx: _ExtractionContext) -> list[Path]:
    dest = ctx.raw_dir / name
    if not (dest.exists() and dest.stat().st_size > 0):
        download.download_file(
            url, dest, ctx.download_cfg.user_agent, ctx.download_cfg.chunk_size
        )

    pcaps = extract.extract_archive(dest, ctx.interim_dir, ctx.extract_cfg)
    if ctx.applications is None:
        return pcaps
    return [
        p
        for p in pcaps
        if _matches_application(p.name, ctx.applications, ctx.filename_pattern)
    ]


def _selective_rar(name: str, url: str, ctx: _ExtractionContext) -> list[Path]:
    with (
        RemoteRangeFile(url, ctx.download_cfg.user_agent, evict=False) as remote_file,
        rarfile.RarFile(remote_file) as archive,
    ):
        if archive.is_solid():
            return _handle_solid_archive(name, url, ctx)

        member_names = select_application_members(
            archive.infolist(),
            list(ctx.applications) if ctx.applications is not None else None,
            ctx.filename_pattern,
        )
        extracted, pending = _partition_members(
            archive, member_names, ctx.interim_dir, ctx.extract_cfg.overwrite
        )
        if not pending:
            return extracted

        fetched = _fetch_member_ranges(remote_file, pending)
        if not fetched:
            return extracted

        sevenzip = resolve_sevenzip(ctx.download_cfg.sevenzip_path)
        extracted.extend(_extract_fetched_members(remote_file, name, sevenzip, fetched))
        return extracted


def _handle_solid_archive(name: str, url: str, ctx: _ExtractionContext) -> list[Path]:
    if ctx.download_cfg.solid_fallback:
        logger.info("Archive %s is solid; falling back to full download", name)
        return _full_download(name, url, ctx)
    logger.warning("Archive %s is solid; skipping (solid_fallback=False)", name)
    return []


def _partition_members(
    archive: rarfile.RarFile,
    member_names: list[str],
    interim_dir: Path,
    overwrite: bool,
) -> tuple[list[Path], list[_PendingMember]]:
    already_extracted: list[Path] = []
    pending: list[_PendingMember] = []

    for member_name in member_names:
        dest = interim_dir / Path(member_name).name
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


def _extract_fetched_members(
    remote_file: RemoteRangeFile,
    archive_name: str,
    sevenzip: str,
    members: list[_PendingMember],
) -> list[Path]:
    extracted: list[Path] = []
    sparse_archive = write_sparse_archive(remote_file, archive_name)
    try:
        for member in members:
            try:
                extract_member_7z(sevenzip, sparse_archive, member.name, member.dest)
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
    finally:
        sparse_archive.unlink(missing_ok=True)
    return extracted

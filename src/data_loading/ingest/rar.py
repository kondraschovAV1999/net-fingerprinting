"""RAR extraction helpers backed by 7-Zip."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from data_loading.ingest.remote_file import RemoteRangeFile


def resolve_sevenzip(configured: Optional[str]) -> str:
    """Return a path to a 7-Zip executable.

    Resolution order: explicit config value, ``7z``/``7zz`` on PATH, then the
    default Windows install location.
    """
    if configured:
        if Path(configured).is_file():
            return str(Path(configured))
        if found := shutil.which(configured):
            return found
        raise RuntimeError(f"7-Zip not found at configured path: {configured}")

    for candidate in ("7z", "7zz"):
        if found := shutil.which(candidate):
            return found

    default = r"C:\Program Files\7-Zip\7z.exe"
    if os.path.isfile(default):
        return default

    raise RuntimeError(
        "7-Zip (7z.exe) is required for RAR extraction but was not found. "
        "Install 7-Zip or set download.sevenzip_path."
    )


def write_sparse_archive(remote_file: RemoteRangeFile, name: str) -> Path:
    """Write fetched HTTP blocks at their original offsets into a temp archive.

    Unfetched regions stay zero-filled so the RAR header chain and the selected
    members' data are present at their original offsets.
    """
    suffix = Path(name).suffix or ".rar"
    fd, path = tempfile.mkstemp(suffix=suffix)
    tmp = Path(path)
    try:
        with os.fdopen(fd, "wb") as out:
            out.truncate(remote_file.size)
            for block_id, content in sorted(remote_file.cache.items()):
                out.seek(block_id * remote_file.block_size)
                out.write(content)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def extract_member_7z(
    sevenzip: str, sparse_archive: Path, member_name: str, dest: Path
) -> Path:
    """Extract a single member from a sparse archive to ``dest`` via 7-Zip."""
    cmd = [sevenzip, "e", "-so", "-bb0", "-p", "--", str(sparse_archive), member_name]
    with open(dest, "wb") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        stderr = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"7-Zip extraction failed for {member_name}: {stderr}")
    return dest

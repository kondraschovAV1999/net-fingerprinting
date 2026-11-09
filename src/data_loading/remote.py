"""Remote seekable file access and selective RAR extraction."""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import requests

from data_loading.parse_filename import parse_filename

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

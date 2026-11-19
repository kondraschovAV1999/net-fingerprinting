"""Seekable read-only file object backed by HTTP Range requests."""

from __future__ import annotations

import io
from typing import Any

import requests


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
        evict: bool = True,
    ) -> None:
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self._evict = evict
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        )
        self.size = self._determine_size()
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def _determine_size(self) -> int:
        with self.session.head(
            self.url, allow_redirects=True, timeout=self.timeout
        ) as response:
            response.raise_for_status()
            if (length := response.headers.get("Content-Length")) is not None:
                return int(length)

        with self.session.get(
            self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout
        ) as response:
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])

        raise RuntimeError(f"Unable to determine file size for {self.url}")

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self.position = max(0, min(target, self.size))
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

    def prefetch(self, offset: int, size: int) -> int:
        """Populate the block cache for ``[offset, offset + size)`` without copying bytes.

        Returns the number of bytes actually available in that range.
        """
        if offset < 0 or size <= 0 or offset >= self.size:
            return 0

        end = min(offset + size, self.size)
        pos = offset
        while pos < end:
            block = self._fetch_block(pos // self.block_size)
            block_offset = pos % self.block_size
            available = min(end - pos, len(block) - block_offset)
            if available <= 0:
                break
            pos += available
        return pos - offset

    def _fetch_block(self, block_id: int) -> bytes:
        if block_id in self.cache:
            return self.cache[block_id]

        start = block_id * self.block_size
        stop = min(start + self.block_size, self.size) - 1
        with self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{stop}"},
            timeout=self.timeout,
        ) as response:
            if response.status_code != 206:
                raise RuntimeError(
                    f"Server does not support HTTP Range requests "
                    f"(status {response.status_code})"
                )
            self.cache[block_id] = response.content
            return self.cache[block_id]

    def _evict_cache(self) -> None:
        if not self._evict:
            return
        current_block = self.position // self.block_size
        for block_id in [b for b in self.cache if b < current_block]:
            del self.cache[block_id]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> RemoteRangeFile:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

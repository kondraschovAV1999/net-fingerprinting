"""Seekable read-only file object backed by HTTP Range requests."""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class RemoteRangeFile:
    """Seekable read-only file object backed by HTTP Range requests.

    Reads are served from a bounded block cache (default 1 MiB blocks) so that
    sequential streaming does not issue one HTTP request per small read.

    Transient failures (rate limiting, server errors, HTML challenge pages) are
    retried with exponential backoff.
    """

    def __init__(
        self,
        url: str,
        user_agent: str,
        block_size: int = 1 << 20,
        timeout: int = 60,
        evict: bool = True,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self._evict = evict
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        )
        self.size = self._determine_size()
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def _backoff(self, attempt: int) -> None:
        time.sleep(self.backoff_base * (2**attempt))

    @staticmethod
    def _should_retry(status: int, content_type: str) -> bool:
        if status in _RETRYABLE_STATUS:
            return True
        # Cloudflare / rate-limit challenge pages often return 200 with HTML.
        return status == 200 and "text/html" in content_type.lower()

    def _determine_size(self) -> int:
        for attempt in range(self.max_retries + 1):
            try:
                with self.session.head(
                    self.url, allow_redirects=True, timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    if (length := response.headers.get("Content-Length")) is not None:
                        return int(length)
                break
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise
                logger.warning("HEAD failed for %s: %s; retrying", self.url, exc)
                self._backoff(attempt)

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
        range_header = f"bytes={start}-{stop}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": range_header},
                    timeout=self.timeout,
                    stream=True,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise
                logger.warning(
                    "Range request failed for %s: %s; retrying", self.url, exc
                )
                self._backoff(attempt)
                continue

            status = response.status_code
            content_type = response.headers.get("Content-Type", "")
            content_length = response.headers.get("Content-Length")

            if status == 206:
                self.cache[block_id] = response.content
                response.close()
                return self.cache[block_id]

            # Whole-file request: some servers reply 200 instead of 206 when the
            # requested range covers the entire (sub-block-size) file.
            if status == 200 and start == 0 and stop == self.size - 1:
                body = response.content
                response.close()
                if len(body) == self.size:
                    self.cache[block_id] = body
                    return body

            logger.warning(
                "Unexpected response for %s (range %s): status=%d "
                "content-type=%r content-length=%s",
                self.url,
                range_header,
                status,
                content_type,
                content_length,
            )
            response.close()

            if self._should_retry(status, content_type) and attempt < self.max_retries:
                self._backoff(attempt)
                continue

            raise RuntimeError(
                f"Server does not support HTTP Range requests (status {status})"
            )

        raise RuntimeError(
            f"Server does not support HTTP Range requests for {self.url}"
        )

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

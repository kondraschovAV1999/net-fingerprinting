"""Parse PCAP filenames into structured metadata."""

from __future__ import annotations

import re

from utils.schema.parsing import FilenameMetadata


def parse_filename(filename: str, pattern: str) -> FilenameMetadata:
    """Parse a PCAP filename into typed metadata.

    The pattern must define named groups: ``capture_timestamp``,
    ``application``, ``download_speed``, and ``upload_speed``.
    """
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match convention: {filename}")
    return FilenameMetadata(
        capture_timestamp=int(match.group("capture_timestamp")),
        application=match.group("application"),
        download_speed=float(match.group("download_speed")),
        upload_speed=float(match.group("upload_speed")),
        file_source=filename,
    )

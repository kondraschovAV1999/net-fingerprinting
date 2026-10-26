"""Config and data-record dataclasses for packet/filename parsing."""

from dataclasses import dataclass


@dataclass
class ParsingConfig:
    packet_limit: int = 100000


@dataclass
class FilenameMetadata:
    capture_timestamp: int
    application: str
    download_speed: float
    upload_speed: float
    file_source: str

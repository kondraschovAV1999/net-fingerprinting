"""Config and data-record dataclasses for packet/filename parsing."""

from dataclasses import dataclass

# Single source of truth for the PCAP filename convention. Named groups are
# consumed by ``data_loading.parsing.parse_filename.parse_filename``. The trailing
# ``_?`` tolerates an occasional stray underscore left by the capture tool
# before the ``.pcap`` extension (e.g. ``..._3.69-3.11_.pcap``).
DEFAULT_FILENAME_PATTERN = (
    r"^(?P<capture_timestamp>\d+)_(?P<application>.+)_"
    r"(?P<download_speed>\d+(?:\.\d+)?)[-_](?P<upload_speed>\d+(?:\.\d+)?)_?\.pcap$"
)


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

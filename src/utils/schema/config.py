"""Root config dataclass, composed of the per-concern config dataclasses."""

from dataclasses import dataclass, field
from typing import Any, List

from utils.schema.data_download import DatasetConfig, DownloadConfig
from utils.schema.extraction import ExtractConfig
from utils.schema.parsing import ParsingConfig
from utils.schema.paths import PathsConfig


@dataclass
class DataLoadingConfig:
    datasets: List[DatasetConfig] = field(default_factory=list)
    # None (all), a single dataset_id string, or a list of dataset_id strings.
    selected_dataset_ids: Any = None
    download: DownloadConfig = field(default_factory=DownloadConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    filename_pattern: str = (
        r"^(?P<capture_timestamp>\d+)_(?P<application>.+)_"
        r"(?P<download_speed>\d+(?:\.\d+)?)[-_](?P<upload_speed>\d+(?:\.\d+)?)\.pcap$"
    )

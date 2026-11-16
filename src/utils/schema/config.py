"""Root config dataclass, composed of the per-concern config dataclasses."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from utils.schema.data_download import DatasetConfig, DownloadConfig
from utils.schema.extraction import ExtractConfig
from utils.schema.parsing import DEFAULT_FILENAME_PATTERN, ParsingConfig
from utils.schema.paths import PathsConfig


@dataclass
class DataLoadingConfig:
    datasets: List[DatasetConfig] = field(default_factory=list)
    # None (all), a single dataset_id string, or a list of dataset_id strings.
    selected_dataset_ids: Any = None
    # None or [] = all applications (no filtering).
    applications: Optional[List[str]] = None
    download: DownloadConfig = field(default_factory=DownloadConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    filename_pattern: str = DEFAULT_FILENAME_PATTERN

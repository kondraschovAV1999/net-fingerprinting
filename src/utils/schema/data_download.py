"""Config dataclasses for downloading dataset archives."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    dataset_id: str
    name: str
    version: int = 1


@dataclass
class DownloadConfig:
    base_url: str = "https://api.data.mendeley.com/datasets"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
    archive_extensions: list[str] = field(default_factory=lambda: [".rar", ".zip"])
    chunk_size: int = 8192
    max_files: Optional[int] = None

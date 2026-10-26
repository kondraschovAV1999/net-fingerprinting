"""Config dataclass for output paths."""

from dataclasses import dataclass


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    combined_csv: str = "data/processed/combined.csv"

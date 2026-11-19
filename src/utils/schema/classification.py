"""Config dataclasses for the classification stage."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelSpec:
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationConfig:
    models: list[ModelSpec] = field(default_factory=list)
    zero_division: int = 0
    applications: Optional[list[str]] = None
    max_pcaps_per_app: Optional[int] = None
    combined_csv: str = "data/processed/combined.csv"
    metrics_csv: str = "data/processed/metrics.csv"
    models_dir: str = "data/processed/models"

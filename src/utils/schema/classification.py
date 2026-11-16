"""Config dataclasses for the classification stage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationConfig:
    models: list[ModelSpec] = field(default_factory=list)
    zero_division: int = 0
    feature_config: str = "statistical"
    rate_interval: float = 1.0
    interactive_bins: int = 20
    ratios: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    dataset_scope: str = "combined"
    combined_csv: str = "data/processed/combined.csv"
    metrics_csv: str = "data/processed/metrics.csv"

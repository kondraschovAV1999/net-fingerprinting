"""Config dataclass for the feature-extraction stage."""

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    feature_config: str = "statistical"
    rate_interval: float = 1.0
    interactive_bins: int = 20

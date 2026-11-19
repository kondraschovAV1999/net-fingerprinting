"""Config dataclass for the data-splitting stage."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataSplittingConfig:
    test_size: float = 0.15
    n_folds: Optional[int] = None
    seed: int = 42
    dataset_scope: str = "combined"

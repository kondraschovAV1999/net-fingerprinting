"""Top-level config for the classification training run."""

from dataclasses import dataclass, field

from utils.schema.classification import ClassificationConfig
from utils.schema.data_splitting import DataSplittingConfig
from utils.schema.preprocessing import PreprocessingConfig


@dataclass
class TrainConfig:
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    data_splitting: DataSplittingConfig = field(default_factory=DataSplittingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

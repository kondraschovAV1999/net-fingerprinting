"""Config dataclass for archive extraction."""

from dataclasses import dataclass


@dataclass
class ExtractConfig:
    overwrite: bool = False

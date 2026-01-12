from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataSplitConfig:
    root_dir: Path
    data_dir: Path

@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    data_dir: Path
    train: Path
    test: Path
    geocode_cache: Path
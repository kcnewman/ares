from pathlib import Path

import yaml

CONFIG_PATH = Path("config/config.yaml")
SCHEMA_PATH = Path("config/schema.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)

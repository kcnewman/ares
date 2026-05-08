import os
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/config.yaml")
SCHEMA_PATH = Path("config/schema.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if "LLM_SERVICE_URL" in os.environ:
        config["llm_service_url"] = os.environ["LLM_SERVICE_URL"]
    return config


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)

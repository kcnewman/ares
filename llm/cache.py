import json
from pathlib import Path


class JsonFileCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self._save()

    def get_all(self) -> dict:
        return dict(self._data)

    def has(self, key: str) -> bool:
        return key in self._data

    def set_batch(self, mapping: dict) -> None:
        self._data.update(mapping)
        self._save()

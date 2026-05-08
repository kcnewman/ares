"""Tests for lib/utils.py"""

import json

from lib.utils import load_json, save_json


class TestSaveAndLoadJson:
    def test_roundtrip(self, tmp_path):
        data = {"key": "value", "num": 42}
        path = tmp_path / "test.json"
        save_json(path, data)
        assert path.exists()
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "test.json"
        save_json(path, {"a": 1})
        assert path.exists()
        assert json.loads(path.read_text()) == {"a": 1}

from pathlib import Path

import pytest
import yaml

from inference.supervisor_agent.config_io import (
    read_global_config,
    read_service_config,
    resolve_service_config_path,
    write_global_config,
    write_service_config,
)


def test_write_and_read_service_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "download.yaml").write_text(
        yaml.safe_dump({"service_id": "download", "config": {"civitai_token": "old"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("inference.supervisor_agent.config_io.CONFIG_DIR", config_dir)

    merged = write_service_config("download", {"config": {"civitai_token": "new"}})
    assert merged[0]["service_id"] == "download"
    assert merged[0]["config"]["civitai_token"] == "new"

    loaded, _path = read_service_config("download")
    assert loaded["config"]["civitai_token"] == "new"


def test_read_service_config_by_service_id_field(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "service_text_qwen.yaml").write_text(
        yaml.safe_dump({"service_id": "service_text", "service_type": "text", "config": {"runtime": {"backend": "vllm"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("inference.supervisor_agent.config_io.CONFIG_DIR", config_dir)

    loaded, path = read_service_config("service_text")
    assert path == "service_text_qwen.yaml"
    assert loaded["service_type"] == "text"


def test_read_missing_service_config_returns_skeleton(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr("inference.supervisor_agent.config_io.CONFIG_DIR", config_dir)

    loaded, path = read_service_config("service_text")
    assert loaded == {"service_id": "service_text"}
    assert path == "service_text.yaml"
    assert resolve_service_config_path("service_text") is None


def test_write_global_config_merges(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "inference.yaml").write_text(
        yaml.safe_dump({"api_base_url": "http://old", "models_dir": "resources/models"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("inference.supervisor_agent.config_io.CONFIG_DIR", config_dir)

    merged = write_global_config({"api_base_url": "http://new"})
    assert merged["api_base_url"] == "http://new"
    assert merged["models_dir"] == "resources/models"
    assert read_global_config()["api_base_url"] == "http://new"

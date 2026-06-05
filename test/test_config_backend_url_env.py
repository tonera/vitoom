"""VITOOM_BACKEND_URL maps to server.public_base_url."""

import os

import pytest

from backend.core.config import ConfigManager


@pytest.fixture()
def reset_config_singleton():
    import backend.core.config as config_module

    config_module._config_manager = None
    yield
    config_module._config_manager = None


def test_vitoom_backend_url_sets_public_base_url(monkeypatch, reset_config_singleton):
    monkeypatch.setenv("VITOOM_BACKEND_URL", "http://192.168.1.10:8888/")
    manager = ConfigManager()
    assert manager.get("server.public_base_url") == "http://192.168.1.10:8888"


def test_empty_vitoom_backend_url_does_not_override_app_yaml(monkeypatch, reset_config_singleton, tmp_path):
    import backend.core.config as config_module

    app_yaml = tmp_path / "app.yaml"
    app_yaml.write_text("server:\n  public_base_url: http://from-yaml:9000\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "APP_CONFIG_FILE", app_yaml)
    monkeypatch.delenv("VITOOM_BACKEND_URL", raising=False)

    manager = ConfigManager()
    assert manager.get("server.public_base_url") == "http://from-yaml:9000"

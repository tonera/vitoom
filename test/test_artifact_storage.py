"""artifact_storage 工具测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.artifact_storage import (
    normalize_storage_for_write,
    normalize_storage_label,
    resolve_artifact_public_url,
)


def test_normalize_storage_for_write():
    assert normalize_storage_for_write("local") == "server"
    assert normalize_storage_for_write("") == "server"
    assert normalize_storage_for_write("s3") == "s3"
    assert normalize_storage_for_write("oss") == "oss"
    assert normalize_storage_for_write("server") == "server"


def test_normalize_storage_label():
    assert normalize_storage_label("local") == "local"
    assert normalize_storage_label("server") == "server"
    assert normalize_storage_label("bad") == "local"


def test_resolve_artifact_public_url_local():
    assert resolve_artifact_public_url("local", "a/b.png") is None
    assert resolve_artifact_public_url("server", "") is None


def test_resolve_artifact_public_url_server_relative(monkeypatch):
    monkeypatch.setattr(
        "backend.utils.artifact_storage.get_config",
        lambda key, default=None: "" if key == "server.public_base_url" else default,
    )
    url = resolve_artifact_public_url("server", "2026/06/03/x.png")
    assert url == "/outputs/2026/06/03/x.png"


def test_resolve_artifact_public_url_server_absolute(monkeypatch):
    def fake_get(key, default=None):
        if key == "server.public_base_url":
            return "https://api.example.com"
        if key == "storage.local.http_base_url":
            return "/outputs"
        return default

    monkeypatch.setattr("backend.utils.artifact_storage.get_config", fake_get)
    url = resolve_artifact_public_url("server", "uploads/202601/a.png")
    assert url == "https://api.example.com/outputs/uploads/202601/a.png"


def test_resolve_artifact_public_url_s3(monkeypatch):
    monkeypatch.setattr(
        "backend.utils.artifact_storage.get_config",
        lambda key, default=None: "https://bucket.s3.amazonaws.com"
        if key == "storage.s3.public_base_url"
        else default,
    )
    url = resolve_artifact_public_url("s3", "k/f.png")
    assert url == "https://bucket.s3.amazonaws.com/k/f.png"

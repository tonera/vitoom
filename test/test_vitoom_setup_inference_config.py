"""Tests for local inference.yaml generation during setup."""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vitoom_setup.inference_config import (  # noqa: E402
    render_local_inference_yaml,
    write_local_inference_yaml,
)


def test_render_local_inference_yaml_uses_env_values():
    content = render_local_inference_yaml(
        {
            "VITOOM_BACKEND_URL": "http://192.168.1.10:8888",
            "VITOOM_WS_URL": "ws://192.168.1.10:8888",
            "VITOOM_INFERENCE_UPLOAD_AUTH_SECRET": "test-secret-value",
            "VITOOM_PIPELINE_CACHE_TTL_SECONDS": "300",
        }
    )
    cfg = yaml.safe_load(content)

    assert cfg["models_dir"] == "resources/models"
    assert cfg["weights_dir"] == "resources/weights"
    assert cfg["pipeline_cache_ttl_seconds"] == 300
    assert cfg["api_base_url"] == "http://192.168.1.10:8888"
    assert cfg["ws_url"] == "ws://192.168.1.10:8888"
    assert cfg["storage"]["server"]["auth"]["secret"] == "test-secret-value"


def test_write_local_inference_yaml_creates_file_when_missing(tmp_path):
    env = {
        "VITOOM_BACKEND_URL": "http://10.0.0.1:8888",
        "VITOOM_WS_URL": "ws://10.0.0.1:8888",
        "VITOOM_INFERENCE_UPLOAD_AUTH_SECRET": "secret",
    }

    written = write_local_inference_yaml(env, repo_root=tmp_path)

    assert written == (tmp_path / "inference" / "config" / "inference.yaml").resolve()
    assert written.is_file()


def test_write_local_inference_yaml_skips_existing_file(tmp_path):
    path = tmp_path / "inference" / "config" / "inference.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("models_dir: resources/custom\n", encoding="utf-8")

    written = write_local_inference_yaml(
        {"VITOOM_BACKEND_URL": "http://10.0.0.1:8888"},
        repo_root=tmp_path,
    )

    assert written is None
    assert path.read_text(encoding="utf-8") == "models_dir: resources/custom\n"

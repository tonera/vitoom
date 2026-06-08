from pathlib import Path
import sys
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vitoom_setup.docker_images import (  # noqa: E402
    _docker_pull_supports_plain_progress,
    components_to_env_keys,
    docker_pull,
    format_size,
    resolve_image_ref,
    tar_path,
)


def test_tar_path_for_aarch64_backend():
    root = Path("/repo")
    assert tar_path(root, "aarch64", "VITOOM_BACKEND_IMAGE") == (
        root / "images" / "aarch64" / "vitoom-backend-latest-aarch64.tar"
    )


def test_components_to_env_keys_order():
    keys = components_to_env_keys({"mini", "backend", "visual"})
    assert keys == [
        "VITOOM_BACKEND_IMAGE",
        "VITOOM_VISUAL_IMAGE",
        "VITOOM_MINI_IMAGE",
    ]


def test_format_size_gb():
    assert format_size(11 * 1024**3).endswith("GB")


def test_resolve_image_ref_prefers_env():
    env = {"VITOOM_BACKEND_IMAGE": "custom/backend:tag"}
    assert resolve_image_ref(env, "x86_64", "VITOOM_BACKEND_IMAGE") == "custom/backend:tag"


def test_docker_pull_omits_progress_on_old_cli():
    import vitoom_setup.docker_images as docker_images

    docker_images._PULL_PLAIN_PROGRESS = None
    with patch(
        "vitoom_setup.docker_images.subprocess.run",
        side_effect=[
            type("R", (), {"stdout": "Usage: docker pull", "stderr": ""})(),
            type("R", (), {"returncode": 0})(),
        ],
    ) as run:
        docker_pull("tonera/vitoom-backend:latest-aarch64")
    assert run.call_args_list[-1].args[0] == [
        "docker",
        "pull",
        "tonera/vitoom-backend:latest-aarch64",
    ]


def test_docker_pull_uses_plain_progress_when_supported():
    import vitoom_setup.docker_images as docker_images

    docker_images._PULL_PLAIN_PROGRESS = None
    with patch(
        "vitoom_setup.docker_images.subprocess.run",
        side_effect=[
            type("R", (), {"stdout": "  --progress string", "stderr": ""})(),
            type("R", (), {"returncode": 0})(),
        ],
    ) as run:
        docker_pull("tonera/vitoom-backend:latest-aarch64")
    assert run.call_args_list[-1].args[0] == [
        "docker",
        "pull",
        "--progress=plain",
        "tonera/vitoom-backend:latest-aarch64",
    ]


def test_docker_pull_supports_plain_progress_is_cached():
    import vitoom_setup.docker_images as docker_images

    docker_images._PULL_PLAIN_PROGRESS = None
    with patch(
        "vitoom_setup.docker_images.subprocess.run",
        return_value=type("R", (), {"stdout": "  --progress string", "stderr": ""})(),
    ) as run:
        assert _docker_pull_supports_plain_progress() is True
        assert _docker_pull_supports_plain_progress() is True
    assert run.call_count == 1

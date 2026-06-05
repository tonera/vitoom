from pathlib import Path

from vitoom_setup.docker_images import (
    components_to_env_keys,
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

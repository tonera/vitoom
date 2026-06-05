"""Shared constants for Vitoom setup."""

from __future__ import annotations

BUILD_COMPONENT_IDS = ("backend", "visual", "mini", "audio")
INSTALL_COMPONENT_IDS = ("backend", "visual", "text", "audio", "mini", "download")
DEPLOY_INFERENCE_IDS = ("visual", "text", "audio", "download", "mini")
CUDA_EXEMPT_INFERENCE = frozenset({"download"})

SUPERVISOR_PORTS: dict[str, int] = {
    "visual": 9001,
    "text": 9002,
    "audio": 9003,
    "download": 9004,
    "mini": 9005,
}

SUPERVISOR_ENV_KEYS: dict[str, str] = {
    "visual": "VITOOM_VISUAL_SUPERVISOR_URL",
    "text": "VITOOM_TEXT_SUPERVISOR_URL",
    "audio": "VITOOM_AUDIO_SUPERVISOR_URL",
    "download": "VITOOM_DOWNLOAD_SUPERVISOR_URL",
    "mini": "VITOOM_MINI_SUPERVISOR_URL",
}

SECRET_PLACEHOLDERS = frozenset(
    {
        "",
        "Please fill in the uploaded key.",
        "Please enter a random long string.",
    }
)
MIN_SECRET_LENGTH = 24

DEFAULT_ADMIN_PASSWORD_PLACEHOLDER = "admin123456"
DEFAULT_ADMIN_PASSWORD_LENGTH = 10

DOCKERHUB_USER = "tonera"

CN_MIRROR = {
    "APT_MIRROR": "https://mirrors.aliyun.com/debian",
    "PIP_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/",
}


def hub_image(name: str, tag: str) -> str:
    return f"{DOCKERHUB_USER}/{name}:{tag}"


ARCH_IMAGE_TAGS: dict[str, dict[str, str]] = {
    "x86_64": {
        "VITOOM_BACKEND_IMAGE": hub_image("vitoom-backend", "latest-x86_64"),
        "VITOOM_VISUAL_IMAGE": hub_image(
            "vitoom-inference-visual", "experimental-cu130-torch2.11-x86_64"
        ),
        "VITOOM_TEXT_IMAGE": hub_image(
            "vitoom-inference-text", "experimental-cu130-torch2.11-x86_64"
        ),
        "VITOOM_AUDIO_IMAGE": hub_image(
            "vitoom-inference-audio", "experimental-cu130-torch2.9.1-x86_64"
        ),
        "VITOOM_MINI_IMAGE": hub_image(
            "vitoom-inference-mini", "experimental-cu130-torch2.11-x86_64"
        ),
        "VITOOM_DOWNLOAD_IMAGE": hub_image(
            "vitoom-inference-download", "experimental-x86_64"
        ),
    },
    "aarch64": {
        "VITOOM_BACKEND_IMAGE": hub_image("vitoom-backend", "latest-aarch64"),
        "VITOOM_VISUAL_IMAGE": hub_image(
            "vitoom-inference-visual",
            "experimental-cu130-torch2.11-aarch64-nvidia-spark",
        ),
        "VITOOM_TEXT_IMAGE": hub_image(
            "vitoom-inference-text",
            "experimental-cu130-torch2.11-aarch64-nvidia-spark",
        ),
        "VITOOM_AUDIO_IMAGE": hub_image(
            "vitoom-inference-audio",
            "experimental-cu130-torch2.9.1-aarch64-nvidia-spark",
        ),
        "VITOOM_MINI_IMAGE": hub_image(
            "vitoom-inference-mini",
            "experimental-cu130-torch2.11-aarch64-nvidia-spark",
        ),
        "VITOOM_DOWNLOAD_IMAGE": hub_image(
            "vitoom-inference-download", "experimental-aarch64-nvidia-spark"
        ),
    },
}

DEFAULT_SERVER_PORT = 8888

IMAGE_ENV_KEYS = (
    "VITOOM_BACKEND_IMAGE",
    "VITOOM_VISUAL_IMAGE",
    "VITOOM_TEXT_IMAGE",
    "VITOOM_AUDIO_IMAGE",
    "VITOOM_MINI_IMAGE",
    "VITOOM_DOWNLOAD_IMAGE",
)

COMPONENT_TO_IMAGE_ENV: dict[str, str] = {
    "backend": "VITOOM_BACKEND_IMAGE",
    "visual": "VITOOM_VISUAL_IMAGE",
    "text": "VITOOM_TEXT_IMAGE",
    "audio": "VITOOM_AUDIO_IMAGE",
    "mini": "VITOOM_MINI_IMAGE",
    "download": "VITOOM_DOWNLOAD_IMAGE",
}

ARCH_IMAGE_TARS: dict[str, dict[str, str]] = {
    "x86_64": {
        "VITOOM_BACKEND_IMAGE": "vitoom-backend-latest-x86_64.tar",
        "VITOOM_VISUAL_IMAGE": "vitoom-inference-visual-cu130-torch2.11-x86_64.tar",
        "VITOOM_TEXT_IMAGE": "vitoom-inference-text-cu130-torch2.11-x86_64.tar",
        "VITOOM_AUDIO_IMAGE": "vitoom-inference-audio-cu130-torch2.9.1-x86_64.tar",
        "VITOOM_MINI_IMAGE": "vitoom-inference-mini-cu130-torch2.11-x86_64.tar",
        "VITOOM_DOWNLOAD_IMAGE": "vitoom-inference-download-experimental-x86_64.tar",
    },
    "aarch64": {
        "VITOOM_BACKEND_IMAGE": "vitoom-backend-latest-aarch64.tar",
        "VITOOM_VISUAL_IMAGE": "vitoom-inference-visual-cu130-torch2.11-aarch64-nvidia-spark.tar",
        "VITOOM_TEXT_IMAGE": "vitoom-inference-text-cu130-torch2.11-aarch64-nvidia-spark.tar",
        "VITOOM_AUDIO_IMAGE": "vitoom-inference-audio-cu130-torch2.9.1-aarch64-nvidia-spark.tar",
        "VITOOM_MINI_IMAGE": "vitoom-inference-mini-cu130-torch2.11-aarch64-nvidia-spark.tar",
        "VITOOM_DOWNLOAD_IMAGE": "vitoom-inference-download-experimental-aarch64.tar",
    },
}

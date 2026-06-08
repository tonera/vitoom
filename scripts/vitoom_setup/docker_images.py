"""Load or pull Vitoom Docker images for deployment."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from vitoom_setup.constants import (
    ARCH_IMAGE_TAGS,
    ARCH_IMAGE_TARS,
    COMPONENT_TO_IMAGE_ENV,
    IMAGE_ENV_KEYS,
)

EmitFn = Callable[..., str]
LogFn = Callable[[str], None]

IMAGE_ENV_TO_COMPONENT = {value: key for key, value in COMPONENT_TO_IMAGE_ENV.items()}


def resolve_arch(env: dict[str, str], *, fallback: str) -> str:
    arch = (env.get("VITOOM_TARGET_ARCH") or "").strip()
    if arch in ARCH_IMAGE_TAGS:
        return arch
    return fallback


def resolve_image_ref(env: dict[str, str], arch: str, env_key: str) -> str:
    value = (env.get(env_key) or "").strip()
    if value:
        return value
    return ARCH_IMAGE_TAGS[arch][env_key]


def tar_path(repo_root: Path, arch: str, env_key: str) -> Path:
    return repo_root / "images" / arch / ARCH_IMAGE_TARS[arch][env_key]


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.1f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def docker_image_exists(image_ref: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def docker_load(tar_file: Path) -> None:
    subprocess.run(["docker", "load", "-i", str(tar_file)], check=True)


_PULL_PLAIN_PROGRESS: bool | None = None


def _docker_pull_supports_plain_progress() -> bool:
    """Docker Engine 23+ supports `docker pull --progress=plain`; older CLIs do not."""
    global _PULL_PLAIN_PROGRESS
    if _PULL_PLAIN_PROGRESS is not None:
        return _PULL_PLAIN_PROGRESS
    result = subprocess.run(
        ["docker", "pull", "--help"],
        capture_output=True,
        text=True,
    )
    help_text = f"{result.stdout}\n{result.stderr}"
    _PULL_PLAIN_PROGRESS = "--progress" in help_text
    return _PULL_PLAIN_PROGRESS


def docker_pull(image_ref: str) -> None:
    cmd = ["docker", "pull", image_ref]
    if _docker_pull_supports_plain_progress():
        cmd = ["docker", "pull", "--progress=plain", image_ref]
    subprocess.run(cmd, check=True)


def _say(log: LogFn | None, emit: EmitFn | None, key: str, **kwargs: object) -> None:
    if emit is None and log is None:
        return
    message = emit(key, **kwargs) if emit is not None else key
    if log is not None:
        log(message)
    else:
        print(message, flush=True)


def ensure_docker_image(
    repo_root: Path,
    env: dict[str, str],
    arch: str,
    env_key: str,
    *,
    skip_if_present: bool = True,
    current: int = 1,
    total: int = 1,
    log: LogFn | None = None,
    emit: EmitFn | None = None,
) -> str:
    """Return action: skipped | loaded | pulled."""
    image_ref = resolve_image_ref(env, arch, env_key)
    component = IMAGE_ENV_TO_COMPONENT.get(env_key, env_key)

    if skip_if_present and docker_image_exists(image_ref):
        _say(
            log,
            emit,
            "setup.status.docker_image_skip",
            current=current,
            total=total,
            component=component,
            image=image_ref,
        )
        return "skipped"

    local_tar = tar_path(repo_root, arch, env_key)
    _say(
        log,
        emit,
        "setup.status.docker_image_begin",
        current=current,
        total=total,
        component=component,
        image=image_ref,
    )

    started = time.monotonic()
    if local_tar.is_file():
        tar_size = format_size(local_tar.stat().st_size)
        _say(
            log,
            emit,
            "setup.status.docker_image_tar",
            path=str(local_tar),
            size=tar_size,
        )
        docker_load(local_tar)
        action = "loaded"
    else:
        _say(log, emit, "setup.status.docker_image_hub", image=image_ref)
        docker_pull(image_ref)
        action = "pulled"

    elapsed = time.monotonic() - started
    action_label = (
        emit(f"setup.status.docker_image_action_{action}")
        if emit is not None
        else action
    )
    _say(
        log,
        emit,
        "setup.status.docker_image_done",
        action=action_label,
        image=image_ref,
        seconds=f"{elapsed:.1f}",
    )
    return action


def components_to_env_keys(components: set[str]) -> list[str]:
    keys: list[str] = []
    for component_id in COMPONENT_TO_IMAGE_ENV:
        if component_id in components:
            keys.append(COMPONENT_TO_IMAGE_ENV[component_id])
    return keys


def ensure_docker_images(
    repo_root: Path,
    env: dict[str, str],
    arch: str,
    *,
    components: set[str] | None = None,
    skip_if_present: bool = True,
    log: LogFn | None = None,
    emit: EmitFn | None = None,
) -> dict[str, str]:
    """Ensure images for selected components. Returns env_key -> action."""
    if components is None:
        env_keys = list(IMAGE_ENV_KEYS)
    else:
        env_keys = components_to_env_keys(components)
        if not env_keys:
            return {}

    total = len(env_keys)
    results: dict[str, str] = {}
    for index, env_key in enumerate(env_keys, start=1):
        results[env_key] = ensure_docker_image(
            repo_root,
            env,
            arch,
            env_key,
            skip_if_present=skip_if_present,
            current=index,
            total=total,
            log=log,
            emit=emit,
        )
    return results

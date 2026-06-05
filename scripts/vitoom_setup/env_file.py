"""Read and update the deployment .env file."""

from __future__ import annotations

import secrets
import shutil
import string
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from vitoom_setup.constants import (
    DEFAULT_ADMIN_PASSWORD_LENGTH,
    DEFAULT_ADMIN_PASSWORD_PLACEHOLDER,
    MIN_SECRET_LENGTH,
    SECRET_PLACEHOLDERS,
)


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def is_placeholder_secret(value: str | None) -> bool:
    if value is None:
        return True
    trimmed = value.strip()
    if trimmed in SECRET_PLACEHOLDERS:
        return True
    return len(trimmed) < MIN_SECRET_LENGTH


def generate_upload_secret() -> str:
    return secrets.token_urlsafe(32)


def resolve_upload_secret(existing: dict[str, str]) -> tuple[str, bool]:
    """Return (secret, was_preserved)."""
    current = existing.get("VITOOM_INFERENCE_UPLOAD_AUTH_SECRET")
    if not is_placeholder_secret(current):
        return current or "", True
    return generate_upload_secret(), False


def is_placeholder_admin_password(value: str | None) -> bool:
    if value is None:
        return True
    trimmed = value.strip()
    if not trimmed:
        return True
    return trimmed == DEFAULT_ADMIN_PASSWORD_PLACEHOLDER


def generate_admin_password() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(DEFAULT_ADMIN_PASSWORD_LENGTH))


def resolve_admin_password(existing: dict[str, str]) -> tuple[str, bool]:
    """Return (password, was_preserved)."""
    current = existing.get("DEFAULT_ADMIN_PASSWORD")
    if not is_placeholder_admin_password(current):
        return (current or "").strip(), True
    return generate_admin_password(), False


def ensure_env_file(env_path: Path, example_path: Path) -> None:
    if env_path.is_file():
        return
    if not example_path.is_file():
        raise FileNotFoundError(f"Missing template: {example_path}")
    shutil.copy2(example_path, env_path)


def backup_env(env_path: Path) -> Path | None:
    if not env_path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = env_path.with_name(f"{env_path.name}.bak.{stamp}")
    shutil.copy2(env_path, backup)
    return backup


def upsert_env_file(env_path: Path, updates: dict[str, str]) -> None:
    """Merge key=value pairs into .env, preserving comments and unrelated keys."""
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    new_lines: list[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            new_lines.append(f"{key}={remaining.pop(key)}")
            seen.add(key)
        else:
            new_lines.append(line)

    if remaining:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        for key in sorted(remaining.keys()):
            new_lines.append(f"{key}={remaining[key]}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def backend_url_to_ws_url(backend_url: str) -> str:
    url = backend_url.strip()
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    raise ValueError(f"Unsupported backend URL scheme: {backend_url}")


def supervisor_url(host: str, component: str) -> str:
    from vitoom_setup.constants import SUPERVISOR_PORTS

    port = SUPERVISOR_PORTS[component]
    return f"http://{host}:{port}"


def load_env_state(env_path: Path, example_path: Path) -> dict[str, str]:
    ensure_env_file(env_path, example_path)
    return parse_env_file(env_path)


BACKEND_URL_DEFAULT_INPUTS = frozenset({"", "1", "y", "yes", "ok", "default"})


def parse_backend_url(value: str) -> str | None:
    """Return normalized URL if valid, else None."""
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.hostname:
        return url
    return None


def resolve_backend_url_from_input(raw: str, default_url: str) -> str | None:
    """Resolve user input against a default URL. None means invalid (re-prompt)."""
    if raw.strip().lower() in BACKEND_URL_DEFAULT_INPUTS:
        return parse_backend_url(default_url) or default_url.rstrip("/")
    return parse_backend_url(raw)


def is_valid_backend_url(value: str | None) -> bool:
    if not value or not value.strip():
        return False
    trimmed = value.strip()
    if "127.0.0.1" in trimmed or "localhost" in trimmed.lower():
        return False
    return parse_backend_url(trimmed) is not None


def supervisor_env_updates(host: str, selected: set[str]) -> dict[str, str]:
    from vitoom_setup.constants import DEPLOY_INFERENCE_IDS, SUPERVISOR_ENV_KEYS

    updates: dict[str, str] = {}
    for component in DEPLOY_INFERENCE_IDS:
        key = SUPERVISOR_ENV_KEYS[component]
        if component in selected:
            updates[key] = supervisor_url(host, component)
        else:
            updates[key] = ""
    return updates


def selection_to_build_components(selected: set[str]) -> set[str]:
    """Map install selection to manifest build component ids (for artifact download)."""
    build: set[str] = set()
    if "backend" in selected:
        build.add("backend")
    if "visual" in selected:
        build.add("visual")
    if "mini" in selected:
        build.add("mini")
    if "audio" in selected:
        build.add("audio")
    return build

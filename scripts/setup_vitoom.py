#!/usr/bin/env python3
"""Interactive Vitoom setup: configure .env and optionally download Docker build artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from backend.i18n.locale import detect_cli_locale  # noqa: E402
from backend.i18n.translator import t  # noqa: E402
from vitoom_setup import REPO_ROOT as ROOT  # noqa: E402
from vitoom_setup.build_artifacts import (  # noqa: E402
    detect_arch,
    maybe_download_mini_model,
    run_build_download,
)
from vitoom_setup.constants import (  # noqa: E402
    CUDA_EXEMPT_INFERENCE,
    DEFAULT_SERVER_PORT,
    DEPLOY_INFERENCE_IDS,
    INSTALL_COMPONENT_IDS,
)
from vitoom_setup.cuda import cuda_is_available  # noqa: E402
from vitoom_setup.env_file import (  # noqa: E402
    BACKEND_URL_DEFAULT_INPUTS,
    backend_url_to_ws_url,
    backup_env,
    is_placeholder_secret,
    is_valid_backend_url,
    load_env_state,
    resolve_backend_url_from_input,
    resolve_admin_password,
    resolve_upload_secret,
    selection_to_build_components,
    supervisor_env_updates,
    upsert_env_file,
)
from vitoom_setup.network import is_port_in_use, pick_ipv4  # noqa: E402
from vitoom_setup.region import (  # noqa: E402
    locale_from_env_value,
    region_env_updates,
    region_from_env_value,
    region_from_locale,
)

ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"


def _print(msg: str) -> None:
    print(msg)


def _prompt(msg: str) -> str:
    return input(msg)


def _merge_env(env: dict[str, str], updates: dict[str, str]) -> None:
    env.update(updates)


def prompt_locale(hint_locale: str) -> str:
    _print(t("setup.prompt.select_locale", hint_locale))
    _print(t("setup.prompt.select_locale_zh", hint_locale))
    _print(t("setup.prompt.select_locale_ja", hint_locale))
    _print(t("setup.prompt.select_locale_en", hint_locale))
    while True:
        raw = _prompt("> ").strip().lower()
        if raw in ("1", "zh", "zh-cn", "cn", "china"):
            return "zh-CN"
        if raw in ("2", "ja", "ja-jp", "jp", "japan"):
            return "ja-JP"
        if raw in ("3", "en", "en-us", "intl", "global", ""):
            return "en-US"
        _print(t("setup.error.invalid_choice", hint_locale))


def prompt_region(locale: str) -> str:
    _print(t("setup.prompt.select_region", locale))
    _print(t("setup.prompt.select_region_cn", locale))
    _print(t("setup.prompt.select_region_intl", locale))
    while True:
        raw = _prompt("> ").strip().lower()
        if raw in ("1", "cn", "china"):
            return "cn"
        if raw in ("2", "intl", "global", ""):
            return "intl"
        _print(t("setup.error.invalid_choice", locale))


def resolve_locale_region(env: dict[str, str]) -> tuple[str, str]:
    """Return (locale, region). Prompt when values are missing in env."""
    hint_locale = detect_cli_locale()

    locale_from_file = locale_from_env_value(env.get("VITOOM_LOCALE"))
    locale_from_shell = locale_from_env_value(os.environ.get("VITOOM_LOCALE"))
    locale = locale_from_file or locale_from_shell
    if not locale:
        locale = prompt_locale(hint_locale)
        env["VITOOM_LOCALE"] = locale

    region_from_file = region_from_env_value(env.get("VITOOM_REGION"))
    if region_from_file:
        return locale, region_from_file

    region_from_shell = region_from_env_value(os.environ.get("VITOOM_REGION"))
    if region_from_shell:
        _merge_env(env, region_env_updates(region_from_shell))
        return locale, region_from_shell

    if locale_from_file or locale_from_shell:
        region = region_from_locale(locale)
        env.setdefault("VITOOM_REGION", region)
        return locale, region

    region = prompt_region(locale)
    _merge_env(env, region_env_updates(region))
    return locale, region


def expand_selection(tokens: set[str], locale: str) -> set[str]:
    if "all" in tokens:
        return {"backend", *DEPLOY_INFERENCE_IDS}
    unknown = tokens - set(INSTALL_COMPONENT_IDS) - {"all"}
    if unknown:
        raise SystemExit(
            t("setup.error.unknown_component", locale, component=", ".join(sorted(unknown)))
        )
    if not tokens - {"all"}:
        raise SystemExit(t("setup.error.no_component_selected", locale))
    return tokens - {"all"}


def prompt_install_components(locale: str) -> set[str]:
    _print(t("setup.prompt.select_install", locale))
    labels = {
        "backend": t("setup.install.backend", locale),
        "visual": t("setup.install.visual", locale),
        "text": t("setup.install.text", locale),
        "audio": t("setup.install.audio", locale),
        "mini": t("setup.install.mini", locale),
        "download": t("setup.install.download", locale),
    }
    for idx, component_id in enumerate(INSTALL_COMPONENT_IDS, start=1):
        _print(f"  [{idx}] {component_id} — {labels[component_id]}")
    _print(t("setup.prompt.select_all", locale))
    raw = _prompt("> ").strip().lower()
    if not raw or raw in {"a", "all", "*"}:
        return expand_selection({"all"}, locale)
    selected: set[str] = set()
    tokens = [token.strip() for token in raw.replace(" ", ",").split(",") if token.strip()]
    name_map = {str(i): cid for i, cid in enumerate(INSTALL_COMPONENT_IDS, start=1)}
    for token in tokens:
        if token in name_map:
            selected.add(name_map[token])
        elif token in INSTALL_COMPONENT_IDS or token == "all":
            selected.add(token)
        else:
            raise SystemExit(t("setup.error.unknown_component", locale, component=token))
    return expand_selection(selected, locale)


def inference_needs_cuda(selected: set[str]) -> bool:
    gpu_inference = set(DEPLOY_INFERENCE_IDS) - CUDA_EXEMPT_INFERENCE
    return bool(selected & gpu_inference)


def require_cuda(locale: str) -> None:
    if cuda_is_available():
        _print(t("setup.status.cuda_ok", locale))
        return
    raise SystemExit(t("setup.error.cuda_required", locale))


def prompt_server_port(locale: str, env: dict[str, str]) -> int:
    raw_port = env.get("VITOOM_SERVER_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_SERVER_PORT
    except ValueError:
        port = DEFAULT_SERVER_PORT

    while True:
        if is_port_in_use(port):
            _print(t("setup.error.port_in_use", locale, port=port))
            raw = _prompt(t("setup.prompt.enter_port", locale)).strip()
            if not raw:
                port += 1
                continue
            try:
                port = int(raw)
            except ValueError:
                _print(t("setup.error.invalid_port", locale))
            continue

        raw = _prompt(t("setup.prompt.port_available", locale, port=port)).strip()
        if not raw:
            return port
        try:
            port = int(raw)
        except ValueError:
            _print(t("setup.error.invalid_port", locale))
            continue
        if not 1 <= port <= 65535:
            _print(t("setup.error.invalid_port", locale))


def pick_host_ip(locale: str) -> str:
    return pick_ipv4(
        _prompt,
        _print,
        empty_message=t("setup.error.no_ip", locale),
        select_message=t("setup.prompt.select_ip", locale),
        manual_message=t("setup.prompt.manual_ip", locale),
    )


def confirm_backend_url(locale: str, default_url: str) -> str:
    while True:
        _print(t("setup.prompt.confirm_backend_url", locale, url=default_url))
        raw = _prompt("> ").strip()
        resolved = resolve_backend_url_from_input(raw, default_url)
        if resolved:
            return resolved
        _print(t("setup.error.invalid_backend_url", locale))


def resolve_backend_url(locale: str, env: dict[str, str], *, host_ip: str, port: int) -> str:
    existing = env.get("VITOOM_BACKEND_URL", "")
    if is_valid_backend_url(existing):
        _print(t("setup.prompt.backend_url_existing", locale, url=existing))
        raw = _prompt("> ").strip()
        if raw.strip().lower() in BACKEND_URL_DEFAULT_INPUTS:
            return existing.rstrip("/")
        if raw:
            return confirm_backend_url(locale, raw)
        return existing.rstrip("/")

    default = f"http://{host_ip}:{port}"
    return confirm_backend_url(locale, default)


def resolve_secret(locale: str, env: dict[str, str], *, has_backend: bool) -> str:
    current = env.get("VITOOM_INFERENCE_UPLOAD_AUTH_SECRET", "")
    if not is_placeholder_secret(current):
        _print(t("setup.status.secret_preserved", locale))
        return current

    if not has_backend:
        _print(t("setup.prompt.secret_paste", locale))
        while True:
            pasted = _prompt("> ").strip()
            if not is_placeholder_secret(pasted):
                return pasted
            _print(t("setup.error.secret_too_short", locale))

    secret, _ = resolve_upload_secret(env)
    _print(t("setup.status.secret_generated", locale))
    return secret


def arch_env_updates(arch: str) -> dict[str, str]:
    from vitoom_setup.constants import ARCH_IMAGE_TAGS

    tags = ARCH_IMAGE_TAGS.get(arch, ARCH_IMAGE_TAGS["x86_64"])
    return {"VITOOM_TARGET_ARCH": arch, **tags}


def configure_env(
    locale: str,
    region: str,
    env: dict[str, str],
    selected: set[str],
) -> None:
    has_backend = "backend" in selected
    inference = selected & set(DEPLOY_INFERENCE_IDS)
    has_inference = bool(inference)

    if inference_needs_cuda(inference):
        require_cuda(locale)
    elif "download" in inference:
        _print(t("setup.status.cuda_skipped_download", locale))

    host_ip: str | None = None
    port = DEFAULT_SERVER_PORT

    if has_backend:
        port = prompt_server_port(locale, env)
        host_ip = pick_host_ip(locale)
        backend_url = resolve_backend_url(locale, env, host_ip=host_ip, port=port)
        _merge_env(
            env,
            {
                "VITOOM_SERVER_PORT": str(port),
                "VITOOM_BACKEND_URL": backend_url,
                "VITOOM_WS_URL": backend_url_to_ws_url(backend_url),
            },
        )
    elif has_inference:
        if not is_valid_backend_url(env.get("VITOOM_BACKEND_URL")):
            _print(t("setup.prompt.backend_url_remote", locale))
            while True:
                raw = _prompt("> ").strip()
                if raw:
                    backend_url = confirm_backend_url(locale, raw)
                    break
                _print(t("setup.error.backend_url_required", locale))
        else:
            backend_url = env["VITOOM_BACKEND_URL"].rstrip("/")
        _merge_env(
            env,
            {
                "VITOOM_BACKEND_URL": backend_url,
                "VITOOM_WS_URL": backend_url_to_ws_url(backend_url),
            },
        )
        host_ip = pick_host_ip(locale)

    secret_needed = has_backend or has_inference
    if secret_needed:
        secret = resolve_secret(locale, env, has_backend=has_backend)
        env["VITOOM_INFERENCE_UPLOAD_AUTH_SECRET"] = secret

    arch = detect_arch(locale)
    _print(t("setup.status.detected_arch", locale, arch=arch))
    _merge_env(env, arch_env_updates(arch))

    if region == "cn" and not env.get("VITOOM_WHEEL_BASE_URL", "").strip():
        _print(t("setup.prompt.wheel_base_url", locale))
        raw = _prompt("> ").strip()
        if raw:
            env["VITOOM_WHEEL_BASE_URL"] = raw

    if has_inference:
        if host_ip is None:
            host_ip = pick_host_ip(locale)
        _merge_env(env, supervisor_env_updates(host_ip, inference))
    else:
        _merge_env(env, supervisor_env_updates("127.0.0.1", set()))


def write_env_file(locale: str, env: dict[str, str]) -> None:
    from vitoom_setup.constants import SUPERVISOR_ENV_KEYS

    updates: dict[str, str] = {"VITOOM_LOCALE": locale}

    managed = {
        "VITOOM_LOCALE",
        "VITOOM_REGION",
        "APT_MIRROR",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "VITOOM_WHEEL_BASE_URL",
        "VITOOM_TARGET_ARCH",
        "VITOOM_SERVER_PORT",
        "VITOOM_BACKEND_URL",
        "VITOOM_WS_URL",
        "VITOOM_INFERENCE_UPLOAD_AUTH_SECRET",
        "DEFAULT_ADMIN_PASSWORD",
        "VITOOM_BACKEND_IMAGE",
        "VITOOM_VISUAL_IMAGE",
        "VITOOM_TEXT_IMAGE",
        "VITOOM_AUDIO_IMAGE",
        "VITOOM_MINI_IMAGE",
        "VITOOM_DOWNLOAD_IMAGE",
        *SUPERVISOR_ENV_KEYS.values(),
    }
    for key in managed:
        if key in env:
            updates[key] = env[key]

    backup = backup_env(ENV_PATH)
    if backup:
        _print(t("setup.status.env_backup", locale, path=str(backup.relative_to(ROOT))))
    upsert_env_file(ENV_PATH, updates)
    _print(t("setup.status.env_written", locale, path=".env"))


def maybe_write_local_inference_config(locale: str, env: dict[str, str], selected: set[str]) -> None:
    if not (selected & {"backend", *DEPLOY_INFERENCE_IDS}):
        return

    from vitoom_setup.inference_config import LOCAL_INFERENCE_YAML, write_local_inference_yaml

    written = write_local_inference_yaml(env, repo_root=ROOT)
    rel = LOCAL_INFERENCE_YAML.as_posix()
    if written:
        _print(t("setup.status.inference_config_written", locale, path=rel))
    else:
        _print(t("setup.status.inference_config_skipped", locale, path=rel))


def maybe_ensure_docker_images(
    locale: str,
    env: dict[str, str],
    selected: set[str],
) -> None:
    from vitoom_setup.build_artifacts import detect_arch
    from vitoom_setup.constants import COMPONENT_TO_IMAGE_ENV
    from vitoom_setup.docker_images import ensure_docker_images

    components = selected & set(COMPONENT_TO_IMAGE_ENV)
    if not components:
        return

    _print(t("setup.prompt.ensure_docker_images", locale))
    raw = _prompt("> ").strip().lower()
    if raw in ("n", "no"):
        _print(t("setup.status.docker_images_skipped", locale))
        return

    arch = env.get("VITOOM_TARGET_ARCH") or detect_arch(locale)

    def emit(key: str, **kwargs: object) -> str:
        return t(key, locale, **kwargs)

    def log(message: str) -> None:
        _print(message)

    _print(t("setup.status.docker_images_start", locale, arch=arch))
    results = ensure_docker_images(
        ROOT,
        env,
        arch,
        components=components,
        log=log,
        emit=emit,
    )
    if results:
        _print(t("setup.status.docker_images_finished", locale, count=len(results)))


def maybe_download_artifacts(
    locale: str,
    region: str,
    env: dict[str, str],
    selected: set[str],
) -> None:
    build_components = selection_to_build_components(selected)
    if not build_components:
        _print(t("setup.status.no_artifacts_for_selection", locale))
        return

    _print(t("setup.prompt.download_artifacts", locale))
    raw = _prompt("> ").strip().lower()
    if raw in ("n", "no"):
        _print(t("setup.status.download_skipped", locale))
        return

    arch = env.get("VITOOM_TARGET_ARCH") or detect_arch(locale)
    exit_code = run_build_download(
        locale,
        arch=arch,
        components=build_components,
        region=region,
        env_values=env,
        skip_confirm=True,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def main() -> int:
    env = load_env_state(ENV_PATH, ENV_EXAMPLE_PATH)
    locale, region = resolve_locale_region(env)
    _print(t("setup.welcome", locale))

    selected = prompt_install_components(locale)
    configure_env(locale, region, env, selected)
    admin_password, admin_preserved = resolve_admin_password(env)
    env["DEFAULT_ADMIN_PASSWORD"] = admin_password
    if not admin_preserved:
        _print(t("setup.status.admin_password_generated", locale))
    write_env_file(locale, env)
    maybe_write_local_inference_config(locale, env, selected)
    maybe_download_artifacts(locale, region, env, selected)
    maybe_download_mini_model(locale, region, env, selected)
    maybe_ensure_docker_images(locale, env, selected)

    _print(t("setup.done", locale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

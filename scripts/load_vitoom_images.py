#!/usr/bin/env python3
"""Load Vitoom Docker images from images/<arch>/ tar or Docker Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from backend.i18n.locale import detect_cli_locale  # noqa: E402
from backend.i18n.translator import t  # noqa: E402
from vitoom_setup import REPO_ROOT as ROOT  # noqa: E402
from vitoom_setup.build_artifacts import detect_arch  # noqa: E402
from vitoom_setup.constants import INSTALL_COMPONENT_IDS  # noqa: E402
from vitoom_setup.docker_images import ensure_docker_images  # noqa: E402
from vitoom_setup.env_file import load_env_state  # noqa: E402
from vitoom_setup.region import locale_from_env_value  # noqa: E402

ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"


def _parse_components(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    tokens = {token.strip() for token in raw.split(",") if token.strip()}
    unknown = tokens - set(INSTALL_COMPONENT_IDS)
    if unknown:
        raise SystemExit(f"unknown component(s): {', '.join(sorted(unknown))}")
    return tokens


def _resolve_locale(env: dict[str, str]) -> str:
    return locale_from_env_value(env.get("VITOOM_LOCALE")) or detect_cli_locale()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load or pull Vitoom Docker images.")
    parser.add_argument(
        "--components",
        help="Comma-separated components: backend,visual,text,audio,mini,download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Load or pull even when the image already exists locally",
    )
    args = parser.parse_args()

    env = load_env_state(ENV_PATH, ENV_EXAMPLE_PATH)
    locale = _resolve_locale(env)
    arch = (env.get("VITOOM_TARGET_ARCH") or "").strip() or detect_arch(locale)
    components = _parse_components(args.components)

    def emit(key: str, **kwargs: object) -> str:
        return t(key, locale, **kwargs)

    def log(message: str) -> None:
        print(message, flush=True)

    log(t("setup.status.docker_images_start", locale, arch=arch))
    results = ensure_docker_images(
        ROOT,
        env,
        arch,
        components=components,
        skip_if_present=not args.force,
        log=log,
        emit=emit,
    )

    if not results:
        log(t("setup.status.docker_images_none", locale))
    else:
        log(t("setup.status.docker_images_finished", locale, count=len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

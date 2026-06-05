#!/usr/bin/env python3
"""Download initial model bundles after Vitoom Docker install."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from backend.i18n.locale import detect_cli_locale  # noqa: E402
from backend.i18n.translator import t  # noqa: E402
from vitoom_setup.env_file import parse_env_file  # noqa: E402
from vitoom_setup.initial_models import ENV_PATH  # noqa: E402
from vitoom_setup.model_hub import ensure_initial_download_dependencies  # noqa: E402
from vitoom_setup.region import locale_from_env_value, region_from_env_value, region_from_locale  # noqa: E402


def main() -> int:
    if not ENV_PATH.is_file():
        raise SystemExit(t("initial_models.error.env_missing", detect_cli_locale()))
    env = parse_env_file(ENV_PATH)
    locale = locale_from_env_value(env.get("VITOOM_LOCALE")) or "en-US"
    region = region_from_env_value(env.get("VITOOM_REGION")) or region_from_locale(locale)
    ensure_initial_download_dependencies(locale)
    from vitoom_setup.initial_models import (  # noqa: E402
        prompt_categories,
        run_initial_model_download,
    )

    print(t("initial_models.welcome", locale))
    categories = prompt_categories(locale)
    return run_initial_model_download(locale, region, categories)


if __name__ == "__main__":
    raise SystemExit(main())

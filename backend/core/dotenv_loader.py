"""Load project root ``.env`` for local backend runs."""

from __future__ import annotations

from pathlib import Path

_DOTENV_LOADED = False


def load_project_dotenv(*, override: bool = False) -> bool:
    """Load ``<repo>/.env`` into ``os.environ`` before reading config env vars."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED and not override:
        return False

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.is_file():
        _DOTENV_LOADED = True
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=override)
        _DOTENV_LOADED = True
        return True
    except ImportError:
        _DOTENV_LOADED = True
        return False

"""
应用版本号读取。

优先级：环境变量 VITOOM_VERSION > VERSION 文件 > fallback。
VERSION 文件依次查找：仓库根目录、当前工作目录。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK_VERSION = "0.0.0-dev"


def _read_version_file(path: Path) -> str | None:
    try:
        if path.is_file():
            version = path.read_text(encoding="utf-8").strip()
            if version:
                return version
    except OSError:
        pass
    return None


def _version_file_candidates() -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in (_REPO_ROOT / "VERSION", Path.cwd() / "VERSION"):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)
    return candidates


@lru_cache(maxsize=1)
def get_version() -> str:
    env_version = os.environ.get("VITOOM_VERSION", "").strip()
    if env_version:
        return env_version

    for path in _version_file_candidates():
        version = _read_version_file(path)
        if version:
            return version

    return _FALLBACK_VERSION

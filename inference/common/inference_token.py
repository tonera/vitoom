"""推理器 WS 连接 token 解析。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DOTENV_LOADED = False


def load_project_dotenv(*, override: bool = False) -> bool:
    """加载项目根目录 ``.env`` 到 ``os.environ``（本地直跑推理器时必需）。"""
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


def resolve_inference_token(explicit: Optional[str] = None) -> str:
    """解析 ``VITOOM_INFERENCE_TOKEN``（显式非空参数优先于环境变量）。"""
    load_project_dotenv()
    if explicit is not None:
        text = str(explicit).strip()
        if text:
            return text
    return str(os.environ.get("VITOOM_INFERENCE_TOKEN") or "").strip()

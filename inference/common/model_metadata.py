from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json


def read_json_dict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def read_json_value(path: Path, key: str, default: Any = None) -> Any:
    data = read_json_dict(path)
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def read_int_from_json(path: Path, key: str) -> Optional[int]:
    v = read_json_value(path, key, None)
    return int(v) if isinstance(v, int) and v > 0 else None


def read_family_name(root: Path) -> Optional[str]:
    path = root / "model_index.json"
    v = read_json_value(path, "_class_name", None)
    return str(v) if isinstance(v, str) and v.strip() else None


def safetensors_has_any_prefixes(path: Path, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return False
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open  # type: ignore

        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        return True
        return False
    except Exception:
        return False


def safetensors_list_keys(path: Path) -> list[str]:
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return []
    try:
        from safetensors import safe_open  # type: ignore

        with safe_open(str(path), framework="np") as f:
            return list(f.keys())
    except Exception:
        return []


def safetensors_get_shape(path: Path, key: str) -> Optional[tuple[int, ...]]:
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return None
    try:
        from safetensors import safe_open  # type: ignore

        with safe_open(str(path), framework="pt", device="cpu") as f:
            if key not in f.keys():
                return None
            shape = f.get_slice(key).get_shape()
        if not isinstance(shape, (list, tuple)):
            return None
        return tuple(int(x) for x in shape)
    except Exception:
        return None


def safetensors_find_first_shape(path: Path, keys: tuple[str, ...]) -> tuple[Optional[tuple[int, ...]], Optional[str]]:
    for key in keys:
        shape = safetensors_get_shape(path, key)
        if shape is not None:
            return shape, key
    return None, None


def safetensors_load_subset(path: Path, prefixes: tuple[str, ...]) -> dict[str, Any]:
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return {}
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能从单文件读取权重") from e

    out: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefixes):
                out[key] = f.get_tensor(key)
    return out


def safetensors_load_excluding_prefixes(path: Path, excluded_prefixes: tuple[str, ...]) -> dict[str, Any]:
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return {}
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能从单文件读取权重") from e

    out: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(excluded_prefixes):
                continue
            out[key] = f.get_tensor(key)
    return out

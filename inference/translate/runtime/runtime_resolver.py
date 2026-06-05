from __future__ import annotations

from typing import Any

_TRANSLATE_BACKEND_RUNTIME_BLOCKS = frozenset({"transformers"})


def merge_translate_runtime_cfg(
    runtime: dict[str, Any] | None,
    *,
    backend: str,
) -> dict[str, Any]:
    """Merge ``config.runtime`` into a flat dict for translate policy."""
    if not isinstance(runtime, dict):
        return {}
    merged: dict[str, Any] = {}
    for key, value in runtime.items():
        if key in _TRANSLATE_BACKEND_RUNTIME_BLOCKS:
            continue
        merged[key] = value
    section = runtime.get(backend)
    if isinstance(section, dict):
        merged = {**merged, **section}
    merged.pop("backend", None)
    return merged


def resolve_translate_backend(service_cfg: dict[str, Any] | None) -> str:
    runtime = service_cfg.get("runtime") if isinstance(service_cfg, dict) else {}
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    backend = str(runtime.get("backend") or "transformers").strip().lower()
    if backend in {"hf", "huggingface", "transformers"}:
        return "transformers"
    raise ValueError(
        f"Unsupported translate runtime.backend={backend!r}; expected transformers"
    )

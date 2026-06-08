from __future__ import annotations

from typing import Any

# mini OCR 的 `config.runtime` 与 `service_text_qwen.yaml` 一致：公共项与 `backend` 同级；
# `transformers` / `vllm` 为各推理后端专属子 dict。
_MINI_OCR_BACKEND_RUNTIME_BLOCKS = frozenset({"transformers", "vllm"})


def merge_mini_ocr_runtime_cfg(
    runtime: dict[str, Any] | None,
    *,
    backend: str,
) -> dict[str, Any]:
    """将 ``config.runtime`` 合并为 OCR policy 使用的扁平 dict。

    - 与 ``backend`` 同级的键为公共项（以及向后兼容的顶层 legacy 项）。
    - ``transformers`` / ``vllm`` 子 dict 为对应后端专属，覆盖同名公共键。
    - 返回结果不含 ``backend``。
    """
    if not isinstance(runtime, dict):
        return {}
    merged: dict[str, Any] = {}
    for key, value in runtime.items():
        if key in _MINI_OCR_BACKEND_RUNTIME_BLOCKS:
            continue
        merged[key] = value
    section = runtime.get(backend)
    if isinstance(section, dict):
        merged = {**merged, **section}
    merged.pop("backend", None)
    return merged

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union


_VARIANT_FILE_RE = re.compile(
    r"\.(?P<variant>[A-Za-z0-9_-]+)\.(?P<ext>safetensors|bin|pt)$"
)


def infer_variant_and_use_safetensors(weights_dir: Union[str, Path]) -> Tuple[Optional[str], bool]:
    """
    侦测 HuggingFace/Diffusers 风格的权重目录，推断：
    - variant: 例如 "fp16"/"bf16"/"fp32"（没有就返回 None）
    - use_safetensors: 目录内是否存在 safetensors 权重（含 sharded/index 形式）

    设计目标是复刻外部项目里 infer_setting 的核心能力，但命名更清晰，且只依赖标准库。
    """
    p = Path(weights_dir)
    if not p.exists():
        raise FileNotFoundError(f"weights_dir not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"weights_dir is not a directory: {p}")

    filenames = [f.name for f in p.iterdir() if f.is_file()]

    # sharded safetensors 可能只有 index.json + 分片；这里两者都算 safetensors。
    use_safetensors = any(
        name.endswith(".safetensors") or name.endswith(".safetensors.index.json") for name in filenames
    )

    variants: list[str] = []
    for name in filenames:
        m = _VARIANT_FILE_RE.search(name)
        if not m:
            continue
        v = m.group("variant")
        # 排除一些“看起来像 variant 但其实是文件名中间段”的噪声（保守一点，只过滤常见无意义值）
        if v in {"model", "pytorch_model", "diffusion_pytorch_model"}:
            continue
        variants.append(v)

    # diffusers 常见优先级：fp16/bf16/fp32；其余保留但不强行解释。
    preferred = ("fp16", "bf16", "fp32", "fp8", "int8", "int4")
    for v in preferred:
        if v in variants:
            return v, use_safetensors

    if variants:
        # 去重后取字典序最小，保证 deterministic（比“第一命中”更稳定）
        return sorted(set(variants))[0], use_safetensors

    return None, use_safetensors


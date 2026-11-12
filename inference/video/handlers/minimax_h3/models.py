"""H3 缓存对象。这个模块不导入 torch，方便缓存释放路径做类型判断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class H3Models:
    """只放 Qwen3-VL processor（含分词器）和元信息。文本编码器、DiT、VAE 按阶段加载、用完即丢，不进缓存。

    仍要经过 PipelineCache：key 变化时它会先驱逐常驻的 Wan2 / Turbo pipeline，否则 H3 会和它们抢显存。
    """

    model_root: str = ""
    tokenizer_dir: str = ""
    processor: Any = None

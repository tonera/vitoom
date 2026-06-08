"""
LoRA 兼容性规则（从 routes.py 抽离）。

职责：
- 仅负责根据“底模 family”推导 LoRA 可用的 family 集合。
"""

from typing import List

from backend.models.model_family import family_aliases, resolve_family


def resolve_lora_compatible_families(base_family: str) -> List[str]:
    """
    根据“底模 family”推导 LoRA 可用的 family 集合（兼容集合）。
    注意：这里返回的是用于查询 LoRA（Model.asset_type == 'lora'）的 family 列表。
    """
    base = str(base_family or "").strip()
    if not base:
        return []

    fam = resolve_family(base)
    if not fam:
        return []

    # 特殊规则：qwen / qwen.edit 共用一套 LoRA 兼容集合（两者互通）
    if fam in {"qwen", "qwen.edit"}:
        merged = [*family_aliases("qwen"), *family_aliases("qwen.edit")]
        seen = set()
        out: List[str] = []
        for x in merged:
            sx = str(x).strip()
            k = sx.lower()
            if not sx or k in seen:
                continue
            seen.add(k)
            out.append(sx)
        return out

    # 其他：同一 family 的 alias 列表
    return family_aliases(fam)


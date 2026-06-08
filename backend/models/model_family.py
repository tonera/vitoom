"""
模型家族（model_family）相关工具（backend 单点维护）。

注意：
- backend 可能独立于 inference 部署，因此此处不依赖 inference 的归一化逻辑。
- 当前项目中 family 是自由字符串（如 "Pony"/"SDXL 1.0"/"Flux.1 D"/"qwen-edit" 等）。
- 本文件提供唯一来源：canonical family -> aliases（有序列表），用于：
  1) /api/models?model_family=... 的 DB 过滤（family IN ...）
  2) LoRA 兼容集合推导（见 lora_compat.py）
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _norm(v: object) -> str:
    return str(v or "").strip().lower()


#
# canonical family -> aliases (ordered list)
#
# 这里尽量对齐原先 backend/models/lora_compat.py 的 MODEL_* 定义顺序，
# 以保证 UI 展示与兼容集合行为稳定。
#
_FAMILY_ALIASES_LIST: Dict[str, List[str]] = {
    "sd15": ["sd15", "SD15", "sd21"],
    "sdxl": ["sdxl", "SDXL 1.0", "Illustrious", "NoobAI", "Pony"],
    "sd3": ["sd3", "presd3", "SDXL 3.0"],
    "flux": ["flux", "flux-d", "flux-s", "Flux.1 D", "Flux.1 S"],
    "flux_kontext": ["flux_kontext", "Flux.1 Kontext"],
    "flux2": ["flux2", "Flux2", "Flux.2 D"],
    "flux2_klein": ["flux2_klein", "Flux.2 Klein"],
    "zimage": ["zimage", "ZImageTurbo"],
    "qwen": ["qwen", "Qwen"],
    "qwen.edit": ["qwen.edit", "qwen-edit", "Qwen-edit"],
    "wan": ["wan", "Wan", "Wan-edit", "Wan-painting", "Wan-sketch"],
    "wan_video": ["wan_video", "Wan-video"],
}


def resolve_family(v: object) -> str:
    """
    将输入的 family 或 alias 归一为 canonical family。
    - 空/None 返回 ""
    - 未知值：返回小写后的原值（由上游决定是否报错/是否当作自定义 family）
    """
    s = _norm(v)
    if not s:
        return ""
    if s in _FAMILY_ALIASES_LIST:
        return s
    for fam, aliases in _FAMILY_ALIASES_LIST.items():
        if s in {_norm(x) for x in aliases}:
            return fam
    return s


def family_aliases(family: str) -> List[str]:
    """
    输入 family（或其别名），返回该 family 的 alias 列表（去重，保留顺序）。
    - 未知值：返回 [原值]（用于“自定义 family”场景）
    """
    fam = resolve_family(family)
    if not fam:
        return []
    aliases = _FAMILY_ALIASES_LIST.get(fam)
    if not aliases:
        return [str(family).strip()]
    seen = set()
    out: List[str] = []
    for x in aliases:
        sx = str(x).strip()
        k = _norm(sx)
        if not sx or not k or k in seen:
            continue
        seen.add(k)
        out.append(sx)
    return out


def parse_model_families_param(model_family: Optional[str]) -> List[str]:
    """
    解析 query 参数 model_family，支持：
    - "sdxl"
    - "sdxl,flux"
    - "sdxl flux"
    返回 canonical family 列表（去重，按输入顺序）。
    """
    raw = str(model_family or "").strip()
    if not raw:
        return []
    # 允许逗号/空格分隔
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    seen = set()
    out: List[str] = []
    for p in parts:
        key = resolve_family(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def family_in_for_families(families: List[str]) -> List[str]:
    """
    将 family 列表展开为 family_in（alias 列表），用于 DB 过滤：
      lower(family) IN (...)
    返回值为“原样字符串列表”（由 SQLAlchemy lower() 去做大小写处理）。
    """
    seen = set()
    out: List[str] = []
    for fam in families:
        for a in family_aliases(fam):
            key = _norm(a)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(a).strip())
    return out


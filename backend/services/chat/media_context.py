from __future__ import annotations

import re
from typing import Any, Dict


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

_MEDIA_RULES_TEXT = (
    "媒体 URL 处理（硬约束）：\n"
    "- 本轮含『描述/分析/识别/看看/理解/是什么/讲了什么/总结/对比/差异』"
    "等要求理解画面的动词 → 调 analyze_media；\n"
    "- 本轮仅裸 URL 或含『记住/保存/暂存/先收着/待会儿用/这是第 N 张』"
    "等仅提供素材的表述 → 不调任何工具，只回『已记录。』一句即可；\n"
    "- 不要输出推理过程或规则判读；直接给最终答案。"
)


def build_conditional_context(prompt_with_history: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    if _URL_RE.search(prompt_with_history or ""):
        ctx["media_rules"] = _MEDIA_RULES_TEXT
    else:
        ctx["media_rules"] = ""
    return ctx

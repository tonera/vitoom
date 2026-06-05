"""LLM 工具调用参数清洗的公共 helper。

LLM（GPT/Qwen/Claude 等）在调用 tool 时常出现这一类共性 quirk：
- 把"无值"渲染成字面量 ``"None"`` / ``"null"`` / ``"undefined"`` 而不是真正的 JSON ``null``；
- 把可选数字字段渲染成字符串（``"None"`` / ``"30"`` / ``""``）；
- 在不同字段之间不一致地混用大小写。

这些场景如果每个工具自己写一份清洗代码，会出现 audio_tts/audio_asr 那种"字面相同的
``_EMPTY_LIKE_TOKENS`` + ``_clean_optional_str`` 复制两份"的折旧。本模块提供一组小函数，
工具内部统一 import 即可。

设计取舍：
- 不做"宽松到吃掉所有奇怪输入"——只兜住明显是"LLM 把 None 序列化错"的几个 token；
  避免误把业务真值（比如名为 "None" 的有效 ID，虽然概率很低）也吞掉。
- 同时提供两套语义（见 ``coerce_optional`` 与 ``clean_optional_str``）；
  pydantic 前置校验和直接业务清洗两种场景各取所需。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# LLM 常见误用为"空值"的字面量（比较时统一小写 + strip）。
EMPTY_LIKE_TOKENS: frozenset[str] = frozenset(
    {"", "none", "null", "nil", "nan", "n/a", "na", "undefined"}
)


def coerce_optional(value: Any) -> Any:
    """把 LLM 常见的字符串形态空值正规化为 ``None``，**其余值原样返回**。

    用法（典型）：作为 ``pydantic.field_validator(mode='before')`` 的清洗前置，
    在校验 ``Optional[float]`` / ``Optional[int]`` 等强类型字段前先把字符串
    ``"None"`` 这类吃掉，避免抛 ``could not convert string to float: 'None'``。

    与 ``clean_optional_str`` 的差别：本函数不强制把非字符串值转成 str；
    传入 ``30`` / ``[1, 2]`` / ``{"k": "v"}`` 等都原样返回。
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in EMPTY_LIKE_TOKENS:
            return None
    return value


def clean_optional_str(value: Any) -> Optional[str]:
    """把 LLM 给的可选字符串归一化：``None`` / 空白 / ``"None"`` 等都视作未设置。

    与 ``coerce_optional`` 的差别：本函数**保证返回 ``str`` 或 ``None``**，并对
    非空值做 ``strip()``，适合直接喂给业务消费方。

    传入非 str（如 list/dict）也会被 ``str()`` 化，调用方应自行决定是否需要这种语义。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in EMPTY_LIKE_TOKENS:
        return None
    return text


def coerce_timeout_seconds(value: Any, *, get_default: Callable[[], float]) -> float:
    """把 LLM 传入的 ``timeout``（可能为 ``"None"`` 字符串等）安全转为正数秒数。"""
    c = coerce_optional(value)
    if c is None:
        return get_default()
    try:
        t = float(c)
    except (TypeError, ValueError):
        return get_default()
    if t <= 0:
        return get_default()
    return t


__all__ = [
    "EMPTY_LIKE_TOKENS",
    "coerce_optional",
    "clean_optional_str",
    "coerce_timeout_seconds",
]

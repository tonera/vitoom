"""``backend.services.agent.tools.builtin._arg_utils`` 的测试。

把 LLM 调用工具时常见的 ``"None"`` / ``"null"`` / ``"undefined"`` 等字符串形态空值
正规化为真正的 ``None`` 是公共需求；本套测试锁住两种语义（``coerce_optional`` 保持
非字符串原样、``clean_optional_str`` 强制返回 str/None），确保后续如果有人调整公共
helper 时不会破坏 audio_tts/audio_asr/document_to_markdown 等工具的依赖。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin import _arg_utils as au


# ---------------------------------------------------------------------------
# coerce_optional：保留非字符串原样
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        "None",
        "none",
        "NONE",
        "null",
        "Null",
        "NULL",
        "undefined",
        "n/a",
        "N/A",
        "nil",
        "nan",
        "",
        "   ",
        "\tnone\n",
    ],
)
def test_coerce_optional_treats_string_nones_as_none(value):
    assert au.coerce_optional(value) is None


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        False,
        [],
        {},
        "real string",
        "0",
        "false",
        30,
        30.5,
        ["a", "b"],
        {"k": "v"},
    ],
)
def test_coerce_optional_keeps_real_values(value):
    assert au.coerce_optional(value) == value


# ---------------------------------------------------------------------------
# clean_optional_str：强制返回 str/None，并 strip 输出
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, "", "  ", "None", "null", "undefined", "n/a", "NIL", "NaN"],
)
def test_clean_optional_str_returns_none_for_unset(value):
    assert au.clean_optional_str(value) is None


def test_clean_optional_str_strips_real_strings():
    assert au.clean_optional_str("  hello  ") == "hello"
    assert au.clean_optional_str("name") == "name"


def test_clean_optional_str_stringifies_non_str_inputs():
    """与 coerce_optional 不同，clean_optional_str 拿到非 str 时会 str() 化。"""
    assert au.clean_optional_str(42) == "42"
    assert au.clean_optional_str(["x"]) == "['x']"


# ---------------------------------------------------------------------------
# 锁住公共常量：避免后续修改时无意中收窄 token 集
# ---------------------------------------------------------------------------


def test_empty_like_tokens_covers_common_llm_quirks():
    expected = {"", "none", "null", "nil", "nan", "n/a", "na", "undefined"}
    assert expected.issubset(au.EMPTY_LIKE_TOKENS)

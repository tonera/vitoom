# pyright: reportMissingImports=false

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from text.runtime.qwen_tool_parser import QwenToolCallParser


def _feed_in_fragments(parser: QwenToolCallParser, text: str, *, chunk_size: int = 7):
    content_parts = []
    tool_calls = []
    for i in range(0, len(text), chunk_size):
        content, calls = parser.feed(text[i : i + chunk_size])
        content_parts.append(content)
        tool_calls.extend(calls)
    flushed_content, flushed_calls = parser.flush()
    content_parts.append(flushed_content)
    tool_calls.extend(flushed_calls)
    return "".join(content_parts), tool_calls


def test_parser_extracts_single_tool_call_across_chunks():
    parser = QwenToolCallParser()
    text = (
        "Thought: I should call the tool."
        "<tool_call>\n"
        '{"name": "analyze_media", "arguments": {"url": "http://x/a.jpg", "q": "?"}}\n'
        "</tool_call>"
    )
    content, calls = _feed_in_fragments(parser, text, chunk_size=5)
    assert calls, "expected a parsed tool call"
    assert calls[0].name == "analyze_media"
    # arguments 应该是 JSON 字符串
    assert '"url": "http://x/a.jpg"' in calls[0].arguments
    # tool_call 标签及其内部 JSON 不应出现在 content 流里
    assert "<tool_call>" not in content
    assert "analyze_media" not in content
    assert "Thought: I should call the tool." in content


def test_parser_extracts_multiple_tool_calls():
    parser = QwenToolCallParser()
    text = (
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
        "between"
        '<tool_call>\n{"name": "b", "arguments": {"y": 2}}\n</tool_call>'
    )
    content, calls = _feed_in_fragments(parser, text, chunk_size=4)
    assert [c.name for c in calls] == ["a", "b"]
    assert "between" in content
    assert "<tool_call>" not in content


def test_parser_accepts_string_arguments():
    parser = QwenToolCallParser()
    text = '<tool_call>\n{"name": "raw", "arguments": "{\\"k\\": 1}"}\n</tool_call>'
    _content, calls = parser.feed(text)
    flushed_content, flushed_calls = parser.flush()
    calls.extend(flushed_calls)
    assert len(calls) == 1
    assert calls[0].name == "raw"
    assert calls[0].arguments == '{"k": 1}'


def test_parser_degrades_invalid_json_back_to_content():
    parser = QwenToolCallParser()
    text = "<tool_call>\nnot json here\n</tool_call>after"
    content, calls = parser.feed(text)
    flushed_content, flushed_calls = parser.flush()
    calls.extend(flushed_calls)
    content += flushed_content
    assert calls == []
    # 原文应该被保留下来
    assert "<tool_call>" in content
    assert "not json here" in content
    assert "after" in content


def test_parser_preserves_plain_text_with_no_tool_call():
    parser = QwenToolCallParser()
    text = "just a normal markdown reply without tools"
    content, calls = _feed_in_fragments(parser, text, chunk_size=8)
    assert calls == []
    assert content == text


def test_parser_extracts_xml_style_tool_call():
    """Qwen3 / Llama-3.1 风格 XML payload 应该被识别为 tool call。"""
    import json

    parser = QwenToolCallParser()
    text = (
        "<tool_call><function=analyze_media>\n"
        "<parameter=url>\n"
        "http://local.fluxsd.cn/2026/04/12/301716522266017792_0_s.jpeg\n"
        "</parameter>\n"
        "<parameter=urls>\n\n"
        "</parameter>\n"
        "<parameter=question>\n"
        "这张图片中是什么？\n"
        "</parameter>\n"
        "</function></tool_call>"
    )
    content, calls = _feed_in_fragments(parser, text, chunk_size=11)
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "analyze_media"
    args = json.loads(call.arguments)
    assert args["url"] == "http://local.fluxsd.cn/2026/04/12/301716522266017792_0_s.jpeg"
    assert args["urls"] == ""
    assert args["question"] == "这张图片中是什么？"
    assert "<tool_call>" not in content
    assert "<function=" not in content
    assert "analyze_media" not in content


def test_parser_xml_coerces_numeric_values_but_keeps_plain_strings():
    """XML 值若是合法 JSON 字面量（数字/布尔）应解析为原生类型，普通字符串保持原样。"""
    import json

    parser = QwenToolCallParser()
    text = (
        "<tool_call><function=demo>"
        "<parameter=count>42</parameter>"
        "<parameter=flag>true</parameter>"
        "<parameter=label>true-ish</parameter>"
        "<parameter=note>hello world</parameter>"
        "</function></tool_call>"
    )
    _content, calls = parser.feed(text)
    flushed_content, flushed_calls = parser.flush()
    calls.extend(flushed_calls)
    assert len(calls) == 1
    args = json.loads(calls[0].arguments)
    assert args["count"] == 42
    assert args["flag"] is True
    # "true-ish" 不是合法 JSON 字面量，必须保留为字符串
    assert args["label"] == "true-ish"
    assert args["note"] == "hello world"


def test_parser_xml_as_first_output_emits_no_content():
    """模型第一个 token 就开启 <tool_call> 时，content 必须为空，不能泄漏任何标签。"""
    import json

    parser = QwenToolCallParser()
    text = (
        "<tool_call><function=analyze_media>"
        "<parameter=url>http://x/a.jpg</parameter>"
        "<parameter=question>what</parameter>"
        "</function></tool_call>"
    )
    content, calls = _feed_in_fragments(parser, text, chunk_size=6)
    assert content == ""
    assert len(calls) == 1
    args = json.loads(calls[0].arguments)
    assert args == {"url": "http://x/a.jpg", "question": "what"}


def test_parser_xml_across_chunk_boundaries():
    """把完整的 XML tool call 切成极碎的片段也能完整解析出来。"""
    import json

    parser = QwenToolCallParser()
    text = (
        "<tool_call><function=foo>"
        "<parameter=x>1</parameter>"
        "<parameter=y>hello</parameter>"
        "</function></tool_call>"
    )
    content, calls = _feed_in_fragments(parser, text, chunk_size=3)
    assert len(calls) == 1
    args = json.loads(calls[0].arguments)
    assert args == {"x": 1, "y": "hello"}
    assert "<tool_call>" not in content

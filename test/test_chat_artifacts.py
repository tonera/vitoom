"""``backend.services.chat.artifacts`` 解析层的回归测试。

重点保护 ``try_parse_tool_result_payload`` 的三种形态识别能力——
其中『前缀文本 + fenced JSON』是 ``document_to_markdown`` 工具在 ``_run`` 返回值
开头注入『最终回复风格指令』之后必须支持的形态。如果这条路径退化，前端附件
下载区会再次失踪。
"""

from __future__ import annotations

import json

import pytest

from backend.services.chat.artifacts import (
    build_chat_files_from_tool_result,
    try_parse_tool_result_payload,
)


_PAYLOAD = {
    "tool": "document_to_markdown",
    "status": "completed",
    "items": [
        {
            "task_id": "abc",
            "status": "completed",
            "preview_md": "# Hi",
            "files": [
                {
                    "file_id": "f1",
                    "file_name": "doc.zip",
                    "url": "https://example.com/doc.zip",
                    "mime_type": "application/zip",
                    "file_size": 660 * 1024,
                }
            ],
        }
    ],
}


def _payload_json() -> str:
    return json.dumps(_PAYLOAD, ensure_ascii=False)


@pytest.mark.parametrize(
    "label,body",
    [
        ("legacy_fenced", f"```json\n{_payload_json()}\n```"),
        ("bare_json", _payload_json()),
        (
            "instruction_prefix_then_fenced",
            (
                "[Final Reply Instruction · 必须严格遵守]\n"
                "请把每个 items[].preview_md 原样作为流式正文输出……\n\n"
                f"```json\n{_payload_json()}\n```"
            ),
        ),
        (
            "fenced_without_lang_tag",
            f"```\n{_payload_json()}\n```",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_try_parse_tool_result_payload_recognises_known_shapes(
    label: str, body: str
) -> None:
    parsed = try_parse_tool_result_payload(body)
    assert parsed is not None, f"shape={label!r} should be parsable"
    assert parsed.get("tool") == "document_to_markdown"
    assert parsed.get("items") and len(parsed["items"]) == 1


def test_try_parse_tool_result_payload_returns_none_for_non_json() -> None:
    assert try_parse_tool_result_payload("just some plain text") is None
    assert try_parse_tool_result_payload("") is None
    assert try_parse_tool_result_payload(None) is None
    assert try_parse_tool_result_payload(12345) is None


def test_try_parse_tool_result_payload_passes_through_dict() -> None:
    assert try_parse_tool_result_payload(_PAYLOAD) is _PAYLOAD


def test_build_chat_files_from_tool_result_extracts_zip_artifact() -> None:
    parsed = try_parse_tool_result_payload(
        "[Final Reply Instruction · 必须严格遵守]\n指令略\n\n"
        f"```json\n{_payload_json()}\n```"
    )
    assert parsed is not None

    files = build_chat_files_from_tool_result(
        parsed, source_tool="document_to_markdown"
    )
    assert len(files) == 1
    item = files[0]
    assert item["url"] == "https://example.com/doc.zip"
    assert item["category"] == "archive"
    assert item["file_size"] == 660 * 1024
    assert item["source_tool"] == "document_to_markdown"
    # archive 类型不回填 preview_text（zip 节点本身就是下载入口，无需正文预览）
    assert item.get("preview_text") in (None, "")


def test_build_chat_files_from_tool_result_backfills_preview_for_md() -> None:
    """category=document 时，应把 item.preview_md 回填到 file.preview_text。"""

    payload = {
        "tool": "document_to_markdown",
        "status": "completed",
        "items": [
            {
                "task_id": "abc",
                "status": "completed",
                "preview_md": "# Hello world",
                "files": [
                    {
                        "file_id": "f-md",
                        "file_name": "doc.md",
                        "url": "https://example.com/doc.md",
                        "mime_type": "text/markdown",
                        "file_size": 1234,
                    }
                ],
            }
        ],
    }

    files = build_chat_files_from_tool_result(
        payload, source_tool="document_to_markdown"
    )
    assert len(files) == 1
    item = files[0]
    assert item["category"] == "document"
    assert item["preview_text"] == "# Hello world"

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin import process_url_content as puc


def test_inspect_url_detects_pdf_by_suffix():
    info = puc.inspect_url("https://example.com/report.pdf")

    assert info.kind == "pdf"
    assert info.suffix == ".pdf"


def test_classify_task_uses_output_format_hint_for_markdown():
    inspections = [puc.UrlInspection(url="https://example.com/report.pdf", kind="pdf")]
    decision = puc.classify_task(
        question="please convert this document to markdown",
        inspections=inspections,
        output_format_hint="markdown",
    )

    assert decision.task == "convert_to_markdown"


def test_classify_task_detects_table_excel_request():
    inspections = [puc.UrlInspection(url="https://example.com/table.png", kind="image")]
    decision = puc.classify_task(
        question="把图中的表格导出为 excel",
        inspections=inspections,
        output_format_hint="excel",
    )

    assert decision.task == "extract_table"


def test_classify_task_summarizes_existing_markdown_document_by_default():
    inspections = [puc.UrlInspection(url="https://example.com/report.md", kind="markdown")]
    decision = puc.classify_task(
        question="请基于这份文档给出回复",
        inspections=inspections,
    )

    assert decision.task == "summarize"


def test_classify_task_ignores_markdown_hint_without_conversion_intent():
    inspections = [puc.UrlInspection(url="https://example.com/report.txt", kind="text")]
    decision = puc.classify_task(
        question="请基于这份文档总结一下病人情况",
        inspections=inspections,
        output_format_hint="markdown",
    )

    assert decision.task == "summarize"


def test_process_url_content_reads_existing_markdown_without_generating_files(monkeypatch):
    monkeypatch.setattr(
        puc,
        "inspect_url",
        lambda url, timeout=5.0: puc.UrlInspection(url=url, kind="markdown", suffix=".md"),
    )
    monkeypatch.setattr(
        puc,
        "_read_text_url_sync",
        lambda url, timeout, max_chars=60000: {
            "tool": "process_url_content",
            "status": "completed",
            "url": url,
            "content": "# 报告\n检测结果：EGFR 阳性",
        },
    )

    payload = puc.process_url_content_sync(
        user_id="u1",
        question="请总结一下病人情况",
        urls=["https://example.com/report.md"],
    )

    assert payload["route"] == "text_url_reader"
    assert payload["result"]["items"][0]["content"].startswith("# 报告")
    assert "files" not in payload


def test_normalize_urls_accepts_json_array_string():
    urls = puc._normalize_urls(urls='["https://example.com/a.pdf", "https://example.com/b.docx"]')

    assert urls == ["https://example.com/a.pdf", "https://example.com/b.docx"]


def test_expose_nested_artifacts_lifts_document_item_files():
    payload = {
        "tool": "process_url_content",
        "status": "completed",
        "route": "document_to_markdown",
        "result": {
            "tool": "document_to_markdown",
            "status": "completed",
            "items": [
                {
                    "task_id": "task-1",
                    "preview_md": "hello",
                    "files": [
                        {
                            "file_id": "file-1",
                            "file_name": "converted.zip",
                            "url": "https://example.com/converted.zip",
                            "mime_type": "application/zip",
                        }
                    ],
                }
            ],
        },
    }

    exposed = puc._expose_nested_artifacts(payload)

    assert exposed["total_files"] == 1
    assert exposed["files"][0]["file_name"] == "converted.zip"
    assert exposed["files"][0]["task_id"] == "task-1"
    assert exposed["files"][0]["preview_md"] == "hello"


def test_invoke_atomic_tool_emits_nested_events():
    events = []

    def started(name, args, event_id, parent):
        events.append(("start", name, args, event_id, parent))

    def finished(name, output, event_id, parent):
        events.append(("finish", name, output, event_id, parent))

    result = puc._invoke_atomic_tool(
        context={
            "session_nested_tool_hooks": {
                "emit_tool_started": started,
                "emit_tool_finished": finished,
            }
        },
        tool_name="document_to_markdown",
        arguments={"urls": ["https://example.com/a.pdf"]},
        func=lambda: {"status": "completed"},
    )

    assert result == {"status": "completed"}
    assert events[0][0] == "start"
    assert events[0][1] == "document_to_markdown"
    assert events[0][4] == "process_url_content"
    assert events[1][0] == "finish"
    assert events[1][1] == "document_to_markdown"
    assert events[1][3] == events[0][3]

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin import web_page_reader as wpr


def test_read_web_page_sync_uses_scrape_website_tool(monkeypatch):
    seen: dict[str, object] = {}

    class FakeScrapeWebsiteTool:
        def run(self, **kwargs):
            seen.update(kwargs)
            return "页面标题\n\n这是一篇关于 CrewAI 网页抓取工具的文章。"

    fake_module = types.SimpleNamespace(ScrapeWebsiteTool=FakeScrapeWebsiteTool)
    monkeypatch.setitem(sys.modules, "crewai_tools", fake_module)

    payload = wpr.read_web_page_sync(
        url="https://example.com/article",
        question="主要内容是什么？",
        max_chars=5000,
    )

    assert seen == {"website_url": "https://example.com/article"}
    assert payload["status"] == "completed"
    assert payload["url"] == "https://example.com/article"
    assert payload["question"] == "主要内容是什么？"
    assert "CrewAI 网页抓取工具" in payload["content"]
    assert payload["truncated"] is False


def test_read_web_page_sync_rejects_non_page_resources(monkeypatch):
    with pytest.raises(RuntimeError, match="ordinary HTML web pages"):
        wpr.read_web_page_sync(url="https://example.com/image.jpg")


def test_read_web_page_sync_truncates_long_content(monkeypatch):
    class FakeScrapeWebsiteTool:
        def run(self, **kwargs):
            return "x" * 5000

    fake_module = types.SimpleNamespace(ScrapeWebsiteTool=FakeScrapeWebsiteTool)
    monkeypatch.setitem(sys.modules, "crewai_tools", fake_module)

    payload = wpr.read_web_page_sync(url="https://example.com/post", max_chars=1000)

    assert payload["truncated"] is True
    assert payload["original_length"] == 5000
    assert payload["returned_length"] == 1000
    assert len(payload["content"]) == 1000


def test_tool_args_schema_exposes_expected_parameters():
    try:
        import crewai  # noqa: F401
    except Exception:
        return

    tool = wpr.build_web_page_reader_tool()
    fields = set(tool.args_schema.model_fields.keys())
    assert fields == {"url", "question", "max_chars"}

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tool_catalog import ToolCatalog  # noqa: E402
from backend.services.agent.tools.builtin import knowledge_base_archive as kb_archive_tool  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core import archive  # noqa: E402
from backend.services.agent.tools.registry import get_tool_plugin_registry  # noqa: E402


class _NoopClient:
    def __init__(self):
        self.requests = []

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        return {"acknowledged": True}


def test_knowledge_base_archive_registered_in_registry_and_catalog():
    registrations = get_tool_plugin_registry().all_registrations()
    assert kb_archive_tool.KNOWLEDGE_BASE_ARCHIVE_TOOL_NAME in registrations

    entry = ToolCatalog().get(kb_archive_tool.KNOWLEDGE_BASE_ARCHIVE_TOOL_NAME)
    assert entry is not None
    assert entry.enabled is True
    assert "归档" in entry.description


def test_save_markdown_to_archive_creates_markdown_file(monkeypatch, tmp_path):
    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_canonical_root", lambda: tmp_path)

    path = archive.save_markdown_to_archive("# 归档\n\n内容", user_id="u1", title="测试对话")

    assert path.suffix == ".md"
    assert path.read_text(encoding="utf-8").startswith("# 归档")
    assert "用户归档" in str(path)


def test_archive_file_path_rejects_media_file(tmp_path):
    media = tmp_path / "photo.png"
    media.write_bytes(b"fake image")

    try:
        archive.archive_file_path(media, user_id="u1", client=_NoopClient())
    except RuntimeError as exc:
        assert "不入库" in str(exc)
    else:
        raise AssertionError("media file should be rejected")


def test_archive_file_path_classifies_ingests_and_refreshes(monkeypatch, tmp_path):
    source = tmp_path / "doc.md"
    source.write_text("知识库归档测试", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    client = _NoopClient()

    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_canonical_root", lambda: tmp_path)
    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_manifest_path", lambda: manifest)
    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_document_index", lambda: "kb_document_v1")
    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_chunk_index", lambda: "kb_chunk_v1")
    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_embedding_dims", lambda: 384)
    monkeypatch.setattr(
        archive,
        "classify_source_row",
        lambda row, user_id: {
            "domain": "项目",
            "topic": "归档",
            "subtopic": "",
            "summary": "",
            "classification_confidence": 0.9,
            "classification_reason": "",
            "tags": ["归档"],
            "classification_status": "classified",
        },
    )

    result = archive.archive_file_path(source, user_id="u1", client=client)

    assert result["document"]["domain"] == "项目"
    assert result["ingest"]["documents"] == 1
    assert manifest.exists()
    assert any(path.endswith("_refresh") for _method, path, _kwargs in client.requests)


def test_archive_conversation_can_skip_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(archive.agent_settings, "get_knowledge_base_canonical_root", lambda: tmp_path)
    captured = {}

    def fake_archive_file_path(path, **kwargs):
        captured["text"] = path.read_text(encoding="utf-8")
        return {"document": {"document_id": "doc_1", "file_name": path.name}, "ingest": {"documents": 1}}

    monkeypatch.setattr(archive, "archive_file_path", fake_archive_file_path)
    archive.archive_conversation("# 已总结", user_id="u1", title="对话", summarize=False)

    assert "# 已总结" in captured["text"]

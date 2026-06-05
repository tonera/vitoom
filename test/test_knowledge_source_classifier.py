from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin.knowledge_base_core import classifier  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.source_manifest import read_manifest  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.source_organizer import organize_sources  # noqa: E402


def test_normalize_classification_accepts_high_confidence_project():
    result = classifier.normalize_classification(
        {
            "domain": "项目",
            "topic": "架构设计",
            "subtopic": "Vitoom",
            "summary": "Vitoom 架构设计文档",
            "confidence": 0.91,
            "reason": "文件名包含架构设计",
            "suggested_tags": ["Vitoom", "架构设计"],
        },
        threshold=0.75,
    )

    assert result["domain"] == "项目"
    assert result["topic"] == "架构设计"
    assert result["classification_status"] == "classified"
    assert result["tags"] == ["Vitoom", "架构设计"]


def test_normalize_classification_low_confidence_goes_to_pending():
    result = classifier.normalize_classification(
        {
            "domain": "技术",
            "topic": "未知",
            "confidence": 0.2,
            "reason": "证据不足",
            "suggested_tags": ["猜测"],
        },
        threshold=0.75,
    )

    assert result["domain"] == "未分类_待确认"
    assert result["topic"] == ""
    assert result["classification_status"] == "low_confidence"


def test_classify_source_row_uses_internal_llm_contract(monkeypatch):
    def fake_completion(messages, *, user_id, error_label, **_kwargs):
        assert user_id == "tester"
        assert error_label == "knowledge source classifier"
        assert "分类器" in messages[0]["content"]
        return '{"domain":"项目","topic":"交付资料","subtopic":"德信科技","summary":"产品介绍","confidence":0.86,"reason":"文件名命中","suggested_tags":["德信科技"]}'

    monkeypatch.setattr(classifier, "run_agent_planner_completion", fake_completion)
    result = classifier.classify_source_row(
        {"file_name": "德信科技-秘火产品介绍_成本管理.zip", "extension": ".zip"},
        threshold=0.75,
        user_id="tester",
    )

    assert result["domain"] == "项目"
    assert result["subtopic"] == "德信科技"
    assert result["classification_confidence"] == 0.86


def test_classify_source_row_resolves_internal_effective_user(monkeypatch):
    seen = {}

    def fake_completion(messages, *, user_id, error_label, **_kwargs):
        del messages, error_label
        seen["user_id"] = user_id
        return '{"domain":"项目","topic":"交付资料","subtopic":"","summary":"产品介绍","confidence":0.86,"reason":"文件名命中","suggested_tags":[]}'

    monkeypatch.setattr(classifier, "resolve_classifier_user_id", lambda user_id="": "real-user")
    monkeypatch.setattr(classifier, "run_agent_planner_completion", fake_completion)
    classifier.classify_source_row({"file_name": "产品介绍.pdf"}, user_id="agent-system")

    assert seen["user_id"] == "real-user"


def test_resolve_classifier_user_id_prefers_explicit_user():
    assert classifier.resolve_classifier_user_id("real-user") == "real-user"


def test_organize_sources_can_classify_rows(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "Vitoom架构设计.md").write_text("知识库架构设计", encoding="utf-8")

    def fake_classify_source_row(row, *, threshold, user_id, preview_text=""):
        assert row["file_name"] == "Vitoom架构设计.md"
        return {
            "domain": "项目",
            "topic": "架构设计",
            "subtopic": "Vitoom",
            "summary": "Vitoom 架构设计",
            "classification_confidence": 0.9,
            "classification_reason": "文件名命中",
            "tags": ["Vitoom"],
            "classification_status": "classified",
        }

    monkeypatch.setattr(classifier, "classify_source_row", fake_classify_source_row)
    manifest_path = tmp_path / "canonical" / "manifest.jsonl"
    summary = organize_sources(
        [source_root],
        canonical_root=tmp_path / "canonical",
        manifest_path=manifest_path,
        copy_files=False,
        classify=True,
    )
    rows = list(read_manifest(manifest_path))

    assert summary["classification"]["classified"] == 1
    assert rows[0]["domain"] == "项目"
    assert rows[0]["topic"] == "架构设计"
    assert rows[0]["classification_status"] == "classified"


def test_classify_rows_resume_skips_completed_and_retries_failed(monkeypatch):
    calls = []
    rows = [
        {"file_name": "done.md", "classification_status": "classified"},
        {"file_name": "retry.md", "classification_status": "failed"},
    ]

    def fake_classify_source_row(row, *, threshold, user_id, preview_text=""):
        del threshold, user_id, preview_text
        calls.append(row["file_name"])
        return {
            "domain": "项目",
            "topic": "交付资料",
            "subtopic": "",
            "summary": "",
            "classification_confidence": 0.9,
            "classification_reason": "",
            "tags": [],
            "classification_status": "classified",
        }

    monkeypatch.setattr(classifier, "classify_source_row", fake_classify_source_row)
    summary = classifier.classify_rows(rows, resume=True)

    assert calls == ["retry.md"]
    assert summary["skipped"] == 1
    assert summary["classified"] == 2


def test_classify_rows_checkpoint_after_each_processed(monkeypatch):
    checkpoint_count = 0
    rows = [{"file_name": "a.md"}, {"file_name": "b.md"}]

    def fake_classify_source_row(row, *, threshold, user_id, preview_text=""):
        del row, threshold, user_id, preview_text
        return {
            "domain": "项目",
            "topic": "",
            "subtopic": "",
            "summary": "",
            "classification_confidence": 0.9,
            "classification_reason": "",
            "tags": [],
            "classification_status": "classified",
        }

    def checkpoint():
        nonlocal checkpoint_count
        checkpoint_count += 1

    monkeypatch.setattr(classifier, "classify_source_row", fake_classify_source_row)
    classifier.classify_rows(rows, resume=True, checkpoint_callback=checkpoint)

    assert checkpoint_count == 2


def test_classify_rows_reports_classification_result(monkeypatch):
    rows = [{"file_name": "方案.md"}]
    events = []

    def fake_classify_source_row(row, *, threshold, user_id, preview_text=""):
        del row, threshold, user_id, preview_text
        return {
            "domain": "项目",
            "topic": "方案",
            "subtopic": "Vitoom",
            "summary": "",
            "classification_confidence": 0.88,
            "classification_reason": "",
            "tags": [],
            "classification_status": "classified",
        }

    monkeypatch.setattr(classifier, "classify_source_row", fake_classify_source_row)
    classifier.classify_rows(rows, progress_callback=lambda event, payload: events.append((event, payload)), progress_every=1)

    result_events = [payload for event, payload in events if event == "classify_result"]
    assert result_events
    assert result_events[0]["current"] == "方案.md"
    assert result_events[0]["domain"] == "项目"
    assert result_events[0]["topic"] == "方案"

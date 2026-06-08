from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin.knowledge_base_core.source_manifest import read_manifest, write_manifest  # noqa: E402
from backend.services.agent.tools.builtin.knowledge_base_core.source_organizer import organize_sources  # noqa: E402


def test_source_organizer_records_archive_as_single_metadata_candidate(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    archive_path = source_root / "项目交付资料.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("内部文件.txt", "不要展开内部清单")

    canonical_root = tmp_path / "canonical"
    manifest_path = canonical_root / "manifest.jsonl"
    summary = organize_sources([source_root], canonical_root=canonical_root, manifest_path=manifest_path, copy_files=True)

    rows = list(read_manifest(manifest_path))
    assert summary["files"] == 1
    assert summary["metadata_only_candidates"] == 1
    assert len(rows) == 1
    assert rows[0]["file_name"] == "项目交付资料.zip"
    assert rows[0]["extension"] == ".zip"
    assert rows[0]["tags"] == ["archive"]
    assert (canonical_root / "项目交付资料.zip").exists()
    assert not (canonical_root / "内部文件.txt").exists()


def test_source_organizer_skips_media_files(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "photo.png").write_bytes(b"fake image")
    (source_root / "design.psd").write_bytes(b"fake psd")
    (source_root / "poster.ai").write_bytes(b"fake illustrator")
    (source_root / "voice.wav").write_bytes(b"fake audio")
    (source_root / "movie.mp4").write_bytes(b"fake video")
    (source_root / ".remote.png.icloud").write_bytes(b"icloud placeholder")
    (source_root / "notes.md").write_text("should ingest", encoding="utf-8")

    manifest_path = tmp_path / "canonical" / "manifest.jsonl"
    summary = organize_sources(
        [source_root],
        canonical_root=tmp_path / "canonical",
        manifest_path=manifest_path,
        copy_files=False,
    )
    rows = list(read_manifest(manifest_path))

    assert summary["files"] == 1
    assert [row["file_name"] for row in rows] == ["notes.md"]


def test_source_organizer_marks_exact_duplicates(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.md").write_text("same", encoding="utf-8")
    (source_root / "b.md").write_text("same", encoding="utf-8")

    summary = organize_sources(
        [source_root],
        canonical_root=tmp_path / "canonical",
        manifest_path=tmp_path / "canonical" / "manifest.jsonl",
        copy_files=False,
    )
    rows = list(read_manifest(tmp_path / "canonical" / "manifest.jsonl"))

    assert summary["files"] == 2
    assert summary["duplicates"] == 1
    assert sum(1 for row in rows if row["is_duplicate"]) == 1


def test_source_organizer_scan_state_skips_unchanged_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.md").write_text("hello", encoding="utf-8")
    canonical_root = tmp_path / "canonical"
    manifest_path = canonical_root / "manifest.jsonl"
    scan_state_path = canonical_root / "scan_state.json"

    first = organize_sources(
        [source_root],
        canonical_root=canonical_root,
        manifest_path=manifest_path,
        scan_state_path=scan_state_path,
        copy_files=False,
    )
    second = organize_sources(
        [source_root],
        canonical_root=canonical_root,
        manifest_path=manifest_path,
        scan_state_path=scan_state_path,
        copy_files=False,
    )

    assert first["scanned_roots"] == [str(source_root)]
    assert second["scanned_roots"] == []
    assert second["skipped_roots"] == [str(source_root)]
    assert len(list(read_manifest(manifest_path))) == 1


def test_source_organizer_resume_keeps_previous_classification(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "a.md"
    source.write_text("hello", encoding="utf-8")
    canonical_root = tmp_path / "canonical"
    manifest_path = canonical_root / "manifest.jsonl"

    first = organize_sources([source_root], canonical_root=canonical_root, manifest_path=manifest_path, copy_files=False)
    rows = list(read_manifest(manifest_path))
    rows[0].update({"classification_status": "classified", "domain": "项目", "topic": "架构设计", "tags": ["Vitoom"]})
    write_manifest(manifest_path, rows)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("classified row should be skipped in resume mode")

    monkeypatch.setattr(
        "backend.services.agent.tools.builtin.knowledge_base_core.classifier.classify_source_row",
        fail_if_called,
    )
    second = organize_sources(
        [source_root],
        canonical_root=canonical_root,
        manifest_path=manifest_path,
        copy_files=False,
        classify=True,
        resume=True,
    )
    resumed_rows = list(read_manifest(manifest_path))

    assert first["files"] == 1
    assert second["classification"]["skipped"] == 1
    assert resumed_rows[0]["domain"] == "项目"
    assert resumed_rows[0]["topic"] == "架构设计"

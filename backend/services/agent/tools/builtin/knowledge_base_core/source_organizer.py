from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .source_manifest import read_manifest, write_manifest


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}
SKIP_FILES = {".DS_Store"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz"}
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg", ".ico", ".heic", ".heif", ".raw",
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
}
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def stderr_progress(event: str, payload: Dict[str, Any]) -> None:
    elapsed = payload.get("elapsed_seconds")
    elapsed_text = f" elapsed={elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
    current = str(payload.get("current") or "")
    current_text = f" current={current}" if current else ""
    counts = " ".join(f"{key}={value}" for key, value in payload.items() if key not in {"elapsed_seconds", "current"})
    print(f"[knowledge-base:{event}]{elapsed_text} {counts}{current_text}".rstrip(), file=sys.stderr, flush=True)


def _merge_resume_rows(rows: List[Dict[str, Any]], manifest_path: Path, *, resume: bool) -> List[Dict[str, Any]]:
    if not resume or not manifest_path.exists():
        return rows
    previous_by_sha = {str(row.get("sha256") or ""): row for row in read_manifest(manifest_path) if str(row.get("sha256") or "")}
    merged: List[Dict[str, Any]] = []
    resumable_fields = {
        "domain",
        "topic",
        "subtopic",
        "summary",
        "classification_confidence",
        "classification_reason",
        "classification_status",
        "classification_error",
        "tags",
    }
    for row in rows:
        previous = previous_by_sha.get(str(row.get("sha256") or ""))
        if not previous:
            merged.append(row)
            continue
        combined = dict(row)
        for field in resumable_fields:
            if field in previous:
                combined[field] = previous[field]
        merged.append(combined)
    return merged


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_skipped_media_name(name: str) -> bool:
    lower_name = str(name or "").lower()
    return any(lower_name.endswith(extension) or lower_name.endswith(f"{extension}.icloud") for extension in MEDIA_EXTENSIONS)


def is_skipped_media_file(path: Path) -> bool:
    extension = path.suffix.lower()
    if is_skipped_media_name(path.name):
        return True
    if extension in MEDIA_EXTENSIONS:
        return True
    mime_type = mimetypes.guess_type(str(path))[0] or ""
    return mime_type.startswith(("image/", "audio/", "video/"))


def iter_source_files(scan_roots: Iterable[Path]) -> Iterable[Path]:
    for root in scan_roots:
        if not root.exists():
            continue
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [item for item in dirs if item not in SKIP_DIRS]
            for file_name in files:
                if file_name in SKIP_FILES or file_name.startswith("~$"):
                    continue
                path = Path(current_root) / file_name
                if is_skipped_media_file(path):
                    continue
                yield path


def directory_signature(root: Path) -> Dict[str, Any]:
    file_count = 0
    total_size = 0
    max_mtime = 0.0
    for path in iter_source_files([root]):
        try:
            stat = path.stat()
        except OSError:
            continue
        file_count += 1
        total_size += stat.st_size
        max_mtime = max(max_mtime, stat.st_mtime)
    return {"path": str(root), "file_count": file_count, "total_size": total_size, "max_mtime": max_mtime}


def load_scan_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"roots": {}}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {"roots": {}}
    except Exception:
        return {"roots": {}}


def write_scan_state(path: Path, roots: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"roots": roots, "updated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest_rows(
    scan_roots: Iterable[Path],
    *,
    canonical_root: Path,
    max_files: int = 0,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    roots = [Path(item) for item in scan_roots]
    started = time.perf_counter()
    for path in iter_source_files(roots):
        if max_files > 0 and len(rows) >= max_files:
            break
        try:
            stat = path.stat()
            digest = sha256_file(path)
        except OSError:
            continue
        source_root = next((root for root in roots if path.is_relative_to(root)), path.parent)
        relative_path = path.relative_to(source_root)
        document_id = "doc_" + hashlib.sha1(digest.encode("ascii")).hexdigest()[:20]
        duplicate_of = seen_hashes.get(digest, "")
        if not duplicate_of:
            seen_hashes[digest] = document_id
        extension = path.suffix.lower()
        canonical_path = canonical_root / relative_path
        rows.append(
            {
                "document_id": document_id,
                "document_group_id": document_id,
                "source_path": str(path),
                "canonical_path": str(canonical_path),
                "sha256": digest,
                "size_bytes": stat.st_size,
                "file_name": path.name,
                "file_stem": path.stem,
                "extension": extension,
                "mime_type": mimetypes.guess_type(str(path))[0] or "",
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "source_root": str(source_root),
                "relative_source_path": str(relative_path),
                "domain": "未分类_待确认",
                "topic": "",
                "subtopic": "",
                "tags": ["archive"] if extension in ARCHIVE_EXTENSIONS else [],
                "is_duplicate": bool(duplicate_of),
                "duplicate_of": duplicate_of,
                "version_rank": 1,
                "version_label": "latest",
                "tenant_id": "default",
                "owner_user_id": "agent-system",
                "access_level": "public",
                "active": True,
                "deleted": False,
                "archived": False,
            }
        )
        if progress_callback and (len(rows) == 1 or len(rows) % max(1, progress_every) == 0):
            progress_callback(
                "scan",
                {
                    "files": len(rows),
                    "duplicates": sum(1 for row in rows if row.get("is_duplicate")),
                    "elapsed_seconds": time.perf_counter() - started,
                    "current": path.name,
                },
            )
    if progress_callback:
        progress_callback(
            "scan_done",
            {
                "files": len(rows),
                "duplicates": sum(1 for row in rows if row.get("is_duplicate")),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    return rows


def copy_canonical_files(
    rows: Iterable[Dict[str, Any]],
    *,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
) -> int:
    copied = 0
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        if row.get("is_duplicate"):
            continue
        source = Path(str(row.get("source_path") or ""))
        target = Path(str(row.get("canonical_path") or ""))
        if not source.exists() or not target:
            continue
        if dry_run:
            copied += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        shutil.copy2(source, target)
        copied += 1
        if progress_callback and (copied == 1 or copied % max(1, progress_every) == 0):
            progress_callback(
                "copy",
                {
                    "processed": index,
                    "copied": copied,
                    "elapsed_seconds": time.perf_counter() - started,
                    "current": target.name,
                },
            )
    if progress_callback:
        progress_callback("copy_done", {"copied": copied, "elapsed_seconds": time.perf_counter() - started})
    return copied


def organize_sources(
    scan_roots: Iterable[Path],
    *,
    canonical_root: Path,
    manifest_path: Path,
    scan_state_path: Path | None = None,
    copy_files: bool = True,
    skip_previously_scanned_dirs: bool = True,
    classify: bool = False,
    low_confidence_threshold: float = 0.75,
    classifier_user_id: str = "",
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
    resume: bool = True,
    dry_run: bool = False,
    max_files: int = 0,
) -> Dict[str, Any]:
    overall_started = time.perf_counter()
    requested_roots = [Path(item) for item in scan_roots]
    state = load_scan_state(scan_state_path) if scan_state_path else {"roots": {}}
    previous_roots = state.get("roots") if isinstance(state.get("roots"), dict) else {}
    current_signatures = {str(root): directory_signature(root) for root in requested_roots if root.exists()}
    skipped_roots = []
    changed_roots = []
    for root in requested_roots:
        signature = current_signatures.get(str(root))
        if not signature:
            continue
        previous = previous_roots.get(str(root))
        if skip_previously_scanned_dirs and previous == signature:
            skipped_roots.append(str(root))
        else:
            changed_roots.append(root)

    rows = build_manifest_rows(
        changed_roots,
        canonical_root=canonical_root,
        max_files=max_files,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )
    if skipped_roots and manifest_path.exists():
        skipped = set(skipped_roots)
        rows = [row for row in read_manifest(manifest_path) if str(row.get("source_root") or "") in skipped] + rows
    rows = _merge_resume_rows(rows, manifest_path, resume=resume)
    classification_summary = {"classified": 0, "low_confidence": 0, "failed": 0, "skipped": 0}
    if classify:
        from .classifier import classify_rows

        def checkpoint() -> None:
            if not dry_run:
                write_manifest(manifest_path, rows)

        classification_summary = classify_rows(
            rows,
            threshold=low_confidence_threshold,
            user_id=classifier_user_id,
            progress_callback=progress_callback,
            progress_every=progress_every,
            resume=resume,
            checkpoint_callback=checkpoint,
        )
    copied = copy_canonical_files(rows, dry_run=dry_run, progress_callback=progress_callback, progress_every=progress_every) if copy_files else 0
    if not dry_run:
        write_manifest(manifest_path, rows)
        if scan_state_path:
            merged_roots = dict(previous_roots)
            merged_roots.update(current_signatures)
            write_scan_state(scan_state_path, merged_roots)
    duplicates = sum(1 for row in rows if row.get("is_duplicate"))
    metadata_only = sum(1 for row in rows if str(row.get("extension") or "").lower() in ARCHIVE_EXTENSIONS or str(row.get("extension") or "").lower() == ".key")
    return {
        "files": len(rows),
        "duplicates": duplicates,
        "metadata_only_candidates": metadata_only,
        "classification": classification_summary,
        "copied": copied,
        "scanned_roots": [str(root) for root in changed_roots],
        "skipped_roots": skipped_roots,
        "manifest_path": str(manifest_path),
        "scan_state_path": str(scan_state_path or ""),
        "dry_run": dry_run,
        "elapsed_seconds": round(time.perf_counter() - overall_started, 3),
    }

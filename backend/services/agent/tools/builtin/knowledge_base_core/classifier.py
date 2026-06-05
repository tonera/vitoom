from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List

from backend.services.agent.settings import get_agent_internal_user_id
from backend.services.agent.tools.builtin.business_query_core.planner_base import parse_llm_json_object, run_agent_planner_completion


DEFAULT_TAXONOMY = ("财务", "项目", "人事", "合同法务", "产品", "技术", "未分类_待确认")
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clean_tags(value: Any) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def preview_text_for_classification(row: Dict[str, Any], *, max_chars: int = 3000) -> str:
    path = Path(str(row.get("source_path") or row.get("canonical_path") or ""))
    if path.suffix.lower() not in {".md", ".markdown", ".txt", ".html", ".htm"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def resolve_classifier_user_id(user_id: str = "") -> str:
    configured = str(user_id or "").strip()
    if configured and configured != get_agent_internal_user_id():
        return configured
    try:
        from backend.database.models import User

        users = User.list_all(limit=20)
    except Exception:
        users = []
    active_users = [user for user in users if str(user.get("status") or "active") == "active"]
    admin_users = [user for user in active_users if bool(user.get("is_admin"))]
    selected = (admin_users or active_users or users or [{}])[0]
    resolved = str(selected.get("id") or "").strip()
    if not resolved:
        raise RuntimeError("知识库 LLM 分类需要有效用户 ID。请传入 --classifier-user-id <用户ID>。")
    return resolved


def normalize_classification(raw: Dict[str, Any], *, threshold: float = 0.75) -> Dict[str, Any]:
    domain = str(raw.get("domain") or "").strip()
    if domain not in DEFAULT_TAXONOMY:
        domain = "未分类_待确认"
    confidence = _coerce_confidence(raw.get("confidence"))
    low_confidence = confidence < threshold
    return {
        "domain": "未分类_待确认" if low_confidence else domain,
        "topic": "" if low_confidence else str(raw.get("topic") or "").strip(),
        "subtopic": "" if low_confidence else str(raw.get("subtopic") or "").strip(),
        "summary": str(raw.get("summary") or "").strip(),
        "classification_confidence": confidence,
        "classification_reason": str(raw.get("reason") or "").strip(),
        "tags": _clean_tags(raw.get("suggested_tags") or raw.get("tags")),
        "classification_status": "low_confidence" if low_confidence else "classified",
    }


def classify_source_row(
    row: Dict[str, Any],
    *,
    threshold: float = 0.75,
    user_id: str = "agent-system",
    preview_text: str = "",
) -> Dict[str, Any]:
    preview = preview_text if preview_text else preview_text_for_classification(row)
    effective_user_id = resolve_classifier_user_id(user_id)
    messages = [
        {
            "role": "system",
            "content": (
                "你是企业本地知识库源文件分类器。只根据用户给出的文件元数据和少量预览分类，不要臆测。"
                "可选 domain 只能是：财务、项目、人事、合同法务、产品、技术、未分类_待确认。"
                "必须只输出 JSON 对象，字段为 domain/topic/subtopic/summary/confidence/reason/suggested_tags。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "file_name": row.get("file_name") or "",
                    "relative_source_path": row.get("relative_source_path") or "",
                    "source_path": row.get("source_path") or "",
                    "extension": row.get("extension") or "",
                    "mime_type": row.get("mime_type") or "",
                    "size_bytes": row.get("size_bytes") or 0,
                    "modified_at": row.get("modified_at") or "",
                    "preview": preview[:3000],
                },
                ensure_ascii=False,
            ),
        },
    ]
    raw = run_agent_planner_completion(messages, user_id=effective_user_id, error_label="knowledge source classifier")
    parsed = parse_llm_json_object(raw, error_message="knowledge source classifier must return a JSON object")
    return normalize_classification(parsed, threshold=threshold)


def classify_rows(
    rows: List[Dict[str, Any]],
    *,
    threshold: float = 0.75,
    user_id: str = "",
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 25,
    resume: bool = True,
    checkpoint_callback: Callable[[], None] | None = None,
) -> Dict[str, int]:
    classified = 0
    low_confidence = 0
    failed = 0
    skipped = 0
    started = time.perf_counter()
    total = sum(1 for row in rows if not row.get("is_duplicate"))
    if progress_callback:
        progress_callback(
            "classify_start",
            {
                "total": total,
                "resume": resume,
                "elapsed_seconds": 0.0,
            },
        )
    processed = 0
    visited = 0
    for row in rows:
        if row.get("is_duplicate"):
            continue
        visited += 1
        status = str(row.get("classification_status") or "").strip()
        if resume and status in {"classified", "low_confidence"}:
            skipped += 1
            if status == "classified":
                classified += 1
            else:
                low_confidence += 1
            if progress_callback and (visited == 1 or visited % max(1, progress_every) == 0 or visited == total):
                progress_callback(
                    "classify",
                    {
                        "visited": visited,
                        "processed": processed,
                        "total": total,
                        "skipped": skipped,
                        "classified": classified,
                        "low_confidence": low_confidence,
                        "failed": failed,
                        "elapsed_seconds": time.perf_counter() - started,
                        "current": row.get("file_name") or "",
                    },
                )
            continue
        processed += 1
        try:
            result = classify_source_row(row, threshold=threshold, user_id=user_id)
            existing_tags = _clean_tags(row.get("tags"))
            row.update(result)
            row["tags"] = sorted(set(existing_tags + _clean_tags(result.get("tags"))))
            if result["classification_status"] == "low_confidence":
                low_confidence += 1
            else:
                classified += 1
        except Exception as exc:
            row["classification_status"] = "failed"
            row["classification_error"] = f"{type(exc).__name__}: {exc}"
            row["domain"] = "未分类_待确认"
            failed += 1
        if progress_callback:
            progress_callback(
                "classify_result",
                {
                    "status": row.get("classification_status") or "",
                    "domain": row.get("domain") or "",
                    "topic": row.get("topic") or "",
                    "subtopic": row.get("subtopic") or "",
                    "confidence": row.get("classification_confidence", ""),
                    "current": row.get("file_name") or "",
                },
            )
        if checkpoint_callback:
            checkpoint_callback()
        if progress_callback and (processed == 1 or processed % max(1, progress_every) == 0 or processed == total):
            progress_callback(
                "classify",
                {
                    "visited": visited,
                    "processed": processed,
                    "total": total,
                    "skipped": skipped,
                    "classified": classified,
                    "low_confidence": low_confidence,
                    "failed": failed,
                    "elapsed_seconds": time.perf_counter() - started,
                    "current": row.get("file_name") or "",
                },
            )
    if progress_callback:
        progress_callback(
            "classify_done",
            {
                    "visited": visited,
                "processed": processed,
                "total": total,
                "skipped": skipped,
                "classified": classified,
                "low_confidence": low_confidence,
                "failed": failed,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    return {"classified": classified, "low_confidence": low_confidence, "failed": failed, "skipped": skipped}

"""process_url_content 工具：URL 内容处理 facade。

普通 Master Agent 只需要面对这个高层入口；网页读取、文档转换、多模态理解、
OCR 和表格导出等底层工具由本工具内部按 URL 类型与用户目标编排。
"""

from __future__ import annotations

import json
import logging
import math
import mimetypes
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import httpx

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import clean_optional_str, coerce_optional, coerce_timeout_seconds
from ._media_common import submit_and_collect

logger = logging.getLogger(__name__)

PROCESS_URL_CONTENT_TOOL_NAME = "process_url_content"

PROCESS_URL_CONTENT_DESCRIPTION = (
    "统一处理用户提供的网页、文档、图片、视频、PDF、Office、文本或压缩包 URL。"
    "适合总结/分析 URL 内容、文档转 Markdown/PDF、图片/PDF OCR 文字/表格/公式、"
    "指定字段抽取、zip 清单检查等 URL 内容任务。普通对话中遇到 URL 内容处理时优先调用本工具，"
    "不要再直接调用 web_page_reader/analyze_media/document_to_markdown/document_to_pdf/table_to_excel。"
)

PROCESS_URL_CONTENT_DOCSTRING = (
    "Facade tool for URL content processing. Invoke with `question` and `urls`; "
    "optional `output_format_hint`, `fields_hint`, and `context`. The tool inspects URL kind, "
    "classifies the content task, and routes to webpage reading, media analysis, document conversion, "
    "mini OCR, or table export as needed. Returns JSON for the agent to summarize or surface files."
)

_REMOTE_URL_PREFIXES = ("http://", "https://")
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv", ".avi", ".flv", ".m4v", ".3gp"})
_AUDIO_EXTS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".amr"})
_PDF_EXTS = frozenset({".pdf"})
_OFFICE_EXTS = frozenset({".doc", ".docx", ".docm", ".ppt", ".pptx", ".pptm"})
_SPREADSHEET_EXTS = frozenset({".xls", ".xlsx", ".xlsm", ".csv", ".tsv"})
_TEXT_EXTS = frozenset({".txt", ".rtf", ".epub"})
_MARKDOWN_EXTS = frozenset({".md", ".markdown", ".mdown", ".mkd", ".rmd", ".qmd"})
_ZIP_EXTS = frozenset({".zip"})

_ALL_TASKS = (
    "summarize",
    "analyze",
    "convert_to_markdown",
    "convert_to_pdf",
    "extract_text",
    "extract_formula",
    "extract_table",
    "extract_fields",
    "inspect_zip",
)

_ALLOWED_TASKS_BY_KIND: Dict[str, tuple[str, ...]] = {
    "web_page": ("summarize", "analyze", "extract_fields"),
    "image": ("analyze", "extract_text", "extract_table", "extract_formula", "extract_fields"),
    "video": ("summarize", "analyze"),
    "pdf": (
        "summarize",
        "analyze",
        "convert_to_markdown",
        "extract_text",
        "extract_table",
        "extract_formula",
        "extract_fields",
    ),
    "office": ("summarize", "analyze", "convert_to_markdown", "convert_to_pdf", "extract_fields"),
    "spreadsheet": ("summarize", "analyze", "convert_to_markdown", "convert_to_pdf", "extract_fields"),
    "text": ("summarize", "analyze", "convert_to_markdown", "convert_to_pdf", "extract_fields"),
    "markdown": ("summarize", "analyze", "convert_to_pdf", "extract_fields"),
    "zip": ("inspect_zip", "convert_to_markdown", "convert_to_pdf"),
    "unknown": _ALL_TASKS,
}

_TASK_ANCHORS: Dict[str, tuple[str, ...]] = {
    "summarize": (
        "summarize the content",
        "what is this document about",
        "explain the main points",
        "总结主要内容",
        "这个网页讲了什么",
        "この文書の内容を要約する",
    ),
    "analyze": (
        "analyze this content",
        "what is this",
        "describe this image or video",
        "分析一下内容",
        "这是什么",
        "この内容を分析する",
    ),
    "convert_to_markdown": (
        "convert this file to markdown",
        "export as md",
        "save as markdown",
        "转成 Markdown 文档",
        "导出为 md",
        "Markdown形式に変換する",
    ),
    "convert_to_pdf": (
        "convert this file to pdf",
        "export as pdf",
        "save as pdf",
        "转成 PDF 文档",
        "导出为 pdf",
        "PDF形式に変換する",
    ),
    "extract_text": (
        "extract text with ocr",
        "recognize text in image or pdf",
        "提取文字",
        "识别图片文字",
        "文字認識",
    ),
    "extract_formula": (
        "extract formulas",
        "recognize math formulas",
        "提取公式",
        "识别数学公式",
        "数式を認識する",
    ),
    "extract_table": (
        "extract tables to excel",
        "export table from image or pdf",
        "save table as xlsx",
        "提取表格并导出 Excel",
        "表を抽出する",
    ),
    "extract_fields": (
        "extract specified fields",
        "extract name age date amount",
        "抽取指定字段",
        "提取姓名年龄籍贯",
        "指定項目を抽出する",
    ),
    "inspect_zip": (
        "list files in zip",
        "inspect archive contents",
        "压缩包里有什么文件",
        "统计 zip 文件类型和数量",
        "zipの中身を確認する",
    ),
}

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")
_CONVERT_INTENT_RE = re.compile(
    r"(?:convert|export|save\s+as|转(?:为|成)|转换|导出|保存(?:为|成))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UrlInspection:
    url: str
    kind: str
    suffix: str = ""
    mime: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class TaskDecision:
    task: str
    confidence: float
    allowed_tasks: tuple[str, ...]
    reason: str


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _clean_str(value: Any) -> str:
    return clean_optional_str(value) or ""


def _normalize_urls(url: Any = None, urls: Any = None) -> List[str]:
    def _coerce(value: Any) -> List[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, (list, tuple, set)):
            return [item for item in (_clean_str(v) for v in value) if item]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    return [item for item in (_clean_str(v) for v in parsed) if item]
            return [text]
        normalized = _clean_str(value)
        return [normalized] if normalized else []

    result: List[str] = []
    seen = set()
    for item in _coerce(url) + _coerce(urls):
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _kind_from_suffix(suffix: str) -> str:
    normalized = str(suffix or "").lower()
    if normalized in _IMAGE_EXTS:
        return "image"
    if normalized in _VIDEO_EXTS:
        return "video"
    if normalized in _AUDIO_EXTS:
        return "audio"
    if normalized in _PDF_EXTS:
        return "pdf"
    if normalized in _OFFICE_EXTS:
        return "office"
    if normalized in _SPREADSHEET_EXTS:
        return "spreadsheet"
    if normalized in _MARKDOWN_EXTS:
        return "markdown"
    if normalized in _TEXT_EXTS:
        return "text"
    if normalized in _ZIP_EXTS:
        return "zip"
    return ""


def _kind_from_mime(mime: str) -> str:
    normalized = str(mime or "").split(";", 1)[0].strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith("video/"):
        return "video"
    if normalized.startswith("audio/"):
        return "audio"
    if normalized in {"application/pdf", "application/x-pdf"}:
        return "pdf"
    if normalized in {"application/zip", "application/x-zip-compressed"}:
        return "zip"
    if normalized in {"text/markdown", "text/x-markdown"}:
        return "markdown"
    if normalized.startswith("text/"):
        return "web_page" if normalized == "text/html" else "text"
    if "spreadsheet" in normalized or "excel" in normalized or normalized.endswith("/csv"):
        return "spreadsheet"
    if "wordprocessingml" in normalized or "presentationml" in normalized or "msword" in normalized or "powerpoint" in normalized:
        return "office"
    return ""


def _sniff_remote_mime(url: str, *, timeout: float) -> str:
    if not str(url or "").lower().startswith(_REMOTE_URL_PREFIXES):
        return ""
    try:
        with httpx.Client(timeout=min(max(timeout, 1.0), 8.0), follow_redirects=True) as client:
            try:
                response = client.head(url)
                content_type = str(response.headers.get("content-type") or "").strip().lower()
                if content_type:
                    return content_type
            except Exception:
                pass
            response = client.get(url, headers={"Range": "bytes=0-0"})
            return str(response.headers.get("content-type") or "").strip().lower()
    except Exception:
        return ""


def _read_text_url_sync(url: str, *, timeout: float, max_chars: int = 60000) -> Dict[str, Any]:
    text_url = _clean_str(url)
    if not text_url.lower().startswith(_REMOTE_URL_PREFIXES):
        raise RuntimeError(f"only http(s) text URLs are supported, got: {text_url}")
    with httpx.Client(timeout=min(max(timeout, 1.0), 30.0), follow_redirects=True) as client:
        response = client.get(text_url)
        response.raise_for_status()
    content = response.text or ""
    clipped = content[:max_chars].rstrip()
    return {
        "tool": PROCESS_URL_CONTENT_TOOL_NAME,
        "status": "completed",
        "url": text_url,
        "content": clipped,
        "truncated": len(content) > len(clipped),
        "original_length": len(content),
        "returned_length": len(clipped),
    }


def inspect_url(url: str, *, timeout: float = 5.0) -> UrlInspection:
    text = _clean_str(url)
    parsed = urlparse(text)
    suffix = PurePosixPath(parsed.path or "").suffix.lower()
    suffix_kind = _kind_from_suffix(suffix)
    mime = ""
    mime_kind = ""

    if not suffix_kind or suffix_kind == "web_page":
        mime = _sniff_remote_mime(text, timeout=timeout)
        mime_kind = _kind_from_mime(mime)
    elif suffix_kind in {"pdf", "zip", "office", "spreadsheet", "text", "markdown"}:
        mime = mimetypes.guess_type(parsed.path or "")[0] or ""

    kind = mime_kind or suffix_kind or "web_page"
    confidence = 0.9 if suffix_kind else 0.75 if mime_kind else 0.45
    return UrlInspection(url=text, kind=kind, suffix=suffix, mime=mime, confidence=confidence)


def _ngrams(text: str, *, min_n: int = 2, max_n: int = 3) -> List[str]:
    length = len(text)
    result: List[str] = []
    for size in range(min_n, min(max_n, length) + 1):
        result.extend(text[idx : idx + size] for idx in range(length - size + 1))
    return result


def _tokens(text: str) -> Dict[str, int]:
    raw = str(text or "").lower().strip()
    counts: Dict[str, int] = {}
    if not raw:
        return counts
    for match in _ASCII_TOKEN_RE.finditer(raw):
        token = match.group(0)
        counts[token] = counts.get(token, 0) + 1
        for piece in re.split(r"[_-]+", token):
            if piece:
                counts[piece] = counts.get(piece, 0) + 1
    for match in _CJK_RE.finditer(raw):
        chunk = re.sub(r"\s+", "", match.group(0))
        if not chunk:
            continue
        counts[chunk] = counts.get(chunk, 0) + 1
        for ngram in _ngrams(chunk):
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _cosine(left: Dict[str, int], right: Dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in shared)
    if numerator <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_output_format_hint(value: Any) -> str:
    text = _clean_str(value).lower()
    aliases = {
        "md": "markdown",
        "xlsx": "excel",
        "xls": "excel",
        "zip_summary": "zip",
    }
    return aliases.get(text, text or "auto")


def _looks_like_conversion_request(question: str) -> bool:
    return bool(_CONVERT_INTENT_RE.search(str(question or "")))


def _coerce_fields_hint(value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return _coerce_fields_hint(parsed)
        return [item.strip() for item in re.split(r"[,，、\n]+", text) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            if isinstance(item, dict):
                name = _clean_str(item.get("name") or item.get("field") or item.get("key"))
            else:
                name = _clean_str(item)
            if name:
                result.append(name)
        return result
    return []


def _public_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(context or {})
    allowed = ("conversation_id", "turn_id", "input_mode", "source")
    return {key: raw.get(key) for key in allowed if raw.get(key) not in (None, "")}


def _allowed_tasks_for(inspections: Sequence[UrlInspection]) -> tuple[str, ...]:
    if not inspections:
        return _ALL_TASKS
    allowed_sets = [
        set(_ALLOWED_TASKS_BY_KIND.get(item.kind, _ALLOWED_TASKS_BY_KIND["unknown"]))
        for item in inspections
    ]
    shared = set.intersection(*allowed_sets) if allowed_sets else set(_ALL_TASKS)
    if shared:
        return tuple(task for task in _ALL_TASKS if task in shared)
    # 多 URL 类型混杂时用并集，后续逐 item 执行。
    union = set().union(*allowed_sets)
    return tuple(task for task in _ALL_TASKS if task in union)


def classify_task(
    *,
    question: str,
    inspections: Sequence[UrlInspection],
    output_format_hint: Any = None,
    fields_hint: Any = None,
) -> TaskDecision:
    allowed = _allowed_tasks_for(inspections)
    output_format = _normalize_output_format_hint(output_format_hint)
    fields = _coerce_fields_hint(fields_hint)
    kinds = {item.kind for item in inspections}

    hint_task = ""
    if fields and "extract_fields" in allowed:
        hint_task = "extract_fields"
    elif output_format == "pdf" and "convert_to_pdf" in allowed and _looks_like_conversion_request(question):
        hint_task = "convert_to_pdf"
    elif output_format == "markdown" and "convert_to_markdown" in allowed and _looks_like_conversion_request(question):
        hint_task = "convert_to_markdown"
    elif output_format == "excel" and "extract_table" in allowed:
        hint_task = "extract_table"
    elif output_format == "zip" and "inspect_zip" in allowed:
        hint_task = "inspect_zip"
    if hint_task:
        return TaskDecision(hint_task, 0.95, allowed, f"hint matched {hint_task}")

    if kinds <= {"markdown", "text"} and not fields and not _looks_like_conversion_request(question) and "summarize" in allowed:
        return TaskDecision("summarize", 0.75, allowed, "existing text document default")

    query_vec = _tokens(question)
    scored: List[tuple[str, float]] = []
    for task in allowed:
        best = 0.0
        for anchor in _TASK_ANCHORS.get(task, ()):
            best = max(best, _cosine(query_vec, _tokens(anchor)))
        scored.append((task, best))
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored and scored[0][1] > 0:
        top_task, top_score = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        confidence = min(0.95, 0.55 + top_score + max(0.0, top_score - second))
        if confidence >= 0.62:
            return TaskDecision(top_task, confidence, allowed, "intent anchor match")

    if "zip" in kinds and "inspect_zip" in allowed:
        return TaskDecision("inspect_zip", 0.7, allowed, "zip default")
    if kinds <= {"image", "video"} and "analyze" in allowed:
        return TaskDecision("analyze", 0.65, allowed, "media default")
    if "summarize" in allowed:
        return TaskDecision("summarize", 0.6, allowed, "summary default")
    return TaskDecision(allowed[0] if allowed else "analyze", 0.45, allowed, "fallback")


def _default_timeout() -> float:
    return 300.0


def _mini_model_name() -> str:
    try:
        from backend.services.agent.settings import get_document_to_markdown_pdf_model_name

        return str(get_document_to_markdown_pdf_model_name() or "").strip()
    except Exception:
        return "GLM-OCR"


def _run_mini_ocr_sync(
    *,
    user_id: str,
    urls: Sequence[str],
    ocr_task: str,
    fields: Sequence[str],
    timeout: float,
    storage: str,
    task_event_callback: Any = None,
) -> Dict[str, Any]:
    from backend.api.tasks.routes import TaskCreateRequest

    extract: Dict[str, Any] = {"task": "extract" if ocr_task == "extract_fields" else ocr_task}
    if ocr_task == "extract_fields":
        extract["schema"] = {
            field: {"type": "string", "description": field}
            for field in fields
        } or {"key_information": {"type": "string", "description": "关键结构化信息"}}
    file_type = "json" if ocr_task == "extract_fields" else "md"
    request = TaskCreateRequest(
        task_type="mini",
        job_type="OCR",
        load_name=_mini_model_name(),
        tpl_list=list(urls),
        extract=extract,
        file_type=file_type,
        storage=storage,
    )
    return submit_and_collect(
        tool_name=PROCESS_URL_CONTENT_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=max(1, len(urls)),
        effective_timeout=timeout,
        task_kind_label=f"mini-ocr-{ocr_task}",
        file_fields=(
            "file_id",
            "url",
            "thumb_url",
            "storage_path",
            "file_name",
            "file_size",
            "mime_type",
            "index",
        ),
        extra_result_fields=("content", "file_type"),
        task_event_callback=task_event_callback if callable(task_event_callback) else None,
    )


def _inspect_zip_urls(urls: Sequence[str], inspections: Sequence[UrlInspection]) -> Dict[str, Any]:
    return {
        "tool": PROCESS_URL_CONTENT_TOOL_NAME,
        "status": "completed",
        "task": "inspect_zip",
        "message": "已识别为压缩包 URL。当前轻量检查只返回压缩包条目；如需读取内部清单，请使用转换任务触发下载解包。",
        "items": [
            {
                "url": item.url,
                "kind": item.kind,
                "suffix": item.suffix,
                "mime": item.mime,
            }
            for item in inspections
        ],
        "total": len(urls),
    }


def _collect_result_files(value: Any) -> List[Dict[str, Any]]:
    """从底层工具结果里递归收集 files，提升给聊天附件解析器使用。"""

    results: List[Dict[str, Any]] = []
    seen = set()

    def _append(info: Any, *, task_id: str = "", preview_md: str = "") -> None:
        if not isinstance(info, dict):
            return
        item = dict(info)
        if task_id and not item.get("task_id") and not item.get("derived_task_id"):
            item["task_id"] = task_id
        if preview_md and not item.get("preview_md") and not item.get("preview_text"):
            item["preview_md"] = preview_md
        key = (
            str(item.get("file_id") or item.get("id") or "").strip()
            or str(item.get("url") or item.get("http_url") or "").strip()
            or str(item.get("storage_path") or "").strip()
            or json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        )
        if key in seen:
            return
        seen.add(key)
        results.append(item)

    def _walk(payload: Any, *, inherited_task_id: str = "", inherited_preview: str = "") -> None:
        if isinstance(payload, dict):
            task_id = _clean_str(payload.get("task_id")) or inherited_task_id
            preview = _clean_str(payload.get("preview_md") or payload.get("preview_text")) or inherited_preview
            for file_info in payload.get("files") or []:
                _append(file_info, task_id=task_id, preview_md=preview)
            for item in payload.get("items") or []:
                _walk(item, inherited_task_id=task_id, inherited_preview=preview)
            for key, child in payload.items():
                if key in {"files", "items"}:
                    continue
                if isinstance(child, (dict, list)):
                    _walk(child, inherited_task_id=task_id, inherited_preview=preview)
        elif isinstance(payload, list):
            for item in payload:
                _walk(item, inherited_task_id=inherited_task_id, inherited_preview=inherited_preview)

    _walk(value)
    return results


def _expose_nested_artifacts(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把 facade 内部 result 的文件提升到顶层，兼容现有前端附件渲染。"""

    files = _collect_result_files(payload.get("result"))
    if not files:
        return payload
    exposed = dict(payload)
    exposed["files"] = files
    exposed["total_files"] = len(files)
    return exposed


def _nested_tool_hooks(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = (context or {}).get("session_nested_tool_hooks") if isinstance(context, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _invoke_atomic_tool(
    *,
    context: Optional[Dict[str, Any]],
    tool_name: str,
    arguments: Dict[str, Any],
    func: Callable[[], Any],
) -> Any:
    """执行 facade 内部原子能力，并向前端事件窗口暴露子工具事件。"""

    hooks = _nested_tool_hooks(context)
    event_id = f"{tool_name}:{uuid.uuid4().hex[:8]}"
    parent = PROCESS_URL_CONTENT_TOOL_NAME
    started = hooks.get("emit_tool_started")
    finished = hooks.get("emit_tool_finished")
    failed = hooks.get("emit_tool_failed")
    if callable(started):
        try:
            started(tool_name, arguments, event_id, parent)
        except Exception:
            logger.debug("process_url_content nested start event failed: %s", tool_name)
    try:
        result = func()
    except Exception as exc:
        if callable(failed):
            try:
                failed(tool_name, str(exc), event_id, parent)
            except Exception:
                logger.debug("process_url_content nested failed event failed: %s", tool_name)
        raise
    if callable(finished):
        try:
            finished(tool_name, result, event_id, parent)
        except Exception:
            logger.debug("process_url_content nested finish event failed: %s", tool_name)
    return result


def process_url_content_sync(
    *,
    user_id: str,
    question: Any,
    urls: Any = None,
    url: Any = None,
    output_format_hint: Any = "auto",
    fields_hint: Any = None,
    context: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    storage: str = "local",
) -> Dict[str, Any]:
    if not _clean_str(user_id):
        return {"tool": PROCESS_URL_CONTENT_TOOL_NAME, "status": "failed", "error": "process_url_content requires a user_id"}
    source_urls = _normalize_urls(url=url, urls=urls)
    if not source_urls:
        return {"tool": PROCESS_URL_CONTENT_TOOL_NAME, "status": "failed", "error": "at least one URL is required"}

    effective_timeout = coerce_timeout_seconds(timeout, get_default=_default_timeout)
    inspections = [inspect_url(item, timeout=min(effective_timeout, 8.0)) for item in source_urls]
    decision = classify_task(
        question=_clean_str(question),
        inspections=inspections,
        output_format_hint=output_format_hint,
        fields_hint=fields_hint,
    )
    fields = _coerce_fields_hint(fields_hint)
    kinds = {item.kind for item in inspections}
    payload_base = {
        "tool": PROCESS_URL_CONTENT_TOOL_NAME,
        "status": "completed",
        "task": decision.task,
        "decision": {
            "task": decision.task,
            "confidence": decision.confidence,
            "allowed_tasks": list(decision.allowed_tasks),
            "reason": decision.reason,
        },
        "urls": source_urls,
        "inspections": [item.__dict__ for item in inspections],
        "context": _public_context(context),
    }

    try:
        if decision.task == "convert_to_markdown":
            from .document_to_markdown import convert_documents_sync

            result = _invoke_atomic_tool(
                context=context,
                tool_name="document_to_markdown",
                arguments={"urls": source_urls, "timeout": effective_timeout},
                func=lambda: convert_documents_sync(
                    user_id=user_id,
                    urls=source_urls,
                    timeout=effective_timeout,
                    storage=storage,
                ),
            )
            return _expose_nested_artifacts({**payload_base, "route": "document_to_markdown", "result": result})
        if decision.task == "convert_to_pdf":
            from .document_to_pdf import convert_documents_to_pdf_sync

            result = _invoke_atomic_tool(
                context=context,
                tool_name="document_to_pdf",
                arguments={"urls": source_urls, "timeout": effective_timeout},
                func=lambda: convert_documents_to_pdf_sync(
                    user_id=user_id,
                    urls=source_urls,
                    timeout=effective_timeout,
                    storage=storage,
                ),
            )
            return _expose_nested_artifacts({**payload_base, "route": "document_to_pdf", "result": result})
        if decision.task == "inspect_zip":
            result = _invoke_atomic_tool(
                context=context,
                tool_name="inspect_zip",
                arguments={"urls": source_urls},
                func=lambda: _inspect_zip_urls(source_urls, inspections),
            )
            return _expose_nested_artifacts({**payload_base, "route": "inspect_zip", "result": result})

        if decision.task in {"extract_text", "extract_formula", "extract_table", "extract_fields"}:
            allowed_ocr_kinds = {"image", "pdf"}
            if not kinds <= allowed_ocr_kinds:
                return {
                    **payload_base,
                    "status": "failed",
                    "error": f"{decision.task} only supports image/PDF URLs, got: {sorted(kinds)}",
                }
            ocr_task = {
                "extract_text": "text",
                "extract_formula": "formula",
                "extract_table": "table",
                "extract_fields": "extract_fields",
            }[decision.task]
            ocr_result = _invoke_atomic_tool(
                context=context,
                tool_name="mini_ocr",
                arguments={"urls": source_urls, "ocr_task": ocr_task, "fields": fields},
                func=lambda: _run_mini_ocr_sync(
                    user_id=user_id,
                    urls=source_urls,
                    ocr_task=ocr_task,
                    fields=fields,
                    timeout=effective_timeout,
                    storage=storage,
                    task_event_callback=(context or {}).get("task_event_callback") if isinstance(context, dict) else None,
                ),
            )
            result: Dict[str, Any] = {"ocr": ocr_result}
            if decision.task == "extract_table" and _normalize_output_format_hint(output_format_hint) == "excel":
                content = _clean_str(ocr_result.get("content"))
                if content:
                    from .table_to_excel import export_table_to_excel_sync

                    result["excel"] = _invoke_atomic_tool(
                        context=context,
                        tool_name="table_to_excel",
                        arguments={"filename": "extracted-table"},
                        func=lambda: export_table_to_excel_sync(
                            user_id=user_id,
                            content=content,
                            filename="extracted-table",
                            storage=storage,
                        ),
                    )
            return _expose_nested_artifacts({**payload_base, "route": f"mini_ocr:{ocr_task}", "result": result})

        if kinds <= {"image", "video"}:
            from .analyze_media import _do_analyze_media

            result_text = _invoke_atomic_tool(
                context=context,
                tool_name="analyze_media",
                arguments={"urls": source_urls, "question": _clean_str(question)},
                func=lambda: _do_analyze_media(
                    effective_user_id=user_id,
                    url=source_urls,
                    question=_clean_str(question),
                ),
            )
            return _expose_nested_artifacts({**payload_base, "route": "analyze_media", "result": {"content": result_text}})

        if kinds <= {"web_page"}:
            from .web_page_reader import read_web_page_sync

            def _read_pages() -> List[Dict[str, Any]]:
                return [
                    read_web_page_sync(url=item, question=question, max_chars=12000)
                    for item in source_urls
                ]

            items = _invoke_atomic_tool(
                context=context,
                tool_name="web_page_reader",
                arguments={"urls": source_urls, "question": _clean_str(question)},
                func=_read_pages,
            )
            return _expose_nested_artifacts({**payload_base, "route": "web_page_reader", "result": {"items": items, "total": len(items)}})

        if decision.task in {"summarize", "analyze", "extract_fields"} and kinds <= {"markdown", "text"}:
            def _read_text_items() -> List[Dict[str, Any]]:
                return [
                    _read_text_url_sync(item, timeout=effective_timeout)
                    for item in source_urls
                ]

            items = _invoke_atomic_tool(
                context=context,
                tool_name="text_url_reader",
                arguments={"urls": source_urls, "question": _clean_str(question)},
                func=_read_text_items,
            )
            return _expose_nested_artifacts({**payload_base, "route": "text_url_reader", "result": {"items": items, "total": len(items)}})

        # 文档类总结/分析：先转 Markdown，LLM 再基于 preview/files 整理回答。
        from .document_to_markdown import convert_documents_sync

        result = _invoke_atomic_tool(
            context=context,
            tool_name="document_to_markdown",
            arguments={"urls": source_urls, "timeout": effective_timeout},
            func=lambda: convert_documents_sync(
                user_id=user_id,
                urls=source_urls,
                timeout=effective_timeout,
                storage=storage,
            ),
        )
        return _expose_nested_artifacts({**payload_base, "route": "document_to_markdown", "result": result})
    except Exception as exc:
        logger.exception("process_url_content failed")
        return {**payload_base, "status": "failed", "error": str(exc)}


@register_tool(
    name=PROCESS_URL_CONTENT_TOOL_NAME,
    description=PROCESS_URL_CONTENT_DESCRIPTION,
    tags=["url", "document", "webpage", "media", "ocr", "pdf", "网页", "文档", "图片", "视频", "OCR"],
    provider="local",
    enabled=True,
)
def build_process_url_content_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()

    try:
        from crewai.tools import BaseTool
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, ConfigDict, Field, field_validator
    except Exception as exc:
        raise RuntimeError("pydantic is required to build process_url_content tool") from exc

    class ProcessUrlContentArgs(BaseModel):
        model_config = ConfigDict(extra="ignore")

        question: str = Field(default="", description="User's original request about the URL, or rewritten request from context.")
        urls: Optional[Any] = Field(default=None, description="One or more URLs; string array or JSON array string supported.")
        url: Optional[str] = Field(default=None, description="Single URL.")
        output_format_hint: Optional[str] = Field(
            default="auto",
            description="Coarse output format hint: text/markdown/pdf/excel/json/zip/auto.",
        )
        fields_hint: Optional[Any] = Field(default=None, description="Field extraction hints, e.g. ['name','age'].")
        timeout: Optional[float] = Field(default=None, description="Processing timeout in seconds.")

        @field_validator("question", "urls", "url", "output_format_hint", "fields_hint", "timeout", mode="before")
        @classmethod
        def _normalize_llm_string_nones(cls, value: Any) -> Any:
            return coerce_optional(value)

    class ProcessUrlContentTool(BaseTool):
        name: str = PROCESS_URL_CONTENT_TOOL_NAME
        description: str = PROCESS_URL_CONTENT_DESCRIPTION
        args_schema: type = ProcessUrlContentArgs

        def _run(self, **kwargs: Any) -> str:
            try:
                args = ProcessUrlContentArgs.model_validate(kwargs)
            except Exception as exc:
                payload = {
                    "tool": PROCESS_URL_CONTENT_TOOL_NAME,
                    "status": "failed",
                    "error": f"invalid tool arguments: {exc}",
                }
                return f"```json\n{_json_dumps(payload)}\n```"
            payload = process_url_content_sync(
                user_id=bound_user_id,
                context=ctx,
                **args.model_dump(),
            )
            return f"```json\n{_json_dumps(payload)}\n```"

    tool_instance = ProcessUrlContentTool()
    tool_instance.__doc__ = PROCESS_URL_CONTENT_DOCSTRING
    return tool_instance

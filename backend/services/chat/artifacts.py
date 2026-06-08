"""chat 结果文件（artifacts / files）归一化辅助。

目标：
1. 普通 Master 工具调用与 slash command 共用同一套文件结构；
2. 让 `/ws/chat` 的 `artifact_created` 与 `message_completed.payload.files`
   使用稳定一致的字段，便于前端直接复用任务通道的文件渲染逻辑。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
_ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}
_DOCUMENT_EXTS = {
    ".md",
    ".txt",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".html",
    ".htm",
}


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def infer_chat_file_category(
    *,
    file_name: str = "",
    mime_type: str = "",
    default_category: str = "text",
) -> str:
    mime = _clean_str(mime_type).lower()
    ext = Path(_clean_str(file_name)).suffix.lower()

    if mime.startswith("image/") or ext in _IMAGE_EXTS:
        return "image"
    if mime.startswith("video/") or ext in _VIDEO_EXTS:
        return "video"
    if mime.startswith("audio/") or ext in _AUDIO_EXTS:
        return "audio"
    if (
        mime in {"application/zip", "application/x-zip-compressed"}
        or ext in _ARCHIVE_EXTS
    ):
        return "archive"
    if mime == "application/pdf" or ext in _DOCUMENT_EXTS:
        return "document"
    return _clean_str(default_category) or "text"


def default_category_for_tool(tool_name: str) -> str:
    normalized = _clean_str(tool_name).lower()
    if "image" in normalized:
        return "image"
    if "video" in normalized:
        return "video"
    if "audio" in normalized:
        return "audio"
    return "text"


def chat_file_identity(file_info: Dict[str, Any]) -> str:
    """给 Turn 级聚合提供稳定去重 key。"""
    file_id = _clean_str(file_info.get("file_id") or file_info.get("id"))
    if file_id:
        return f"file_id:{file_id}"

    url = _clean_str(file_info.get("url") or file_info.get("http_url"))
    if url:
        return f"url:{url}"

    storage_path = _clean_str(file_info.get("storage_path"))
    if storage_path:
        return f"storage:{storage_path}"

    task_id = _clean_str(file_info.get("derived_task_id") or file_info.get("task_id"))
    file_name = _clean_str(file_info.get("file_name"))
    if task_id or file_name:
        return f"name:{task_id}:{file_name}"

    return json.dumps(file_info, ensure_ascii=False, sort_keys=True, default=str)


def normalize_chat_file(
    info: Dict[str, Any],
    *,
    default_category: str = "text",
    derived_task_id: Optional[str] = None,
    source_tool: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(info, dict):
        return None

    file_name = _clean_str(info.get("file_name"))
    mime_type = _clean_str(info.get("mime_type") or info.get("mime"))
    url = _clean_str(info.get("url") or info.get("http_url"))
    http_url = _clean_str(info.get("http_url") or info.get("url"))
    thumb_url = _clean_str(info.get("thumb_url"))
    storage_path = _clean_str(info.get("storage_path"))
    thumbnail_path = _clean_str(info.get("thumbnail_path"))
    file_id = _clean_str(info.get("file_id") or info.get("id"))

    if not any([file_id, file_name, url, storage_path]):
        return None

    normalized: Dict[str, Any] = {
        "file_id": file_id or None,
        "file_name": file_name or None,
        "file_size": info.get("file_size"),
        "mime_type": mime_type or None,
        "url": url or None,
        "http_url": http_url or None,
        "thumb_url": thumb_url or None,
        "storage_path": storage_path or None,
        "thumbnail_path": thumbnail_path or None,
        "category": _clean_str(info.get("category"))
        or infer_chat_file_category(
            file_name=file_name,
            mime_type=mime_type,
            default_category=default_category,
        ),
        "derived_task_id": _clean_str(
            info.get("derived_task_id") or info.get("task_id") or derived_task_id
        )
        or None,
        "source_tool": _clean_str(info.get("source_tool") or source_tool) or None,
        "preview_text": _clean_str(
            info.get("preview_text") or info.get("preview_md")
        )
        or None,
    }

    metadata = info.get("metadata")
    if isinstance(metadata, dict) and metadata:
        normalized["metadata"] = metadata

    return normalized


def build_chat_files_from_tool_result(
    payload: Dict[str, Any],
    *,
    default_category: str = "text",
    source_tool: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    results: List[Dict[str, Any]] = []
    seen = set()

    def _append_from(single_payload: Dict[str, Any]) -> None:
        task_id = _clean_str(single_payload.get("task_id"))
        preview_text = _clean_str(
            single_payload.get("preview_text") or single_payload.get("preview_md")
        )
        for info in single_payload.get("files") or []:
            normalized = normalize_chat_file(
                info,
                default_category=default_category,
                derived_task_id=task_id or None,
                source_tool=source_tool,
            )
            if not normalized:
                continue
            if preview_text and not normalized.get("preview_text") and normalized.get(
                "category"
            ) in {"document", "text"}:
                normalized["preview_text"] = preview_text
            key = chat_file_identity(normalized)
            if key in seen:
                continue
            seen.add(key)
            results.append(normalized)

    _append_from(payload)

    for item in payload.get("items") or []:
        if isinstance(item, dict):
            _append_from(item)
    return results


def try_parse_tool_result_payload(output: Any) -> Optional[Dict[str, Any]]:
    """尽量从工具输出中提取结构化 payload。

    兼容三种形态：
    1. 整段就是 fenced JSON：``\\`\\`\\`json\\n{...}\\n\\`\\`\\```（旧格式）；
    2. 前缀文本 + fenced JSON：例如工具结果开头加了一段『最终回复风格指令』，
       后面再跟 ``\\`\\`\\`json\\n{...}\\n\\`\\`\\``；
    3. 裸 JSON：``{...}`` 直接作为字符串返回。

    任何一种成功即返回解析后的 dict；全部失败返回 ``None``。
    """
    if isinstance(output, dict):
        return output

    if not isinstance(output, str):
        return None

    text = output.strip()
    if not text:
        return None

    candidates: List[str] = []

    # 优先抽出 fenced 代码块的内容（允许前缀文本，例如指令）。
    # 非贪婪 `[\s\S]*?` 保证抓到第一个 ``` 闭合就停，避免误吞后续内容。
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        body = match.group(1).strip()
        if body.startswith("{"):
            candidates.append(body)

    if text.startswith("{"):
        candidates.append(text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


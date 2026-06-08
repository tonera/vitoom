"""
文档转文本 API（供翻译页等直接调用 document_to_markdown）。
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from requests.exceptions import ReadTimeout, RequestException

from backend.auth import get_current_user_id
from backend.core.config import get_config
from backend.core.logger import get_app_logger
from backend.core.response import ok
from backend.services.agent.settings import (
    get_document_to_markdown_default_timeout,
    get_document_to_markdown_pdf_model_name,
)

logger = get_app_logger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["Documents"])

_MAX_TEXT_BYTES = 20 * 1024 * 1024


class DocumentToTextRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Uploaded document absolute URL or outputs path")


def _download_timeout_seconds() -> float:
    base = float(get_document_to_markdown_default_timeout())
    return max(base + 60.0, 300.0)


def _outputs_root() -> Path:
    outputs_dir = Path(get_config("storage.local.base_path", "resources/outputs"))
    if not outputs_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        outputs_dir = (project_root / outputs_dir).resolve()
    return outputs_dir


def _resolve_outputs_relative(path_or_url: str) -> Optional[str]:
    raw = str(path_or_url or "").strip()
    if not raw:
        return None

    path_part = raw
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        path_part = parsed.path or ""

    markers = ("/outputs/", "outputs/")
    for marker in markers:
        idx = path_part.find(marker)
        if idx >= 0:
            return path_part[idx + len(marker) :].lstrip("/")

    normalized = raw.replace("\\", "/").lstrip("/")
    if normalized.startswith("outputs/"):
        return normalized[len("outputs/") :]
    return normalized if "/" in normalized and "://" not in raw else None


def _try_read_local_outputs(url: str, *, storage_path: Optional[str] = None) -> Optional[str]:
    candidates = []
    for candidate in (storage_path, url):
        rel = _resolve_outputs_relative(str(candidate or "").strip())
        if rel:
            candidates.append(rel)
    if not candidates:
        return None

    outputs_root = _outputs_root()
    for rel in candidates:
        local_path = (outputs_root / rel).resolve()
        try:
            local_path.relative_to(outputs_root)
        except ValueError:
            continue
        if not local_path.is_file():
            continue
        if local_path.suffix.lower() == ".zip":
            return _read_zip_text(local_path.read_bytes())
        data = local_path.read_bytes()
        if len(data) > _MAX_TEXT_BYTES:
            raise ValueError(f"document too large (> {_MAX_TEXT_BYTES} bytes)")
        return data.decode("utf-8", errors="replace")
    return None


def _read_zip_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in sorted(names):
            lower = name.lower()
            if lower.endswith(".md") or lower.endswith(".txt") or lower.endswith(".markdown"):
                payload = zf.read(name)
                if len(payload) > _MAX_TEXT_BYTES:
                    raise ValueError(f"document too large (> {_MAX_TEXT_BYTES} bytes)")
                return payload.decode("utf-8", errors="replace")
    raise ValueError("archive contains no markdown/text file")


def _read_remote_text(url: str) -> str:
    timeout = _download_timeout_seconds()
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
    except ReadTimeout as exc:
        raise TimeoutError(
            f"download timed out after {timeout:.0f}s: {url}"
        ) from exc
    except RequestException as exc:
        raise RuntimeError(f"download failed: {url}: {exc}") from exc

    data = resp.content
    if len(data) > _MAX_TEXT_BYTES:
        raise ValueError(f"document too large (> {_MAX_TEXT_BYTES} bytes)")
    content_type = str(resp.headers.get("content-type") or "").lower()
    if url.lower().endswith(".zip") or "zip" in content_type:
        return _read_zip_text(data)
    return data.decode("utf-8", errors="replace")


def _read_download_text(url: str, *, storage_path: Optional[str] = None) -> str:
    local = _try_read_local_outputs(url, storage_path=storage_path)
    if local is not None:
        return local
    return _read_remote_text(url)


def _pick_completed_item(result: Dict[str, Any]) -> Dict[str, Any]:
    items = result.get("items") or []
    if not isinstance(items, list):
        items = []
    for item in items:
        if isinstance(item, dict) and str(item.get("status") or "").lower() == "completed":
            return item
    if items and isinstance(items[0], dict):
        return items[0]
    return {}


def _extract_text_from_convert_result(result: Dict[str, Any]) -> str:
    item = _pick_completed_item(result)
    if not item:
        raise ValueError(str(result.get("error") or "document conversion produced no result"))

    status = str(item.get("status") or "").lower()
    if status and status not in {"completed"}:
        raise ValueError(str(item.get("error") or f"document conversion status={status}"))

    inline_content = str(item.get("content") or result.get("content") or "").strip()
    if inline_content:
        return inline_content

    files = item.get("files") or []
    if isinstance(files, list) and files:
        first = files[0] if isinstance(files[0], dict) else {}
        download_url = str(first.get("url") or first.get("http_url") or "").strip()
        storage_path = str(first.get("storage_path") or "").strip() or None
        if download_url or storage_path:
            return _read_download_text(download_url, storage_path=storage_path).strip()

    preview = str(item.get("preview_md") or "").strip()
    if preview:
        logger.warning("document_to_text falling back to preview_md (truncated preview)")
        return preview

    raise ValueError(str(item.get("error") or "document conversion returned no downloadable text"))


def _convert_and_extract_text(*, user_id: str, url: str, timeout: float) -> Dict[str, Any]:
    from backend.services.agent.tools.builtin.document_to_markdown import convert_documents_sync

    result = convert_documents_sync(user_id=user_id, url=url, timeout=timeout)
    overall = str(result.get("status") or "").lower()
    if overall == "failed" and not (result.get("items") or []):
        raise ValueError(str(result.get("error") or "document conversion failed"))
    item = _pick_completed_item(result)
    text = _extract_text_from_convert_result(result)
    return {
        "text": text,
        "provider": item.get("provider"),
        "status": item.get("status") or result.get("status") or "completed",
    }


@router.get("/convert-config")
async def document_convert_config():
    """文档转换相关配置（供前端 PDF OCR 等场景使用）。"""
    return ok(
        data={
            "timeout_seconds": float(get_document_to_markdown_default_timeout()),
            "pdf_ocr_model": str(get_document_to_markdown_pdf_model_name() or "").strip(),
        },
        msg="ok",
    )


@router.post("/to-text")
async def document_to_text(
    request: DocumentToTextRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    将已上传文档（docx/pdf/txt 等）转为纯文本/Markdown 全文，供翻译等场景使用。
    底层复用 ``document_to_markdown`` 工具（pandoc / markitdown 等）。
    """
    url = str(request.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    timeout = float(get_document_to_markdown_default_timeout())
    try:
        payload = await asyncio.to_thread(
            _convert_and_extract_text,
            user_id=user_id,
            url=url,
            timeout=timeout,
        )
    except TimeoutError as exc:
        logger.warning("document_to_text timed out url=%s", url)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("document_to_text failed url=%s", url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ok(
        data={
            "text": payload["text"],
            "source_url": url,
            "provider": payload.get("provider"),
            "status": payload.get("status"),
        },
        msg="ok",
    )

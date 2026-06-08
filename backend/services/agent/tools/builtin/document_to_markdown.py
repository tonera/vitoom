"""document_to_markdown 工具：把文档转换为 Markdown。

目标：
1. 普通文档（doc/docx/txt/xlsx/pptx/...）在后端本地转换：
   遗留二进制 ``.doc``（Word 97–2003）优先经 LibreOffice 导出 Markdown；
   其余格式使用 markitdown；
2. PDF 因 markitdown 质量较差，改走 mini OCR 文档链路；
3. 无论哪条路径，最终都要有可下载产物；
4. 聊天内联只返回首页预览，避免把整份文档正文塞进上下文。
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from backend.services.agent.tools.registry import register_tool
from backend.services.chat.user_messages import unavailable_message
from backend.storage import get_storage_manager_by_mode
from backend.utils import generate_uuid, safe_filename

from ._arg_utils import clean_optional_str, coerce_optional, coerce_timeout_seconds
from ._libreoffice import (
    SOFFICE_LIGHT_HEADLESS_ARGS,
    locate_soffice_binary,
    pick_convert_output_file,
    rewrite_libreoffice_markdown_sidecars,
    run_soffice_convert_sync,
)
from ._media_common import submit_and_collect
from ._vendor.mtef import mathtype_ole_to_latex

logger = logging.getLogger(__name__)

DOCUMENT_TO_MARKDOWN_TOOL_NAME = "document_to_markdown"

DOCUMENT_TO_MARKDOWN_DESCRIPTION = (
    "把用户提供的文档链接或对话内容转换/导出为 Markdown。"
    "适合 doc/docx/pdf/txt/xlsx/xls/pptx 等文档格式。"
    "当用户明确要求『把这个文档转为 md/markdown』，或『总结/整理以上信息并导出为 md 文档』时使用。"
    "如果是 PDF，优先走 OCR 文档转换链路；如果是其他常见办公/文本文档，"
    "使用本地文档转换器；如果输入是聊天内容，先整理正文，再把正文传入 markdown/content 并直接保存为 .md。"
    "支持单文件或多文件。"
    "返回 ``files[]``：向用户写可点击链接时**用 ``url``**；须原样复制，禁止用 ``storage_path`` 自拼域名。"
    " | 最终回复风格（严格遵守）："
    "逐 item 输出，仅含 ``preview_md`` 原文（**禁止**用 ``` 或 ~~~ 包裹）；"
    "若 ``files[0].url`` 以 ``.zip`` 结尾，末尾另起一行附一条 markdown 下载链接，"
    "锚文本一两个词、与对话语言一致，URL 取 ``files[0].url`` 原值、禁止改写；"
    "多 item 用 ``---`` 分隔；``status=processing`` 只输出 ``message``；"
    "``status=failed`` 只输出 ``error``；"
    "禁止任何寒暄/包装/元信息（如『由于…/已转换/下载链接：/压缩包内含/"
    "预览片段/document.md/images/』）。"
)

DOCUMENT_TO_MARKDOWN_DOCSTRING = (
    "Convert document URL(s) into Markdown. "
    "Supports single `url` or multiple `urls`, and generated chat content via "
    "`markdown` or `content` when the user asks to export the conversation as a Markdown file. "
    "Returns a JSON code block containing per-file task ids, preview markdown, "
    "and downloadable artifact URLs. "
    "Final reply MUST stream each item's `preview_md` verbatim (do NOT wrap in code "
    "fences); when `files[0].url` ends with `.zip`, append exactly one markdown "
    "download link on a new line whose anchor text is one or two words in the user "
    "conversation's language, using `files[0].url` verbatim; join multiple items "
    "with `---`; for `status=processing` items emit only `message`; for "
    "`status=failed` items emit only `error`; no preamble/wrap-up."
)

_REMOTE_URL_PREFIXES = ("http://", "https://")
_PDF_MIME_MARKERS = ("application/pdf", "application/x-pdf")
_PROCESSING_MESSAGE = "任务已提交，正在处理中"
_DOC_FILE_FIELDS = (
    "file_id",
    "url",
    "thumb_url",
    "storage_path",
    "file_name",
    "file_size",
    "mime_type",
    "index",
)
_PDF_OCR_MIN_TIMEOUT_SECONDS = 300.0

# docx 走 pandoc 高保真路径（含 OMML 数学公式），其他 OOXML 走 markitdown + 兜底抽图
_PANDOC_DOCX_EXTENSIONS = (".docx", ".docm")
_LO_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_OOXML_MEDIA_DIRS = {
    ".docx": "word/media/",
    ".docm": "word/media/",
    ".pptx": "ppt/media/",
    ".pptm": "ppt/media/",
    ".xlsx": "xl/media/",
    ".xlsm": "xl/media/",
}
# OMML→`$...$`（tex_math_dollars）+ GFM 表格 + GFM 任务列表；保持纯文本输出（不加 raw_attribute 噪声）
_PANDOC_DOCX_TO_FORMAT = (
    "gfm+tex_math_dollars+pipe_tables+task_lists+raw_html-raw_attribute"
)
# markdown 中 base64 内嵌图（mammoth 默认行为），用于 markitdown 路径下抽图
_DATA_URI_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(data:image/(?P<ext>png|jpe?g|gif|webp|bmp);base64,(?P<b64>[^)]+)\)",
    re.IGNORECASE,
)


class _PandocBinaryNotFound(RuntimeError):
    """pandoc 系统二进制找不到——只在这种情况下降级到 markitdown，避免吞掉真实转换错误。"""


def _default_timeout() -> float:
    try:
        from backend.services.agent.settings import (
            get_document_to_markdown_default_timeout,
        )

        return float(get_document_to_markdown_default_timeout())
    except Exception:
        return 300.0


def _default_pdf_model_name() -> str:
    try:
        from backend.services.agent.settings import (
            get_document_to_markdown_pdf_model_name,
        )

        return str(get_document_to_markdown_pdf_model_name() or "").strip()
    except Exception:
        return "GLM-OCR"


def _effective_pdf_timeout(timeout: float) -> float:
    """PDF OCR is page-count dependent; avoid accidental short LLM timeouts."""

    try:
        value = float(timeout)
    except (TypeError, ValueError):
        value = _default_timeout()
    if value <= 0:
        value = _default_timeout()
    return max(value, _PDF_OCR_MIN_TIMEOUT_SECONDS)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _clean_str(value: Any) -> str:
    return clean_optional_str(value) or ""


def _coerce_export_content(markdown: Any = None, content: Any = None) -> str:
    text = clean_optional_str(markdown)
    if text:
        return text
    return clean_optional_str(content) or ""


def _safe_export_stem(
    *, filename: Any = None, title: Any = None, default: str = "llm-content"
) -> str:
    raw = clean_optional_str(filename) or clean_optional_str(title) or default
    name = Path(str(raw or default)).name.strip() or default
    stem = Path(name).stem if Path(name).suffix else name
    return safe_filename(stem) or default


def _normalize_source_list(url: Any = None, urls: Any = None) -> List[str]:
    def _coerce(value: Any) -> List[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, (list, tuple, set)):
            cleaned: List[str] = []
            for item in value:
                normalized = clean_optional_str(item)
                if normalized:
                    cleaned.append(normalized)
            return cleaned
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
                    cleaned: List[str] = []
                    for item in parsed:
                        normalized = clean_optional_str(item)
                        if normalized:
                            cleaned.append(normalized)
                    return cleaned
            normalized = clean_optional_str(text)
            return [normalized] if normalized else []
        normalized = clean_optional_str(value)
        return [normalized] if normalized else []

    merged: List[str] = []
    seen = set()
    for item in _coerce(url) + _coerce(urls):
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _is_remote_url(source: str) -> bool:
    lowered = str(source or "").strip().lower()
    return lowered.startswith(_REMOTE_URL_PREFIXES)


def _guess_file_name(source: str, *, content_type: str = "") -> str:
    parsed = urlparse(source)
    raw_name = Path(parsed.path or source).name
    if raw_name:
        return raw_name
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    return f"document{ext or ''}"


def _sniff_remote_content_type(source: str, *, timeout: float) -> str:
    try:
        with httpx.Client(timeout=min(timeout, 20.0), follow_redirects=True) as client:
            try:
                resp = client.head(source)
                content_type = str(resp.headers.get("content-type") or "").strip().lower()
                if content_type:
                    return content_type
            except Exception:
                pass
            resp = client.get(source, headers={"Range": "bytes=0-0"})
            return str(resp.headers.get("content-type") or "").strip().lower()
    except Exception:
        return ""


def _detect_pdf_source(source: str, *, timeout: float) -> bool:
    suffix = Path(urlparse(source).path or source).suffix.lower()
    if suffix == ".pdf":
        return True
    if _is_remote_url(source):
        content_type = _sniff_remote_content_type(source, timeout=timeout)
        return any(marker in content_type for marker in _PDF_MIME_MARKERS)
    local_path = Path(source)
    if local_path.exists():
        if local_path.suffix.lower() == ".pdf":
            return True
        try:
            with local_path.open("rb") as handle:
                return handle.read(4) == b"%PDF"
        except Exception:
            return False
    return False


# mini OCR 的分页注释（见 ``inference/mini/handlers/ocr_handler.py``）；其他来源
# 可能是 form-feed `\f`、或者 markdown 标题。我们按真实存在的分页边界优先切。
_OCR_PAGE_MARK_RE = re.compile(r"\n?<!--\s*page\s+\d+\s*-->", re.IGNORECASE)


def _extract_first_page_preview(markdown_text: str, *, max_chars: int = 800) -> str:
    """提取"首页预览"——只取第一页内容，避免把整份文档塞进 LLM 上下文。

    分页边界识别顺序：
    1. mini OCR 注释 ``<!-- page N -->``（PDF 路径实际用的分页符）；
    2. ASCII form-feed ``\\f``（部分历史 OCR / pdfminer 输出用）；
    3. 字符上限 ``max_chars``（最后兜底，避免文档完全没分页标记时仍一次性塞满上下文）。

    无分页标记时只截字符，且明确加 ``…`` 表示截断。
    """
    text = str(markdown_text or "").strip()
    if not text:
        return ""

    # PDF/OCR：第一页 = 第一个 <!-- page N --> 与第二个 <!-- page N --> 之间。
    page_marks = list(_OCR_PAGE_MARK_RE.finditer(text))
    if page_marks:
        first_start = page_marks[0].end()
        if len(page_marks) >= 2:
            text = text[first_start : page_marks[1].start()].strip()
        else:
            text = text[first_start:].strip()
    elif "\f" in text:
        text = text.split("\f", 1)[0].strip()

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_task_prompt(source: str) -> str:
    return f"Convert document to markdown: {source}"


def _storage_file_to_tool_file(file_row: Dict[str, Any]) -> Dict[str, Any]:
    storage_path = str(file_row.get("storage_path") or "").strip()
    stored_name = Path(storage_path).name if storage_path else ""
    return {
        "file_id": str(file_row.get("id") or ""),
        "url": file_row.get("http_url") or file_row.get("url"),
        "thumb_url": file_row.get("thumb_url"),
        "storage_path": file_row.get("storage_path"),
        # UI 展示名与真实下载文件保持一致：直接使用 storage_path 最后一段文件名
        "file_name": stored_name or file_row.get("file_name"),
        "file_size": file_row.get("file_size"),
        "mime_type": file_row.get("mime_type"),
        "index": 0,
    }


def _persist_local_markdown_result(
    *,
    user_id: str,
    source: str,
    markdown_text: str,
    started_at: datetime,
    storage: str,
    provider: str = "markitdown",
    source_kind: str = "document",
) -> Dict[str, Any]:
    from backend.database import Task

    completed_at = datetime.utcnow()
    preview_md = _extract_first_page_preview(markdown_text)
    task_id = generate_uuid()
    task_prompt = _render_task_prompt(source)
    task_params = {
        "job_type": "DOC_TO_MD",
        "source_url": source,
        "source_kind": source_kind,
        "provider": provider,
        "preview_chars": len(preview_md),
    }
    created = Task.create(
        id=task_id,
        user_id=user_id,
        task_type="text",
        prompt=task_prompt,
        params=task_params,
        status="completed",
        storage=storage,
    )
    if not created:
        raise RuntimeError("failed to create local document task")

    source_name = _guess_file_name(source)
    source_stem = Path(source_name).stem or "document"
    output_name = safe_filename(f"{source_stem}.md")
    metadata = {
        "job_type": "DOC_TO_MD",
        "source_url": source,
        "source_kind": source_kind,
        "provider": provider,
        "preview_md": preview_md,
    }
    try:
        storage_manager = get_storage_manager_by_mode(str(storage or "local"))
        file_row = asyncio.run(
            storage_manager.save_file(
                file_data=markdown_text.encode("utf-8"),
                user_id=user_id,
                category="text",
                filename=output_name,
                task_id=task_id,
                metadata=metadata,
                storage=storage,
            )
        )
    except Exception as exc:
        Task.update(
            task_id,
            status="failed",
            error=str(exc),
            progress=0,
            started_at=started_at,
            completed_at=completed_at,
        )
        return {
            "source": source,
            "task_id": task_id,
            "status": "failed",
            "provider": provider,
            "preview_md": "",
            "files": [],
            "total": 0,
            "error": str(exc),
        }
    Task.update(
        task_id,
        status="completed",
        progress=100,
        started_at=started_at,
        completed_at=completed_at,
    )
    return {
        "source": source,
        "task_id": task_id,
        "status": "completed",
        "provider": provider,
        "preview_md": preview_md,
        "files": [_storage_file_to_tool_file(file_row)],
        "total": 1,
    }


def _persist_local_zip_result(
    *,
    user_id: str,
    source: str,
    md_text: str,
    zip_bytes: bytes,
    image_count: int,
    provider: str,
    source_kind: str,
    started_at: datetime,
    storage: str,
) -> Dict[str, Any]:
    """与 ``_persist_local_markdown_result`` 等价的"zip 产物"版本。

    与 PDF 路径产物结构一致（``document.md`` + ``images/`` + ``meta.json``），
    所以前端不需要为非 PDF 文档单独再加分支。
    """
    from backend.database import Task

    completed_at = datetime.utcnow()
    preview_md = _extract_first_page_preview(md_text)
    task_id = generate_uuid()
    task_prompt = _render_task_prompt(source)
    task_params = {
        "job_type": "DOC_TO_MD",
        "source_url": source,
        "source_kind": source_kind,
        "provider": provider,
        "preview_chars": len(preview_md),
        "output_kind": "zip",
        "image_count": image_count,
    }
    created = Task.create(
        id=task_id,
        user_id=user_id,
        task_type="text",
        prompt=task_prompt,
        params=task_params,
        status="completed",
        storage=storage,
    )
    if not created:
        raise RuntimeError("failed to create local document task")

    source_name = _guess_file_name(source)
    source_stem = Path(source_name).stem or "document"
    output_name = safe_filename(f"{source_stem}.zip")
    metadata = {
        "job_type": "DOC_TO_MD",
        "source_url": source,
        "provider": provider,
        "preview_md": preview_md,
        "output_kind": "zip",
        "image_count": image_count,
    }
    try:
        storage_manager = get_storage_manager_by_mode(str(storage or "local"))
        file_row = asyncio.run(
            storage_manager.save_file(
                file_data=zip_bytes,
                user_id=user_id,
                category="text",
                filename=output_name,
                task_id=task_id,
                metadata=metadata,
                storage=storage,
            )
        )
    except Exception as exc:
        Task.update(
            task_id,
            status="failed",
            error=str(exc),
            progress=0,
            started_at=started_at,
            completed_at=completed_at,
        )
        return {
            "source": source,
            "task_id": task_id,
            "status": "failed",
            "provider": provider,
            "preview_md": "",
            "files": [],
            "total": 0,
            "error": str(exc),
        }
    Task.update(
        task_id,
        status="completed",
        progress=100,
        started_at=started_at,
        completed_at=completed_at,
    )
    return {
        "source": source,
        "task_id": task_id,
        "status": "completed",
        "provider": provider,
        "preview_md": preview_md,
        "files": [_storage_file_to_tool_file(file_row)],
        "total": 1,
        "output_kind": "zip",
        "image_count": image_count,
    }


def _persist_local_failure(
    *,
    user_id: str,
    source: str,
    error: str,
    started_at: datetime,
    storage: str,
    provider: str = "markitdown",
    source_kind: str = "document",
) -> Dict[str, Any]:
    from backend.database import Task

    completed_at = datetime.utcnow()
    task_id = generate_uuid()
    created = Task.create(
        id=task_id,
        user_id=user_id,
        task_type="text",
        prompt=_render_task_prompt(source),
        params={
            "job_type": "DOC_TO_MD",
            "source_url": source,
            "source_kind": source_kind,
            "provider": provider,
        },
        status="failed",
        storage=storage,
    )
    if created:
        Task.update(
            task_id,
            status="failed",
            error=str(error),
            progress=0,
            started_at=started_at,
            completed_at=completed_at,
        )
    return {
        "source": source,
        "task_id": task_id if created else None,
        "status": "failed",
        "provider": provider,
        "preview_md": "",
        "files": [],
        "total": 0,
        "error": str(error),
    }


def _download_remote_to_path(source: str, local_path: Path, *, timeout: float) -> str:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", source) as response:
            response.raise_for_status()
            content_type = str(response.headers.get("content-type") or "").strip().lower()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
    return content_type


def _materialize_source_locally(
    source: str, *, work_dir: Path, timeout: float
) -> Path:
    """把 url / 本地路径解析到 work_dir 下的真实文件（保留扩展名）。

    与原 `_convert_non_pdf_document_sync` 内的 download/copy 块语义等价，
    抽出来让 pandoc 与 markitdown 两条子路径共用。
    """
    source_text = _clean_str(source)
    if not source_text:
        raise ValueError("document source is required")

    if _is_remote_url(source_text):
        sniffed_type = _sniff_remote_content_type(source_text, timeout=timeout)
        base_name = _guess_file_name(source_text, content_type=sniffed_type)
        local_path = work_dir / safe_filename(base_name or "document")
        content_type = _download_remote_to_path(source_text, local_path, timeout=timeout)
        if not local_path.suffix:
            guessed_ext = mimetypes.guess_extension(
                (content_type or sniffed_type or "").split(";")[0].strip()
            )
            if guessed_ext:
                renamed = local_path.with_suffix(guessed_ext)
                local_path.rename(renamed)
                local_path = renamed
        return local_path
    src_path = Path(source_text)
    if not src_path.exists():
        raise FileNotFoundError(f"document not found: {source_text}")
    local_path = work_dir / safe_filename(src_path.name or "document")
    shutil.copyfile(src_path, local_path)
    return local_path


def _locate_pandoc_binary() -> Optional[str]:
    """按优先级定位 pandoc 二进制。

    1. ``VITOOM_PANDOC_BIN``：显式覆盖（容器/CI 用）
    2. ``shutil.which('pandoc')``：依赖外层 PATH
    3. Python 解释器同目录：conda env 通常把 pandoc 装在与 python 并列的 bin 下，
       这条路径**不依赖** PATH 配置，对后端 worker 进程最稳。
    """
    env_bin = os.environ.get("VITOOM_PANDOC_BIN", "").strip()
    if env_bin and Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
        return env_bin
    found = shutil.which("pandoc")
    if found:
        return found
    sibling = Path(sys.executable).resolve().parent / "pandoc"
    if sibling.is_file() and os.access(str(sibling), os.X_OK):
        return str(sibling)
    return None


def _try_convert_legacy_doc_with_libreoffice_sync(
    *,
    user_id: str,
    source: str,
    timeout: float,
    storage: str,
    soffice_bin: str,
) -> Optional[Dict[str, Any]]:
    """遗留 ``.doc``：LibreOffice headless ``--convert-to md``（Writer Markdown 导出）。

    失败时返回 ``None``（由调用方降级到 markitdown），避免先写入一条 ``failed`` 任务记录。
    """
    started_at = datetime.utcnow()
    try:
        with tempfile.TemporaryDirectory(prefix="doc_to_md_libreoffice_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            lo_profile = tmp_root / "lo_profile"
            lo_profile.mkdir(parents=True, exist_ok=True)
            out_dir = tmp_root / "lo_md_out"
            out_dir.mkdir(parents=True, exist_ok=True)

            local_path = _materialize_source_locally(
                source, work_dir=tmp_root, timeout=timeout
            )
            if local_path.suffix.lower() != ".doc":
                raise RuntimeError(
                    f"expected .doc for LibreOffice path, got {local_path.suffix!r}"
                )

            proc = run_soffice_convert_sync(
                soffice_bin=soffice_bin,
                input_path=local_path,
                out_dir=out_dir,
                convert_to="md",
                profile_dir=lo_profile,
                lo_args=SOFFICE_LIGHT_HEADLESS_ARGS,
                timeout=timeout,
                timeout_floor=1.0,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    detail or f"LibreOffice exited with code {proc.returncode}"
                )

            md_path = pick_convert_output_file(
                out_dir, source_stem=local_path.stem, output_suffix=".md"
            )

            markdown_text = md_path.read_text(encoding="utf-8", errors="replace").strip()
            if not markdown_text:
                raise RuntimeError("LibreOffice produced empty Markdown")

            md_with_refs, inline_imgs = _extract_data_uri_images(markdown_text)
            md_sidecar, sidecar_imgs = rewrite_libreoffice_markdown_sidecars(
                md_with_refs,
                out_dir,
                md_path,
                image_suffixes=_LO_IMAGE_SUFFIXES,
                start_index=len(inline_imgs) + 1,
            )
            seen_hashes: Set[str] = {
                hashlib.sha1(b).hexdigest() for _, b in inline_imgs + sidecar_imgs
            }
            extra_imgs = _extract_archive_media(local_path, skip_hashes=seen_hashes)
            all_images = list(inline_imgs) + list(sidecar_imgs) + list(extra_imgs)

            if not all_images:
                return _persist_local_markdown_result(
                    user_id=user_id,
                    source=source,
                    markdown_text=md_sidecar,
                    started_at=started_at,
                    storage=storage,
                    provider="libreoffice",
                    source_kind="doc",
                )

            zip_bytes = _pack_doc_zip(
                md_text=md_sidecar,
                image_entries=all_images,
                source=source,
                provider="libreoffice",
                source_kind="doc",
            )
            return _persist_local_zip_result(
                user_id=user_id,
                source=source,
                md_text=md_sidecar,
                zip_bytes=zip_bytes,
                image_count=len(all_images),
                provider="libreoffice",
                source_kind="doc",
                started_at=started_at,
                storage=storage,
            )
    except Exception as exc:
        logger.warning(
            "document_to_markdown LibreOffice .doc conversion failed source=%s (%s)",
            source,
            exc,
        )
        return None


# MathType OLE（Equation.DSMT4）虽然挂在 docx 里，pandoc 只读 OMML，不解 MathType
# 二进制（MTEF），所以这类公式会以兜底图（image*.wmf/emf）形式落到 ``--extract-media``
# 输出里。我们这里做"docx-only"的旁路：自己直接从 docx 内部把 OLE → LaTeX 抽出来，
# pandoc 跑完之后用 LaTeX 把对那些图的引用整段替换掉，wmf 不再进 zip。
#
# 解析失败时返回空 dict，调用方继续按 wmf 兜底——绝不阻断转换链路。
_DOCX_OLE_OBJECT_TAG = "{urn:schemas-microsoft-com:office:office}OLEObject"
_DOCX_VML_IMAGEDATA_TAG = "{urn:schemas-microsoft-com:vml}imagedata"
_DOCX_REL_ID_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_DOCX_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _extract_mathtype_latex_map(docx_path: Path) -> Dict[str, str]:
    """从 docx 内部建立 ``imageN.wmf -> "$LaTeX$"`` 映射。

    docx 里每个 MathType 公式都是这样表达：

        <w:object>
          <v:shape>
            <v:imagedata r:id="rIdM"/>          <- 兜底显示图（wmf/emf）
          </v:shape>
          <o:OLEObject ProgID="Equation.DSMT4" r:id="rIdN"/>  <- MTEF 二进制
        </w:object>

    通过 ``word/_rels/document.xml.rels`` 把 rId 解析为 ``Target`` 路径，
    我们就能把 ``imageN.wmf`` 与 ``oleObjectN.bin`` 配对，再调
    :func:`mathtype_ole_to_latex` 把 OLE bin 翻成 LaTeX。

    返回 dict 可能为空（非 docx / 无 MathType / XML 损坏 / 解析失败）；
    调用方应将空 dict 视为"按 wmf 图回退"。
    """
    suffix = docx_path.suffix.lower()
    if suffix not in (".docx", ".docm"):
        return {}

    from xml.etree import ElementTree as ET

    result: Dict[str, str] = {}
    try:
        zf = zipfile.ZipFile(str(docx_path), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        logger.debug("mathtype map: not a valid zip (%s)", exc)
        return {}

    try:
        try:
            rels_xml = zf.read("word/_rels/document.xml.rels")
            doc_xml = zf.read("word/document.xml")
        except KeyError:
            return {}

        try:
            rels_root = ET.fromstring(rels_xml)
            doc_root = ET.fromstring(doc_xml)
        except ET.ParseError as exc:
            logger.debug("mathtype map: XML parse error (%s)", exc)
            return {}

        rid_to_target: Dict[str, str] = {}
        for rel in rels_root.findall(f"{_DOCX_PKG_REL_NS}Relationship"):
            rid = rel.get("Id") or ""
            target = rel.get("Target") or ""
            if rid and target:
                rid_to_target[rid] = target

        for ole_node in doc_root.iter(_DOCX_OLE_OBJECT_TAG):
            prog_id = (ole_node.get("ProgID") or "").lower()
            if "equation" not in prog_id:
                continue
            ole_rid = ole_node.get(_DOCX_REL_ID_ATTR)
            ole_target = rid_to_target.get(ole_rid or "")
            if not ole_target:
                continue

            # imagedata 在同一个 <w:object> 里，但 ElementTree 没有 .parent，
            # 退而求其次：在与 OLEObject **相邻**的兄弟里找 v:imagedata。docx 实际生成里
            # OLEObject 与 v:shape 是 <w:object> 的两个直接子节点，所以这里枚举同级即可。
            # 容错：如果同级找不到（极少见），退回到全局 r:id 命中下面再做去重。
            img_target: Optional[str] = None
            container = None
            for ancestor in doc_root.iter():
                if ole_node in list(ancestor):
                    container = ancestor
                    break
            if container is not None:
                img_node = container.find(f".//{_DOCX_VML_IMAGEDATA_TAG}")
                if img_node is not None:
                    img_rid = img_node.get(_DOCX_REL_ID_ATTR)
                    img_target = rid_to_target.get(img_rid or "")

            if not img_target:
                continue

            ole_path_in_zip = "word/" + ole_target.lstrip("./").lstrip("/")
            try:
                ole_bytes = zf.read(ole_path_in_zip)
            except KeyError:
                continue

            latex = mathtype_ole_to_latex(ole_bytes)
            if not latex:
                continue

            img_filename = img_target.rsplit("/", 1)[-1]
            if img_filename:
                result[img_filename] = latex
    finally:
        zf.close()

    if result:
        logger.info(
            "[doc_to_md] mathtype: recovered %d/%d LaTeX equations from %s",
            len(result),
            len(result),  # 当前实现只统计成功条目；保留位置便于以后填总数
            docx_path.name,
        )
    return result


def _run_pandoc_docx(
    local_path: Path, work_dir: Path, *, timeout: float
) -> Tuple[str, List[Tuple[str, bytes]]]:
    """调 pandoc 把 docx 转为带数学公式的 GFM markdown，并外抽图片。

    Returns ``(md_text, image_entries)``，``image_entries`` 形如 ``[(arcname, bytes), ...]``，
    arcname 已规范成 ``images/img_NNN.ext``，md 中相应引用同步重写。

    在收尾阶段会把 docx 内部 MathType OLE 公式（即 ``Equation.DSMT4`` 二进制）
    旁路解析为 LaTeX，并把 md 中对应的兜底图引用整段替换为 ``$...$``，
    这些图也不会再放进 ``image_entries``（zip 里就不会出现 wmf 占位）。
    解析失败的图按原 wmf 兜底处理，转换链路绝不因此中断。

    对 pandoc 输出的 raw HTML ``<img>`` 会去掉 ``alt`` 属性（常见为站长水印长句/乱码），
    避免污染正文可访问性之外的可读性。正文里还会去掉仍指向 **WMF/EMF** 的 ``<img>`` /
    ``![](…)``（浏览器多无法显示），但 zip 里仍保留对应文件。
    """
    pandoc_bin = _locate_pandoc_binary()
    if not pandoc_bin:
        raise _PandocBinaryNotFound(
            "pandoc binary not found; install via `conda install -c conda-forge pandoc`"
        )

    media_dir = work_dir / "media"
    out_md = work_dir / "document.md"
    cmd = [
        pandoc_bin,
        str(local_path),
        "--from=docx",
        f"--to={_PANDOC_DOCX_TO_FORMAT}",
        f"--extract-media={media_dir}",
        "--wrap=none",
        "-o",
        str(out_md),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(30.0, timeout),
            cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"pandoc timed out after {timeout:.0f}s") from exc

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-500:]
        raise RuntimeError(f"pandoc failed (rc={proc.returncode}): {stderr_tail}")

    md_text = out_md.read_text(encoding="utf-8") if out_md.exists() else ""

    mathtype_latex_map = _extract_mathtype_latex_map(local_path)

    image_entries: List[Tuple[str, bytes]] = []
    if media_dir.exists():
        files = sorted(p for p in media_dir.rglob("*") if p.is_file())
        # 收集 (绝对路径 / work_dir 相对路径 / ./ 前缀版) 的候选键，再统一对 md_text 做字符串替换。
        # pandoc 3.x 在 docx 含显式尺寸图片时会输出 <img src="ABS_PATH" style=".." />（HTML 而非 md
        # 链接），所以只用 markdown 链接正则会漏。这里走"按抽出文件的真实路径替换"路线，HTML 与
        # markdown 两种形式都能命中，且不会误伤正文中同名字符串（路径都包含 work_dir）。
        rewrite_keys: List[Tuple[str, str]] = []  # (search_str, arcname)
        latex_replacements = 0
        for p in files:
            candidates: List[str] = []
            for path_form in (p, p.resolve()):
                candidates.append(str(path_form))
            try:
                rel_from_work = p.relative_to(work_dir).as_posix()
                candidates.append(rel_from_work)
                candidates.append(f"./{rel_from_work}")
            except ValueError:
                pass
            seen: Set[str] = set()
            unique_candidates: List[str] = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)

            latex = mathtype_latex_map.get(p.name)
            if latex:
                # MathType 公式：把所有引用形态（markdown ![](path) / HTML <img src="path" .../>）
                # **整段**替换成 LaTeX，原 wmf 不入 image_entries（zip 里也不再出现）。
                # 用 lambda 作 repl，避免 LaTeX 里的 \frac、\1 之类被 re.sub 当反向引用。
                md_text = _replace_image_refs_with_latex(
                    md_text, unique_candidates, latex
                )
                latex_replacements += 1
                continue

            seq = len(image_entries) + 1
            ext = p.suffix.lower().lstrip(".") or "bin"
            arcname = f"images/img_{seq:03d}.{ext}"
            try:
                image_entries.append((arcname, p.read_bytes()))
            except Exception:
                continue

            for c in unique_candidates:
                rewrite_keys.append((c, arcname))

        # 长字符串优先替换，避免相对路径先把绝对路径里的尾巴吃掉。
        rewrite_keys.sort(key=lambda kv: len(kv[0]), reverse=True)
        for search_str, arcname in rewrite_keys:
            md_text = md_text.replace(search_str, arcname)

        if latex_replacements:
            logger.info(
                "[doc_to_md] mathtype: replaced %d image refs with LaTeX (kept %d images)",
                latex_replacements,
                len(image_entries),
            )

    # pandoc 3.x 对 docx 里带标定尺寸/标题的图片输出 raw ``<img src="…" style="…" alt="…">``，
    # alt 里常是整段水印/站名，正文中会显示成噪声或乱码，统一去掉只保留图本身。
    md_text = _strip_html_img_alt_attributes(md_text)
    # 普通图若仍是 wmf/emf（非 MathType、无法替换为 LaTeX），浏览器与常见 MD 预览都画不出，
    # 内联 ``<img>``/``![](…)`` 只会变成空坑；正文里删掉这类引用，文件仍留在 zip 里供本地下载打开。
    md_text = _strip_unrenderable_vector_image_refs(md_text)
    return md_text, image_entries


def _strip_html_img_alt_attributes(text: str) -> str:
    """从正文里的 raw HTML ``<img …>`` 中移除 ``alt=…`` 属性，不改 ``src`` / ``style`` 等。"""

    def _one_img(m: re.Match[str]) -> str:
        tag = m.group(0)
        # 双引号、单引号、无引号（少见）三种 alt 形式各剥一遍。
        tag = re.sub(
            r'\s+alt\s*=\s*(?:"[^"]*")',
            "",
            tag,
            flags=re.IGNORECASE | re.DOTALL,
        )
        tag = re.sub(
            r"\s+alt\s*=\s*(?:'[^']*')",
            "",
            tag,
            flags=re.IGNORECASE | re.DOTALL,
        )
        tag = re.sub(
            r"\s+alt\s*=\s*[^\s>]+",
            "",
            tag,
            flags=re.IGNORECASE,
        )
        return tag

    return re.sub(r"<img\b[^>]+/?>", _one_img, text, flags=re.IGNORECASE)


def _strip_unrenderable_vector_image_refs(text: str) -> str:
    """Remove inline WMF/EMF image references from markdown body.

    Web 端与 GFM/常规 Markdown 渲染器通常不能显示 these vector formats, so ``<img src=…wmf>`` and
    ``![](…wmf)`` 只会留下空白/破损占位。Zip 中仍包含对应 ``images/img_*.wmf`` 供用户本地
    用专业软件查看，这里只清正文，不误删已打包字节。
    """

    def _is_wmf_or_emf_path(p: str) -> bool:
        s = p.strip().split("?", 1)[0].strip()
        if not s:
            return False
        low = s.lower()
        return low.endswith(".wmf") or low.endswith(".emf")

    def _md_img(m: re.Match[str]) -> str:
        if _is_wmf_or_emf_path(m.group(1)):
            return ""
        return m.group(0)

    out = re.sub(
        r"!\[[^\]]*\]\(([^)]+)\)",
        _md_img,
        text,
    )

    def _html_img(m: re.Match[str]) -> str:
        full = m.group(0)
        sm = re.search(
            r"""\bsrc\s*=\s*([\"'])([^\"'>]+)\1""",
            full,
            re.IGNORECASE,
        )
        if sm and _is_wmf_or_emf_path(sm.group(2)):
            return ""
        return full

    return re.sub(r"<img\b[^>]+/?>", _html_img, out, flags=re.IGNORECASE)


def _replace_image_refs_with_latex(
    md_text: str, path_candidates: List[str], latex: str
) -> str:
    """把 md 里所有引用 ``path_candidates`` 的图片整段替换成 ``latex``。

    覆盖 markdown 链接（``![alt](path)``）与 raw HTML（``<img src="path" .../>``）两种
    pandoc 输出形态。``latex`` 自身已包含 ``$ ... $``，前后各加一个空格避免与邻接
    文字粘连。``re.sub`` 的 ``repl`` 用 lambda 提供——LaTeX 里常见的 ``\\frac`` /
    ``\\1`` 在普通字符串 repl 下会被当成反向引用而炸掉，这里用 lambda 隔离。
    """
    if not path_candidates:
        return md_text
    escaped = "|".join(re.escape(c) for c in path_candidates if c)
    if not escaped:
        return md_text
    pattern = re.compile(
        r"!\[[^\]]*\]\((?:" + escaped + r")\)"
        r"|"
        r"<img\s+src=\"(?:" + escaped + r")\"[^>]*/?>"
    )
    repl = f" {latex} "
    return pattern.sub(lambda _m: repl, md_text)


def _extract_data_uri_images(
    md_text: str,
) -> Tuple[str, List[Tuple[str, bytes]]]:
    """markitdown 路径专用：把 mammoth 默认产出的 base64 内嵌图抽出为独立文件，

    并把 md 中相应引用替换成 ``images/img_NNN.ext``。返回 ``(rewritten_md, entries)``。
    """
    import base64

    entries: List[Tuple[str, bytes]] = []
    seen: Dict[str, str] = {}

    def _sub(match: "re.Match[str]") -> str:
        ext = match.group("ext").lower()
        if ext == "jpeg":
            ext = "jpg"
        try:
            data = base64.b64decode(match.group("b64"), validate=False)
        except Exception:
            return match.group(0)
        digest = hashlib.sha1(data).hexdigest()
        if digest in seen:
            arc = seen[digest]
        else:
            arc = f"images/img_{len(entries) + 1:03d}.{ext}"
            entries.append((arc, data))
            seen[digest] = arc
        alt = match.group("alt") or ""
        return f"![{alt}]({arc})"

    new_md = _DATA_URI_RE.sub(_sub, md_text)
    return new_md, entries


def _extract_archive_media(
    local_path: Path, *, skip_hashes: Set[str]
) -> List[Tuple[str, bytes]]:
    """OOXML（.docx/.pptx/.xlsx）兜底：从源 zip 包里直接读 media/* 没被前面流程引用过的图。

    用 sha1 去重避免与 base64/pandoc 抽出的同一张图重复入包。
    解析失败 / 不是 zip → 静默返回空，绝不抛错。
    """
    media_prefix = _OOXML_MEDIA_DIRS.get(local_path.suffix.lower())
    if not media_prefix:
        return []
    entries: List[Tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(local_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.startswith(media_prefix):
                    continue
                try:
                    data = zf.read(info)
                except Exception:
                    continue
                digest = hashlib.sha1(data).hexdigest()
                if digest in skip_hashes:
                    continue
                skip_hashes.add(digest)
                ext = Path(info.filename).suffix.lower().lstrip(".") or "bin"
                entries.append((f"images/extra_{len(entries) + 1:03d}.{ext}", data))
    except (zipfile.BadZipFile, OSError):
        return []
    return entries


def _pack_doc_zip(
    md_text: str,
    image_entries: List[Tuple[str, bytes]],
    *,
    source: str,
    provider: str,
    source_kind: str,
) -> bytes:
    """与 PDF 路径 build_doc_zip 同形：document.md + meta.json + images/*。"""
    if not md_text.endswith("\n"):
        md_text = md_text + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("document.md", md_text)
        zf.writestr(
            "meta.json",
            json.dumps(
                {
                    "source_url": source,
                    "source_kind": source_kind,
                    "provider": provider,
                    "image_count": len(image_entries),
                    "generated_at": int(datetime.utcnow().timestamp()),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        for arcname, data in image_entries:
            zf.writestr(arcname, data)
    return buf.getvalue()


def _convert_docx_with_pandoc_sync(
    *,
    user_id: str,
    source: str,
    timeout: float,
    storage: str,
) -> Dict[str, Any]:
    """docx/docm 高保真路径：pandoc 同时处理"文本+OMML 公式+图片外抽"。

    - 含图 → 打成 zip（``document.md`` + ``images/`` + ``meta.json``）；
    - 无图 → 仍输出 .md，保持与现有产物一致；
    - 内部找不到 pandoc 二进制时抛 ``_PandocBinaryNotFound``，由调用方决定是否软降级。
    """
    started_at = datetime.utcnow()
    with tempfile.TemporaryDirectory(prefix="doc_to_md_pandoc_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        local_path = _materialize_source_locally(source, work_dir=tmp_root, timeout=timeout)
        md_text, image_entries = _run_pandoc_docx(local_path, tmp_root, timeout=timeout)

        md_text = (md_text or "").strip()
        if not md_text and not image_entries:
            raise RuntimeError("pandoc produced empty markdown content")

        # OOXML 兜底：偶发情况下 word/media/ 里有 pandoc 没引用到的图（比如旧式公式 EMF），
        # 用 sha1 去重后仅追加进 zip（images/extra_*.ext），不在 md 里列引用——避免水印/乱码 alt
        # 等噪声整段出现在正文末尾。
        seen_hashes: Set[str] = {hashlib.sha1(b).hexdigest() for _, b in image_entries}
        extra = _extract_archive_media(local_path, skip_hashes=seen_hashes)
        if extra:
            image_entries.extend(extra)

        if not image_entries:
            return _persist_local_markdown_result(
                user_id=user_id,
                source=source,
                markdown_text=md_text,
                started_at=started_at,
                storage=storage,
            )

        zip_bytes = _pack_doc_zip(
            md_text=md_text,
            image_entries=image_entries,
            source=source,
            provider="pandoc",
            source_kind="docx",
        )
        return _persist_local_zip_result(
            user_id=user_id,
            source=source,
            md_text=md_text,
            zip_bytes=zip_bytes,
            image_count=len(image_entries),
            provider="pandoc",
            source_kind="docx",
            started_at=started_at,
            storage=storage,
        )


def _convert_with_markitdown_sync(
    *,
    user_id: str,
    source: str,
    timeout: float,
    storage: str,
) -> Dict[str, Any]:
    """非 docx 的常规办公/文本格式路径：markitdown + 抽图（含 zip 产物切换）。

    抽图分两步，互补且去重：
    1. ``_extract_data_uri_images``：mammoth 默认把 docx 内嵌图 base64 进 md 文本，这里把它们捞出来；
    2. ``_extract_archive_media``：直接读 OOXML 容器里 ``word/media/`` ``ppt/media/`` ``xl/media/``，
       兜底 markitdown 在 pptx/xlsx 上几乎不引用图片的常见情况。archive 中多出的图只打进
       ``images/extra_*.ext``，**不**在 md 正文末尾追加大段 ``![](… )`` 列表。
    """
    started_at = datetime.utcnow()
    try:
        from markitdown import MarkItDown
    except Exception:
        return _persist_local_failure(
            user_id=user_id,
            source=source,
            error=unavailable_message(),
            started_at=started_at,
            storage=storage,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="doc_to_md_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            local_path = _materialize_source_locally(source, work_dir=tmp_root, timeout=timeout)

            converter = MarkItDown(enable_plugins=False)
            result = converter.convert(str(local_path))
            markdown_text = str(getattr(result, "text_content", "") or "").strip()
            if not markdown_text:
                raise RuntimeError("markitdown returned empty markdown content")

            md_with_refs, inline_imgs = _extract_data_uri_images(markdown_text)
            seen_hashes: Set[str] = {hashlib.sha1(b).hexdigest() for _, b in inline_imgs}
            extra_imgs = _extract_archive_media(local_path, skip_hashes=seen_hashes)

            all_images = list(inline_imgs) + list(extra_imgs)
            if not all_images:
                return _persist_local_markdown_result(
                    user_id=user_id,
                    source=source,
                    markdown_text=md_with_refs,
                    started_at=started_at,
                    storage=storage,
                )

            source_kind = (local_path.suffix.lower().lstrip(".") or "document")
            zip_bytes = _pack_doc_zip(
                md_text=md_with_refs,
                image_entries=all_images,
                source=source,
                provider="markitdown",
                source_kind=source_kind,
            )
            return _persist_local_zip_result(
                user_id=user_id,
                source=source,
                md_text=md_with_refs,
                zip_bytes=zip_bytes,
                image_count=len(all_images),
                provider="markitdown",
                source_kind=source_kind,
                started_at=started_at,
                storage=storage,
            )
    except Exception as exc:
        logger.exception("document_to_markdown local conversion failed source=%s", source)
        return _persist_local_failure(
            user_id=user_id,
            source=source,
            error=str(exc),
            started_at=started_at,
            storage=storage,
        )


def _peek_source_suffix(source: str, *, timeout: float) -> str:
    """轻量探测源文件后缀，用于决定走 pandoc 还是 markitdown 子路径。

    远程 URL 直接看 path 后缀，不再发 HEAD 探测——只为了路由决策不值得多一次网络往返；
    若 path 没有后缀，会落到 markitdown 路径（也能处理大多数情况）。
    """
    if _is_remote_url(source):
        return Path(urlparse(source).path or "").suffix.lower()
    try:
        return Path(source).suffix.lower()
    except Exception:
        return ""


def _convert_non_pdf_document_sync(
    *,
    user_id: str,
    source: str,
    timeout: float,
    storage: str,
) -> Dict[str, Any]:
    """非 PDF 文档的转换入口：按文件后缀路由到 pandoc、LibreOffice 或 markitdown。

    路由策略：
    - ``.docx`` / ``.docm``：pandoc（保留 OMML 数学公式，``--extract-media`` 抽图）；
      pandoc 二进制找不到时**软降级**回 markitdown，并打 WARN。
    - ``.doc``（遗留 Word 二进制）：若存在 LibreOffice ``soffice``，优先 ``--convert-to md``；
      失败或未安装时降级 markitdown。
    - 其他后缀：markitdown，叠加 base64 抽图 + OOXML 兜底，必要时切换 zip 产物。
    """
    suffix = _peek_source_suffix(source, timeout=min(timeout, 20.0))
    if suffix in _PANDOC_DOCX_EXTENSIONS:
        try:
            return _convert_docx_with_pandoc_sync(
                user_id=user_id,
                source=source,
                timeout=timeout,
                storage=storage,
            )
        except _PandocBinaryNotFound as exc:
            logger.warning(
                "[doc_to_md] pandoc not found, fall back to markitdown for source=%s (%s)",
                source,
                exc,
            )
        except Exception as exc:
            # pandoc 在但转换失败：这通常意味着 docx 损坏或参数问题，markitdown 也大概率失败；
            # 直接落 failure，不浪费一次 markitdown 调用。
            logger.exception(
                "document_to_markdown pandoc conversion failed source=%s", source
            )
            return _persist_local_failure(
                user_id=user_id,
                source=source,
                error=str(exc),
                started_at=datetime.utcnow(),
                storage=storage,
            )

    if suffix == ".doc":
        lo_bin = locate_soffice_binary()
        if lo_bin:
            done = _try_convert_legacy_doc_with_libreoffice_sync(
                user_id=user_id,
                source=source,
                timeout=timeout,
                storage=storage,
                soffice_bin=lo_bin,
            )
            if done is not None:
                return done

    return _convert_with_markitdown_sync(
        user_id=user_id,
        source=source,
        timeout=timeout,
        storage=storage,
    )


def _convert_pdf_document_sync(
    *,
    user_id: str,
    source: str,
    timeout: float,
    storage: str,
) -> Dict[str, Any]:
    from backend.api.tasks.routes import TaskCreateRequest

    request_kwargs: Dict[str, Any] = {
        "task_type": "mini",
        "job_type": "OCR",
        "load_name": _default_pdf_model_name(),
        "tpl_list": [source],
        "extract": {"task": "text"},
        "storage": storage,
    }
    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        return {
            "source": source,
            "task_id": None,
            "status": "failed",
            "provider": "mini_ocr",
            "preview_md": "",
            "files": [],
            "total": 0,
            "error": f"invalid mini request params: {exc}",
        }

    result = submit_and_collect(
        tool_name=DOCUMENT_TO_MARKDOWN_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=1,
        effective_timeout=timeout,
        task_kind_label="document-pdf",
        file_fields=_DOC_FILE_FIELDS,
        extra_result_fields=("content", "file_type"),
    )
    content = str(result.get("content") or "").strip()
    if str(result.get("status") or "").strip().lower() == "timeout":
        return {
            "source": source,
            "task_id": result.get("task_id"),
            "status": "processing",
            "provider": "mini_ocr",
            "preview_md": "",
            "files": [],
            "total": 0,
            "message": _PROCESSING_MESSAGE,
            "wait_timeout_seconds": timeout,
        }
    return {
        "source": source,
        "task_id": result.get("task_id"),
        "status": result.get("status") or "failed",
        "provider": "mini_ocr",
        "content": content,
        "preview_md": _extract_first_page_preview(content),
        "files": list(result.get("files") or []),
        "total": int(result.get("total") or 0),
        "file_type": result.get("file_type"),
        "error": result.get("error"),
    }


def _export_markdown_content_sync(
    *,
    user_id: str,
    markdown_text: str,
    filename: Any = None,
    title: Any = None,
    storage: str,
) -> Dict[str, Any]:
    stem = _safe_export_stem(filename=filename, title=title)
    source = f"{stem}.md"
    started_at = datetime.utcnow()
    try:
        return _persist_local_markdown_result(
            user_id=user_id,
            source=source,
            markdown_text=markdown_text,
            started_at=started_at,
            storage=storage,
            provider="content",
            source_kind="content",
        )
    except Exception as exc:
        logger.exception("document_to_markdown content export failed source=%s", source)
        return _persist_local_failure(
            user_id=user_id,
            source=source,
            error=str(exc),
            started_at=started_at,
            storage=storage,
            provider="content",
            source_kind="content",
        )


def convert_documents_sync(
    *,
    user_id: str,
    url: Any = None,
    urls: Any = None,
    markdown: Any = None,
    content: Any = None,
    filename: Any = None,
    title: Any = None,
    timeout: Optional[float] = None,
    storage: str = "local",
) -> Dict[str, Any]:
    tool_result: Dict[str, Any] = {
        "tool": DOCUMENT_TO_MARKDOWN_TOOL_NAME,
        "status": "failed",
        "items": [],
        "total": 0,
    }
    if not _clean_str(user_id):
        tool_result["error"] = "document_to_markdown requires a user_id (bound via agent context)"
        return tool_result

    sources = _normalize_source_list(url=url, urls=urls)
    content_text = _coerce_export_content(markdown=markdown, content=content)
    if not sources and content_text:
        item = _export_markdown_content_sync(
            user_id=user_id,
            markdown_text=content_text,
            filename=filename,
            title=title,
            storage=storage,
        )
        tool_result["items"] = [item]
        tool_result["total"] = 1
        tool_result["completed"] = (
            1 if str(item.get("status") or "").lower() == "completed" else 0
        )
        tool_result["status"] = "completed" if tool_result["completed"] == 1 else "failed"
        return tool_result

    if not sources:
        tool_result["error"] = "at least one document url/path or markdown/content is required"
        return tool_result

    effective_timeout = coerce_timeout_seconds(timeout, get_default=_default_timeout)

    items: List[Dict[str, Any]] = []
    completed_count = 0
    processing_count = 0
    for source in sources:
        source_text = _clean_str(source)
        if not source_text:
            continue
        is_pdf = _detect_pdf_source(source_text, timeout=min(effective_timeout, 20.0))
        if is_pdf:
            item = _convert_pdf_document_sync(
                user_id=user_id,
                source=source_text,
                timeout=_effective_pdf_timeout(effective_timeout),
                storage=storage,
            )
        else:
            item = _convert_non_pdf_document_sync(
                user_id=user_id,
                source=source_text,
                timeout=effective_timeout,
                storage=storage,
            )
        items.append(item)
        item_status = str(item.get("status") or "").lower()
        if item_status == "completed":
            completed_count += 1
        elif item_status in {"processing", "pending", "queued"}:
            processing_count += 1

    tool_result["items"] = items
    tool_result["total"] = len(items)
    tool_result["completed"] = completed_count
    if processing_count:
        tool_result["processing"] = processing_count
    if items and completed_count == len(items):
        tool_result["status"] = "completed"
    elif processing_count > 0 and completed_count + processing_count == len(items):
        tool_result["status"] = "processing"
    elif completed_count > 0:
        tool_result["status"] = "partial"
    else:
        tool_result["status"] = "failed"
    if not items:
        tool_result["error"] = "no valid document inputs"
    return tool_result


@register_tool(
    name=DOCUMENT_TO_MARKDOWN_TOOL_NAME,
    description=DOCUMENT_TO_MARKDOWN_DESCRIPTION,
    tags=[
        "document",
        "markdown",
        "md",
        "pdf",
        "docx",
        "txt",
        "文档",
        "转markdown",
        "转md",
        "文档转换",
        "pdf转md",
        "doc转md",
    ],
    provider="local",
    enabled=True,
)
def build_document_to_markdown_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()

    try:
        from crewai.tools import BaseTool
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, ConfigDict, Field, field_validator
    except Exception as exc:
        raise RuntimeError("pydantic is required to build document_to_markdown tool") from exc

    class DocumentToMarkdownArgs(BaseModel):
        model_config = ConfigDict(extra="ignore")

        url: Optional[str] = Field(
            default=None,
            description="Single document URL or path; for converting one file only.",
        )
        urls: Optional[Any] = Field(
            default=None,
            description=(
                "Multiple document URLs/paths. Pass a string array or JSON array string, "
                'e.g. `["https://a.docx","https://b.pdf"]`.'
            ),
        )
        timeout: Optional[float] = Field(
            default=None,
            description="Total document conversion timeout in seconds; uses backend config by default.",
        )
        markdown: Optional[str] = Field(
            default=None,
            description="Markdown body to export directly as .md file.",
        )
        content: Optional[str] = Field(
            default=None,
            description="Text/Markdown body to export directly as .md file.",
        )
        filename: Optional[str] = Field(
            default=None,
            description="Filename for content export; extension may be omitted.",
        )
        title: Optional[str] = Field(
            default=None,
            description="Title/filename candidate for content export.",
        )

        # LLM quirk：常把 None 渲染成字符串 "None"/"null"/""，需要在校验前正规化。
        # 公共实现见 ``backend.services.agent.tools.builtin._arg_utils.coerce_optional``。
        @field_validator(
            "url",
            "urls",
            "timeout",
            "markdown",
            "content",
            "filename",
            "title",
            mode="before",
        )
        @classmethod
        def _normalize_llm_string_nones(cls, value: Any) -> Any:
            return coerce_optional(value)

    class DocumentToMarkdownTool(BaseTool):
        name: str = DOCUMENT_TO_MARKDOWN_TOOL_NAME
        description: str = DOCUMENT_TO_MARKDOWN_DESCRIPTION
        args_schema: type = DocumentToMarkdownArgs

        def _run(self, **kwargs: Any) -> str:
            try:
                args = DocumentToMarkdownArgs.model_validate(kwargs)
            except Exception as exc:
                return (
                    "```json\n"
                    f"{_json_dumps({'tool': DOCUMENT_TO_MARKDOWN_TOOL_NAME, 'status': 'failed', 'error': f'invalid tool arguments: {exc}'})}\n"
                    "```"
                )
            payload = convert_documents_sync(
                user_id=bound_user_id, **args.model_dump()
            )
            return f"```json\n{_json_dumps(payload)}\n```"

    tool_instance = DocumentToMarkdownTool()
    tool_instance.__doc__ = DOCUMENT_TO_MARKDOWN_DOCSTRING
    return tool_instance


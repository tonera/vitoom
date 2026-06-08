"""document_to_pdf 工具：文档转 PDF（Office->LibreOffice，Markdown->pandoc）。"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from backend.services.agent.tools.registry import register_tool
from backend.storage import get_storage_manager_by_mode
from backend.utils import generate_uuid, safe_filename

from ._arg_utils import clean_optional_str, coerce_optional, coerce_timeout_seconds
from ._libreoffice import (
    SOFFICE_FULL_HEADLESS_ARGS,
    locate_soffice_binary,
    pick_convert_output_file,
    run_soffice_convert_sync,
)

logger = logging.getLogger(__name__)

DOCUMENT_TO_PDF_TOOL_NAME = "document_to_pdf"
DOCUMENT_TO_PDF_DESCRIPTION = (
    "把用户提供的文档或对话内容转换为 PDF：.md/markdown/content 用 pandoc + PDF 引擎"
    "（优先 tectonic，无则本机 xelatex/pdflatex）；Office 用 LibreOffice。"
    "zip 会递归处理其中 .md/.doc/.docx/…/pdf；多份输出为 zip。"
    "当用户要求把聊天总结、上文、回答内容导出为 PDF 时，先整理正文，再把正文传入 markdown/content。"
)
DOCUMENT_TO_PDF_DOCSTRING = (
    "Convert documents to PDF. Markdown: pandoc with tectonic (preferred) or system LaTeX; "
    "office: LibreOffice. Also accepts generated chat content via `markdown` or `content`; "
    "when exporting chat content, pass the final markdown body in that field and optionally "
    "`filename`/`title`. See deploy docs (Tectonic + pandoc)."
)

_REMOTE_URL_PREFIXES = ("http://", "https://")
_PDF_MIME_MARKERS = ("application/pdf", "application/x-pdf")
_ZIP_MIME_MARKERS = ("application/zip", "application/x-zip-compressed")
_OFFICE_EXTS = {".doc", ".docx", ".docm", ".xls", ".xlsx", ".xlsm", ".ppt", ".pptx", ".pptm"}
# 与 .md 相同走 pandoc；含常见别名与大写被 Path.suffix.lower() 处理
_MARKDOWN_EXTS = frozenset({".md", ".markdown", ".mdown", ".mkd", ".rmd", ".qmd"})


def _default_timeout() -> float:
    try:
        from backend.services.agent.settings import get_document_to_pdf_default_timeout

        return float(get_document_to_pdf_default_timeout())
    except Exception:
        return 300.0


def _default_font() -> str:
    try:
        from backend.services.agent.settings import get_document_to_pdf_default_font

        return str(get_document_to_pdf_default_font() or "").strip()
    except Exception:
        return "Noto Sans CJK SC"


def _coerce_timeout_seconds(value: Any) -> float:
    """见 ``_arg_utils.coerce_timeout_seconds``。"""
    return coerce_timeout_seconds(value, get_default=_default_timeout)


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
            return [item for item in (clean_optional_str(v) for v in value) if item]
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
                    return [item for item in (clean_optional_str(v) for v in parsed) if item]
            normalized = clean_optional_str(text)
            return [normalized] if normalized else []
        normalized = clean_optional_str(value)
        return [normalized] if normalized else []

    merged: List[str] = []
    seen = set()
    for item in _coerce(url) + _coerce(urls):
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def _is_remote_url(source: str) -> bool:
    return str(source or "").strip().lower().startswith(_REMOTE_URL_PREFIXES)


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


def _materialize_source_locally(source: str, *, work_dir: Path, timeout: float) -> Path:
    source_text = _clean_str(source)
    if not source_text:
        raise ValueError("document source is required")
    if _is_remote_url(source_text):
        sniffed = _sniff_remote_content_type(source_text, timeout=timeout)
        local_path = work_dir / safe_filename(_guess_file_name(source_text, content_type=sniffed))
        content_type = _download_remote_to_path(source_text, local_path, timeout=timeout)
        if not local_path.suffix:
            guessed = mimetypes.guess_extension((content_type or sniffed).split(";")[0].strip())
            if guessed:
                renamed = local_path.with_suffix(guessed)
                local_path.rename(renamed)
                local_path = renamed
        return local_path
    src_path = Path(source_text)
    if not src_path.exists():
        raise FileNotFoundError(f"document not found: {source_text}")
    local_path = work_dir / safe_filename(src_path.name or "document")
    shutil.copyfile(src_path, local_path)
    return local_path


def _detect_pdf_source(source: str, *, timeout: float) -> bool:
    suffix = Path(urlparse(source).path or source).suffix.lower()
    if suffix == ".pdf":
        return True
    if _is_remote_url(source):
        content_type = _sniff_remote_content_type(source, timeout=timeout)
        return any(marker in content_type for marker in _PDF_MIME_MARKERS)
    path = Path(source)
    if not (path.exists() and path.is_file()):
        return False
    if path.suffix.lower() == ".pdf":
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"%PDF"
    except Exception:
        return False


def _is_zip_source(source: str, *, timeout: float) -> bool:
    """判断输入是否为「zip 压缩包」语义（.zip 或远程 Content-Type 为 zip）。"""
    suffix = Path(urlparse(source).path or source).suffix.lower()
    if suffix == ".zip":
        return True
    if _is_remote_url(source):
        content_type = _sniff_remote_content_type(source, timeout=timeout)
        return any(marker in content_type for marker in _ZIP_MIME_MARKERS)
    path = Path(source)
    return path.exists() and path.is_file() and zipfile.is_zipfile(path)


def _normalize_zip_entry_name(filename: str) -> Tuple[str, bool]:
    """将 zip 内文件名规范为使用正斜杠的相对路径，并判断是否为目录项。

    Windows 工具打的 zip 常用反斜杠分隔；在 Unix 上若不先替换，单段路径里的扩展名会错。"""
    raw = (filename or "").replace("\\", "/")
    is_dir = raw.endswith("/")
    name = raw.rstrip("/").lstrip("/")
    return name, is_dir


def _is_zip_ignored_relpath(rel: str) -> bool:
    """macOS/旧 zip 中 ``__MACOSX``、``._*`` 资源叉，勿当正文转 PDF。"""
    n = (rel or "").replace("\\", "/")
    if "__MACOSX" in n.split("/"):
        return True
    leaf = n.rsplit("/", 1)[-1] if n else ""
    if leaf.startswith("._"):
        return True
    return False


def _inner_pdf_name_from_rel(rel: str) -> str:
    """用 zip 内相对路径生成压缩包内唯一的 .pdf 文件名，避免子目录重名仅 basename 冲突。"""
    p = Path(rel)
    as_pdf = p.with_suffix(".pdf")
    name = as_pdf.as_posix().replace("\\", "/").replace("/", "_")
    return safe_filename(name) or f"out_{abs(hash(rel)) & 0xFFFF}.pdf"


def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    base = dest.resolve()
    base.mkdir(parents=True, exist_ok=True)
    for info in zf.infolist():
        name, is_dir_guess = _normalize_zip_entry_name(str(info.filename or ""))
        is_dir = is_dir_guess
        if not is_dir and hasattr(info, "is_dir"):
            try:
                is_dir = bool(info.is_dir())
            except Exception:
                is_dir = False
        if not name and not is_dir:
            continue
        if ".." in Path(name).parts:
            raise ValueError(f"unsafe path in zip: {info.filename!r}")
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe path in zip: {info.filename!r}")
        target = (base / member).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"zip path traversal: {info.filename!r}") from exc
        if is_dir:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source_f, open(target, "wb") as out_f:
                shutil.copyfileobj(source_f, out_f, length=1024 * 1024)


def _convert_with_libreoffice(local_path: Path, *, timeout: float) -> Path:
    """使用 ``soffice --headless --convert-to pdf``。无文档窗口、无用户交互。

    **仅 macOS**：程序坞仍可能出现 LibreOffice 图标；无头环境变量见共用模块 ``_libreoffice``
    中的 ``libreoffice_headless_env``。
    """
    soffice = locate_soffice_binary()
    if not soffice:
        raise RuntimeError("service_unavailable: libreoffice not installed (soffice not found)")
    out_dir = local_path.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = local_path.parent / "lo_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = run_soffice_convert_sync(
            soffice_bin=soffice,
            input_path=local_path,
            out_dir=out_dir,
            convert_to="pdf",
            profile_dir=profile_dir,
            lo_args=SOFFICE_FULL_HEADLESS_ARGS,
            timeout=timeout,
            cwd=local_path.parent,
            timeout_floor=20.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"libreoffice timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        raise RuntimeError(f"libreoffice conversion failed (rc={proc.returncode}): {tail}")
    return pick_convert_output_file(out_dir, source_stem=local_path.stem, output_suffix=".pdf")


def _locate_pandoc_binary() -> Optional[str]:
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


def _pandoc_markdown_font_vars(default_font: str) -> List[str]:
    """Pandoc 默认 LaTeX 模板在 XeTeX 下需同时设 mainfont 与 CJKmainfont，否则中文会落到
    Latin Modern 并出现缺字、Overfull hbox 乃至 ``Missing $ inserted`` 等错误。
    """
    df = (default_font or "").strip()
    if not df:
        return []
    return ["-V", f"mainfont={df}", "-V", f"CJKmainfont={df}"]


def _pandoc_markdown_pdf_engine_steps(
    *, default_font: str
) -> List[Tuple[str, List[str]]]:
    """Pandoc 的 PDF 引擎尝试顺序：优先 tectonic，否则本机已存在的 xelatex / pdflatex。

    与对外安装文档一致：用户侧以 Tectonic 为准；全量 TeX 仅作已有环境兜底。

    当配置了 mainfont 时，同时设置 CJKmainfont（同值），避免中文无可用字体、仅回退
    到 lmroman 导致编译失败。随后仍对 tectonic/xelatex 各尝试一次无字体参数，以应对
    本机无该 OTF/字体名写错、或需依赖模板默认西文字体的情况。
    """
    steps: List[Tuple[str, List[str]]] = []
    v_font = _pandoc_markdown_font_vars(default_font)

    for engine in ("tectonic", "xelatex"):
        if not shutil.which(engine):
            continue
        if v_font:
            steps.append((engine, list(v_font)))
        steps.append((engine, []))

    if shutil.which("pdflatex"):
        steps.append(("pdflatex", []))
    return steps


def _has_pandoc_markdown_pdf_engine() -> bool:
    return bool(
        shutil.which("tectonic")
        or shutil.which("xelatex")
        or shutil.which("pdflatex")
    )


def _pandoc_engine_attempt_label(engine: str, extra: List[str]) -> str:
    if not extra:
        return engine
    return f"{engine} {' '.join(extra)}"


def _pandoc_output_excerpt(stdout: str, stderr: str, *, max_len: int = 3200) -> str:
    """合并 pandoc/TeX 标准输出，保留尾部；过长时略截断，避免只看见大量 note: downloading。"""
    combined = f"{stdout or ''}\n{stderr or ''}".strip()
    if len(combined) <= max_len:
        return combined
    return f"…(省略 {len(combined) - max_len} 字)\n" + combined[-max_len:]


def _convert_markdown_with_pandoc(md_path: Path, *, timeout: float, default_font: str) -> Path:
    pandoc = _locate_pandoc_binary()
    if not pandoc:
        raise RuntimeError("service_unavailable: pandoc not installed")
    if not _has_pandoc_markdown_pdf_engine():
        raise RuntimeError(
            "无法将 Markdown 转为 PDF：未找到 tectonic，也未找到 xelatex 或 pdflatex。"
            " 请按部署文档安装 Tectonic 与 pandoc；若机器上已有 TeX 发行版则无需再装 tectonic。"
        )
    work_md = md_path
    # Markdown→PDF 始终先经轻量清洗（见 md_sanitize_for_pdf）；非配置项。
    if md_path.suffix.lower() in _MARKDOWN_EXTS:
        from .md_sanitize_for_pdf import sanitize_markdown_for_pdf_text

        try:
            raw = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"read markdown failed: {md_path}: {exc}") from exc
        clean = sanitize_markdown_for_pdf_text(raw)
        if clean != raw:
            work_md = md_path.parent / f"{md_path.stem}.sanitized.md"
            work_md.write_text(clean, encoding="utf-8")
            logger.debug("document_to_pdf: wrote sanitized md %s", work_md)
    out_pdf = md_path.with_suffix(".pdf")
    failed_parts: List[str] = []
    for engine, extra in _pandoc_markdown_pdf_engine_steps(
        default_font=default_font
    ):
        if out_pdf.exists():
            out_pdf.unlink()
        cmd = [
            pandoc,
            work_md.name,
            "-o",
            out_pdf.name,
            # docx/笔记导出的 md 常引用 images/、media/、Figures/ 下资源
            "--resource-path=.:images:media:assets:Figures:figures",
            f"--pdf-engine={engine}",
            *extra,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(20.0, timeout),
            cwd=str(md_path.parent),
            check=False,
        )
        ex = _pandoc_output_excerpt(proc.stdout or "", proc.stderr or "")
        if proc.returncode == 0 and out_pdf.exists() and out_pdf.stat().st_size > 0:
            return out_pdf
        label = _pandoc_engine_attempt_label(engine, extra)
        failed_parts.append(f"[{label}]\n{ex}")
    detail = "\n---\n".join(failed_parts) if failed_parts else ""
    msg = "Markdown 转 PDF 失败（已尝试本机所有可用的 PDF 引擎仍未成功）"
    if detail:
        msg += f"。\n{detail}"
    raise RuntimeError(msg)


def _convert_zip_mixed_to_pdfs(
    zip_path: Path, *, timeout: float, default_font: str
) -> Tuple[List[Tuple[str, Path]], Dict[str, Any]]:
    extract_dir = zip_path.parent / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extractall(zf, extract_dir)

    zip_items: List[Tuple[str, Path]] = []
    used_zip_names: set = set()
    report: Dict[str, Any] = {"converted": [], "skipped": [], "failed": []}

    for file_path in sorted(p for p in extract_dir.rglob("*") if p.is_file()):
        rel = file_path.relative_to(extract_dir).as_posix()
        if _is_zip_ignored_relpath(rel):
            report["skipped"].append(
                {"path": rel, "reason": "zip_metadata_or_appledouble"}
            )
            continue
        base_inner = _inner_pdf_name_from_rel(rel)

        def _unique_entry(name: str) -> str:
            if name not in used_zip_names:
                used_zip_names.add(name)
                return name
            stem = Path(name).stem
            n = 2
            while True:
                alt = f"{stem}_{n}.pdf"
                if alt not in used_zip_names:
                    used_zip_names.add(alt)
                    return alt
                n += 1

        suffix = file_path.suffix.lower()
        try:
            if suffix in _MARKDOWN_EXTS:
                out = _convert_markdown_with_pandoc(file_path, timeout=timeout, default_font=default_font)
                inner = _unique_entry(base_inner)
                zip_items.append((inner, out))
                report["converted"].append(
                    {"path": rel, "engine": "pandoc", "zip_entry": inner}
                )
            elif suffix in _OFFICE_EXTS:
                out = _convert_with_libreoffice(file_path, timeout=timeout)
                inner = _unique_entry(base_inner)
                zip_items.append((inner, out))
                report["converted"].append(
                    {"path": rel, "engine": "libreoffice", "zip_entry": inner}
                )
            elif suffix == ".pdf":
                target = file_path.parent / f"{Path(_inner_pdf_name_from_rel(rel)).stem}.passthrough.pdf"
                shutil.copyfile(file_path, target)
                inner = _unique_entry(base_inner)
                zip_items.append((inner, target))
                report["converted"].append(
                    {"path": rel, "engine": "passthrough", "zip_entry": inner}
                )
            else:
                report["skipped"].append({"path": rel, "reason": "unsupported_suffix"})
        except Exception as exc:
            report["failed"].append({"path": rel, "error": str(exc)})

    if not zip_items:
        n_extracted = sum(1 for p in extract_dir.rglob("*") if p.is_file())
        f_list = list(report.get("failed") or [])
        s_list = list(report.get("skipped") or [])
        if f_list:
            f0 = f_list[0]
            raise RuntimeError(
                f"zip: 未生成任何 PDF；{len(f_list)} 个文件失败。"
                f" 首个错误 [{f0.get('path')}]: {f0.get('error')}"
            )
        if n_extracted == 0:
            raise RuntimeError("zip: 解压后未找到任何文件，请检查压缩包是否损坏或为空")
        if s_list and not f_list:
            sample = ", ".join(str(x.get("path", "")) for x in s_list[:5])
            raise RuntimeError(
                f"zip: 无支持的后缀，已跳过 {len(s_list)} 个文件。示例: {sample}"
            )
        raise RuntimeError("zip input has no convertible files")
    return zip_items, report


def _build_pdf_zip(
    items: List[Tuple[str, Path]], report: Optional[Dict[str, Any]] = None
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, (name, p) in enumerate(items, start=1):
            inner = safe_filename(name) or f"document_{i}.pdf"
            if inner in zf.namelist():
                inner = f"{Path(inner).stem}_{i}.pdf"
            zf.writestr(inner, p.read_bytes())
        if report is not None:
            zf.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return buf.getvalue()


def _storage_file_to_tool_file(file_row: Dict[str, Any]) -> Dict[str, Any]:
    storage_path = str(file_row.get("storage_path") or "").strip()
    return {
        "file_id": str(file_row.get("id") or ""),
        "url": file_row.get("http_url") or file_row.get("url"),
        "thumb_url": file_row.get("thumb_url"),
        "storage_path": storage_path or None,
        "file_name": Path(storage_path).name if storage_path else file_row.get("file_name"),
        "file_size": file_row.get("file_size"),
        "mime_type": file_row.get("mime_type"),
        "index": 0,
    }


def _persist_output_bytes(
    *,
    user_id: str,
    source: str,
    payload_bytes: bytes,
    output_name: str,
    output_kind: str,
    provider: str,
    storage: str,
    started_at: datetime,
    converted_count: int = 1,
    failed_count: int = 0,
) -> Dict[str, Any]:
    from backend.database import Task

    task_id = generate_uuid()
    completed_at = datetime.utcnow()
    Task.create(
        id=task_id,
        user_id=user_id,
        task_type="text",
        prompt=f"Convert document to pdf: {source}",
        params={
            "job_type": "DOC_TO_PDF",
            "provider": provider,
            "output_kind": output_kind,
            "converted_count": converted_count,
            "failed_count": failed_count,
        },
        status="completed",
        storage=storage,
    )
    storage_manager = get_storage_manager_by_mode(str(storage or "local"))
    file_row = asyncio.run(
        storage_manager.save_file(
            file_data=payload_bytes,
            user_id=user_id,
            category="archive" if output_kind == "zip" else "document",
            filename=safe_filename(output_name),
            task_id=task_id,
            metadata={"job_type": "DOC_TO_PDF", "provider": provider, "output_kind": output_kind},
            storage=storage,
        )
    )
    Task.update(task_id, status="completed", progress=100, started_at=started_at, completed_at=completed_at)
    return {
        "source": source,
        "task_id": task_id,
        "status": "completed" if failed_count == 0 else "partial",
        "provider": provider,
        "files": [_storage_file_to_tool_file(file_row)],
        "total": 1,
        "output_kind": output_kind,
        "converted_count": converted_count,
        "failed_count": failed_count,
    }


def _persist_failure(*, user_id: str, source: str, error: str, started_at: datetime, storage: str) -> Dict[str, Any]:
    from backend.database import Task

    completed_at = datetime.utcnow()
    task_id = generate_uuid()
    Task.create(
        id=task_id,
        user_id=user_id,
        task_type="text",
        prompt=f"Convert document to pdf: {source}",
        params={"job_type": "DOC_TO_PDF"},
        status="failed",
        storage=storage,
    )
    Task.update(task_id, status="failed", error=str(error), progress=0, started_at=started_at, completed_at=completed_at)
    return {
        "source": source,
        "task_id": task_id,
        "status": "failed",
        "provider": "local",
        "files": [],
        "total": 0,
        "error": str(error),
    }


def _convert_single_document_sync(
    *,
    user_id: str,
    source: str,
    timeout: float,
    storage: str,
    default_font: str,
) -> Dict[str, Any]:
    started_at = datetime.utcnow()
    try:
        with tempfile.TemporaryDirectory(prefix="doc_to_pdf_") as tmp_dir:
            local_path = _materialize_source_locally(source, work_dir=Path(tmp_dir), timeout=timeout)
            suffix = local_path.suffix.lower()

            if _detect_pdf_source(str(local_path), timeout=min(timeout, 20.0)):
                stem = Path(_guess_file_name(source)).stem or "document"
                return _persist_output_bytes(
                    user_id=user_id,
                    source=source,
                    payload_bytes=local_path.read_bytes(),
                    output_name=f"{stem}.pdf",
                    output_kind="pdf",
                    provider="passthrough",
                    storage=storage,
                    started_at=started_at,
                )

            # 先按显式后缀路由，避免把 docx/xlsx/pptx 这类 OOXML（本质是 zip 容器）
            # 误判成“zip输入包”分支。
            if suffix in _MARKDOWN_EXTS:
                pdf_path = _convert_markdown_with_pandoc(local_path, timeout=timeout, default_font=default_font)
                return _persist_output_bytes(
                    user_id=user_id,
                    source=source,
                    payload_bytes=pdf_path.read_bytes(),
                    output_name=pdf_path.name,
                    output_kind="pdf",
                    provider="pandoc",
                    storage=storage,
                    started_at=started_at,
                )

            # 注意：OOXML（docx/xlsx/pptx）底层也是 zip 容器，不能据此走“zip 批处理分支”。
            # 仅当输入显式为 .zip，或语义判定为 zip 且后缀不属于已知单文档类型时才走 zip 分支。
            should_treat_as_zip = suffix == ".zip" or (
                _is_zip_source(str(source), timeout=min(timeout, 20.0))
                and suffix not in (_OFFICE_EXTS | _MARKDOWN_EXTS | {".pdf"})
            )
            if should_treat_as_zip:
                zip_items, report = _convert_zip_mixed_to_pdfs(
                    local_path, timeout=timeout, default_font=default_font
                )
                stem = Path(_guess_file_name(source)).stem or "document"
                converted_count = len(report.get("converted") or [])
                failed_count = len(report.get("failed") or [])
                if len(zip_items) == 1:
                    _name, one_pdf = zip_items[0]
                    return _persist_output_bytes(
                        user_id=user_id,
                        source=source,
                        payload_bytes=one_pdf.read_bytes(),
                        output_name=f"{stem}.pdf",
                        output_kind="pdf",
                        provider="mixed_zip",
                        storage=storage,
                        started_at=started_at,
                        converted_count=converted_count,
                        failed_count=failed_count,
                    )
                return _persist_output_bytes(
                    user_id=user_id,
                    source=source,
                    payload_bytes=_build_pdf_zip(zip_items, report=report),
                    output_name=f"{stem}.zip",
                    output_kind="zip",
                    provider="mixed_zip",
                    storage=storage,
                    started_at=started_at,
                    converted_count=converted_count,
                    failed_count=failed_count,
                )

            pdf_path = _convert_with_libreoffice(local_path, timeout=timeout)
            return _persist_output_bytes(
                user_id=user_id,
                source=source,
                payload_bytes=pdf_path.read_bytes(),
                output_name=pdf_path.name,
                output_kind="pdf",
                provider="libreoffice",
                storage=storage,
                started_at=started_at,
            )
    except Exception as exc:
        logger.exception("document_to_pdf conversion failed source=%s", source)
        return _persist_failure(user_id=user_id, source=source, error=str(exc), started_at=started_at, storage=storage)


def _convert_markdown_content_to_pdf_sync(
    *,
    user_id: str,
    markdown_text: str,
    filename: Any = None,
    title: Any = None,
    timeout: float,
    storage: str,
    default_font: str,
) -> Dict[str, Any]:
    stem = _safe_export_stem(filename=filename, title=title)
    source = f"content:{stem}.md"
    started_at = datetime.utcnow()
    try:
        with tempfile.TemporaryDirectory(prefix="content_to_pdf_") as tmp_dir:
            md_path = Path(tmp_dir) / f"{stem}.md"
            md_path.write_text(markdown_text, encoding="utf-8")
            pdf_path = _convert_markdown_with_pandoc(md_path, timeout=timeout, default_font=default_font)
            return _persist_output_bytes(
                user_id=user_id,
                source=source,
                payload_bytes=pdf_path.read_bytes(),
                output_name=f"{stem}.pdf",
                output_kind="pdf",
                provider="pandoc_content",
                storage=storage,
                started_at=started_at,
            )
    except Exception as exc:
        logger.exception("document_to_pdf content export failed source=%s", source)
        return _persist_failure(user_id=user_id, source=source, error=str(exc), started_at=started_at, storage=storage)


def convert_documents_to_pdf_sync(
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
    default_font: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"tool": DOCUMENT_TO_PDF_TOOL_NAME, "status": "failed", "items": [], "total": 0}
    if not _clean_str(user_id):
        result["error"] = "document_to_pdf requires a user_id (bound via agent context)"
        return result
    sources = _normalize_source_list(url=url, urls=urls)
    effective_timeout = _coerce_timeout_seconds(timeout)
    effective_default_font = _clean_str(default_font) or _default_font()
    content_text = _coerce_export_content(markdown=markdown, content=content)

    if not sources and content_text:
        item = _convert_markdown_content_to_pdf_sync(
            user_id=user_id,
            markdown_text=content_text,
            filename=filename,
            title=title,
            timeout=effective_timeout,
            storage=storage,
            default_font=effective_default_font,
        )
        result["items"] = [item]
        result["total"] = 1
        result["completed"] = (
            1 if str(item.get("status") or "").lower() in {"completed", "partial"} else 0
        )
        result["status"] = "completed" if result["completed"] == 1 else "failed"
        return result

    if not sources:
        result["error"] = "at least one document url/path or markdown/content is required"
        return result

    completed = 0
    items: List[Dict[str, Any]] = []
    for src in sources:
        item = _convert_single_document_sync(
            user_id=user_id,
            source=_clean_str(src),
            timeout=effective_timeout,
            storage=storage,
            default_font=effective_default_font,
        )
        items.append(item)
        if str(item.get("status") or "").lower() in {"completed", "partial"}:
            completed += 1

    result["items"] = items
    result["total"] = len(items)
    result["completed"] = completed
    if items and completed == len(items):
        result["status"] = "completed"
    elif completed > 0:
        result["status"] = "partial"
    else:
        result["status"] = "failed"
    return result


@register_tool(
    name=DOCUMENT_TO_PDF_TOOL_NAME,
    description=DOCUMENT_TO_PDF_DESCRIPTION,
    tags=["document", "pdf", "markdown", "md", "zip", "libreoffice", "pandoc", "文档", "转pdf"],
    provider="local",
    enabled=True,
)
def build_document_to_pdf_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()

    try:
        from crewai.tools import BaseTool
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, ConfigDict, Field, field_validator
    except Exception as exc:
        raise RuntimeError("pydantic is required to build document_to_pdf tool") from exc

    class DocumentToPdfArgs(BaseModel):
        model_config = ConfigDict(extra="ignore")

        url: Optional[str] = Field(default=None, description="Single document URL or path.")
        urls: Optional[Any] = Field(default=None, description="Multiple document URLs/paths.")
        markdown: Optional[str] = Field(
            default=None, description="Markdown body to export directly as PDF."
        )
        content: Optional[str] = Field(
            default=None, description="Text/Markdown body to export directly as PDF."
        )
        filename: Optional[str] = Field(
            default=None, description="Filename for content export; extension may be omitted."
        )
        title: Optional[str] = Field(default=None, description="Title/filename candidate for content export.")
        timeout: Optional[float] = Field(default=None, description="Total timeout in seconds.")
        default_font: Optional[str] = Field(default=None, description="Pandoc mainfont for Markdown-to-PDF conversion.")

        @field_validator(
            "url",
            "urls",
            "markdown",
            "content",
            "filename",
            "title",
            "timeout",
            "default_font",
            mode="before",
        )
        @classmethod
        def _normalize_llm_string_nones(cls, value: Any) -> Any:
            return coerce_optional(value)

    class DocumentToPdfTool(BaseTool):
        name: str = DOCUMENT_TO_PDF_TOOL_NAME
        description: str = DOCUMENT_TO_PDF_DESCRIPTION
        args_schema: type = DocumentToPdfArgs

        def _run(self, **kwargs: Any) -> str:
            # CrewAI 会把原始 kwargs 传入，不经过 Pydantic；须在此 model_validate 才能应用 field_validator
            #（否则例如 timeout 为字符串 "None" 时会在 float() 处崩溃）。
            try:
                args = DocumentToPdfArgs.model_validate(kwargs)
            except Exception as exc:
                return (
                    "```json\n"
                    f"{_json_dumps({'tool': DOCUMENT_TO_PDF_TOOL_NAME, 'status': 'failed', 'error': f'invalid tool arguments: {exc}'})}\n"
                    "```"
                )
            payload = convert_documents_to_pdf_sync(
                user_id=bound_user_id, **args.model_dump()
            )
            return f"```json\n{_json_dumps(payload)}\n```"

    tool_instance = DocumentToPdfTool()
    tool_instance.__doc__ = DOCUMENT_TO_PDF_DOCSTRING
    return tool_instance
"""
用户上传接口

- 接入点：POST /v1/uploads
- 落盘目录：resources/outputs/uploads/{YYYYMM}/uuid.ext
- 数据库存储相对路径：uploads/{YYYYMM}/uuid.ext
- 类型：图片/视频/音频；常见办公文档；常见压缩包（见 _allowed_upload）
"""

from __future__ import annotations

import html
import secrets
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File as FastFile, Query, Request
from fastapi.responses import HTMLResponse

from backend.auth import get_current_user_id
from backend.core.config import get_config
from backend.core.response import ok
from backend.core.logger import get_app_logger
from backend.utils import generate_uuid
from backend.utils.artifact_storage import normalize_storage_for_write, resolve_artifact_public_url
from backend.storage.object_storage import put_bytes_at_key
from backend.database import UserUpload
from backend.database.db import get_engine


def _ensure_user_uploads_table_exists() -> None:
    """
    为了兼容“未手动执行迁移”的开发环境：上传前确保 user_uploads 表存在。
    """
    try:
        from sqlalchemy import inspect, text

        engine = get_engine()
        inspector = inspect(engine)
        if "user_uploads" not in inspector.get_table_names():
            UserUpload.__table__.create(bind=engine, checkfirst=True)
            return
        cols = {c["name"] for c in inspector.get_columns("user_uploads")}
        if "storage" not in cols:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE user_uploads ADD COLUMN storage VARCHAR(20) "
                        "NOT NULL DEFAULT 'server'"
                    )
                )
    except Exception as e:
        logger.warning(f"Failed to ensure user_uploads table exists: {e}")

logger = get_app_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Uploads"])

_QR_UPLOAD_TTL_SECONDS = 10 * 60
_QR_UPLOADS_LOCK = threading.Lock()
_QR_UPLOADS: Dict[str, Dict[str, Any]] = {}


def _outputs_url(rel_path: str) -> str:
    return f"/outputs/{str(rel_path).lstrip('/')}"


def _cleanup_expired_qr_uploads(now: Optional[float] = None) -> None:
    ts = now if now is not None else time.time()
    with _QR_UPLOADS_LOCK:
        expired = [token for token, item in _QR_UPLOADS.items() if float(item.get("expires_at_ts") or 0) <= ts]
        for token in expired:
            _QR_UPLOADS.pop(token, None)


def _qr_entry(token: str) -> Dict[str, Any]:
    _cleanup_expired_qr_uploads()
    with _QR_UPLOADS_LOCK:
        entry = _QR_UPLOADS.get(token)
        if not entry:
            raise HTTPException(status_code=404, detail="QR upload token not found or expired")
        return entry


def _accept_allows_upload(accept: str, content_type: str, filename: str) -> bool:
    accept_text = str(accept or "").strip().lower()
    if not accept_text:
        return True
    rules = [item.strip() for item in accept_text.split(",") if item.strip()]
    if not rules:
        return True

    ct = str(content_type or "").lower()
    name = str(filename or "").lower()
    for rule in rules:
        if rule == "*/*":
            return True
        if rule.startswith(".") and name.endswith(rule):
            return True
        if rule.endswith("/*") and ct.startswith(rule[:-1]):
            return True
        if ct == rule:
            return True
    return False


async def _save_user_upload(
    request: Request,
    file: UploadFile,
    user_id: str,
    *,
    accept: str = "",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_user_uploads_table_exists()
    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "upload"

    if accept and not _accept_allows_upload(accept, content_type, filename):
        raise HTTPException(status_code=400, detail="file type does not match accept")
    if not _allowed_upload(content_type, filename):
        raise HTTPException(status_code=400, detail="unsupported file type")

    max_size = int(get_config("upload.max_size", 50 * 1024 * 1024))
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > max_size:
        raise HTTPException(status_code=413, detail="file too large")

    effective_storage = normalize_storage_for_write(
        storage or str(get_config("storage.default", "server") or "server")
    )

    yyyymm = datetime.utcnow().strftime("%Y%m")
    ext = _safe_ext(content_type, filename)
    unique = generate_uuid()
    rel_path = f"uploads/{yyyymm}/{unique}{ext}"

    try:
        await put_bytes_at_key(effective_storage, rel_path, raw, content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Upload put failed: storage={effective_storage}, key={rel_path}, err={e}", exc_info=True)
        raise HTTPException(status_code=500, detail="upload storage failed") from e

    upload_id = generate_uuid()
    saved = UserUpload.create(
        id=upload_id,
        user_id=user_id,
        original_name=filename,
        storage_path=rel_path,
        file_size=len(raw),
        mime_type=content_type,
        storage=effective_storage,
    )
    if not saved:
        # DB 写失败不影响文件落盘，但这里按强一致处理：返回 500
        raise HTTPException(status_code=500, detail="failed to record upload")

    url = resolve_artifact_public_url(effective_storage, rel_path, request) or ""
    return {
        "id": upload_id,
        "storage_path": rel_path,
        "storage": effective_storage,
        "url": url,
        "http_url": url,
        "file_name": filename,
        "mime_type": content_type,
        "file_size": len(raw),
    }


@router.post("/uploads/qrcode/init")
async def init_qrcode_upload(
    request: Request,
    payload: Optional[Dict[str, Any]] = Body(default=None),
    user_id: str = Depends(get_current_user_id),
):
    """
    初始化二维码上传 token。

    PC 端登录用户调用该接口拿到 upload_url 并渲染二维码；手机端扫码后无需登录，
    上传文件仍归属创建 token 的用户。
    """
    _cleanup_expired_qr_uploads()
    accept = str((payload or {}).get("accept") or "").strip()
    if len(accept) > 512:
        raise HTTPException(status_code=400, detail="accept is too long")
    storage = str((payload or {}).get("storage") or "").strip() or None

    token = secrets.token_urlsafe(24)
    now = time.time()
    expires_at_ts = now + _QR_UPLOAD_TTL_SECONDS
    expires_at = datetime.utcfromtimestamp(expires_at_ts).isoformat()

    with _QR_UPLOADS_LOCK:
        _QR_UPLOADS[token] = {
            "user_id": user_id,
            "accept": accept,
            "storage": storage,
            "created_at_ts": now,
            "expires_at_ts": expires_at_ts,
            "expires_at": expires_at,
            "status": "pending",
            "result": None,
            "error": "",
        }

    return ok(
        data={
            "token": token,
            "upload_url": str(request.url_for("get_qrcode_upload_page", token=token)),
            "uploadUrl": str(request.url_for("get_qrcode_upload_page", token=token)),
            "poll_url": str(request.url_for("poll_qrcode_upload", token=token)),
            "pollUrl": str(request.url_for("poll_qrcode_upload", token=token)),
            "expires_at": expires_at,
            "expiresAt": expires_at,
            "ttl_seconds": _QR_UPLOAD_TTL_SECONDS,
            "ttlSeconds": _QR_UPLOAD_TTL_SECONDS,
        },
        msg="ok",
    )


@router.get("/uploads/qrcode/{token}", response_class=HTMLResponse)
async def get_qrcode_upload_page(token: str):
    entry = _qr_entry(token)
    accept = html.escape(str(entry.get("accept") or ""), quote=True)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>上传文件</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#0f172a; color:#e5e7eb; }}
    main {{ min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; box-sizing:border-box; }}
    .card {{ width:min(460px,100%); border:1px solid rgba(148,163,184,.28); border-radius:20px; background:rgba(15,23,42,.92); padding:24px; box-shadow:0 20px 60px rgba(0,0,0,.35); }}
    h1 {{ margin:0 0 8px; font-size:22px; }}
    p {{ color:#94a3b8; line-height:1.6; }}
    input, button {{ width:100%; box-sizing:border-box; border-radius:14px; padding:12px; font-size:16px; }}
    input {{ border:1px dashed #64748b; background:#020617; color:#e5e7eb; }}
    button {{ margin-top:14px; border:0; background:#4f46e5; color:white; font-weight:700; }}
    button:disabled {{ opacity:.55; }}
    #msg {{ margin-top:14px; min-height:24px; }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>上传文件</h1>
      <p>选择文件后会上传到电脑端当前会话。请保持电脑端二维码窗口打开。</p>
      <form id="form">
        <input id="file" name="file" type="file" accept="{accept}" required />
        <button id="btn" type="submit">上传</button>
      </form>
      <div id="msg"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById('form');
    const file = document.getElementById('file');
    const btn = document.getElementById('btn');
    const msg = document.getElementById('msg');
    form.addEventListener('submit', async (ev) => {{
      ev.preventDefault();
      if (!file.files || !file.files[0]) return;
      btn.disabled = true;
      msg.textContent = '上传中...';
      const fd = new FormData();
      fd.append('file', file.files[0]);
      try {{
        const res = await fetch(window.location.href, {{ method: 'POST', body: fd }});
        const data = await res.json().catch(() => null);
        if (!res.ok || !data || data.code !== 1) {{
          throw new Error((data && (data.detail || data.msg)) || '上传失败');
        }}
        msg.textContent = '上传成功，可以回到电脑端继续操作。';
        form.reset();
      }} catch (err) {{
        msg.textContent = err && err.message ? err.message : '上传失败';
        btn.disabled = false;
      }}
    }});
  </script>
</body>
</html>"""
    )


@router.post("/uploads/qrcode/{token}")
async def upload_file_by_qrcode(
    request: Request,
    token: str,
    file: UploadFile = FastFile(..., description="Multipart file field name: file"),
):
    entry = _qr_entry(token)
    if entry.get("status") == "uploaded":
        raise HTTPException(status_code=409, detail="QR upload token has already been used")
    try:
        result = await _save_user_upload(
            request,
            file,
            str(entry.get("user_id") or ""),
            accept=str(entry.get("accept") or ""),
            storage=str(entry.get("storage") or "").strip() or None,
        )
        with _QR_UPLOADS_LOCK:
            if token in _QR_UPLOADS:
                _QR_UPLOADS[token]["status"] = "uploaded"
                _QR_UPLOADS[token]["result"] = result
        return ok(data=result, msg="uploaded")
    except HTTPException as exc:
        with _QR_UPLOADS_LOCK:
            if token in _QR_UPLOADS:
                _QR_UPLOADS[token]["status"] = "failed"
                _QR_UPLOADS[token]["error"] = str(exc.detail)
        raise
    except Exception as exc:
        with _QR_UPLOADS_LOCK:
            if token in _QR_UPLOADS:
                _QR_UPLOADS[token]["status"] = "failed"
                _QR_UPLOADS[token]["error"] = str(exc)
        raise
    finally:
        try:
            await file.close()
        except Exception:
            pass


@router.get("/uploads/qrcode/{token}/poll")
async def poll_qrcode_upload(
    token: str,
    user_id: str = Depends(get_current_user_id),
):
    entry = _qr_entry(token)
    if str(entry.get("user_id") or "") != str(user_id):
        raise HTTPException(status_code=403, detail="forbidden")
    status = str(entry.get("status") or "pending")
    if status == "uploaded" and entry.get("result"):
        return ok(data=entry["result"], msg="uploaded")
    if status == "failed":
        return ok(data=None, msg=str(entry.get("error") or "failed"))
    return ok(data=None, msg="PENDING")



@router.get("/uploads")
async def list_uploads(
    request: Request,
    keyword: Optional[str] = Query(None, description="Optional fuzzy search by upload filename (original_name)"),
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user_id),
):
    """
    获取当前用户上传文件列表（用于前端“从上传文件选取”）
    """
    _ensure_user_uploads_table_exists()
    total = UserUpload.count_by_user(user_id, keyword=keyword)
    rows = UserUpload.list_by_user(user_id, keyword=keyword, limit=limit, offset=offset)

    items: List[Dict[str, Any]] = []
    for r in rows:
        storage_path = r.get("storage_path") or ""
        row_storage = r.get("storage") or "server"
        url = resolve_artifact_public_url(row_storage, storage_path, request) or ""
        items.append(
            {
                "id": r.get("id"),
                "file_name": r.get("original_name"),
                "mime_type": r.get("mime_type"),
                "file_size": r.get("file_size"),
                "created_at": r.get("created_at"),
                "storage_path": storage_path,
                "storage": row_storage,
                "url": url,
                "thumb_url": url,
                "http_url": url,
            }
        )
    return ok(data={"items": items, "total": total}, msg="ok")


_DOC_MIMES = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "text/plain",
        "text/csv",
        "application/csv",
        "text/markdown",
        "text/x-markdown",
        "application/rtf",
        "text/rtf",
        "text/html",
    }
)
_ARCHIVE_MIMES = frozenset(
    {
        "application/zip",
        "application/x-zip-compressed",
        "application/vnd.rar",
        "application/x-rar-compressed",
        "application/x-7z-compressed",
        "application/gzip",
        "application/x-gzip",
        "application/x-tar",
        "application/x-bzip2",
        "application/x-xz",
        "application/zstd",
        "application/x-lzip",
        "application/x-compress",
        "application/java-archive",
    }
)
_DOC_SUFFIXES = (
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".csv",
    ".rtf",
    ".md",
    ".markdown",
    ".html",
    ".htm",
)
_ARCHIVE_SUFFIXES = (
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".tbz2",
    ".xz",
    ".zst",
    ".lz4",
    ".cab",
    ".jar",
    ".war",
)
_MEDIA_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".weba",
)


def _allowed_upload(content_type: str, filename: str) -> bool:
    """
    允许：图片、视频、音频；常见办公文档；常见压缩包。
    """
    ct = (content_type or "").lower()
    name = (filename or "").lower()

    if ct.startswith("image/") or ct.startswith("video/") or ct.startswith("audio/"):
        return True
    if ct in _DOC_MIMES or ct in _ARCHIVE_MIMES:
        return True

    # 浏览器常把未知类型标成 octet-stream：按后缀放行
    if ct == "application/octet-stream" and name.endswith(_DOC_SUFFIXES + _ARCHIVE_SUFFIXES + _MEDIA_SUFFIXES):
        return True

    if name.endswith(_DOC_SUFFIXES + _ARCHIVE_SUFFIXES + _MEDIA_SUFFIXES):
        return True
    return False


def _safe_ext(content_type: str, filename: str) -> str:
    """
    推断扩展名（带点），尽量保留用户扩展名；图片默认 .jpeg
    """
    name = filename or ""
    ext = Path(name).suffix.lower()
    if ext:
        return ext

    ct = (content_type or "").lower()
    if ct == "application/pdf":
        return ".pdf"
    if ct in (
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        return ".docx"
    if ct == "text/plain":
        return ".txt"
    if ct in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",):
        return ".xlsx"
    if ct in ("application/vnd.ms-excel",):
        return ".xls"
    if ct in ("application/vnd.openxmlformats-officedocument.presentationml.presentation",):
        return ".pptx"
    if ct in ("application/vnd.ms-powerpoint",):
        return ".ppt"
    if ct in ("application/vnd.oasis.opendocument.text",):
        return ".odt"
    if ct in ("application/vnd.oasis.opendocument.spreadsheet",):
        return ".ods"
    if ct in ("application/vnd.oasis.opendocument.presentation",):
        return ".odp"
    if ct in ("text/csv", "application/csv"):
        return ".csv"
    if ct in ("application/rtf", "text/rtf"):
        return ".rtf"
    if ct in ("text/markdown", "text/x-markdown"):
        return ".md"
    if ct == "text/html":
        return ".html"
    if ct in ("application/zip", "application/x-zip-compressed"):
        return ".zip"
    if ct in ("application/vnd.rar", "application/x-rar-compressed"):
        return ".rar"
    if ct == "application/x-7z-compressed":
        return ".7z"
    if ct in ("application/gzip", "application/x-gzip"):
        return ".gz"
    if ct == "application/x-tar":
        return ".tar"
    if ct == "application/x-bzip2":
        return ".bz2"
    if ct == "application/x-xz":
        return ".xz"
    if ct == "application/zstd":
        return ".zst"
    if ct == "application/java-archive":
        return ".jar"
    if ct.startswith("image/"):
        # 默认兜底
        return ".jpeg"
    if ct.startswith("video/"):
        return ".mp4"
    if ct in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if ct in ("audio/wav", "audio/x-wav", "audio/wave"):
        return ".wav"
    if ct == "audio/mp4":
        return ".m4a"
    if ct == "audio/aac":
        return ".aac"
    if ct == "audio/flac":
        return ".flac"
    if ct in ("audio/ogg", "application/ogg"):
        return ".ogg"
    if ct == "audio/webm":
        return ".weba"
    if ct == "audio/opus":
        return ".opus"
    if ct.startswith("audio/"):
        return ".mp3"
    return ".bin"


@router.post("/uploads", status_code=201)
async def upload_file(
    request: Request,
    file: UploadFile = FastFile(..., description="Multipart file field name: file"),
    storage: Optional[str] = Query(
        None,
        description="Storage target: server | s3 | oss (local is treated as server on backend)",
    ),
    user_id: str = Depends(get_current_user_id),
):
    """
    用户上传文件（支持：图片/视频/音频；常见办公文档；常见压缩包）
    """
    try:
        return ok(
            data=await _save_user_upload(request, file, user_id, storage=storage),
            msg="uploaded",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="upload failed")
    finally:
        try:
            await file.close()
        except Exception:
            pass



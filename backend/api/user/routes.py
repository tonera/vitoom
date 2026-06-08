"""
用户相关接口

- 获取用户生成文件（图片/视频/音频/文件）：GET /v1/user/files
- 删除用户生成文件记录：DELETE /v1/user/files/{file_id}
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from backend.auth import get_current_user_id
from backend.core.logger import get_app_logger
from backend.core.response import ok
from backend.database.db import get_db_context
from backend.database.models import File as FileModel
from backend.database.models import Task as TaskModel
from backend.storage.manager import StorageManager
from backend.utils.artifact_storage import resolve_artifact_public_url

logger = get_app_logger(__name__)

router = APIRouter(prefix="/v1", tags=["User"])


@router.get("/user/files")
async def list_user_files(
    request: Request,
    category: Optional[str] = Query(None, description="Optional filter: image/video/audio/text"),
    keyword: Optional[str] = Query(None, description="Optional fuzzy search by task prompt keyword (returns associated files)"),
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user_id),
):
    """
    返回当前用户生成的文件列表（默认包含 image/video/audio/text）。
    """
    allowed = {"image", "video", "audio", "text"}
    with get_db_context() as db:
        q = db.query(FileModel).filter(FileModel.user_id == user_id)
        if category:
            if category not in allowed:
                # 不抛错，返回空，更利于前端容错
                return ok(data={"items": [], "total": 0}, msg="ok")
            q = q.filter(FileModel.category == category)
        else:
            q = q.filter(FileModel.category.in_(list(allowed)))

        kw = (keyword or "").strip()
        if kw:
            # 关键字搜索：在用户自己的 tasks.prompt 上做模糊匹配，返回关联 files
            # SQLite/Postgres/MySQL 都可（SQLite 会退化为 LIKE）
            q = (
                q.join(FileModel.task)
                .filter(TaskModel.user_id == user_id)
                .filter(TaskModel.prompt.ilike(f"%{kw}%"))
            )

        total = q.count()
        rows = (
            q.order_by(FileModel.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items: List[Dict[str, Any]] = []
        for f in rows:
            d = f.to_dict()
            storage_path = d.get("storage_path") or ""
            meta = d.get("metadata") or {}
            thumb_path = meta.get("thumbnail_path") if isinstance(meta, dict) else None
            task = f.task if f.task and f.task.user_id == user_id else None
            task_params = dict(task.params or {}) if task and isinstance(task.params, dict) else {}
            if task:
                task_params["model_key"] = task.model_key

            file_storage = d.get("storage") or "server"
            url = resolve_artifact_public_url(file_storage, storage_path, request) or ""
            thumb_url = (
                resolve_artifact_public_url(file_storage, thumb_path, request) if thumb_path else url
            ) or url

            items.append(
                {
                    "id": d.get("id"),
                    "task_id": d.get("task_id"),
                    "category": d.get("category"),
                    "file_name": d.get("file_name"),
                    "mime_type": d.get("mime_type"),
                    "file_size": d.get("file_size"),
                    "created_at": d.get("created_at"),
                    "storage_path": storage_path,
                    "url": url,
                    "thumb_url": thumb_url,
                    "http_url": url,
                    "thumb_http_url": thumb_url,
                    "task_type": task.type if task else None,
                    "task_status": task.status if task else None,
                    "prompt": task.prompt if task else None,
                    "task_params": task_params,
                }
            )

        return ok(data={"items": items, "total": total}, msg="ok")


@router.delete("/user/files/{file_id}")
async def delete_user_file(
    file_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """删除当前用户的一条生成文件记录（会先尝试删除存储上的主文件与缩略图）。"""
    mgr = StorageManager()
    storage_path = ""
    thumb_path: Optional[str] = None
    with get_db_context() as db:
        row = db.query(FileModel).filter(FileModel.id == file_id, FileModel.user_id == user_id).first()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="作品不存在或无权删除")
        storage_path = (row.storage_path or "").strip()
        meta = row.file_metadata if isinstance(row.file_metadata, dict) else {}
        raw_thumb = meta.get("thumbnail_path")
        thumb_path = str(raw_thumb).strip() if raw_thumb else None

    try:
        if storage_path:
            await mgr.adapter.delete_file(storage_path)
        if thumb_path:
            await mgr.adapter.delete_file(thumb_path)
    except Exception as e:
        logger.warning("删除作品磁盘文件失败（仍将删除记录）: file_id=%s err=%s", file_id, e)

    with get_db_context() as db:
        row = db.query(FileModel).filter(FileModel.id == file_id, FileModel.user_id == user_id).first()
        if row:
            db.delete(row)
            db.commit()

    return ok(data=None, msg="ok")

"""
模型管理API路由
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from urllib.parse import urlparse

from backend.models.service import get_model_service
from backend.auth import get_current_admin_user_id, get_current_user_id
from backend.core.logger import get_app_logger
from backend.core.exceptions import ModelNotFoundException
from backend.core.response import ok
from backend.database.db import get_db_context
from backend.utils import utc_now
from backend.utils.url_utils import to_absolute_outputs_url
from backend.utils.url_utils import normalize_outputs_path

# 抽离的 helper（保持 routes.py 薄路由）
from backend.models.download_broadcast import (
    broadcast_download_message as _broadcast_download_message,
    normalize_source as _normalize_source,
    validate_source as _validate_source,
)
from backend.models.lora_compat import resolve_lora_compatible_families
from backend.models.remote_info import (
    extract_hf_repo_id as _extract_hf_repo_id,
    extract_ms_repo_id as _extract_ms_repo_id,
    looks_like_repo_id as _looks_like_repo_id,
    probe_url_ok as _probe_url_ok,
    fetch_hf_model_info as _fetch_hf_model_info,
    fetch_ms_model_info as _fetch_ms_model_info,
    extract_hf_thumb_candidates as _extract_hf_thumb_candidates,
    extract_ms_thumb_candidates as _extract_ms_thumb_candidates,
)
from backend.models.thumbs import download_thumb_to_outputs_models as _download_thumb_to_outputs_models
from backend.models.video_profiles import augment_model_with_video_profile, list_video_task_modes

logger = get_app_logger(__name__)

router = APIRouter(prefix="/api/models", tags=["Models"])


# ==================== 请求/响应模型 ====================

class CreateModelRequest(BaseModel):
    """创建模型目录请求。"""
    model_key: Optional[str] = Field(None, description="Global stable model key; auto-generated from load_name/runtime_engine/asset_type if omitted")
    name: str = Field(..., description="Display name")
    modality: str = Field("image", description="Task domain; see GET /api/models/meta")
    asset_type: str = Field("checkpoint", description="Asset type: checkpoint/lora/vae/controlnet/provider/workflow")
    family: str = Field("", description="Stable model family key")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Capability flags")
    runtime_engine: str = Field("", description="Runtime engine or cloud provider")
    runtime_config: Dict[str, Any] = Field(default_factory=dict, description="Runtime configuration")
    load_name: str = Field(..., description="Inference-side load name, filename, or directory name")
    service_status: str = Field("inactive", description="active/inactive/disabled")
    storage_mode: str = Field("cloud", description="local/cloud")
    download_status: str = Field("pending", description="pending/downloading/completed/failed/canceled")
    source: Dict[str, Any] = Field(default_factory=dict, description="Source information")
    thumb: Optional[str] = Field(None, description="Thumbnail relative path or URL")
    tags: List[str] = Field(default_factory=list, description="Tags")
    trigger_words: List[str] = Field(default_factory=list, description="Trigger words")
    description: Optional[str] = Field(None, description="Model description")
    
    class Config:
        populate_by_name = True


class DownloadActionRequest(BaseModel):
    """统一下载动作接口请求体。"""

    action: str = Field(..., description="Action: start/cancel/refresh")
    source: Optional[Dict[str, Any]] = Field(None, description="Override source from model record")
    asset_type: Optional[str] = Field(None, description="Override asset_type from model record")


class UpdateModelRequest(BaseModel):
    """更新模型目录请求。"""
    model_key: Optional[str] = None
    name: Optional[str] = None
    modality: Optional[str] = None
    asset_type: Optional[str] = None
    family: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    runtime_engine: Optional[str] = None
    runtime_config: Optional[Dict[str, Any]] = None
    load_name: Optional[str] = None
    service_status: Optional[str] = None
    storage_mode: Optional[str] = None
    download_status: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    thumb: Optional[str] = Field(None, description="Model thumbnail URL (optional, may be empty)")
    tags: Optional[List[str]] = None
    trigger_words: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True


class RemoteModelInfoRequest(BaseModel):
    """获取远端模型信息请求（HF / ModelScope）"""
    input: str = Field(..., description="Hugging Face / ModelScope URL or repo_id")


# ==================== API端点 ====================

@router.post("", status_code=201)
async def create_model(
    request_http: Request,
    request: CreateModelRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    创建模型记录
    
    直接写入 `model_catalog` 新字段。
    """
    if not str(request.load_name or "").strip():
        raise HTTPException(status_code=400, detail="load_name is required")

    service = get_model_service()

    source = dict(request.source or {})
    provider = str(source.get("provider") or "").strip()
    repo_id = str(source.get("repo_id") or "").strip()
    try:
        from backend.database import Model

        existed = Model.find_existing_for_create(
            name=request.name,
            load_name=request.load_name,
            source=source,
        )
        if existed:
            model_key = str(existed.get("model_key") or "").strip()
            if model_key and provider and repo_id:
                src = existed.get("source") if isinstance(existed.get("source"), dict) else {}
                if not str(src.get("repo_id") or "").strip():
                    try:
                        updated = Model.update(
                            model_key,
                            source={"provider": provider, "repo_id": repo_id},
                        )
                        if updated:
                            existed = updated
                    except Exception as e:
                        logger.warning(f"Failed to backfill model source: {e}", exc_info=True)
            if isinstance(existed, dict) and "thumb" in existed:
                existed["thumb"] = to_absolute_outputs_url(request_http, existed.get("thumb"))
            return ok(data=existed, msg="exists")
    except Exception as e:
        logger.warning(f"Idempotency check for model create failed: {e}", exc_info=True)
    
    # thumb：如果传入 http(s) URL，则下载到 outputs/models 并以相对 outputs 根目录的路径写入表
    thumb_to_store = request.thumb
    if isinstance(thumb_to_store, str) and thumb_to_store.strip().lower().startswith("http"):
        try:
            thumb_to_store = await _download_thumb_to_outputs_models(thumb_to_store)
        except Exception as e:
            logger.error("Create model: invalid thumb url: %s, err=%s", thumb_to_store, str(e), exc_info=True)
            raise HTTPException(status_code=400, detail="invalid thumb url")

    model_dict = service.create_model(
        model_key=request.model_key,
        name=request.name,
        modality=request.modality,
        asset_type=request.asset_type,
        family=request.family,
        capabilities=request.capabilities,
        runtime_engine=request.runtime_engine,
        runtime_config=request.runtime_config,
        load_name=request.load_name,
        service_status=request.service_status,
        storage_mode=request.storage_mode,
        download_status=request.download_status,
        source=source,
        description=request.description,
        thumb=thumb_to_store,
        tags=request.tags,
        trigger_words=request.trigger_words,
    )
    if isinstance(model_dict, dict) and "thumb" in model_dict:
        model_dict["thumb"] = to_absolute_outputs_url(request_http, model_dict.get("thumb"))

    return ok(data=model_dict, msg="created")


@router.get("")
async def list_models(
    request: Request,
    modality: Optional[str] = None,
    storage_mode: Optional[str] = None,
    service_status: Optional[str] = None,
    name: Optional[str] = None,
    asset_type: Optional[str] = None,
    family: Optional[str] = None,
    model_family: Optional[str] = None,
    lora_family: Optional[str] = None,
    editable: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id)
):
    """
    列出模型
    
    查询参数使用 `model_catalog` 新字段。
    """
    service = get_model_service()
    asset_type_norm = str(asset_type or "").strip().lower()
    is_lora = asset_type_norm == "lora"

    family_in: Optional[List[str]] = None
    if model_family and str(model_family).strip():
        try:
            from backend.models.model_family import parse_model_families_param, family_in_for_families

            fams = parse_model_families_param(model_family)
            expanded = family_in_for_families(fams)
            family_in = expanded or None
        except Exception as e:
            logger.warning(f"Failed to parse model_family={model_family}: {e}")

    lora_family_in: Optional[List[str]] = None
    family_exact: Optional[str] = family
    if is_lora:
        base_family = str(family or "").strip()
        if base_family:
            lora_family_in = resolve_lora_compatible_families(base_family)
            family_exact = str(lora_family or "").strip() or None
        else:
            family_exact = str(lora_family or "").strip() or None
    
    models, total = service.list_models(
        modality=modality,
        storage_mode=storage_mode,
        service_status=service_status,
        name=name,
        asset_type=asset_type,
        family=family_exact,
        family_in=(lora_family_in if is_lora else family_in),
        editable=(bool(editable) if editable is not None else None),
        limit=limit,
        offset=offset
    )

    # Normalize thumbs to absolute URLs for cross-machine usage
    try:
        for m in models:
            if isinstance(m, dict) and m.get("thumb"):
                # small optimization: avoid parsing request.base_url repeatedly
                thumb = m.get("thumb")
                m["thumb"] = to_absolute_outputs_url(request, thumb)
    except Exception as e:
        logger.warning(f"Failed to normalize model thumbs: {e}")

    if str(modality or "").strip().lower() == "video":
        try:
            models = [
                augment_model_with_video_profile(m) if isinstance(m, dict) else m
                for m in models
            ]
        except Exception as e:
            logger.warning(f"Failed to attach video profiles: {e}")

    per_page = int(limit) if int(limit) > 0 else 1
    current_page = (int(offset) // per_page) + 1
    last_page = (total + per_page - 1) // per_page if total > 0 else 1
    from_idx = int(offset) + 1 if total > 0 else 0
    to_idx = min(int(offset) + len(models), total) if total > 0 else 0

    # Build filter options under same query filters (without pagination)
    storage_modes: List[str] = []
    families: List[str] = []
    asset_types: List[str] = []
    try:
        with get_db_context() as db:
            from backend.database.models import Model as ModelModel
            from sqlalchemy import or_, func
            q = db.query(ModelModel).filter(ModelModel.deleted_at.is_(None))
            if modality:
                q = q.filter(func.lower(ModelModel.modality) == str(modality).strip().lower())
            if storage_mode:
                q = q.filter(ModelModel.storage_mode == storage_mode)
            if service_status:
                q = q.filter(ModelModel.service_status == service_status)
            if name and str(name).strip():
                kw = str(name).strip()
                like = f"%{kw}%"
                q = q.filter(or_(ModelModel.name.ilike(like), ModelModel.load_name.ilike(like), ModelModel.model_key.ilike(like)))
            if asset_type and str(asset_type).strip() and str(asset_type).strip().lower() != "all":
                q = q.filter(func.lower(ModelModel.asset_type) == str(asset_type).strip().lower())
            if (not is_lora) and family_in:
                vals = [str(x).strip().lower() for x in family_in if str(x).strip()]
                if vals:
                    q = q.filter(func.lower(ModelModel.family).in_(vals))
            if is_lora and lora_family_in:
                vals = [str(x).strip().lower() for x in lora_family_in if str(x).strip()]
                if vals:
                    q = q.filter(func.lower(ModelModel.family).in_(vals))

            q_storage = q
            if is_lora and lora_family and str(lora_family).strip():
                q_storage = q_storage.filter(func.lower(ModelModel.family) == str(lora_family).strip().lower())

            storage_modes = [r[0] for r in q_storage.with_entities(ModelModel.storage_mode).distinct().all() if r and r[0]]
            families = [r[0] for r in q.with_entities(ModelModel.family).distinct().all() if r and r[0]]
            asset_types = [r[0] for r in q.with_entities(ModelModel.asset_type).distinct().all() if r and r[0]]

            storage_modes = sorted(set([str(x).strip() for x in storage_modes if str(x).strip()]), key=lambda x: (x != "local", x))
            families = sorted(set([str(x).strip() for x in families if str(x).strip()]), key=lambda x: x.lower())
            asset_types = sorted(set([str(x).strip() for x in asset_types if str(x).strip()]), key=lambda x: x.lower())
    except Exception as e:
        logger.warning(f"Failed to compute model filter options: {e}")

    return ok(
        data=models,
        msg="ok",
        meta={
            "current_page": current_page,
            "from": from_idx,
            "last_page": last_page,
            "per_page": per_page,
            "to": to_idx,
            "total": total,
            "filter_options": {
                "storage_modes": storage_modes,
                "families": families,
                "asset_types": asset_types,
            },
            "video_task_modes": list_video_task_modes() if str(modality or "").strip().lower() == "video" else None,
        },
    )


@router.get("/meta")
async def get_models_meta(
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    模型管理元数据：合法 modality 与 family 清单（前端下拉 SSOT）。

    数据来源：`config/model_catalog_meta.yaml`
    """
    from backend.models.catalog_meta import get_catalog_meta_payload

    return ok(data=get_catalog_meta_payload(), msg="ok")


@router.get("/{model_key}")
async def get_model(
    request: Request,
    model_key: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    获取模型信息
    
    - **model_key**: 模型稳定键
    """
    service = get_model_service()
    
    model_dict = service.get_model(model_key)
    if not model_dict:
        raise HTTPException(status_code=404, detail="Model not found")

    if isinstance(model_dict, dict) and "thumb" in model_dict:
        model_dict["thumb"] = to_absolute_outputs_url(request, model_dict.get("thumb"))
    if isinstance(model_dict, dict):
        try:
            model_dict = augment_model_with_video_profile(model_dict)
        except Exception as e:
            logger.warning(f"Failed to attach video profile for model_key={model_key}: {e}")
    
    return ok(data=model_dict, msg="ok")


@router.put("/{model_key}/activate")
async def activate_model(
    model_key: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    激活模型
    
    - **model_key**: 模型稳定键
    """
    service = get_model_service()
    
    try:
        model_dict = service.activate_model(model_key)
        return ok(data=model_dict, msg="activated")
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{model_key}/deactivate")
async def deactivate_model(
    model_key: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    停用模型
    
    - **model_key**: 模型稳定键
    """
    service = get_model_service()
    
    try:
        model_dict = service.deactivate_model(model_key)
        return ok(data=model_dict, msg="deactivated")
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{model_key}")
async def update_model(
    request_http: Request,
    model_key: str,
    request: UpdateModelRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """更新模型目录记录。"""
    service = get_model_service()
    
    updates = {}
    if request.model_key is not None:
        updates["model_key"] = request.model_key
    if request.name is not None:
        updates["name"] = request.name
    if request.modality is not None:
        updates["modality"] = request.modality
    if request.asset_type is not None:
        updates["asset_type"] = request.asset_type
    if request.family is not None:
        updates["family"] = request.family
    if request.capabilities is not None:
        updates["capabilities"] = request.capabilities
    if request.runtime_engine is not None:
        updates["runtime_engine"] = request.runtime_engine
    if request.runtime_config is not None:
        updates["runtime_config"] = request.runtime_config
    if request.load_name is not None:
        updates["load_name"] = request.load_name
    if request.service_status is not None:
        updates["service_status"] = request.service_status
    if request.storage_mode is not None:
        updates["storage_mode"] = request.storage_mode
    if request.download_status is not None:
        updates["download_status"] = request.download_status
    if request.source is not None:
        updates["source"] = request.source
    if request.description is not None:
        updates["description"] = request.description
    if request.thumb is not None:
        thumb_v = request.thumb
        if isinstance(thumb_v, str):
            t = thumb_v.strip()
            # 兼容：前端可能回传绝对 URL（例如 http://x:8888/outputs/models/xxx.jpeg）
            # 此时无需再次下载，只需要把 /outputs/ 前缀剥离为可持久化的相对形式（models/xxx.jpeg）
            try:
                from urllib.parse import urlparse

                if t.lower().startswith("http://") or t.lower().startswith("https://"):
                    p = urlparse(t)
                    # "/outputs/models/.." -> "models/.."
                    if p.path and "/outputs/" in p.path:
                        norm = normalize_outputs_path(p.path)
                        if norm.startswith("/outputs/"):
                            thumb_v = norm.replace("/outputs/", "", 1)
                else:
                    # "/outputs/models/.." 或 "outputs/models/.." 或 "models/.."
                    if "/outputs/" in t or t.startswith(("outputs/", "resources/outputs/")):
                        norm = normalize_outputs_path(t)
                        if norm.startswith("/outputs/"):
                            thumb_v = norm.replace("/outputs/", "", 1)
            except Exception:
                pass

        if isinstance(thumb_v, str) and thumb_v.strip().lower().startswith("http"):
            try:
                thumb_v = await _download_thumb_to_outputs_models(thumb_v)
            except Exception as e:
                logger.error("Update model: invalid thumb url: %s, err=%s", thumb_v, str(e), exc_info=True)
                raise HTTPException(status_code=400, detail="invalid thumb url")
        updates["thumb"] = thumb_v
    if request.tags is not None:
        updates["tags"] = request.tags
    if request.trigger_words is not None:
        updates["trigger_words"] = request.trigger_words
    
    try:
        model_dict = service.update_model(model_key, **updates)
        if isinstance(model_dict, dict) and "thumb" in model_dict:
            model_dict["thumb"] = to_absolute_outputs_url(request_http, model_dict.get("thumb"))
        return ok(data=model_dict, msg="updated")
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{model_key}")
async def delete_model(
    model_key: str,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    删除模型（包括数据库记录和本地文件）
    
    - **model_key**: 模型稳定键
    """
    service = get_model_service()
    
    try:
        deleted = service.delete_model(model_key)
        if deleted:
            return ok(data={"model_key": model_key}, msg="deleted")
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model")
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{model_key}/download/action")
async def download_action(
    model_key: str,
    request: DownloadActionRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    统一下载动作接口：start/cancel/refresh
    """
    action = str(request.action or "").strip().lower()
    if action not in {"start", "cancel", "refresh"}:
        raise HTTPException(status_code=400, detail="action must be start/cancel/refresh")

    service = get_model_service()
    model_dict = service.get_model(model_key)
    if not model_dict:
        raise HTTPException(status_code=404, detail="Model not found")

    source = dict(model_dict.get("source") or {})
    if request.source:
        source.update(request.source)
    provider, repo_id = _normalize_source(source)
    asset_type_for_download = str(
        (request.asset_type or model_dict.get("asset_type") or "checkpoint") or ""
    ).strip().lower() or "checkpoint"

    _validate_source(provider, repo_id)

    # ===== action=start：预写 DB（pending） =====
    if action == "start":
        try:
            from backend.database import Model as ModelDB
            from backend.models.download_text import upsert_download_block

            desc = (model_dict.get("description") or "")
            new_desc = upsert_download_block(
                desc,
                status="pending",
                progress="",
                error="",
                worker="",
            )
            ModelDB.update(
                model_key,
                source={**source, "provider": provider, "repo_id": repo_id},
                download_status="pending",
                description=new_desc,
            )
        except Exception as e:
            logger.warning(f"Failed to pre-update model download fields: {e}", exc_info=True)

    # ===== 广播 =====
    msg_type = "download_cancel" if action == "cancel" else "download"
    sent = await _broadcast_download_message(
        model_key=model_key,
        source={**source, "provider": provider, "repo_id": repo_id},
        asset_type=asset_type_for_download,
        message_type=msg_type,
    )

    if sent <= 0:
        if action == "start":
            # 无可用连接：标记 failed 并返回 503
            try:
                from backend.database import Model as ModelDB
                from backend.models.download_text import upsert_download_block

                md = ModelDB.get_by_model_key(model_key) or {}
                desc = md.get("description") or ""
                reason = "No running download service connected"
                new_desc = upsert_download_block(desc, status="failed", progress="", error=reason, worker="")
                ModelDB.update(model_key, download_status="failed", description=new_desc)
            except Exception:
                pass
        raise HTTPException(status_code=503, detail="No running download service connected")

    # ===== action=cancel：立即更新 DB + 推前端 =====
    if action == "cancel":
        try:
            from backend.database import Model as ModelDB
            from backend.models.download_text import upsert_download_block

            desc = (model_dict.get("description") or "")
            new_desc = upsert_download_block(desc, status="canceled", progress="", error="", worker="")
            ModelDB.update(model_key, download_status="canceled", description=new_desc)
        except Exception:
            pass

        try:
            from backend.websocket.manager import get_websocket_manager
            manager = get_websocket_manager()
            await manager.forward_model_message(
                model_key,
                {
                    "type": "download_status",
                    "model_key": model_key,
                    "source": {**source, "provider": provider, "repo_id": repo_id},
                    "status": "canceled",
                    "service_id": "",
                    "timestamp": utc_now().isoformat(),
                },
            )
        except Exception:
            pass

        return ok(data={"action": "cancel", "broadcast_sent": sent, "model_key": model_key}, msg="canceled")

    if action == "refresh":
        return ok(data={"action": "refresh", "broadcast_sent": sent, "model_key": model_key}, msg="refreshed")

    # action == start
    return ok(data={"action": "start", "broadcast_sent": sent, "model_key": model_key}, msg="enqueued")

@router.post("/remote-info")
async def get_remote_model_info(
    request: RemoteModelInfoRequest,
    _admin_id: str = Depends(get_current_admin_user_id),
):
    """
    获取 huggingface / modelscope 远端模型信息。

    判定规则：
    - 输入含 huggingface.co：按 huggingface 处理，并自动解析 repo_id
    - 输入含 modelscope.cn：按 modelscope 处理，并自动解析 repo_id
    - 输入为 repo_id：先探测 huggingface.co/{repo_id}（3s 超时），超时/失败再探测 modelscope.cn/models/{repo_id}
    """
    raw = str(request.input or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="input is required")

    # 1) 明确 URL
    hf_repo = _extract_hf_repo_id(raw)
    if hf_repo:
        info = await _fetch_hf_model_info(hf_repo)
        thumbs = _extract_hf_thumb_candidates(info, hf_repo, limit=10)
        return ok(data={"provider": "huggingface", "repo_id": hf_repo, "info": info, "thumb_candidates": thumbs}, msg="ok")

    ms_repo = _extract_ms_repo_id(raw)
    if ms_repo:
        info = await _fetch_ms_model_info(ms_repo)
        thumbs = _extract_ms_thumb_candidates(info, ms_repo, limit=10)
        return ok(data={"provider": "modelscope", "repo_id": ms_repo, "info": info, "thumb_candidates": thumbs}, msg="ok")

    # 2) repo_id 探测
    if _looks_like_repo_id(raw):
        repo_id = raw
        hf_ok = await _probe_url_ok(f"https://huggingface.co/{repo_id}", timeout_seconds=3.0)
        if hf_ok:
            info = await _fetch_hf_model_info(repo_id)
            thumbs = _extract_hf_thumb_candidates(info, repo_id, limit=10)
            return ok(data={"provider": "huggingface", "repo_id": repo_id, "info": info, "thumb_candidates": thumbs}, msg="ok")

        ms_ok = await _probe_url_ok(f"https://modelscope.cn/models/{repo_id}", timeout_seconds=3.0)
        if ms_ok:
            info = await _fetch_ms_model_info(repo_id)
            thumbs = _extract_ms_thumb_candidates(info, repo_id, limit=10)
            return ok(data={"provider": "modelscope", "repo_id": repo_id, "info": info, "thumb_candidates": thumbs}, msg="ok")

        raise HTTPException(status_code=404, detail="repo_id not found on huggingface or modelscope (or timeout)")

    raise HTTPException(status_code=400, detail="unsupported input format")


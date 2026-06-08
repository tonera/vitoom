"""
推理服务管理API路由
提供推理服务记录的CRUD和状态同步接口
"""
from pathlib import Path
from typing import Optional, Dict, Any

import aiofiles
import hmac
import hashlib
import time

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header, Request
from pydantic import BaseModel, Field

from backend.core.config import get_config

from backend.services.inference.service import InferenceServiceManager, get_inference_service_manager
from backend.core.logger import get_app_logger
from backend.core.exceptions import InferenceServiceNotFoundException
from backend.core.response import ok
from backend.utils.http_utils import resolve_client_ip

logger = get_app_logger(__name__)

# 聚合路由：app.py 只 include_router 这个 router
router = APIRouter()
services_router = APIRouter(prefix="/api/inference/services", tags=["Inference Services"])
upload_router = APIRouter(prefix="/api/inference", tags=["Inference Upload"])


# ==================== 请求/响应模型 ====================

class CreateServiceRequest(BaseModel):
    """创建推理服务请求"""
    name: str = Field(..., description="Service name")
    type: str = Field(..., description="Service type (vllm/ollama/diffusers/DiffSynth/modelscope, etc.)")
    service_type: Optional[str] = Field(None, description="Service content type (image/video/audio/text); maps to tasks.type")
    config: Optional[Dict[str, Any]] = Field(None, description="Service configuration")
    gpu_enabled: bool = Field(False, description="Whether GPU is enabled")
    auto_start: bool = Field(False, description="Whether to auto-start")


class UpdateServiceRequest(BaseModel):
    """更新推理服务请求"""
    name: Optional[str] = Field(None, description="Service name")
    type: Optional[str] = Field(None, description="Service type (vllm/ollama, etc.)")
    service_type: Optional[str] = Field(None, description="Service content type (image/video/audio/text)")
    config: Optional[Dict[str, Any]] = Field(None, description="Service configuration")
    gpu_enabled: Optional[bool] = Field(None, description="Whether GPU is enabled")
    auto_start: Optional[bool] = Field(None, description="Whether to auto-start")


class SyncServiceStartRequest(BaseModel):
    """同步服务启动请求（由推理器调用）"""
    host: str = Field(..., description="Service host address")
    port: Optional[int] = Field(None, description="Service port (text inference only)")
    config: Optional[Dict[str, Any]] = Field(None, description="Service configuration (includes loaded models, etc.)")


# ==================== API端点 ====================

@services_router.post("", status_code=201)
async def create_service(
    request: CreateServiceRequest,
):
    """
    创建推理服务记录
    
    - **name**: 服务名称
    - **type**: 服务类型（vllm/ollama/diffusers/DiffSynth/modelscope等）
    - **service_type**: 服务内容类型（image/video/audio/text），对应tasks表的type字段
    - **config**: 服务配置（可选）
    - **gpu_enabled**: 是否启用GPU
    - **auto_start**: 是否自动启动
    """
    manager = get_inference_service_manager()
    
    try:
        service_dict = manager.create_service(
            name=request.name,
            service_type=request.type,
            content_service_type=request.service_type,
            config=request.config,
            gpu_enabled=request.gpu_enabled,
            auto_start=request.auto_start
        )
        return ok(data=service_dict, msg="created")
    except Exception as e:
        logger.error(f"Failed to create service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@services_router.get("")
async def list_services(
    type: Optional[str] = None,
    service_type: Optional[str] = None,  # image/video/audio/text
    status: Optional[str] = None,
):
    """
    列出推理服务
    
    - **type**: 服务类型（vllm/ollama/diffusers等，可选）
    - **service_type**: 服务内容类型（image/video/audio/text，可选）
    - **status**: 状态（可选）
    """
    manager = get_inference_service_manager()
    
    services = manager.list_services(
        service_type=type,
        content_service_type=service_type,
        status=status
    )
    
    return ok(data={"services": services, "total": len(services)}, msg="ok")


@services_router.get("/{service_id}")
async def get_service(service_id: str):
    """
    获取推理服务信息
    
    - **service_id**: 服务ID
    """
    manager = get_inference_service_manager()
    
    service_dict = manager.get_service(service_id)
    if not service_dict:
        raise HTTPException(status_code=404, detail="Service not found")
    
    return ok(data=service_dict, msg="ok")


@services_router.put("/{service_id}")
async def update_service(
    service_id: str,
    request: UpdateServiceRequest,
):
    """
    更新推理服务配置
    
    - **service_id**: 服务ID
    - **name**: 服务名称（可选）
    - **type**: 服务类型（vllm/ollama等，可选）
    - **service_type**: 服务内容类型（image/video/audio/text，可选）
    - **config**: 服务配置（可选）
    - **gpu_enabled**: 是否启用GPU（可选）
    - **auto_start**: 是否自动启动（可选）
    
    注意：id、host、port、status、process_id字段不可通过此接口修改
    """
    manager = get_inference_service_manager()
    
    try:
        service_dict = manager.update_service(
            service_id=service_id,
            name=request.name,
            service_type=request.type,
            content_service_type=request.service_type,
            config=request.config,
            gpu_enabled=request.gpu_enabled,
            auto_start=request.auto_start
        )
        return ok(data=service_dict, msg="updated")
    except InferenceServiceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@services_router.post("/{service_id}/start")
async def sync_service_start(
    service_id: str,
    request: SyncServiceStartRequest,
    http_request: Request,
):
    """
    同步服务启动状态（由推理器调用）
    
    - **service_id**: 服务ID
    - **host**: 服务主机地址
    - **port**: 服务端口（可选，仅文本推理器需要）
    - **config**: 服务配置（可选，包含已加载的模型等信息）
    
    注意：此接口由推理器在启动时调用，用于同步服务状态
    """
    manager = get_inference_service_manager()
    
    try:
        service_dict = manager.sync_service_start(
            service_id=service_id,
            host=request.host,
            port=request.port,
            config=request.config,
            client_ip=resolve_client_ip(http_request),
        )
        return ok(data=service_dict, msg="started")
    except InferenceServiceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to sync service start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@services_router.post("/{service_id}/stop")
async def sync_service_stop(service_id: str):
    """
    同步服务停止状态（由推理器调用）
    
    - **service_id**: 服务ID
    
    注意：此接口由推理器在停止时调用，用于同步服务状态
    """
    manager = get_inference_service_manager()
    
    try:
        service_dict = manager.sync_service_stop(service_id)
        return ok(data=service_dict, msg="stopped")
    except InferenceServiceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to sync service stop: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@services_router.delete("/{service_id}")
async def delete_service(service_id: str):
    """
    删除推理服务记录
    
    - **service_id**: 服务ID
    
    注意：如果服务正在运行，会先更新状态为stopped，然后删除记录
    """
    manager = get_inference_service_manager()
    
    try:
        success = manager.delete_service(service_id)
        if success:
            return ok(data={"service_id": service_id}, msg="deleted")
        else:
            raise HTTPException(status_code=500, detail="Failed to delete service")
    except InferenceServiceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 推理侧直传文件接收（仅落盘，不写DB）====================

def _normalize_and_validate_key(key: str) -> str:
    """
    key 约定为相对路径，例如：YYYY/MM/DD/taskid_0.png
    禁止绝对路径、目录穿越（..）、空 key。
    """
    if key is None:
        raise HTTPException(status_code=400, detail="key is required")

    k = key.strip().replace("\\", "/")
    while k.startswith("/"):
        k = k[1:]
    if not k:
        raise HTTPException(status_code=400, detail="key is empty")
    if "\x00" in k:
        raise HTTPException(status_code=400, detail="invalid key")

    parts = [p for p in k.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="invalid key: path traversal")

    return "/".join(parts)


@upload_router.post("/upload", status_code=201)
async def upload_inference_file(
    file: UploadFile = File(..., description="File to upload (multipart)"),
    key: str = Form(..., description="Relative storage path key, e.g. YYYY/MM/DD/xxx.png"),
    overwrite: bool = Form(False, description="Whether to overwrite an existing file"),
    content_type: Optional[str] = Form(None, description="Optional override for upload content_type"),
    x_vitoom_upload_timestamp: Optional[str] = Header(
        default=None, alias="X-Vitoom-Upload-Timestamp", description="(Optional) Inference-side upload auth timestamp (seconds)"
    ),
    x_vitoom_upload_signature: Optional[str] = Header(
        default=None, alias="X-Vitoom-Upload-Signature", description="(Optional) Inference-side upload auth signature (hex)"
    ),
):
    """
    推理侧直传文件接收接口（只负责存文件，不写数据库）

    - **file**: multipart 文件字段名固定为 file
    - **key**: 相对路径（将保存到 storage.local.base_path/key）
    - **overwrite**: 默认不覆盖，存在则返回 409
    """
    normalized_key = _normalize_and_validate_key(key)

    base_path = Path(get_config("storage.local.base_path", "resources/outputs")).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    dest = (base_path / normalized_key).resolve()
    try:
        dest.relative_to(base_path)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid key: outside base_path")

    if dest.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="file already exists")

    dest.parent.mkdir(parents=True, exist_ok=True)

    max_size = int(get_config("upload.max_size", 104857600))
    actual_content_type = content_type or file.content_type or "application/octet-stream"

    # ==================== 可选签权校验 ====================
    # 若配置了 secret（非空）则强制校验；为空则保持历史兼容（不校验）
    secret = str(get_config("inference.upload_auth_secret", "") or "").strip()
    if not secret:
        # 兼容：也允许放到 storage.server.auth.secret
        secret = str(get_config("storage.server.auth.secret", "") or "").strip()
    if secret:
        if not x_vitoom_upload_timestamp or not x_vitoom_upload_signature:
            raise HTTPException(status_code=401, detail="missing upload auth headers")
        try:
            ts = int(str(x_vitoom_upload_timestamp).strip())
        except Exception:
            raise HTTPException(status_code=401, detail="invalid upload timestamp")
        now = int(time.time())
        max_skew = int(get_config("inference.upload_auth_max_skew_seconds", 600) or 600)
        if abs(now - ts) > max_skew:
            raise HTTPException(status_code=401, detail="upload auth expired")
        canonical = f"{ts}\n{normalized_key}\n{actual_content_type}\n".encode("utf-8")
        expected = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, str(x_vitoom_upload_signature).strip()):
            raise HTTPException(status_code=401, detail="invalid upload signature")

    written = 0
    try:
        async with aiofiles.open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_size:
                    raise HTTPException(status_code=413, detail="file too large")
                await out.write(chunk)
    except HTTPException:
        # 清理半写入文件
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        logger.error(f"Upload failed: key={normalized_key}, err={e}", exc_info=True)
        raise HTTPException(status_code=500, detail="upload failed")
    finally:
        await file.close()

    return ok(
        data={
            "key": normalized_key,
            "file_name": file.filename,
            "size": written,
            "content_type": actual_content_type,
        },
        msg="uploaded",
    )


# 将子路由挂载到聚合 router 上（供 app.py include_router）
router.include_router(services_router)
router.include_router(upload_router)

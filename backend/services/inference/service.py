"""
推理服务管理服务层
负责推理服务记录的CRUD管理和状态同步
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

from backend.database import InferenceService
from backend.core.logger import get_app_logger
from backend.core.exceptions import InferenceServiceNotFoundException
from backend.utils import generate_uuid

logger = get_app_logger(__name__)


class InferenceServiceManager:
    """推理服务管理器"""
    
    def __init__(self):
        """初始化推理服务管理器"""
        logger.info("InferenceServiceManager initialized")
    
    def create_service(
        self,
        name: str,
        service_type: str,
        config: Optional[Dict[str, Any]] = None,
        gpu_enabled: bool = False,
        auto_start: bool = False,
        content_service_type: Optional[str] = None,  # image/video/audio/text
    ) -> Dict[str, Any]:
        """
        创建推理服务记录
        
        Args:
            name: 服务名称
            service_type: 服务类型（vllm/ollama/diffusers/DiffSynth/modelscope等）
            config: 服务配置（可选）
            gpu_enabled: 是否启用GPU
            auto_start: 是否自动启动
            content_service_type: 服务内容类型（image/video/audio/text），对应tasks表的type字段
        
        Returns:
            服务信息字典
        """
        service_id = generate_uuid()
        
        service_dict = InferenceService.create(
            id=service_id,
            name=name,
            service_type=service_type,
            config=config,
            gpu_enabled=gpu_enabled,
            auto_start=auto_start,
            status="stopped",
            content_service_type=content_service_type
        )
        
        if not service_dict:
            raise Exception("Failed to create inference service record")
        
        logger.info(f"Inference service created: {service_id} ({name}, content_type: {content_service_type})")
        return service_dict
    
    def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """
        获取推理服务信息
        
        Args:
            service_id: 服务ID
        
        Returns:
            服务信息字典，如果不存在则返回None
        """
        return InferenceService.get_by_id(service_id)
    
    def list_services(
        self,
        service_type: Optional[str] = None,
        content_service_type: Optional[str] = None,  # image/video/audio/text
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        列出推理服务
        
        Args:
            service_type: 服务类型（vllm/ollama/diffusers等，可选）
            content_service_type: 服务内容类型（image/video/audio/text，可选）
            status: 状态（可选）
        
        Returns:
            服务列表
        """
        all_services = InferenceService.list_all()
        
        # 过滤服务类型、内容类型和状态
        if service_type:
            all_services = [s for s in all_services if s.get("type") == service_type]
        if content_service_type:
            all_services = [s for s in all_services if s.get("service_type") == content_service_type]
        if status:
            all_services = [s for s in all_services if s.get("status") == status]
        
        return all_services
    
    def update_service(
        self,
        service_id: str,
        name: Optional[str] = None,
        service_type: Optional[str] = None,
        content_service_type: Optional[str] = None,  # image/video/audio/text
        config: Optional[Dict[str, Any]] = None,
        gpu_enabled: Optional[bool] = None,
        auto_start: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        更新推理服务配置（仅允许修改的字段）
        
        Args:
            service_id: 服务ID
            name: 服务名称（可选）
            service_type: 服务类型（vllm/ollama等，可选）
            content_service_type: 服务内容类型（image/video/audio/text，可选）
            config: 服务配置（可选）
            gpu_enabled: 是否启用GPU（可选）
            auto_start: 是否自动启动（可选）
        
        Returns:
            更新后的服务信息字典
        
        Raises:
            InferenceServiceNotFoundException: 服务不存在
        """
        service_dict = InferenceService.get_by_id(service_id)
        if not service_dict:
            raise InferenceServiceNotFoundException(service_id)
        
        updates = {}
        if name is not None:
            updates["name"] = name
        if service_type is not None:
            updates["type"] = service_type
        if content_service_type is not None:
            updates["service_type"] = content_service_type
        if config is not None:
            updates["config"] = config
        if gpu_enabled is not None:
            updates["gpu_enabled"] = gpu_enabled
        if auto_start is not None:
            updates["auto_start"] = auto_start
        
        if not updates:
            return service_dict
        
        updated = InferenceService.update(service_id, **updates)
        if not updated:
            raise Exception("Failed to update inference service")
        
        logger.info(f"Inference service updated: {service_id}")
        return updated
    
    def sync_service_start(
        self,
        service_id: str,
        host: str,
        port: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        同步服务启动状态（由推理器调用）
        
        Args:
            service_id: 服务ID
            host: 服务主机地址
            port: 服务端口（可选，仅文本推理器需要）
            config: 服务配置（可选，包含已加载的模型等信息）
        
        Returns:
            更新后的服务信息字典
        
        Raises:
            InferenceServiceNotFoundException: 服务不存在
        """
        service_dict = InferenceService.get_by_id(service_id)
        if not service_dict:
            # 推理器启动时会调用 /api/inference/services/{service_id}/start。
            # 为了兼容“service_id 来自本地 yaml（例如 service_200）且未提前创建记录”的场景，
            # 这里做一次 upsert：不存在则自动创建并标记 running。
            inferred_name = service_id
            inferred_type = "unknown"
            inferred_content_service_type: Optional[str] = None  # image/video/audio/text

            if isinstance(config, dict):
                inferred_name = str(config.get("name") or inferred_name)
                # type = 引擎/实现类型（vllm/ollama/diffusers/...），不是 image/video
                inferred_type = str(config.get("type") or inferred_type)
                # service_type = 内容类型（image/video/audio/text），用于任务派发匹配
                inferred_content_service_type = config.get("service_type") or config.get("content_service_type")
                if inferred_content_service_type is not None:
                    inferred_content_service_type = str(inferred_content_service_type)

            created = InferenceService.create(
                id=service_id,
                name=inferred_name,
                service_type=inferred_type,
                port=port,
                host=host,
                client_ip=client_ip,
                config=config,
                status="running",
                gpu_enabled=False,
                auto_start=False,
                content_service_type=inferred_content_service_type,
            )
            if not created:
                raise Exception("Failed to auto-create inference service record on start")

            logger.info(
                f"Inference service auto-created on start: {service_id} "
                f"(content_type: {inferred_content_service_type}, type: {inferred_type})"
            )
            return created
        
        updates = {
            "status": "running",
            "host": host,
        }

        if client_ip:
            updates["client_ip"] = client_ip
        
        if port is not None:
            updates["port"] = port
        
        if config is not None:
            # 合并配置
            current_config = service_dict.get("config") or {}
            current_config.update(config)
            updates["config"] = current_config

            # 若推理器上报了内容类型（image/video/...），同步到 service_type 字段，保证任务派发可匹配
            reported_content_type = config.get("service_type") if isinstance(config, dict) else None
            if reported_content_type and service_dict.get("service_type") != reported_content_type:
                updates["service_type"] = str(reported_content_type)

            # 若推理器上报了引擎类型（vllm/ollama/...），同步到 type 字段
            reported_engine_type = config.get("type") if isinstance(config, dict) else None
            if reported_engine_type and service_dict.get("type") != reported_engine_type:
                updates["type"] = str(reported_engine_type)
            
            reported_name = config.get("name") if isinstance(config, dict) else None
            if reported_name and service_dict.get("name") != reported_name:
                updates["name"] = str(reported_name)
        
        updated = InferenceService.update(service_id, **updates)
        if not updated:
            raise Exception("Failed to sync service start status")
        
        logger.info(f"Inference service started: {service_id} ({host}:{port or 'N/A'})")
        return updated
    
    def sync_service_stop(self, service_id: str) -> Dict[str, Any]:
        """
        同步服务停止状态（由推理器调用）
        
        Args:
            service_id: 服务ID
        
        Returns:
            更新后的服务信息字典
        
        Raises:
            InferenceServiceNotFoundException: 服务不存在
        """
        service_dict = InferenceService.get_by_id(service_id)
        if not service_dict:
            raise InferenceServiceNotFoundException(service_id)
        
        updated = InferenceService.update(service_id, status="stopped")
        if not updated:
            raise Exception("Failed to sync service stop status")
        
        logger.info(f"Inference service stopped: {service_id}")
        return updated

    def sync_service_registration(
        self,
        service_id: str,
        *,
        content_service_type: Optional[str] = None,
        supports_task: bool = True,
        supported_models: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        fixed_model: Optional[str] = None,
        fixed_family: Optional[str] = None,
    ) -> Dict[str, Any]:
        """同步推理服务的注册信息。

        对 audio 大类服务（``service_type in {audio, asr, tts}``）强制要求
        ``supported_models`` 与 ``capabilities`` 均非空：前者是静态模型能力清单，
        后者是本实例对外声明提供的子能力（``tts`` / ``asr`` 等），由 dispatch 在
        pin 路径（空 ``model_name``）下按 capability 过滤候选，避免 TTS 请求被
        随机派发到只提供 ASR 的服务。
        """
        service_dict = InferenceService.get_by_id(service_id)
        if not service_dict:
            raise InferenceServiceNotFoundException(service_id)

        normalized_service_type = str(
            content_service_type or service_dict.get("service_type") or ""
        ).strip().lower()
        normalized_supported_models = [
            str(item).strip()
            for item in (supported_models or [])
            if str(item).strip()
        ]
        normalized_supported_models = list(dict.fromkeys(normalized_supported_models))

        normalized_capabilities = [
            str(item).strip().lower()
            for item in (capabilities or [])
            if str(item).strip()
        ]
        normalized_capabilities = list(dict.fromkeys(normalized_capabilities))
        normalized_fixed_model = str(fixed_model or "").strip()
        normalized_fixed_family = str(fixed_family or "").strip()

        is_audio_service = normalized_service_type in {"audio", "asr", "tts"}
        if is_audio_service and not normalized_supported_models:
            raise ValueError(
                f"Audio inference service '{service_id}' must register non-empty supported_models"
            )
        if is_audio_service and not normalized_capabilities:
            raise ValueError(
                f"Audio inference service '{service_id}' must register non-empty capabilities "
                "(e.g. ['tts'] or ['asr'])"
            )
        if is_audio_service:
            # 三种合法注册（与 inference/audio/inferrer.py::_enforce_fixed_model_consistency 对齐）：
            #   - 都为空：完全无 pin，无服务级默认 family；
            #   - 仅 fixed_family：声明本服务的"默认 family"，不 pin 权重；
            #   - 都设置：传统 pin 模式，dispatch + 推理侧都按 fixed_model 硬收窄。
            # 非法：仅 fixed_model 不带 fixed_family——pin 模式必须知道 runtime。
            if normalized_fixed_model and not normalized_fixed_family:
                raise ValueError(
                    f"Audio inference service '{service_id}' fixed_model={normalized_fixed_model!r} "
                    "requires fixed_family to also be registered (pin mode must know runtime)"
                )
            if normalized_fixed_model and normalized_fixed_model not in normalized_supported_models:
                raise ValueError(
                    f"Audio inference service '{service_id}' fixed_model={normalized_fixed_model!r} "
                    f"is not listed in supported_models={normalized_supported_models}"
                )

        updates: Dict[str, Any] = {
            "status": "running",
            "supports_task": bool(supports_task),
            "last_heartbeat_at": datetime.utcnow(),
        }
        if content_service_type:
            updates["service_type"] = str(content_service_type)
        if normalized_supported_models or normalized_capabilities or normalized_fixed_model or normalized_fixed_family:
            current_config = dict(service_dict.get("config") or {})
            if normalized_supported_models:
                current_config["supported_models"] = normalized_supported_models
            if normalized_capabilities:
                current_config["capabilities"] = normalized_capabilities
            # fixed_model/fixed_family 对 audio 是 dispatch pin 规则；对 avatar 等 sidecar
            # 是服务元数据。两者都应持久化，但只有 audio 需要上面的严格 pin 校验。
            if normalized_fixed_model:
                current_config["fixed_model"] = normalized_fixed_model
            else:
                current_config.pop("fixed_model", None)
            if normalized_fixed_family:
                current_config["fixed_family"] = normalized_fixed_family
            else:
                current_config.pop("fixed_family", None)
            updates["config"] = current_config

        updated = InferenceService.update(service_id, **updates)
        if not updated:
            raise Exception("Failed to sync service registration")

        logger.info(
            "Inference service registration synced: %s (service_type=%s, supports_task=%s, "
            "supported_models=%s, capabilities=%s, fixed_model=%s, fixed_family=%s)",
            service_id,
            updated.get("service_type"),
            updated.get("supports_task"),
            normalized_supported_models,
            normalized_capabilities,
            normalized_fixed_model or None,
            normalized_fixed_family or None,
        )
        return updated

    def sync_service_heartbeat(self, service_id: str) -> Dict[str, Any]:
        """刷新推理服务最近心跳时间。"""
        service_dict = InferenceService.get_by_id(service_id)
        if not service_dict:
            raise InferenceServiceNotFoundException(service_id)

        updated = InferenceService.update(
            service_id,
            status="running",
            last_heartbeat_at=datetime.utcnow(),
        )
        if not updated:
            raise Exception("Failed to sync service heartbeat")
        return updated
    
    def delete_service(self, service_id: str) -> bool:
        """
        删除推理服务记录
        
        Args:
            service_id: 服务ID
        
        Returns:
            是否删除成功
        
        Raises:
            InferenceServiceNotFoundException: 服务不存在
        """
        service_dict = InferenceService.get_by_id(service_id)
        if not service_dict:
            raise InferenceServiceNotFoundException(service_id)
        
        # 如果服务正在运行，先停止（更新状态）
        if service_dict["status"] == "running":
            self.sync_service_stop(service_id)
        
        # 删除服务记录
        from backend.database.db import get_db_context
        with get_db_context() as db:
            from backend.database.models import InferenceService as InferenceServiceModel
            service = db.query(InferenceServiceModel).filter(
                InferenceServiceModel.id == service_id
            ).first()
            if service:
                db.delete(service)
                db.commit()
                logger.info(f"Inference service deleted: {service_id}")
                return True
        
        return False
    
    def reset_all_status_on_startup(self):
        """
        系统启动时重置所有服务状态为stopped
        由FastAPI应用启动时调用
        """
        all_services = InferenceService.list_all()
        for service in all_services:
            if service["status"] == "running":
                InferenceService.update(service["id"], status="stopped")
                logger.info(f"Reset service status to stopped: {service['id']}")


# 全局推理服务管理器实例
_inference_service_manager: Optional[InferenceServiceManager] = None


def get_inference_service_manager() -> InferenceServiceManager:
    """
    获取全局推理服务管理器实例（单例模式）
    
    Returns:
        InferenceServiceManager实例
    """
    global _inference_service_manager
    
    if _inference_service_manager is None:
        _inference_service_manager = InferenceServiceManager()
    
    return _inference_service_manager

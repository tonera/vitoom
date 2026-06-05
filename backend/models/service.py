"""模型目录服务层。"""
from datetime import datetime
from typing import Optional, Dict, Any, List

from backend.database import Model
from backend.database.db import get_db_context
from backend.core.logger import get_app_logger
from backend.core.exceptions import ModelNotFoundException

logger = get_app_logger(__name__)


class ModelService:
    """统一模型主数据服务。"""

    def create_model(
        self,
        name: str,
        modality: str,
        asset_type: str,
        family: str,
        load_name: str,
        runtime_engine: str = "",
        runtime_config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        service_status: str = "inactive",
        storage_mode: str = "cloud",
        download_status: str = "pending",
        source: Optional[Dict[str, Any]] = None,
        thumb: Optional[str] = None,
        tags: Optional[List[str]] = None,
        trigger_words: Optional[List[str]] = None,
        description: Optional[str] = None,
        model_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建模型目录记录。"""
        model_dict = Model.create(
            model_key=model_key,
            name=name,
            modality=modality,
            asset_type=asset_type,
            family=family,
            capabilities=capabilities,
            runtime_engine=runtime_engine,
            runtime_config=runtime_config,
            load_name=load_name,
            service_status=service_status,
            storage_mode=storage_mode,
            download_status=download_status,
            source=source,
            thumb=thumb,
            tags=tags,
            trigger_words=trigger_words,
            description=description,
        )

        if not model_dict:
            raise Exception("Failed to create model catalog record")

        logger.info("Model catalog record created: %s (%s)", model_dict.get("id"), name)
        return model_dict
    
    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """获取模型目录记录。"""
        return Model.get_by_model_key(model_key)
    
    def list_models(
        self,
        modality: Optional[str] = None,
        storage_mode: Optional[str] = None,
        service_status: Optional[str] = None,
        name: Optional[str] = None,
        asset_type: Optional[str] = None,
        family: Optional[str] = None,
        family_in: Optional[List[str]] = None,
        editable: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> tuple[List[Dict[str, Any]], int]:
        """列出模型目录记录。"""
        with get_db_context() as db:
            from backend.database.models import Model as ModelModel
            from sqlalchemy import or_, func
            query = db.query(ModelModel)

            query = query.filter(ModelModel.deleted_at.is_(None))
            if modality:
                query = query.filter(func.lower(ModelModel.modality) == str(modality).strip().lower())
            if storage_mode:
                query = query.filter(ModelModel.storage_mode == storage_mode)
            if service_status:
                query = query.filter(ModelModel.service_status == service_status)
            if name and str(name).strip():
                kw = str(name).strip()
                like = f"%{kw}%"
                query = query.filter(or_(ModelModel.name.ilike(like), ModelModel.load_name.ilike(like), ModelModel.model_key.ilike(like)))
            if asset_type and str(asset_type).strip() and str(asset_type).strip().lower() != "all":
                query = query.filter(func.lower(ModelModel.asset_type) == str(asset_type).strip().lower())
            if family_in is not None:
                vals = [str(x).strip().lower() for x in (family_in or []) if str(x).strip()]
                if vals:
                    query = query.filter(func.lower(ModelModel.family).in_(vals))
            if family and str(family).strip():
                query = query.filter(func.lower(ModelModel.family) == str(family).strip().lower())

            total = query.order_by(None).count()

            models = query.order_by(ModelModel.created_at.desc()).offset(offset).limit(limit).all()
            rows = [model.to_dict() for model in models]
            if editable is not None:
                rows = [
                    row for row in rows
                    if bool((row.get("capabilities") or {}).get("editable")) == bool(editable)
                ]
                total = len(rows)
            return rows, int(total)
    
    def activate_model(self, model_key: str) -> Dict[str, Any]:
        """
        激活模型
        
        Args:
            model_key: 模型稳定键
        
        Returns:
            更新后的模型信息字典
        
        Raises:
            ModelNotFoundException: 模型不存在
        """
        model_dict = Model.get_by_model_key(model_key)
        if not model_dict:
            raise ModelNotFoundException(model_key)
        
        updated_model = Model.update(model_key, service_status="active")
        
        if not updated_model:
            raise ModelNotFoundException(model_key)
        
        logger.info(f"Model activated: {model_key}")
        return updated_model
    
    def deactivate_model(self, model_key: str) -> Dict[str, Any]:
        """
        停用模型
        
        Args:
            model_key: 模型稳定键
        
        Returns:
            更新后的模型信息字典
        
        Raises:
            ModelNotFoundException: 模型不存在
        """
        model_dict = Model.get_by_model_key(model_key)
        if not model_dict:
            raise ModelNotFoundException(model_key)
        
        updated_model = Model.update(model_key, service_status="inactive")
        
        if not updated_model:
            raise ModelNotFoundException(model_key)
        
        logger.info(f"Model deactivated: {model_key}")
        return updated_model
    
    def delete_model(self, model_key: str) -> bool:
        model_dict = Model.get_by_model_key(model_key)
        if not model_dict:
            raise ModelNotFoundException(model_key)

        deleted = Model.update(model_key, deleted_at=datetime.utcnow(), service_status="disabled")
        if deleted:
            logger.info("Model catalog record soft-deleted: %s", model_key)
            return True
        return False
    
    def update_model(
        self,
        model_key: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        更新模型信息
        
        Args:
            model_key: 模型稳定键
            **kwargs: 要更新的字段
        
        Returns:
            更新后的模型信息字典
        
        Raises:
            ModelNotFoundException: 模型不存在
        """
        model_dict = Model.get_by_model_key(model_key)
        if not model_dict:
            raise ModelNotFoundException(model_key)
        
        updated_model = Model.update(model_key, **kwargs)
        
        if not updated_model:
            raise ModelNotFoundException(model_key)
        
        logger.info(f"Model updated: {model_key}")
        return updated_model
    
# 全局模型服务实例
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """
    获取全局模型服务实例（单例模式）
    
    Returns:
        ModelService实例
    """
    global _model_service
    
    if _model_service is None:
        _model_service = ModelService()
    
    return _model_service


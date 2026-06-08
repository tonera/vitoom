"""
数据库模型定义 - SQLAlchemy ORM版本
提供ORM模型和兼容的静态方法接口
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Integer, Boolean, Text, DateTime, BigInteger, ForeignKey, JSON, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, Text as SQLText

from .db import Base, get_db_context

logger = logging.getLogger(__name__)


class JSONEncodedDict(TypeDecorator):
    """JSON类型装饰器，自动序列化/反序列化JSON数据"""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value, ensure_ascii=False)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            # 如果 value 为空字符串，视为 null
            if value == '':
                return None
            return json.loads(value)
        return value


# 根据数据库类型选择JSON类型
def get_json_type():
    """根据数据库类型返回合适的JSON类型"""
    from .db import get_database_url
    db_url = get_database_url()
    if db_url.startswith("postgresql"):
        return JSONB
    elif db_url.startswith("mysql"):
        return JSON
    else:
        return JSONEncodedDict


def _serialize_text_json(value: Any) -> Any:
    """把 dict/list 序列化为 JSON 文本，兼容 Text 列存储。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _deserialize_text_json(value: Any) -> Any:
    """尽量把 Text 列里的 JSON 对象/数组还原出来。"""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

    return parsed if isinstance(parsed, (dict, list)) else value


# ==================== SQLAlchemy ORM 模型定义 ====================

class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    nickname = Column(String(100), nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    status = Column(String(20), default="active")
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    # 关系
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    files = relationship("File", back_populates="user", cascade="all, delete-orphan")
    uploads = relationship("UserUpload", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "nickname": self.nickname,
            "email": self.email,
            "password_hash": self.password_hash,
            "status": self.status,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    # ==================== 静态方法接口（兼容旧代码） ====================
    
    @staticmethod
    def create(
        id: str,
        email: str,
        password_hash: str,
        nickname: Optional[str] = None,
        status: str = "active",
        is_admin: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """创建用户"""
        try:
            with get_db_context() as db:
                user = User(
                    id=id,
                    email=email,
                    password_hash=password_hash,
                    nickname=nickname,
                    status=status,
                    is_admin=is_admin,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                return user.to_dict()
        except Exception as e:
            logger.error(f"Failed to create user: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取用户"""
        with get_db_context() as db:
            user = db.query(User).filter(User.id == user_id).first()
            return user.to_dict() if user else None
    
    @staticmethod
    def get_by_email(email: str) -> Optional[Dict[str, Any]]:
        """根据邮箱获取用户"""
        with get_db_context() as db:
            user = db.query(User).filter(User.email == email).first()
            return user.to_dict() if user else None
    
    @staticmethod
    def update(user_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新用户信息"""
        allowed_fields = ["nickname", "email", "password_hash", "status", "is_admin"]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return User.get_by_id(user_id)
        
        updates["updated_at"] = datetime.utcnow()
        
        try:
            with get_db_context() as db:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return None
                for key, value in updates.items():
                    setattr(user, key, value)
                db.commit()
                db.refresh(user)
                return user.to_dict()
        except Exception as e:
            logger.error(f"Failed to update user: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _apply_keyword_filter(query, keyword: Optional[str]):
        kw = (keyword or "").strip()
        if not kw:
            return query
        pattern = f"%{kw}%"
        return query.filter(
            (User.email.ilike(pattern)) | (User.nickname.ilike(pattern))
        )

    @staticmethod
    def count_all(keyword: Optional[str] = None) -> int:
        """统计用户数量"""
        with get_db_context() as db:
            q = User._apply_keyword_filter(db.query(User), keyword)
            return int(q.count())

    @staticmethod
    def list_all(
        limit: int = 100,
        offset: int = 0,
        keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取用户列表"""
        with get_db_context() as db:
            q = User._apply_keyword_filter(db.query(User), keyword)
            users = q.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
            return [user.to_dict() for user in users]

    @staticmethod
    def delete(user_id: str) -> bool:
        """删除用户"""
        try:
            with get_db_context() as db:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return False
                db.delete(user)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete user: {e}", exc_info=True)
            return False


class ApiKey(Base):
    """用户自定义 API Key，仅保存 HMAC-SHA256 摘要。"""

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_prefix = Column(String(24), nullable=False, index=True)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="api_keys")

    def to_dict(self) -> Dict[str, Any]:
        now = datetime.utcnow()
        is_expired = self.expires_at is not None and self.expires_at <= now
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_expired": is_expired,
        }


class Task(Base):
    """任务模型"""
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    params = Column(get_json_type(), nullable=True)
    progress = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    priority = Column(Integer, default=5)
    model_key = Column(String(64), ForeignKey("model_catalog.model_key"), nullable=True, index=True)
    agent_run_id = Column(String(36), ForeignKey("agent_runs.id"), nullable=True, index=True)
    storage = Column(String(20), nullable=False)  # 任务产物存储目标：local/server/oss/s3
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 关系
    user = relationship("User", back_populates="tasks")
    model = relationship("Model", back_populates="tasks")
    files = relationship("File", back_populates="task", cascade="all, delete-orphan")
    error_logs = relationship("ErrorLog", back_populates="task", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "status": self.status,
            "prompt": self.prompt,
            "params": self.params if self.params is not None else {},
            "progress": self.progress,
            "error": self.error,
            "priority": self.priority,
            "model_key": self.model_key,
            "agent_run_id": self.agent_run_id,
            "storage": self.storage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @staticmethod
    def create(
        id: str,
        user_id: str,
        task_type: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        model_key: Optional[str] = None,
        status: str = "pending",
        storage: Optional[str] = None,
        agent_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """创建任务"""
        if storage is None:
            from backend.core.config import get_config
            storage = get_config("storage.default", "server")
        
        try:
            with get_db_context() as db:
                task = Task(
                    id=id,
                    user_id=user_id,
                    type=task_type,
                    prompt=prompt,
                    params=params,
                    priority=priority,
                    model_key=model_key,
                    agent_run_id=agent_run_id,
                    status=status,
                    storage=storage,
                    created_at=datetime.utcnow(),
                )
                db.add(task)
                db.commit()
                db.refresh(task)
                return task.to_dict()
        except Exception as e:
            logger.error(f"Failed to create task: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(task_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取任务"""
        with get_db_context() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            return task.to_dict() if task else None
    
    @staticmethod
    def update(task_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新任务信息"""
        allowed_fields = ["status", "progress", "error", "storage", "started_at", "completed_at", "agent_run_id"]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return Task.get_by_id(task_id)
        
        # 处理时间字段
        if "started_at" in updates and updates["started_at"]:
            if isinstance(updates["started_at"], str):
                updates["started_at"] = datetime.fromisoformat(updates["started_at"])
            elif not isinstance(updates["started_at"], datetime):
                updates["started_at"] = datetime.utcnow()
        if "completed_at" in updates and updates["completed_at"]:
            if isinstance(updates["completed_at"], str):
                updates["completed_at"] = datetime.fromisoformat(updates["completed_at"])
            elif not isinstance(updates["completed_at"], datetime):
                updates["completed_at"] = datetime.utcnow()
        
        try:
            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if not task:
                    return None
                for key, value in updates.items():
                    setattr(task, key, value)
                db.commit()
                db.refresh(task)
                return task.to_dict()
        except Exception as e:
            logger.error(f"Failed to update task: {e}", exc_info=True)
            return None
    
    @staticmethod
    def list_by_user(user_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取用户的任务列表"""
        with get_db_context() as db:
            tasks = db.query(Task).filter(
                Task.user_id == user_id
            ).order_by(Task.created_at.desc()).offset(offset).limit(limit).all()
            return [task.to_dict() for task in tasks]
    
    @staticmethod
    def list_by_status(status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """根据状态获取任务列表"""
        with get_db_context() as db:
            tasks = db.query(Task).filter(
                Task.status == status
            ).order_by(Task.priority.desc(), Task.created_at.asc()).limit(limit).all()
            return [task.to_dict() for task in tasks]

    @staticmethod
    def list_by_agent_run_id(agent_run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """列出绑定到某个 AgentRun 的任务。"""
        normalized = str(agent_run_id or "").strip()
        if not normalized:
            return []
        with get_db_context() as db:
            tasks = (
                db.query(Task)
                .filter(Task.agent_run_id == normalized)
                .order_by(Task.created_at.asc())
                .limit(limit)
                .all()
            )
            return [task.to_dict() for task in tasks]


class Model(Base):
    """统一模型目录。"""
    __tablename__ = "model_catalog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_key = Column(String(64), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    modality = Column(String(20), nullable=False, default="image", index=True)
    asset_type = Column(String(30), nullable=False, default="checkpoint", index=True)
    family = Column(String(100), nullable=False, default="", index=True)
    capabilities = Column(get_json_type(), nullable=False, default=dict)
    runtime_engine = Column(String(100), nullable=False, default="", index=True)
    runtime_config = Column(get_json_type(), nullable=False, default=dict)
    load_name = Column(Text, nullable=False)
    service_status = Column(String(20), nullable=False, default="inactive", index=True)
    storage_mode = Column(String(20), nullable=False, default="cloud", index=True)
    download_status = Column(String(20), nullable=False, default="pending")
    source = Column(get_json_type(), nullable=False, default=dict)
    thumb = Column(Text, nullable=True)
    tags = Column(get_json_type(), nullable=False, default=list)
    trigger_words = Column(get_json_type(), nullable=False, default=list)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True, index=True)

    tasks = relationship("Task", back_populates="model")

    @staticmethod
    def _normalize_model_key_part(value: Any, *, lower: bool = True) -> str:
        text = "" if value is None else str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower() if lower else text

    @staticmethod
    def build_model_key(load_name: str, runtime_engine: str, asset_type: str) -> str:
        """与迁移脚本 / 种子数据一致：load_name 保留大小写，engine/asset_type 小写。"""
        key_source = "|".join(
            [
                Model._normalize_model_key_part(load_name, lower=False),
                Model._normalize_model_key_part(runtime_engine),
                Model._normalize_model_key_part(asset_type),
            ]
        )
        return hashlib.md5(key_source.encode("utf-8")).hexdigest()

    @staticmethod
    def _coerce_id(id_value: Any) -> Optional[int]:
        text = str(id_value or "").strip()
        if not text:
            return None
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为新模型主数据字典。"""
        return {
            "id": self.id,
            "model_key": self.model_key,
            "name": self.name,
            "modality": self.modality,
            "asset_type": self.asset_type,
            "family": self.family,
            "capabilities": self.capabilities if self.capabilities is not None else {},
            "runtime_engine": self.runtime_engine,
            "runtime_config": self.runtime_config if self.runtime_config is not None else {},
            "load_name": self.load_name,
            "service_status": self.service_status,
            "storage_mode": self.storage_mode,
            "download_status": self.download_status,
            "source": self.source if self.source is not None else {},
            "thumb": self.thumb,
            "tags": self.tags if self.tags is not None else [],
            "trigger_words": self.trigger_words if self.trigger_words is not None else [],
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
        }

    @staticmethod
    def create(
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
    ) -> Optional[Dict[str, Any]]:
        """创建模型目录记录。"""
        normalized_load_name = str(load_name or "").strip()
        normalized_engine = str(runtime_engine or "").strip()
        normalized_asset_type = str(asset_type or "checkpoint").strip() or "checkpoint"
        normalized_model_key = str(model_key or "").strip() or Model.build_model_key(
            normalized_load_name,
            normalized_engine,
            normalized_asset_type,
        )

        try:
            with get_db_context() as db:
                model = Model(
                    model_key=normalized_model_key,
                    name=name,
                    modality=str(modality or "image").strip() or "image",
                    asset_type=normalized_asset_type,
                    family=str(family or "").strip(),
                    capabilities=capabilities or {},
                    runtime_engine=normalized_engine,
                    runtime_config=runtime_config or {},
                    load_name=normalized_load_name,
                    service_status=str(service_status or "inactive").strip() or "inactive",
                    storage_mode=str(storage_mode or "cloud").strip() or "cloud",
                    download_status=str(download_status or "pending").strip() or "pending",
                    source=source or {},
                    thumb=thumb,
                    tags=tags or [],
                    trigger_words=trigger_words or [],
                    description=description,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(model)
                db.commit()
                db.refresh(model)
                return model.to_dict()
        except Exception as e:
            logger.error(f"Failed to create model catalog record: {e}", exc_info=True)
            return None

    @staticmethod
    def _repo_id_basename(repo_id: str) -> str:
        rid = str(repo_id or "").strip()
        if not rid:
            return ""
        return rid.rsplit("/", 1)[-1].strip()

    @staticmethod
    def get_by_source(provider: str, repo_id: str) -> Optional[Dict[str, Any]]:
        """通过 source.provider + source.repo_id 查找模型。"""
        normalized_provider = str(provider or "").strip().lower()
        normalized_repo_id = str(repo_id or "").strip()
        if not normalized_provider or not normalized_repo_id:
            return None
        with get_db_context() as db:
            rows = db.query(Model).filter(Model.deleted_at.is_(None)).order_by(Model.created_at.desc()).all()
            for row in rows:
                source = row.source if isinstance(row.source, dict) else {}
                if (
                    str(source.get("provider") or "").strip().lower() == normalized_provider
                    and str(source.get("repo_id") or "").strip() == normalized_repo_id
                ):
                    return row.to_dict()
            return None

    @staticmethod
    def find_existing_for_create(
        *,
        name: str,
        load_name: str,
        source: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        创建前去重：优先 source，再 load_name / name，再 repo_id 与本地名匹配。
        兼容初始安装种子（source 为空、load_name 为短名）与导入（repo_id 为 org/name）。
        """
        src = source if isinstance(source, dict) else {}
        provider = str(src.get("provider") or "").strip()
        repo_id = str(src.get("repo_id") or "").strip()

        if provider and repo_id:
            existed = Model.get_by_source(provider, repo_id)
            if existed:
                return existed

        normalized_load_name = str(load_name or "").strip()
        if normalized_load_name:
            existed = Model.get_by_load_name(normalized_load_name)
            if existed:
                return existed

        normalized_name = str(name or "").strip()
        if normalized_name and normalized_name != normalized_load_name:
            existed = Model.get_by_name(normalized_name)
            if existed:
                return existed

        if repo_id:
            basename = Model._repo_id_basename(repo_id)
            for candidate in (basename, repo_id):
                if not candidate:
                    continue
                existed = Model.get_by_load_name(candidate)
                if existed:
                    return existed
                existed = Model.get_by_name(candidate)
                if existed:
                    return existed

        return None

    @staticmethod
    def get_by_id(id_value: Any) -> Optional[Dict[str, Any]]:
        """根据本地自增 ID 获取模型。"""
        coerced_id = Model._coerce_id(id_value)
        if coerced_id is None:
            return None
        with get_db_context() as db:
            model = db.query(Model).filter(Model.id == coerced_id, Model.deleted_at.is_(None)).first()
            return model.to_dict() if model else None

    @staticmethod
    def get_by_model_key(model_key: str) -> Optional[Dict[str, Any]]:
        key = (model_key or "").strip()
        if not key:
            return None
        with get_db_context() as db:
            model = db.query(Model).filter(Model.model_key == key, Model.deleted_at.is_(None)).first()
            return model.to_dict() if model else None

    @staticmethod
    def get_by_name(model_name: str) -> Optional[Dict[str, Any]]:
        name = (model_name or "").strip()
        if not name:
            return None
        with get_db_context() as db:
            model = (
                db.query(Model)
                .filter(Model.name == name, Model.deleted_at.is_(None))
                .order_by(Model.created_at.desc())
                .first()
            )
            return model.to_dict() if model else None

    @staticmethod
    def get_by_load_name(load_name: str) -> Optional[Dict[str, Any]]:
        name = (load_name or "").strip()
        if not name:
            return None
        with get_db_context() as db:
            model = (
                db.query(Model)
                .filter(Model.load_name == name, Model.deleted_at.is_(None))
                .order_by(Model.created_at.desc())
                .first()
            )
            return model.to_dict() if model else None

    @staticmethod
    def update(model_key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新模型目录记录。"""
        allowed_fields = [
            "model_key",
            "name",
            "modality",
            "asset_type",
            "family",
            "capabilities",
            "runtime_engine",
            "runtime_config",
            "load_name",
            "service_status",
            "storage_mode",
            "download_status",
            "source",
            "thumb",
            "tags",
            "trigger_words",
            "description",
            "deleted_at",
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return Model.get_by_model_key(model_key)

        updates["updated_at"] = datetime.utcnow()

        try:
            with get_db_context() as db:
                key = str(model_key or "").strip()
                if not key:
                    return None
                model = db.query(Model).filter(Model.model_key == key, Model.deleted_at.is_(None)).first()
                if not model:
                    return None
                for field, value in updates.items():
                    setattr(model, field, value)
                identity_fields = {"load_name", "runtime_engine", "asset_type"}
                if identity_fields.intersection(updates) and "model_key" not in updates:
                    new_key = Model.build_model_key(
                        model.load_name, model.runtime_engine, model.asset_type
                    )
                    if new_key != model.model_key:
                        model.model_key = new_key
                db.commit()
                db.refresh(model)
                return model.to_dict()
        except Exception as e:
            logger.error(f"Failed to update model catalog record: {e}", exc_info=True)
            return None

    @staticmethod
    def list_by_modality(modality: str, service_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """根据任务领域获取模型列表。"""
        with get_db_context() as db:
            query = db.query(Model).filter(Model.modality == modality, Model.deleted_at.is_(None))
            if service_status:
                query = query.filter(Model.service_status == service_status)
            models = query.order_by(Model.created_at.desc()).all()
            return [model.to_dict() for model in models]


class File(Base):
    """文件模型"""
    __tablename__ = "files"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    category = Column(String(20), nullable=False, index=True)
    storage = Column(String(20), nullable=False)
    storage_path = Column(Text, nullable=False)
    file_name = Column(String(255), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    http_url = Column(Text, nullable=True)
    file_metadata = Column("metadata", get_json_type(), nullable=True)  # 数据库列名保持为metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # 关系
    task = relationship("Task", back_populates="files")
    user = relationship("User", back_populates="files")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "category": self.category,
            "storage": self.storage,
            "storage_path": self.storage_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "http_url": self.http_url,
            "metadata": self.file_metadata if self.file_metadata is not None else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def create(
        id: str,
        user_id: str,
        category: str,
        storage: str,
        storage_path: str,
        file_name: Optional[str] = None,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
        http_url: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """创建文件记录"""
        try:
            with get_db_context() as db:
                file = File(
                    id=id,
                    task_id=task_id,
                    user_id=user_id,
                    category=category,
                    storage=storage,
                    storage_path=storage_path,
                    file_name=file_name,
                    file_size=file_size,
                    mime_type=mime_type,
                    http_url=http_url,
                    file_metadata=metadata,
                    created_at=datetime.utcnow(),
                )
                db.add(file)
                db.commit()
                db.refresh(file)
                return file.to_dict()
        except Exception as e:
            logger.error(f"Failed to create file: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(file_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文件"""
        with get_db_context() as db:
            file = db.query(File).filter(File.id == file_id).first()
            return file.to_dict() if file else None
    
    @staticmethod
    def list_by_user(user_id: str, category: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取用户的文件列表"""
        with get_db_context() as db:
            query = db.query(File).filter(File.user_id == user_id)
            if category:
                query = query.filter(File.category == category)
            files = query.order_by(File.created_at.desc()).offset(offset).limit(limit).all()
            return [file.to_dict() for file in files]
    
    @staticmethod
    def list_by_task(task_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取任务的文件列表"""
        with get_db_context() as db:
            files = db.query(File).filter(
                File.task_id == task_id
            ).order_by(File.created_at.desc()).offset(offset).limit(limit).all()
            return [file.to_dict() for file in files]
    
    @staticmethod
    def list_by_category(category: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """根据类别获取文件列表"""
        with get_db_context() as db:
            files = db.query(File).filter(
                File.category == category
            ).order_by(File.created_at.desc()).offset(offset).limit(limit).all()
            return [file.to_dict() for file in files]
    
    @staticmethod
    def list_all(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取所有文件列表"""
        with get_db_context() as db:
            files = db.query(File).order_by(File.created_at.desc()).offset(offset).limit(limit).all()
            return [file.to_dict() for file in files]
    
    @staticmethod
    def delete(file_id: str) -> bool:
        """删除文件记录"""
        try:
            with get_db_context() as db:
                file = db.query(File).filter(File.id == file_id).first()
                if file:
                    db.delete(file)
                    db.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete file: {e}", exc_info=True)
            return False


class UserUpload(Base):
    """用户上传文件"""
    __tablename__ = "user_uploads"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    original_name = Column(String(255), nullable=True)
    storage_path = Column(Text, nullable=False)  # uploads/{YYYYMM}/uuid.ext
    storage = Column(String(20), nullable=False, default="server")  # server | s3 | oss（Backend 侧）
    file_size = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="uploads")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "original_name": self.original_name,
            "storage_path": self.storage_path,
            "storage": self.storage,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def create(
        id: str,
        user_id: str,
        original_name: Optional[str],
        storage_path: str,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
        storage: str = "server",
    ) -> Optional[Dict[str, Any]]:
        try:
            with get_db_context() as db:
                rec = UserUpload(
                    id=id,
                    user_id=user_id,
                    original_name=original_name,
                    storage_path=storage_path,
                    storage=storage,
                    file_size=file_size,
                    mime_type=mime_type,
                    created_at=datetime.utcnow(),
                )
                db.add(rec)
                db.commit()
                db.refresh(rec)
                return rec.to_dict()
        except Exception as e:
            logger.error(f"Failed to create user upload: {e}", exc_info=True)
            return None

    @staticmethod
    def list_by_user(
        user_id: str,
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取用户上传文件列表（按 original_name 模糊搜索）"""
        with get_db_context() as db:
            q = db.query(UserUpload).filter(UserUpload.user_id == user_id)
            kw = (keyword or "").strip()
            if kw:
                q = q.filter(UserUpload.original_name.ilike(f"%{kw}%"))
            rows = q.order_by(UserUpload.created_at.desc()).offset(offset).limit(limit).all()
            return [r.to_dict() for r in rows]

    @staticmethod
    def count_by_user(user_id: str, keyword: Optional[str] = None) -> int:
        """统计用户上传文件数量"""
        with get_db_context() as db:
            q = db.query(UserUpload).filter(UserUpload.user_id == user_id)
            kw = (keyword or "").strip()
            if kw:
                q = q.filter(UserUpload.original_name.ilike(f"%{kw}%"))
            return int(q.count())


class Config(Base):
    """配置模型"""
    __tablename__ = "config"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    updated_by = Column(String(36), nullable=True)

    @staticmethod
    def get(key: str, default: Optional[Any] = None) -> Optional[Any]:
        """获取配置值"""
        with get_db_context() as db:
            config = db.query(Config).filter(Config.key == key).first()
            if config and config.value:
                try:
                    return json.loads(config.value)
                except (json.JSONDecodeError, TypeError):
                    return config.value
            return default
    
    @staticmethod
    def set(key: str, value: Any, description: Optional[str] = None, 
            category: Optional[str] = None, updated_by: Optional[str] = None):
        """设置配置值"""
        value_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        with get_db_context() as db:
            config = db.query(Config).filter(Config.key == key).first()
            if config:
                config.value = value_str
                config.description = description
                config.category = category
                config.updated_by = updated_by
                config.updated_at = datetime.utcnow()
            else:
                config = Config(
                    key=key,
                    value=value_str,
                    description=description,
                    category=category,
                    updated_by=updated_by,
                    updated_at=datetime.utcnow(),
                )
                db.add(config)
            db.commit()
    
    @staticmethod
    def get_all_by_category(category: str) -> Dict[str, Any]:
        """获取指定分类的所有配置"""
        with get_db_context() as db:
            configs = db.query(Config).filter(Config.category == category).all()
            result = {}
            for config in configs:
                try:
                    result[config.key] = json.loads(config.value)
                except (json.JSONDecodeError, TypeError):
                    result[config.key] = config.value
            return result


class ErrorLog(Base):
    """错误日志模型"""
    __tablename__ = "error_logs"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True, index=True)
    error_type = Column(String(50), nullable=True)
    message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    severity = Column(String(20), default="error")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # 关系
    task = relationship("Task", back_populates="error_logs")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "error_type": self.error_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "severity": self.severity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def create(
        id: str,
        error_type: str,
        message: str,
        task_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        severity: str = "error",
    ) -> Optional[Dict[str, Any]]:
        """创建错误日志"""
        try:
            with get_db_context() as db:
                error_log = ErrorLog(
                    id=id,
                    task_id=task_id,
                    error_type=error_type,
                    message=message,
                    stack_trace=stack_trace,
                    severity=severity,
                    created_at=datetime.utcnow(),
                )
                db.add(error_log)
                db.commit()
                db.refresh(error_log)
                return error_log.to_dict()
        except Exception as e:
            logger.error(f"Failed to create error log: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(log_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取错误日志"""
        with get_db_context() as db:
            error_log = db.query(ErrorLog).filter(ErrorLog.id == log_id).first()
            return error_log.to_dict() if error_log else None
    
    @staticmethod
    def list_by_task(task_id: str) -> List[Dict[str, Any]]:
        """获取任务相关的错误日志"""
        with get_db_context() as db:
            error_logs = db.query(ErrorLog).filter(
                ErrorLog.task_id == task_id
            ).order_by(ErrorLog.created_at.desc()).all()
            return [log.to_dict() for log in error_logs]


class InferenceService(Base):
    """推理服务模型"""
    __tablename__ = "inference_services"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # 服务类型：vllm/ollama/diffusers/DiffSynth/modelscope等
    service_type = Column(String(20), nullable=True, index=True)  # 服务内容类型：image/video/audio/text
    status = Column(String(20), default="stopped", index=True)
    port = Column(Integer, nullable=True)
    host = Column(String(255), default="127.0.0.1")
    client_ip = Column(String(45), nullable=True)
    config = Column(get_json_type(), nullable=True)
    supports_task = Column(Boolean, default=True, nullable=False)
    process_id = Column(Integer, nullable=True)
    gpu_enabled = Column(Boolean, default=False)
    auto_start = Column(Boolean, default=False)
    last_heartbeat_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    # 关系
    logs = relationship("InferenceServiceLog", back_populates="service", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "service_type": self.service_type,
            "status": self.status,
            "port": self.port,
            "host": self.host,
            "client_ip": self.client_ip,
            "config": self.config if self.config is not None else {},
            "supports_task": self.supports_task,
            "process_id": self.process_id,
            "gpu_enabled": self.gpu_enabled,
            "auto_start": self.auto_start,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def create(
        id: str,
        name: str,
        service_type: str,
        port: Optional[int] = None,
        host: str = "127.0.0.1",
        client_ip: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        status: str = "stopped",
        supports_task: bool = True,
        gpu_enabled: bool = False,
        auto_start: bool = False,
        content_service_type: Optional[str] = None,  # image/video/audio/text
    ) -> Optional[Dict[str, Any]]:
        """创建推理服务记录"""
        try:
            with get_db_context() as db:
                service = InferenceService(
                    id=id,
                    name=name,
                    type=service_type,
                    service_type=content_service_type,
                    status=status,
                    port=port,
                    host=host,
                    client_ip=client_ip,
                    config=config,
                    supports_task=supports_task,
                    process_id=None,
                    gpu_enabled=gpu_enabled,
                    auto_start=auto_start,
                    last_heartbeat_at=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(service)
                db.commit()
                db.refresh(service)
                return service.to_dict()
        except Exception as e:
            logger.error(f"Failed to create inference service: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(service_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取推理服务"""
        with get_db_context() as db:
            service = db.query(InferenceService).filter(InferenceService.id == service_id).first()
            return service.to_dict() if service else None
    
    @staticmethod
    def update(service_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新推理服务信息"""
        allowed_fields = [
            "name",
            "type",
            "service_type",
            "status",
            "port",
            "host",
            "client_ip",
            "config",
            "supports_task",
            "process_id",
            "gpu_enabled",
            "auto_start",
            "last_heartbeat_at",
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return InferenceService.get_by_id(service_id)

        if "last_heartbeat_at" in updates and updates["last_heartbeat_at"]:
            if isinstance(updates["last_heartbeat_at"], str):
                updates["last_heartbeat_at"] = datetime.fromisoformat(updates["last_heartbeat_at"])
            elif not isinstance(updates["last_heartbeat_at"], datetime):
                updates["last_heartbeat_at"] = datetime.utcnow()
        
        updates["updated_at"] = datetime.utcnow()
        
        try:
            with get_db_context() as db:
                service = db.query(InferenceService).filter(InferenceService.id == service_id).first()
                if not service:
                    return None
                for key, value in updates.items():
                    setattr(service, key, value)
                db.commit()
                db.refresh(service)
                return service.to_dict()
        except Exception as e:
            logger.error(f"Failed to update inference service: {e}", exc_info=True)
            return None
    
    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        """获取所有推理服务"""
        with get_db_context() as db:
            services = db.query(InferenceService).order_by(InferenceService.created_at.desc()).all()
            return [service.to_dict() for service in services]


class InferenceServiceLog(Base):
    """推理服务日志模型"""
    __tablename__ = "inference_service_logs"

    id = Column(String(36), primary_key=True)
    service_id = Column(String(36), ForeignKey("inference_services.id"), nullable=False)
    level = Column(String(20), nullable=False)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # 关系
    service = relationship("InferenceService", back_populates="logs")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "service_id": self.service_id,
            "level": self.level,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def create(
        id: str,
        service_id: str,
        level: str,
        message: str,
    ) -> Optional[Dict[str, Any]]:
        """创建推理服务日志"""
        try:
            with get_db_context() as db:
                log = InferenceServiceLog(
                    id=id,
                    service_id=service_id,
                    level=level,
                    message=message,
                    created_at=datetime.utcnow(),
                )
                db.add(log)
                db.commit()
                db.refresh(log)
                return log.to_dict()
        except Exception as e:
            logger.error(f"Failed to create inference service log: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(log_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取日志"""
        with get_db_context() as db:
            log = db.query(InferenceServiceLog).filter(InferenceServiceLog.id == log_id).first()
            return log.to_dict() if log else None
    
    @staticmethod
    def list_by_service(service_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取服务的日志列表"""
        with get_db_context() as db:
            logs = db.query(InferenceServiceLog).filter(
                InferenceServiceLog.service_id == service_id
            ).order_by(InferenceServiceLog.created_at.desc()).limit(limit).all()
            return [log.to_dict() for log in logs]


class Agent(Base):
    """AI Agent模型"""
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String(50), nullable=False, index=True)
    config = Column(get_json_type(), nullable=False)
    status = Column(String(20), default="active", index=True)
    is_preset = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "config": self.config if self.config is not None else {},
            "status": self.status,
            "is_preset": self.is_preset,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def create(
        id: str,
        name: str,
        agent_type: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        status: str = "active",
        is_preset: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """创建Agent记录"""
        try:
            with get_db_context() as db:
                agent = Agent(
                    id=id,
                    name=name,
                    description=description,
                    type=agent_type,
                    config=config,
                    status=status,
                    is_preset=is_preset,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(agent)
                db.commit()
                db.refresh(agent)
                return agent.to_dict()
        except Exception as e:
            logger.error(f"Failed to create agent: {e}", exc_info=True)
            return None
    
    @staticmethod
    def get_by_id(agent_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取Agent"""
        with get_db_context() as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            return agent.to_dict() if agent else None
    
    @staticmethod
    def update(agent_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新Agent信息"""
        allowed_fields = ["name", "description", "type", "config", "status"]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return Agent.get_by_id(agent_id)
        
        updates["updated_at"] = datetime.utcnow()
        
        try:
            with get_db_context() as db:
                agent = db.query(Agent).filter(Agent.id == agent_id).first()
                if not agent:
                    return None
                for key, value in updates.items():
                    setattr(agent, key, value)
                db.commit()
                db.refresh(agent)
                return agent.to_dict()
        except Exception as e:
            logger.error(f"Failed to update agent: {e}", exc_info=True)
            return None
    
    @staticmethod
    def list_all(status: Optional[str] = None, is_preset: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取Agent列表"""
        with get_db_context() as db:
            query = db.query(Agent)
            if status:
                query = query.filter(Agent.status == status)
            if is_preset is not None:
                query = query.filter(Agent.is_preset == is_preset)
            agents = query.order_by(Agent.created_at.desc()).all()
            return [agent.to_dict() for agent in agents]


class AgentRun(Base):
    """Agent 运行记录模型"""
    __tablename__ = "agent_runs"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False, index=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True, index=True)
    turn_id = Column(String(36), nullable=True, index=True)
    source_type = Column(String(30), nullable=False, default="web", index=True)
    source_ref = Column(String(255), nullable=True, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    parent_run_id = Column(String(36), ForeignKey("agent_runs.id"), nullable=True, index=True)
    crew_tool_name = Column(String(100), nullable=True, index=True)
    status = Column(String(30), nullable=False, default="created", index=True)
    input_payload = Column(get_json_type(), nullable=False)
    runtime_config = Column(get_json_type(), nullable=True)
    result_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    usage_metrics = Column(get_json_type(), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "conversation_id": self.conversation_id,
            "parent_run_id": self.parent_run_id,
            "crew_tool_name": self.crew_tool_name,
            "status": self.status,
            "input_payload": self.input_payload if self.input_payload is not None else {},
            "runtime_config": self.runtime_config if self.runtime_config is not None else {},
            "result_summary": _deserialize_text_json(self.result_summary),
            "error_message": self.error_message,
            "usage_metrics": self.usage_metrics if self.usage_metrics is not None else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def create(
        id: str,
        user_id: str,
        agent_id: str,
        input_payload: Dict[str, Any],
        task_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        source_type: str = "web",
        source_ref: Optional[str] = None,
        status: str = "created",
        runtime_config: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        crew_tool_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """创建 AgentRun 记录"""
        try:
            with get_db_context() as db:
                agent_run = AgentRun(
                    id=id,
                    user_id=user_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    turn_id=turn_id,
                    source_type=source_type,
                    source_ref=source_ref,
                    conversation_id=conversation_id,
                    parent_run_id=parent_run_id,
                    crew_tool_name=crew_tool_name,
                    status=status,
                    input_payload=input_payload,
                    runtime_config=runtime_config,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(agent_run)
                db.commit()
                db.refresh(agent_run)
                return agent_run.to_dict()
        except Exception as e:
            logger.error(f"Failed to create agent run: {e}", exc_info=True)
            return None

    @staticmethod
    def get_by_id(run_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取 AgentRun"""
        with get_db_context() as db:
            agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
            return agent_run.to_dict() if agent_run else None

    @staticmethod
    def get_by_task_id(task_id: str) -> Optional[Dict[str, Any]]:
        """根据任务 ID 获取 AgentRun"""
        with get_db_context() as db:
            agent_run = db.query(AgentRun).filter(AgentRun.task_id == task_id).first()
            return agent_run.to_dict() if agent_run else None

    @staticmethod
    def update(run_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """更新 AgentRun 信息"""
        allowed_fields = [
            "task_id",
            "turn_id",
            "source_type",
            "source_ref",
            "status",
            "input_payload",
            "runtime_config",
            "result_summary",
            "error_message",
            "usage_metrics",
            "started_at",
            "completed_at",
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return AgentRun.get_by_id(run_id)

        if "result_summary" in updates:
            updates["result_summary"] = _serialize_text_json(updates["result_summary"])

        updates["updated_at"] = datetime.utcnow()

        try:
            with get_db_context() as db:
                agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
                if not agent_run:
                    return None
                for key, value in updates.items():
                    setattr(agent_run, key, value)
                db.commit()
                db.refresh(agent_run)
                return agent_run.to_dict()
        except Exception as e:
            logger.error(f"Failed to update agent run: {e}", exc_info=True)
            return None

    @staticmethod
    def list_by_user(
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按用户获取 AgentRun 列表"""
        with get_db_context() as db:
            query = db.query(AgentRun).filter(AgentRun.user_id == user_id)
            if status:
                query = query.filter(AgentRun.status == status)
            runs = query.order_by(AgentRun.created_at.desc()).offset(offset).limit(limit).all()
            return [run.to_dict() for run in runs]

    @staticmethod
    def list_by_parent(parent_run_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """列出某次主 Run 派生的子 Run。"""
        with get_db_context() as db:
            rows = (
                db.query(AgentRun)
                .filter(AgentRun.parent_run_id == parent_run_id)
                .order_by(AgentRun.created_at.asc())
                .limit(limit)
                .all()
            )
            return [row.to_dict() for row in rows]

    @staticmethod
    def list_by_turn(turn_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """列出同一 Turn 下的 Run（主 Run + 子 Run）。"""
        if not turn_id:
            return []
        with get_db_context() as db:
            rows = (
                db.query(AgentRun)
                .filter(AgentRun.turn_id == turn_id)
                .order_by(AgentRun.created_at.asc())
                .limit(limit)
                .all()
            )
            return [row.to_dict() for row in rows]


class Conversation(Base):
    """多轮会话模型 - 记录用户与 Agent 的连续对话。"""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=True, index=True)
    title = Column(String(255), nullable=True)
    status = Column(String(20), default="active", index=True)
    conversation_metadata = Column("metadata", get_json_type(), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "title": self.title,
            "status": self.status,
            "metadata": self.conversation_metadata if self.conversation_metadata is not None else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def create(
        id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        title: Optional[str] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            with get_db_context() as db:
                conversation = Conversation(
                    id=id,
                    user_id=user_id,
                    agent_id=agent_id,
                    title=title,
                    status=status,
                    conversation_metadata=metadata,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)
                return conversation.to_dict()
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}", exc_info=True)
            return None

    @staticmethod
    def get_by_id(conversation_id: str) -> Optional[Dict[str, Any]]:
        with get_db_context() as db:
            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            return conversation.to_dict() if conversation else None

    @staticmethod
    def update(conversation_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        allowed_fields = ["agent_id", "title", "status"]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if "metadata" in kwargs:
            updates["conversation_metadata"] = kwargs["metadata"]
        if not updates:
            return Conversation.get_by_id(conversation_id)
        updates["updated_at"] = datetime.utcnow()
        try:
            with get_db_context() as db:
                conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
                if not conv:
                    return None
                for k, v in updates.items():
                    setattr(conv, k, v)
                db.commit()
                db.refresh(conv)
                return conv.to_dict()
        except Exception as e:
            logger.error(f"Failed to update conversation: {e}", exc_info=True)
            return None

    @staticmethod
    def list_by_user(
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        title_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按最近活动时间倒序；可选 ``title_query`` 在会话标题中子串匹配（大小写不敏感）。"""
        raw_q = str(title_query or "").strip()
        with get_db_context() as db:
            activity = func.coalesce(Conversation.updated_at, Conversation.created_at)
            q = db.query(Conversation).filter(Conversation.user_id == user_id)
            if raw_q:
                escaped = (
                    raw_q.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                title_expr = func.coalesce(Conversation.title, "")
                q = q.filter(title_expr.ilike(f"%{escaped}%", escape="\\"))
            rows = (
                q.order_by(activity.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [row.to_dict() for row in rows]


class ConversationMessage(Base):
    """会话消息模型 - 每条 user / assistant / system 消息。"""

    __tablename__ = "conversation_messages"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False, index=True)
    content = Column(Text, nullable=False)
    agent_run_id = Column(String(36), ForeignKey("agent_runs.id"), nullable=True, index=True)
    turn_id = Column(String(36), nullable=True, index=True)
    message_metadata = Column("metadata", get_json_type(), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "agent_run_id": self.agent_run_id,
            "turn_id": self.turn_id,
            "metadata": self.message_metadata if self.message_metadata is not None else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def create(
        id: str,
        conversation_id: str,
        role: str,
        content: str,
        agent_run_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            with get_db_context() as db:
                message = ConversationMessage(
                    id=id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    agent_run_id=agent_run_id,
                    turn_id=turn_id,
                    message_metadata=metadata,
                    created_at=datetime.utcnow(),
                )
                db.add(message)
                db.commit()
                db.refresh(message)
                return message.to_dict()
        except Exception as e:
            logger.error(f"Failed to create conversation message: {e}", exc_info=True)
            return None

    @staticmethod
    def list_by_conversation(
        conversation_id: str,
        limit: int = 200,
        offset: int = 0,
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        with get_db_context() as db:
            query = db.query(ConversationMessage).filter(
                ConversationMessage.conversation_id == conversation_id
            )
            if ascending:
                query = query.order_by(ConversationMessage.created_at.asc())
            else:
                query = query.order_by(ConversationMessage.created_at.desc())
            rows = query.offset(offset).limit(limit).all()
            return [row.to_dict() for row in rows]

    @staticmethod
    def update(message_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        allowed_fields = ["content", "agent_run_id", "role", "turn_id"]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if "metadata" in kwargs:
            updates["message_metadata"] = kwargs["metadata"]
        if not updates:
            return None
        try:
            with get_db_context() as db:
                msg = db.query(ConversationMessage).filter(
                    ConversationMessage.id == message_id
                ).first()
                if not msg:
                    return None
                for k, v in updates.items():
                    setattr(msg, k, v)
                db.commit()
                db.refresh(msg)
                return msg.to_dict()
        except Exception as e:
            logger.error(f"Failed to update conversation message: {e}", exc_info=True)
            return None


class AgentRunEvent(Base):
    """Agent 执行步骤事件日志。

    按时间顺序记录一次 AgentRun 的关键步骤，便于 UI 展示"思考过程"、
    问题复盘以及审计。该表只追加 (append-only)，由 Worker / Tool 层写入。

    典型 event_type：
        - run_started / run_completed / run_failed / run_cancelled
        - tool_selected          # 工具召回阶段挑出的工具
        - tool_call_started      # 某个工具开始执行
        - tool_call_completed    # 工具执行结束（含输出摘要）
        - tool_call_failed
        - crew_tool_invoked      # Master Agent 调用子 Crew
        - llm_message            # 重要 LLM 输出节点（可选）
        - waiting_input          # 预留：Human-in-the-loop 等待用户输入
        - input_received         # 预留：恢复执行
    """

    __tablename__ = "agent_run_events"

    id = Column(String(36), primary_key=True)
    agent_run_id = Column(String(36), ForeignKey("agent_runs.id"), nullable=False, index=True)
    sequence = Column(Integer, nullable=False, default=0, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    tool_name = Column(String(100), nullable=True, index=True)
    content = Column(Text, nullable=True)
    payload = Column(get_json_type(), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_run_id": self.agent_run_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "content": self.content,
            "payload": self.payload if self.payload is not None else {},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def append(
        agent_run_id: str,
        event_type: str,
        *,
        tool_name: Optional[str] = None,
        content: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """追加一条事件记录，sequence 基于同一 run 下已有最大值 + 1。"""
        try:
            from backend.utils import generate_uuid
            with get_db_context() as db:
                max_seq = (
                    db.query(AgentRunEvent.sequence)
                    .filter(AgentRunEvent.agent_run_id == agent_run_id)
                    .order_by(AgentRunEvent.sequence.desc())
                    .limit(1)
                    .scalar()
                )
                next_seq = int(max_seq or 0) + 1
                event = AgentRunEvent(
                    id=generate_uuid(),
                    agent_run_id=agent_run_id,
                    sequence=next_seq,
                    event_type=event_type,
                    tool_name=tool_name,
                    content=content,
                    payload=payload,
                    started_at=started_at,
                    completed_at=completed_at,
                    created_at=datetime.utcnow(),
                )
                db.add(event)
                db.commit()
                db.refresh(event)
                return event.to_dict()
        except Exception as e:
            logger.error(f"Failed to append agent_run_event: {e}", exc_info=True)
            return None

    @staticmethod
    def list_by_run(
        agent_run_id: str,
        *,
        limit: int = 500,
        offset: int = 0,
        ascending: bool = True,
        after_sequence: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """查询某个 run 的事件流。

        ``after_sequence``：仅返回 ``sequence > after_sequence`` 的事件，便于前端
        增量轮询（拿到上一批最大 sequence 后，下次请求只会收到新增事件）。
        """
        with get_db_context() as db:
            query = db.query(AgentRunEvent).filter(
                AgentRunEvent.agent_run_id == agent_run_id
            )
            if after_sequence is not None:
                query = query.filter(AgentRunEvent.sequence > int(after_sequence))
            if ascending:
                query = query.order_by(AgentRunEvent.sequence.asc())
            else:
                query = query.order_by(AgentRunEvent.sequence.desc())
            rows = query.offset(offset).limit(limit).all()
            return [row.to_dict() for row in rows]

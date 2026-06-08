"""
存储管理器
统一管理文件存储，集成数据库记录
"""
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import mimetypes

from .adapter import StorageAdapter
from .local_storage import LocalStorageAdapter
from .factory import create_storage_adapter
from backend.database import File
from backend.database.db import get_db_context
from backend.core.logger import get_app_logger
from backend.core.config import get_config
from backend.utils import generate_uuid
from backend.core.exceptions import ResourceNotFoundException
from backend.utils.artifact_storage import normalize_storage_for_write, resolve_artifact_public_url

logger = get_app_logger(__name__)


class StorageManager:
    """存储管理器"""
    
    def __init__(self, adapter: Optional[StorageAdapter] = None):
        """
        初始化存储管理器
        
        Args:
            adapter: 存储适配器，如果为None则根据配置创建
        """
        if adapter is None:
            # 与 factory 保持一致：server 等与「本地 outputs」对齐时回落到 LocalStorageAdapter
            storage_mode = str(get_config("storage.default", "server") or "server")
            adapter = create_storage_adapter(storage_mode)
        
        self.adapter = adapter
        logger.info(f"StorageManager initialized with adapter: {type(adapter).__name__}")
    
    async def save_file(
        self,
        file_data: bytes,
        user_id: str,
        category: str,
        filename: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        storage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        保存文件并创建数据库记录
        
        Args:
            file_data: 文件数据（字节）
            user_id: 用户ID
            category: 文件类别（image/video/audio/text/upload）
            filename: 文件名
            task_id: 关联任务ID（可选）
            metadata: 文件元数据（可选）
            storage: 存储位置（可选，如果不指定则从配置获取）
        
        Returns:
            文件信息字典（包含数据库记录）
        """
        # 获取文件大小和MIME类型
        file_size = len(file_data)
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "application/octet-stream"
        
        effective_storage = normalize_storage_for_write(
            storage or str(get_config("storage.default", "server") or "server")
        )
        adapter = (
            create_storage_adapter(effective_storage)
            if storage is not None
            else self.adapter
        )

        storage_info = await adapter.save_file(
            file_data=file_data,
            category=category,
            filename=filename,
            metadata=metadata,
        )

        http_url = resolve_artifact_public_url(effective_storage, storage_info.get("storage_path"))

        file_id = generate_uuid()
        file_dict = File.create(
            id=file_id,
            task_id=task_id,
            user_id=user_id,
            category=category,
            storage=effective_storage,
            storage_path=storage_info["storage_path"],
            file_name=filename,
            file_size=file_size,
            mime_type=mime_type,
            http_url=http_url,
            metadata=metadata,
        )
        
        if not file_dict:
            await adapter.delete_file(storage_info["storage_path"])
            raise Exception("Failed to create file record in database")
        
        logger.info(f"File saved and recorded: {file_id} ({filename})")
        
        return file_dict
    
    async def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件信息字典，如果不存在则返回None
        """
        return File.get_by_id(file_id)
    
    async def get_file_url(self, file_id: str) -> str:
        """
        获取文件访问URL
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件访问URL
        
        Raises:
            ResourceNotFoundException: 文件不存在
        """
        file_dict = File.get_by_id(file_id)
        if not file_dict:
            raise ResourceNotFoundException(file_id, "file")
        
        url = resolve_artifact_public_url(
            file_dict.get("storage"),
            file_dict.get("storage_path"),
        )
        if url:
            return url
        if file_dict.get("http_url"):
            return file_dict["http_url"]
        return await self.adapter.get_file_url(file_dict["storage_path"])
    
    async def delete_file(self, file_id: str) -> bool:
        """
        删除文件（包括数据库记录和实际文件）
        
        Args:
            file_id: 文件ID
        
        Returns:
            是否删除成功
        
        Raises:
            ResourceNotFoundException: 文件不存在
        """
        file_dict = File.get_by_id(file_id)
        if not file_dict:
            raise ResourceNotFoundException(file_id, "file")
        
        # 删除实际文件
        deleted = await self.adapter.delete_file(file_dict["storage_path"])
        
        # 删除数据库记录
        File.delete(file_id)
        
        logger.info(f"File deleted: {file_id}")
        
        return deleted
    
    async def read_file(self, file_id: str) -> bytes:
        """
        读取文件内容
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件数据（字节）
        
        Raises:
            ResourceNotFoundException: 文件不存在
        """
        file_dict = File.get_by_id(file_id)
        if not file_dict:
            raise ResourceNotFoundException(file_id, "file")
        
        return await self.adapter.read_file(file_dict["storage_path"])
    
    async def list_files(
        self,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Dict[str, Any]]:
        """
        列出文件
        
        Args:
            user_id: 用户ID（可选，用于过滤）
            category: 文件类别（可选，用于过滤）
            task_id: 任务ID（可选，用于过滤）
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            文件列表
        """
        if user_id:
            return File.list_by_user(user_id, limit=limit, offset=offset)
        elif task_id:
            return File.list_by_task(task_id, limit=limit, offset=offset)
        elif category:
            return File.list_by_category(category, limit=limit, offset=offset)
        else:
            return File.list_all(limit=limit, offset=offset)
    
    async def file_exists(self, file_id: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件是否存在
        """
        file_dict = File.get_by_id(file_id)
        if not file_dict:
            return False
        
        return await self.adapter.file_exists(file_dict["storage_path"])


# 全局存储管理器实例
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """
    获取全局存储管理器实例（单例模式）
    
    Returns:
        StorageManager实例
    """
    global _storage_manager
    
    if _storage_manager is None:
        _storage_manager = StorageManager()
    
    return _storage_manager


"""
本地存储适配器实现
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .adapter import StorageAdapter
from backend.core.logger import get_app_logger
from backend.core.config import get_config
from backend.utils import generate_uuid

logger = get_app_logger(__name__)


class LocalStorageAdapter(StorageAdapter):
    """本地存储适配器"""
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        http_base_url: Optional[str] = None
    ):
        """
        初始化本地存储适配器
        
        Args:
            base_path: 存储基础路径，如果为None则从配置读取
            http_base_url: HTTP访问基础URL，如果为None则从配置读取
        """
        self.base_path = Path(base_path or get_config("storage.local.base_path", "resources/outputs"))

        # http_base_url 支持两种形式：
        # 1) 相对路径："/outputs"（同源）
        # 2) 绝对地址： "http://192.168.0.105:8888/outputs"（跨机器访问）
        cfg_http_base_url = get_config("storage.local.http_base_url", "/outputs")
        raw_http_base_url = http_base_url if (isinstance(http_base_url, str) and http_base_url.strip()) else cfg_http_base_url
        raw_http_base_url = raw_http_base_url if (isinstance(raw_http_base_url, str) and raw_http_base_url.strip()) else "/outputs"
        raw_http_base_url = raw_http_base_url.strip().rstrip("/")

        # 若 http_base_url 是相对路径，且配置了 server.public_base_url，则拼接成绝对地址
        public_base_url = str(get_config("server.public_base_url", "") or "").strip().rstrip("/")
        if raw_http_base_url.startswith("/") and public_base_url:
            self.http_base_url = f"{public_base_url}{raw_http_base_url}"
        else:
            self.http_base_url = raw_http_base_url
        
        # 确保基础路径存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LocalStorageAdapter initialized: base_path={self.base_path}, http_base_url={self.http_base_url}")
    
    def _get_category_path(self, category: str) -> Path:
        """
        获取类别目录路径
        
        Args:
            category: 文件类别
        
        Returns:
            类别目录路径
        """
        return self.base_path / category
    
    def _get_date_path(self) -> Path:
        """
        获取日期路径（年/月/日）
        
        Returns:
            日期路径
        """
        now = datetime.utcnow()
        return Path(str(now.year)) / f"{now.month:02d}" / f"{now.day:02d}"
    
    def _generate_storage_path(self, category: str, filename: str) -> Path:
        """
        生成存储路径
        
        Args:
            category: 文件类别
            filename: 文件名
        
        Returns:
            存储路径
        """
        # 生成唯一文件名（避免冲突）
        file_ext = Path(filename).suffix
        unique_filename = f"{generate_uuid()}{file_ext}"
        
        # 构建路径：category/年/月/日/filename
        date_path = self._get_date_path()
        storage_path = self._get_category_path(category) / date_path / unique_filename
        
        return storage_path
    
    async def save_file(
        self,
        file_data: bytes,
        category: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        保存文件到本地存储
        
        Args:
            file_data: 文件数据（字节）
            category: 文件类别
            filename: 文件名
            metadata: 文件元数据（可选）
        
        Returns:
            包含文件信息的字典
        """
        # 生成存储路径
        storage_path = self._generate_storage_path(category, filename)
        
        # 确保目录存在
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        try:
            storage_path.write_bytes(file_data)
            file_size = len(file_data)
            
            # 生成HTTP URL
            relative_path = storage_path.relative_to(self.base_path)
            http_url = f"{self.http_base_url}/{relative_path.as_posix()}"
            
            logger.info(f"File saved: {storage_path} ({file_size} bytes)")
            
            return {
                "storage_path": str(relative_path),
                "http_url": http_url,
                "file_size": file_size
            }
        
        except Exception as e:
            logger.error(f"Failed to save file: {e}", exc_info=True)
            raise
    
    async def get_file_url(self, storage_path: str) -> str:
        """
        获取文件访问URL
        
        Args:
            storage_path: 存储路径（相对路径）
        
        Returns:
            文件访问URL
        """
        # storage_path已经是相对路径，直接拼接
        if storage_path.startswith("/"):
            storage_path = storage_path[1:]
        
        return f"{self.http_base_url}/{storage_path}"
    
    async def delete_file(self, storage_path: str) -> bool:
        """
        删除文件
        
        Args:
            storage_path: 存储路径（相对路径）
        
        Returns:
            是否删除成功
        """
        try:
            # 构建完整路径
            full_path = self.base_path / storage_path
            
            if full_path.exists():
                full_path.unlink()
                logger.info(f"File deleted: {full_path}")
                return True
            else:
                logger.warning(f"File not found: {full_path}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to delete file: {e}", exc_info=True)
            return False
    
    async def file_exists(self, storage_path: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            storage_path: 存储路径（相对路径）
        
        Returns:
            文件是否存在
        """
        full_path = self.base_path / storage_path
        return full_path.exists()
    
    async def get_file_size(self, storage_path: str) -> int:
        """
        获取文件大小
        
        Args:
            storage_path: 存储路径（相对路径）
        
        Returns:
            文件大小（字节）
        """
        full_path = self.base_path / storage_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {storage_path}")
        
        return full_path.stat().st_size
    
    async def read_file(self, storage_path: str) -> bytes:
        """
        读取文件内容
        
        Args:
            storage_path: 存储路径（相对路径）
        
        Returns:
            文件数据（字节）
        """
        full_path = self.base_path / storage_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {storage_path}")
        
        return full_path.read_bytes()


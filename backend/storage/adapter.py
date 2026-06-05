"""
存储适配器抽象基类
定义统一的存储接口
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path


class StorageAdapter(ABC):
    """存储适配器抽象基类"""
    
    @abstractmethod
    async def save_file(
        self,
        file_data: bytes,
        category: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        保存文件
        
        Args:
            file_data: 文件数据（字节）
            category: 文件类别（image/video/audio/text/upload）
            filename: 文件名
            metadata: 文件元数据（可选）
        
        Returns:
            包含文件信息的字典：
            {
                "storage_path": "相对路径或云存储key",
                "http_url": "HTTP访问URL",
                "file_size": 文件大小（字节）
            }
        """
        pass
    
    @abstractmethod
    async def get_file_url(self, storage_path: str) -> str:
        """
        获取文件访问URL
        
        Args:
            storage_path: 存储路径（相对路径或云存储key）
        
        Returns:
            文件访问URL
        """
        pass
    
    @abstractmethod
    async def delete_file(self, storage_path: str) -> bool:
        """
        删除文件
        
        Args:
            storage_path: 存储路径（相对路径或云存储key）
        
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    async def file_exists(self, storage_path: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            storage_path: 存储路径（相对路径或云存储key）
        
        Returns:
            文件是否存在
        """
        pass
    
    @abstractmethod
    async def get_file_size(self, storage_path: str) -> int:
        """
        获取文件大小
        
        Args:
            storage_path: 存储路径（相对路径或云存储key）
        
        Returns:
            文件大小（字节）
        """
        pass
    
    @abstractmethod
    async def read_file(self, storage_path: str) -> bytes:
        """
        读取文件内容
        
        Args:
            storage_path: 存储路径（相对路径或云存储key）
        
        Returns:
            文件数据（字节）
        """
        pass


"""
文件存储模块
"""
from .adapter import StorageAdapter
from .local_storage import LocalStorageAdapter
from .manager import StorageManager, get_storage_manager
from .factory import create_storage_adapter, get_storage_manager_by_mode

__all__ = [
    "StorageAdapter",
    "LocalStorageAdapter",
    "StorageManager",
    "get_storage_manager",
    "create_storage_adapter",
    "get_storage_manager_by_mode",
]


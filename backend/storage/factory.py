"""
存储适配器工厂
根据 storage_mode 创建对应的存储适配器（Backend 写入侧不认 local，会规范为 server）
"""
from typing import TYPE_CHECKING

from .adapter import StorageAdapter
from .local_storage import LocalStorageAdapter
from backend.core.logger import get_app_logger
from backend.core.config import get_config
from backend.utils.artifact_storage import normalize_storage_for_write

if TYPE_CHECKING:
    from .manager import StorageManager

logger = get_app_logger(__name__)


def create_storage_adapter(storage_mode: str) -> StorageAdapter:
    """
    根据存储类型创建存储适配器。

    Args:
        storage_mode: server | s3 | oss（local 会自动转为 server）

    Returns:
        存储适配器实例
    """
    mode = normalize_storage_for_write(storage_mode)

    if mode == "server":
        base_path = get_config("storage.local.base_path", "resources/outputs")
        http_base_url = get_config("storage.local.http_base_url", "/outputs")
        return LocalStorageAdapter(base_path=base_path, http_base_url=http_base_url)

    if mode == "s3":
        from .object_storage import S3StorageAdapter

        return S3StorageAdapter()

    if mode == "oss":
        from .object_storage import OSSStorageAdapter

        return OSSStorageAdapter()

    raise ValueError(f"Unsupported storage type: {storage_mode}")


def get_storage_manager_by_mode(storage_mode: str) -> "StorageManager":
    """根据存储类型创建存储管理器。"""
    from .manager import StorageManager

    adapter = create_storage_adapter(storage_mode)
    return StorageManager(adapter=adapter)

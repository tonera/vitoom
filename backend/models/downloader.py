"""
模型下载器
处理模型下载、断点续传、完整性校验等
"""
import asyncio
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import hashlib

from backend.core.logger import get_app_logger
from backend.core.config import get_config
from backend.utils.http_utils import HTTPClient

logger = get_app_logger(__name__)


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self):
        """初始化模型下载器"""
        self.models_base_path = Path(get_config("models.storage_path", "resources/models"))
        self.models_base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelDownloader initialized: base_path={self.models_base_path}")
    
    async def download_model(
        self,
        model_key: str,
        source: Dict[str, Any],
        modality: str,
        load_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        下载模型文件
        
        Args:
            model_key: 模型稳定键
            source: 模型来源信息，下载直链使用 source 中的下载地址
            modality: 任务领域
            load_name: 推理加载名
            progress_callback: 进度回调函数 (downloaded, total)
            expected_size: 预期文件大小（字节）
            expected_hash: 预期文件哈希值（SHA256）
        
        Returns:
            下载结果字典
        """
        source_url = str((source or {}).get("download_url") or "").strip()
        if not source_url:
            raise ValueError("source.download_url is required")
        # 构建保存路径
        save_dir = self.models_base_path / str(modality or "model").strip() / str(load_name or model_key).strip()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 从URL提取文件名
        filename = Path(source_url).name or "model.bin"
        save_path = save_dir / filename
        
        try:
            # 检查是否已存在文件（支持断点续传）
            resume_pos = 0
            if save_path.exists():
                resume_pos = save_path.stat().st_size
                if expected_size and resume_pos >= expected_size:
                    logger.info(f"File already exists and complete: {save_path}")
                    return {
                        "success": True,
                        "path": str(save_path),
                        "size": resume_pos
                    }
                logger.info(f"Resuming download from position: {resume_pos}")
            
            # 下载文件
            headers = {}
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
            
            async with HTTPClient() as client:
                response = await client.get(source_url, headers=headers, follow_redirects=True)
                
                if response.status_code == 206:  # Partial Content
                    content_range = response.headers.get("Content-Range", "")
                    if "/" in content_range:
                        total_size = int(content_range.split("/")[-1])
                    else:
                        total_size = expected_size or 0
                elif response.status_code == 200:
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length else (expected_size or 0)
                else:
                    response.raise_for_status()
                    total_size = expected_size or 0
                
                # 打开文件（追加模式以支持断点续传）
                mode = "ab" if resume_pos > 0 else "wb"
                with open(save_path, mode) as f:
                    downloaded = resume_pos
                    
                    # 使用iter_bytes进行流式下载
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 调用进度回调
                        if progress_callback:
                            progress_callback(downloaded, total_size or downloaded)
            
            # 验证文件大小
            actual_size = save_path.stat().st_size
            if expected_size and actual_size != expected_size:
                raise ValueError(f"File size mismatch: expected {expected_size}, got {actual_size}")
            
            # 验证文件哈希（如果提供）
            if expected_hash:
                actual_hash = self._calculate_file_hash(save_path)
                if actual_hash != expected_hash:
                    raise ValueError(f"File hash mismatch: expected {expected_hash}, got {actual_hash}")
            
            logger.info(f"Model downloaded successfully: {save_path} ({actual_size} bytes)")
            
            return {
                "success": True,
                "path": str(save_path),
                "size": actual_size
            }
        
        except Exception as e:
            logger.error(f"Failed to download model: {e}", exc_info=True)
            # 如果下载失败，删除不完整的文件
            if save_path.exists() and resume_pos == 0:
                save_path.unlink()
            raise
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        计算文件哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法（sha256, md5等）
        
        Returns:
            文件哈希值（十六进制字符串）
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def check_disk_space(self, required_size: int) -> bool:
        """
        检查磁盘空间是否足够
        
        Args:
            required_size: 需要的空间大小（字节）
        
        Returns:
            是否有足够空间
        """
        import shutil
        
        try:
            stat = shutil.disk_usage(self.models_base_path)
            available = stat.free
            return available >= required_size
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}", exc_info=True)
            return False


# 全局模型下载器实例
_model_downloader: Optional[ModelDownloader] = None


def get_model_downloader() -> ModelDownloader:
    """
    获取全局模型下载器实例（单例模式）
    
    Returns:
        ModelDownloader实例
    """
    global _model_downloader
    
    if _model_downloader is None:
        _model_downloader = ModelDownloader()
    
    return _model_downloader


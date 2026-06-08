"""
消息缓存模块
负责将消息写入缓存目录和从缓存目录读取消息
"""
import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from .logger import get_logger

logger = get_logger(__name__)


class MessageCache:
    """消息缓存类"""
    
    def __init__(self, cache_dir: str = "resources/cache/messages"):
        """
        初始化消息缓存
        
        Args:
            cache_dir: 缓存目录路径（相对于项目根目录）
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Message cache directory: {self.cache_dir}")
    
    async def save_message(self, task_id: str, message: Dict[str, Any]) -> Optional[Path]:
        """
        异步保存消息到缓存文件
        
        Args:
            task_id: 任务ID
            message: 消息字典（全量任务数据）
        
        Returns:
            保存的文件路径，如果失败则返回None
        
        注意：如果已存在该task_id的缓存文件，会先删除旧文件（避免重复）
        """
        try:
            # 先检查并删除已存在的该task_id的缓存文件（避免重复）
            existing_file = self.get_cache_file_by_task_id(task_id)
            if existing_file:
                await self.delete_message(existing_file)
                logger.debug(f"Deleted existing cache file for task {task_id}: {existing_file}")
            
            # 文件名格式：task_{task_id}_{unix_timestamp}.json
            # 添加"task_"前缀便于区分，时间戳用于排序和去重
            unix_timestamp = int(datetime.now().timestamp())
            filename = f"task_{task_id}_{unix_timestamp}.json"
            file_path = self.cache_dir / filename
            
            # 异步写入文件
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(message, ensure_ascii=False, indent=2))
            
            logger.debug(f"Message cached: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save message cache: {e}", exc_info=True)
            return None
    
    async def load_message(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        异步从缓存文件加载消息
        
        Args:
            file_path: 文件路径
        
        Returns:
            消息字典，如果失败则返回None
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load message cache from {file_path}: {e}", exc_info=True)
            return None
    
    async def delete_message(self, file_path: Path) -> bool:
        """
        异步删除缓存文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否成功删除
        """
        try:
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)
                logger.debug(f"Message cache deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete message cache {file_path}: {e}", exc_info=True)
            return False
    
    async def save_status_result(self, task_id: str, status: str, result_data: Dict[str, Any]) -> Optional[Path]:
        """
        保存状态结果到缓存文件（用于WS断开时的状态同步）
        
        Args:
            task_id: 任务ID
            status: 状态（completed/failed/cancelled）
            result_data: 结果数据字典
        
        Returns:
            保存的文件路径，如果失败则返回None
        
        注意：添加时间戳避免同一task_id多次状态更新时相互覆盖
        """
        try:
            # 文件名格式：res_{task_id}_{status}_{unix_timestamp}.json
            # 添加时间戳避免同一task_id多次状态更新时相互覆盖
            # 注意：秒级时间戳在同一秒内多次写入会产生同名覆盖（看起来像“写了两次相同文件”）
            # 这里改为毫秒，保证同秒内多次保存也不会冲突。
            unix_timestamp = int(datetime.now().timestamp() * 1000)
            filename = f"res_{task_id}_{status}_{unix_timestamp}.json"
            file_path = self.cache_dir / filename
            
            message = {
                "type": "status_result",
                "task_id": task_id,
                "status": status,
                "result": result_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 异步写入文件
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(message, ensure_ascii=False, indent=2))
            
            logger.info(f"Status result cached: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save status result cache: {e}", exc_info=True)
            return None
    
    async def scan_cache_files(self) -> list[Path]:
        """
        扫描缓存目录中的所有消息文件
        
        Returns:
            文件路径列表
        """
        try:
            files = []
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file() and file_path.suffix == '.json':
                    files.append(file_path)
            return sorted(files)  # 按文件名排序（包含时间戳）
        except Exception as e:
            logger.error(f"Failed to scan cache files: {e}", exc_info=True)
            return []
    
    def get_cache_file_by_task_id(self, task_id: str) -> Optional[Path]:
        """
        根据任务ID查找缓存文件（同步方法，用于快速查找）
        
        Args:
            task_id: 任务ID
        
        Returns:
            文件路径，如果不存在则返回None（返回最新的文件，如果有多个）
        """
        try:
            matching_files = []
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file():
                    # 匹配格式：task_{task_id}_{timestamp}.json
                    if file_path.name.startswith(f"task_{task_id}_") and file_path.suffix == '.json':
                        matching_files.append(file_path)
            
            if not matching_files:
                return None
            
            # 如果有多个文件，返回最新的（按文件名中的时间戳排序）
            if len(matching_files) > 1:
                matching_files.sort(key=lambda p: p.name, reverse=True)
                logger.warning(f"Found {len(matching_files)} cache files for task {task_id}, using latest: {matching_files[0].name}")
            
            return matching_files[0]
        except Exception as e:
            logger.error(f"Failed to find cache file for task {task_id}: {e}", exc_info=True)
            return None


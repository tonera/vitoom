"""
任务工作线程
提供任务处理的基础框架
"""
from typing import Dict, Any, Callable, Awaitable, Optional
from backend.core.logger import get_task_logger

task_logger = get_task_logger(__name__)


class TaskWorker:
    """任务工作线程基类"""
    
    def __init__(self, task_type: str):
        """
        初始化工作线程
        
        Args:
            task_type: 任务类型
        """
        self.task_type = task_type
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理任务（子类应实现此方法）
        
        Args:
            task: 任务字典
        
        Returns:
            结果字典，应包含 "result" 字段
        """
        raise NotImplementedError("Subclass must implement process method")
    
    async def update_progress(
        self,
        task_id: str,
        progress: int,
        message: Optional[str] = None
    ):
        """
        更新任务进度（便捷方法）
        
        Args:
            task_id: 任务ID
            progress: 进度（0-100）
            message: 进度消息（可选）
        """
        from backend.queue.queue import get_task_queue
        queue = get_task_queue()
        await queue.update_progress(task_id, progress, message)


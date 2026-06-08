"""
任务队列核心类
实现异步任务队列、优先级队列、任务状态管理
"""
import asyncio
import heapq
from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from enum import IntEnum

from backend.database import Task
from backend.database.db import get_db_context
from backend.core.logger import get_task_logger, get_app_logger
from backend.core.exceptions import TaskNotFoundException, TaskQueueFullException
from backend.utils import generate_uuid, utc_now
from backend.core.config import get_config

logger = get_app_logger(__name__)
task_logger = get_task_logger(__name__)


class TaskStatus(IntEnum):
    """任务状态枚举"""
    PENDING = 0
    QUEUED = 1
    PROCESSING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5


# 状态字符串映射
STATUS_MAP = {
    "pending": TaskStatus.PENDING,
    "queued": TaskStatus.QUEUED,
    "processing": TaskStatus.PROCESSING,
    "completed": TaskStatus.COMPLETED,
    "failed": TaskStatus.FAILED,
    "cancelled": TaskStatus.CANCELLED,
}


@dataclass
class QueueItem:
    """队列项（用于优先级队列）"""
    priority: int = field(compare=False)
    created_at: datetime = field(compare=False)
    task_id: str = field(compare=False)
    
    def __lt__(self, other):
        """优先级比较（数字越大优先级越高，相同优先级按创建时间）"""
        if self.priority != other.priority:
            return self.priority > other.priority  # 优先级高的先出队
        return self.created_at < other.created_at  # 相同优先级，早创建的先出队


class TaskQueue:
    """任务队列类"""
    
    def __init__(
        self,
        max_size: Optional[int] = None,
        max_workers: int = 2,
        default_timeout: Optional[int] = None
    ):
        """
        初始化任务队列
        
        Args:
            max_size: 队列最大大小，如果为None则不限制
            max_workers: 最大并发工作线程数
            default_timeout: 默认任务超时时间（秒），如果为None则使用配置值
        """
        self.max_size = max_size or get_config("tasks.queue.max_size", 1000)
        self.max_workers = max_workers or get_config("tasks.queue.max_workers", 2)
        self.default_timeout = default_timeout or get_config("tasks.queue.timeout", 3600)
        
        # 优先级队列（使用heapq实现）
        self._queue: list[QueueItem] = []
        self._queue_lock = asyncio.Lock()
        
        # 正在处理的任务
        self._processing: Dict[str, asyncio.Task] = {}
        self._processing_lock = asyncio.Lock()
        
        # 任务处理器注册表
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        
        # 工作线程池
        self._workers: list[asyncio.Task] = []
        self._running = False
        
        logger.info(f"TaskQueue initialized: max_size={self.max_size}, max_workers={self.max_workers}")

    @property
    def is_running(self) -> bool:
        """当前工作线程是否已启动。"""
        return self._running
    
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ):
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型（如 "image", "video", "audio", "text"）
            handler: 异步处理函数，接收任务字典，返回结果字典
        """
        self._handlers[task_type] = handler
        logger.info(f"Task handler registered for type: {task_type}")
    
    async def add_task(
        self,
        task_id: str,
        user_id: str,
        task_type: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        model_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        添加任务到队列
        
        Args:
            task_id: 任务ID
            user_id: 用户ID
            task_type: 任务类型
            prompt: 提示词
            params: 任务参数
            priority: 优先级（1-10，数字越大优先级越高）
            model_key: 模型稳定键
        
        Returns:
            任务字典
        
        Raises:
            TaskQueueFullException: 队列已满
        """
        # 检查队列大小
        async with self._queue_lock:
            if len(self._queue) >= self.max_size:
                raise TaskQueueFullException(self.max_size)
        
        # 创建任务记录（数据库）
        task_dict = Task.create(
            id=task_id,
            user_id=user_id,
            task_type=task_type,
            prompt=prompt,
            params=params,
            priority=priority,
            model_key=model_key,
            status="pending"
        )
        
        if not task_dict:
            raise Exception("Failed to create task in database")
        
        # 添加到队列
        async with self._queue_lock:
            queue_item = QueueItem(
                priority=priority,
                created_at=datetime.utcnow(),
                task_id=task_id
            )
            heapq.heappush(self._queue, queue_item)
            
            # 更新任务状态为queued
            Task.update(task_id, status="queued")
        
        task_logger.info(f"Task added to queue: {task_id} (type: {task_type}, priority: {priority})")
        
        return task_dict
    
    async def get_next_task(self) -> Optional[str]:
        """
        获取下一个要处理的任务ID
        
        Returns:
            任务ID，如果队列为空则返回None
        """
        async with self._queue_lock:
            if not self._queue:
                return None
            
            queue_item = heapq.heappop(self._queue)
            return queue_item.task_id
    
    async def start_workers(self):
        """启动工作线程"""
        if self._running:
            logger.warning("Workers already running")
            return
        
        self._running = True
        
        # 启动工作线程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i+1}"))
            self._workers.append(worker)
        
        logger.info(f"Started {self.max_workers} workers")
    
    async def stop_workers(self):
        """停止工作线程"""
        if not self._running:
            return
        
        self._running = False
        
        # 等待所有工作线程完成
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
        
        logger.info("Workers stopped")
    
    async def _worker_loop(self, worker_name: str):
        """工作线程循环"""
        task_logger.info(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # 获取下一个任务
                task_id = await self.get_next_task()
                
                if not task_id:
                    # 队列为空，等待一段时间
                    await asyncio.sleep(0.1)
                    continue
                
                # 检查是否已有工作线程在处理
                async with self._processing_lock:
                    if task_id in self._processing:
                        # 任务已在处理中，重新放回队列
                        task_dict = Task.get_by_id(task_id)
                        if task_dict:
                            queue_item = QueueItem(
                                priority=task_dict.get("priority", 5),
                                created_at=datetime.fromisoformat(task_dict["created_at"]),
                                task_id=task_id
                            )
                            heapq.heappush(self._queue, queue_item)
                        continue
                    
                    # 创建处理任务
                    process_task = asyncio.create_task(self._process_task(task_id, worker_name))
                    self._processing[task_id] = process_task
                
                # 等待任务完成（但不阻塞其他任务）
                await process_task
                
                # 从处理中移除
                async with self._processing_lock:
                    self._processing.pop(task_id, None)
            
            except Exception as e:
                task_logger.error(f"Worker {worker_name} error: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        task_logger.info(f"Worker {worker_name} stopped")
    
    async def _process_task(self, task_id: str, worker_name: str):
        """处理单个任务"""
        try:
            # 获取任务信息
            task_dict = Task.get_by_id(task_id)
            if not task_dict:
                task_logger.error(f"Task not found: {task_id}")
                return
            
            task_type = task_dict["type"]
            
            # 检查是否有处理器
            if task_type not in self._handlers:
                task_logger.error(f"No handler registered for task type: {task_type}")
                Task.update(task_id, status="failed", error=f"No handler for type: {task_type}")
                return
            
            # 更新状态为processing
            Task.update(
                task_id,
                status="processing",
                started_at=datetime.utcnow()
            )
            
            task_logger.info(f"Worker {worker_name} processing task: {task_id} (type: {task_type})")
            
            # 调用处理器
            handler = self._handlers[task_type]
            
            # 设置超时
            try:
                result = await asyncio.wait_for(
                    handler(task_dict),
                    timeout=self.default_timeout
                )
                
                # 更新任务为完成
                Task.update(
                    task_id,
                    status="completed",
                    progress=100,
                    completed_at=datetime.utcnow()
                )
                
                task_logger.info(f"Task completed: {task_id}")
                
                # 推送WebSocket消息
                try:
                    from backend.websocket import get_websocket_manager
                    ws_manager = get_websocket_manager()
                    await ws_manager.send_task_update(
                        task_id,
                        status="completed",
                        message="Task completed successfully"
                    )
                except Exception as e:
                    task_logger.debug(f"Failed to send WebSocket update: {e}")
                
                # 推送WebSocket任务完成通知
                try:
                    from backend.websocket.manager import get_websocket_manager
                    manager = get_websocket_manager()
                    await manager.send_task_update(
                        task_id,
                        status="completed",
                        message="Task completed"
                    )
                except Exception as e:
                    task_logger.debug(f"Failed to send WebSocket completion: {e}")
            
            except asyncio.TimeoutError:
                error_msg = f"Task timeout after {self.default_timeout}s"
                Task.update(
                    task_id,
                    status="failed",
                    error=error_msg
                )
                task_logger.error(f"Task timeout: {task_id}")
                
                # 推送WebSocket任务失败通知
                try:
                    from backend.websocket.manager import get_websocket_manager
                    manager = get_websocket_manager()
                    await manager.send_task_update(
                        task_id,
                        status="failed",
                        error=error_msg
                    )
                except Exception as e:
                    task_logger.debug(f"Failed to send WebSocket failure: {e}")
            
            except Exception as e:
                error_msg = str(e)
                Task.update(
                    task_id,
                    status="failed",
                    error=error_msg,
                    progress=task_dict.get("progress", 0)
                )
                task_logger.error(f"Task failed: {task_id}, error: {e}", exc_info=True)
                
                # 推送WebSocket任务失败通知
                try:
                    from backend.websocket.manager import get_websocket_manager
                    manager = get_websocket_manager()
                    await manager.send_task_update(
                        task_id,
                        status="failed",
                        error=error_msg
                    )
                except Exception as e2:
                    task_logger.debug(f"Failed to send WebSocket failure: {e2}")
        
        except Exception as e:
            error_msg = str(e)
            task_logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            Task.update(task_id, status="failed", error=error_msg)
            
            # 推送WebSocket消息
            try:
                from backend.websocket import get_websocket_manager
                ws_manager = get_websocket_manager()
                await ws_manager.send_task_update(
                    task_id,
                    status="failed",
                    message="Task processing error",
                    error=error_msg
                )
            except Exception as e:
                task_logger.debug(f"Failed to send WebSocket update: {e}")
    
    async def update_progress(self, task_id: str, progress: int, message: Optional[str] = None):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度（0-100）
            message: 进度消息（可选）
        """
        updates = {"progress": max(0, min(100, progress))}
        Task.update(task_id, **updates)
        
        if message:
            task_logger.info(f"Task {task_id} progress: {progress}% - {message}")
        
        # 推送WebSocket消息
        try:
            from backend.websocket import get_websocket_manager
            ws_manager = get_websocket_manager()
            await ws_manager.send_progress(task_id, progress, message)
        except Exception as e:
            # WebSocket推送失败不影响任务执行
            task_logger.debug(f"Failed to send WebSocket progress: {e}")
        
        # 推送WebSocket进度更新
        try:
            from backend.websocket.manager import get_websocket_manager
            manager = get_websocket_manager()
            await manager.send_progress(task_id, progress, message)
        except Exception as e:
            # WebSocket推送失败不影响任务处理
            task_logger.debug(f"Failed to send WebSocket progress: {e}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否成功取消
        """
        # 检查任务是否存在
        task_dict = Task.get_by_id(task_id)
        if not task_dict:
            raise TaskNotFoundException(task_id)
        
        status = task_dict["status"]
        
        # 如果任务已完成或失败，无法取消
        if status in ["completed", "failed", "cancelled"]:
            return False
        
        # 如果任务正在处理中，尝试取消
        async with self._processing_lock:
            if task_id in self._processing:
                process_task = self._processing[task_id]
                process_task.cancel()
                self._processing.pop(task_id, None)
        
        # 从队列中移除（如果还在队列中）
        async with self._queue_lock:
            # 重建队列，排除要取消的任务
            new_queue = []
            for item in self._queue:
                if item.task_id != task_id:
                    new_queue.append(item)
            self._queue = new_queue
            heapq.heapify(self._queue)
        
        # 更新任务状态
        Task.update(task_id, status="cancelled")
        
        task_logger.info(f"Task cancelled: {task_id}")
        return True
    
    async def get_queue_size(self) -> int:
        """获取队列大小"""
        async with self._queue_lock:
            return len(self._queue)
    
    async def get_processing_count(self) -> int:
        """获取正在处理的任务数"""
        async with self._processing_lock:
            return len(self._processing)
    
    async def recover_tasks(self, task_types: Optional[list[str]] = None):
        """
        恢复未完成的任务（应用重启时调用）
        
        将数据库中状态为pending、queued或processing的任务重新加入队列
        """
        with get_db_context() as db:
            from backend.database.models import Task as TaskModel
            query = db.query(TaskModel).filter(
                TaskModel.status.in_(["pending", "queued", "processing"])
            )
            if task_types:
                query = query.filter(TaskModel.type.in_(task_types))
            pending_tasks = query.all()
            
            recovered_count = 0
            
            async with self._queue_lock:
                for task in pending_tasks:
                    # 在session内访问属性
                    task_status = task.status
                    task_id = task.id
                    task_priority = task.priority
                    task_created_at = task.created_at
                    
                    # 将processing状态的任务重置为queued
                    if task_status == "processing":
                        Task.update(task_id, status="queued")
                    
                    queue_item = QueueItem(
                        priority=task_priority,
                        created_at=task_created_at,
                        task_id=task_id
                    )
                    heapq.heappush(self._queue, queue_item)
                    recovered_count += 1
        
        logger.info(f"Recovered {recovered_count} tasks from database")
        return recovered_count


# 全局任务队列实例
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """
    获取全局任务队列实例（单例模式）
    
    Returns:
        TaskQueue实例
    """
    global _task_queue
    
    if _task_queue is None:
        _task_queue = TaskQueue()
    
    return _task_queue


"""
任务处理模块
从消息队列读取任务（全量数据），验证task_type，更新状态，格式化参数
不再查询数据库，所有数据来自消息队列
"""
import sys
import asyncio
from pathlib import Path
from threading import Event
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from .logger import get_logger
from .message_queue import MessageQueue
from .message_cache import MessageCache
from .task_cancel import TaskCancellationRegistry, TaskCancelledError
from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskEgress(Protocol):
    def is_connected(self) -> bool: ...

    async def send_task_status(self, task_id: str, status: str, error: Optional[str] = None, **kwargs) -> bool: ...

# 导入InferenceRequestParams
sys.path.insert(0, str(Path(__file__).parent.parent))
from schemas import InferenceRequestParams

logger = get_logger(__name__)


class TaskProcessor:
    """任务处理类"""
    
    def __init__(
        self,
        message_queue: MessageQueue,
        message_cache: MessageCache,
        ws_client: TaskEgress,
        service_type: str,
        inference_callback: Callable[[InferenceRequestParams], Any]
    ):
        """
        初始化任务处理器
        
        Args:
            message_queue: 消息队列
            message_cache: 消息缓存
            ws_client: WebSocket客户端（用于发送状态更新）
            service_type: 服务类型（image/video/audio/text），用于验证任务类型
            inference_callback: 推理回调函数，接收InferenceRequestParams，执行实际推理
        """
        self.message_queue = message_queue
        self.message_cache = message_cache
        self.ws_client = ws_client
        self.service_type = service_type
        self.inference_callback = inference_callback
        self._current_task_id: Optional[str] = None
        self._cancellation = TaskCancellationRegistry()
        self._running = False
        self._cache_scan_interval = 30  # 30秒扫描一次缓存目录
        self._last_cache_scan = datetime.now()
    
    def set_current_task_id(self, task_id: str):
        """设置当前正在处理的任务ID"""
        self._current_task_id = task_id
    
    def clear_current_task_id(self):
        """清除当前任务ID"""
        self._current_task_id = None

    def get_current_task_id(self) -> Optional[str]:
        """获取当前正在处理的任务ID"""
        return self._current_task_id
    
    def is_task_cancelled(self, task_id: str) -> bool:
        """检查任务是否已取消"""
        return self._cancellation.is_cancelled(task_id)

    def get_task_cancel_event(self, task_id: str) -> Event:
        """获取可供阻塞线程轮询的取消事件。"""
        return self._cancellation.get_event(task_id)
    
    def mark_task_cancelled(self, task_id: str):
        """标记任务为已取消"""
        self._cancellation.mark_cancelled(task_id)
    
    def clear_cancelled_task(self, task_id: str):
        """清除任务的取消标记"""
        self._cancellation.clear(task_id)

    async def handle_cancel_message(self, task_id: str) -> None:
        """旁路处理取消消息，避免依赖串行队列消费。"""
        normalized = str(task_id or "").strip()
        if not normalized:
            return

        self.mark_task_cancelled(normalized)
        logger.info(f"Task {normalized} marked as cancelled")

        cache_file = self.message_cache.get_cache_file_by_task_id(normalized)
        if cache_file:
            await self.message_cache.delete_message(cache_file)
            logger.info(f"Deleted cache file for cancelled task: {normalized}")

        if self._current_task_id == normalized:
            logger.warning(f"Current task {normalized} is being cancelled")
    
    async def process_message(self, message: Dict[str, Any]):
        """
        处理消息
        
        Args:
            message: 消息字典，包含type、task_id、task_data等字段
        """
        message_type = message.get("type")
        task_id = message.get("task_id")
        
        if message_type == "cancel":
            await self.handle_cancel_message(str(task_id or "").strip())
        
        elif message_type == "task":
            # 处理任务消息（全量数据）
            task_data = message.get("task_data")
            if not task_data:
                logger.error(f"Task message missing task_data: {task_id}")
                return
            
            logger.info("="*60)
            logger.info(f"✓✓✓ PROCESSING TASK MESSAGE ✓✓✓")
            logger.info(f"  task_id: {task_id}")
            logger.info("="*60)
            
            # 立即删除缓存文件（避免重复处理）
            cache_file = self.message_cache.get_cache_file_by_task_id(task_id)
            if cache_file:
                await self.message_cache.delete_message(cache_file)
            
            await self._process_task_message(task_id, task_data)
    
    async def _process_task_message(self, task_id: str, task_data: Dict[str, Any]):
        """
        处理任务消息（使用全量数据，不再查询数据库）
        
        Args:
            task_id: 任务ID
            task_data: 全量任务数据字典
        """
        logger.info(f"收到任务 {task_id} 开始处理")
        print(task_data)
        
        # 1. 验证task_type是否匹配
        task_type = task_data.get("type")
        if task_type != self.service_type:
            logger.warning(
                f"Task type mismatch: task_type={task_type}, "
                f"service_type={self.service_type}, task_id={task_id}"
            )
            # 通过WS发送状态更新
            await self.ws_client.send_task_status(
                task_id=task_id,
                status="failed",
                error=f"Task type mismatch: expected {self.service_type}, got {task_type}"
            )
            return
        
        # 2. 检查任务是否已取消
        if self.is_task_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled, skipping")
            await self.ws_client.send_task_status(
                task_id=task_id,
                status="cancelled"
            )
            self.clear_cancelled_task(task_id)
            return
        
        # 3. 更新任务状态为processing（通过WS发送）
        started_at = datetime.utcnow().isoformat()
        await self.ws_client.send_task_status(
            task_id=task_id,
            status="processing",
            started_at=started_at
        )
        
        # 4. 格式化InferenceRequestParams
        try:
            # 直接输出进入 from_task_dict 前的 tpl_list 快照（最终定位点）
            try:
                params = task_data.get("params") if isinstance(task_data, dict) else None
                tpl = params.get("tpl_list") if isinstance(params, dict) else None
                logger.info(f"[TASK_PROCESSOR_BEFORE_PARSE] task_id={task_id} tpl_list={tpl!r}")
            except Exception:
                logger.info(f"[TASK_PROCESSOR_BEFORE_PARSE] task_id={task_id} tpl_list=<unavailable>")

            inference_params = InferenceRequestParams.from_task_dict(task_data)
        except Exception as e:
            logger.error(f"Failed to format InferenceRequestParams for task {task_id}: {e}", exc_info=True)
            await self.ws_client.send_task_status(
                task_id=task_id,
                status="failed",
                error=f"Failed to format inference parameters: {str(e)}"
            )
            return
        
        # 5. 设置当前任务ID
        self.set_current_task_id(task_id)
        
        # 6. 调用推理回调函数
        try:
            logger.info(f"Starting inference for task {task_id}")
            await self.inference_callback(inference_params)
        except TaskCancelledError as exc:
            logger.info(f"Task {task_id} cancelled during inference: {exc}")
            if self.is_task_cancelled(task_id):
                await self.ws_client.send_task_status(
                    task_id=task_id,
                    status="cancelled",
                )
        except Exception as e:
            logger.error(f"Error in inference callback for task {task_id}: {e}", exc_info=True)
            # 如果任务没有被取消，更新为failed
            if not self.is_task_cancelled(task_id):
                await self.ws_client.send_task_status(
                    task_id=task_id,
                    status="failed",
                    error=str(e)
                )
            # OOM 等异常的 traceback 可能持有巨型中间 tensor 的栈帧引用，
            # 必须主动清掉以便 gc 回收显存。
            try:
                e.__traceback__ = None
            except Exception:
                pass
            del e
        finally:
            # 清除当前任务ID
            self.clear_current_task_id()
            # 清除取消标记（如果存在）
            self.clear_cancelled_task(task_id)
    
    async def _process_cache_file(self, file_path: Path):
        """
        处理缓存文件
        
        Args:
            file_path: 缓存文件路径
        """
        filename = file_path.name
        
        # 检查是否是状态结果文件（res_开头）
        if filename.startswith("res_"):
            # 这是状态结果文件，直接推送到WS Server
            message = await self.message_cache.load_message(file_path)
            if message:
                task_id = message.get("task_id")
                status = message.get("status")
                result = message.get("result", {}) or {}
                
                logger.info(f"Processing status result file: {filename}, task_id={task_id}, status={status}")
                
                # 尝试通过WS发送状态更新
                if self.ws_client and self.ws_client.is_connected():
                    # 兼容历史缓存格式：result 内可能包含 task_id/status/type/timestamp 等字段，
                    # 直接 **result 会导致重复关键字参数报错。
                    extra = dict(result) if isinstance(result, dict) else {}
                    task_id = extra.pop("task_id", task_id)
                    status = extra.pop("status", status)
                    # send_task_status 会自行加 type/timestamp，这里避免用户态字段冲突
                    extra.pop("type", None)
                    extra.pop("timestamp", None)

                    if not task_id or not status:
                        # 文件内容不完整，直接删除避免阻塞扫描
                        await self.message_cache.delete_message(file_path)
                        logger.warning(f"Invalid status result cache (missing task_id/status), deleted: {filename}")
                        return

                    success = await self.ws_client.send_task_status(task_id=task_id, status=status, **extra)
                    if success:
                        # 发送成功，删除文件
                        await self.message_cache.delete_message(file_path)
                        logger.info(f"Status result sent and file deleted: {filename}")
                    else:
                        logger.warning(f"Failed to send status result, keeping file: {filename}")
                else:
                    logger.warning(f"WS not connected, keeping status result file: {filename}")
            else:
                # 文件损坏，删除
                await self.message_cache.delete_message(file_path)
                logger.warning(f"Failed to load status result file, deleted: {filename}")
        elif filename.startswith("task_"):
            # 这是普通任务消息文件（格式：task_{task_id}_{timestamp}.json）
            message = await self.message_cache.load_message(file_path)
            if message:
                task_id = message.get("task_id")
                task_data = message.get("task_data", message)
                
                logger.info(f"Processing cached task file: {filename}, task_id={task_id}")
                
                # 立即删除文件（避免重复处理）
                await self.message_cache.delete_message(file_path)
                
                # 处理任务
                await self._process_task_message(task_id, task_data)
            else:
                # 文件损坏，删除
                await self.message_cache.delete_message(file_path)
                logger.warning(f"Failed to load task file, deleted: {filename}")
        else:
            # 未知格式的文件，记录警告但不删除（可能是其他类型的文件）
            logger.warning(f"Unknown cache file format: {filename}, skipping")
    
    async def _scan_cache_directory(self):
        """扫描缓存目录并处理遗留文件"""
        try:
            cache_files = await self.message_cache.scan_cache_files()
            if cache_files:
                logger.info(f"Found {len(cache_files)} cached files, processing...")
                for file_path in cache_files:
                    await self._process_cache_file(file_path)
        except Exception as e:
            logger.error(f"Error scanning cache directory: {e}", exc_info=True)
    
    async def process_loop(self):
        """任务处理循环（从消息队列读取并处理，队列无消息时定期扫描缓存目录）"""
        self._running = True
        logger.info("Task processor loop started")
        
        while self._running:
            try:
                # 检查队列是否为空
                queue_empty = self.message_queue.empty()
                
                if queue_empty:
                    # 队列为空时，检查是否需要扫描缓存目录
                    now = datetime.now()
                    if (now - self._last_cache_scan).total_seconds() >= self._cache_scan_interval:
                        await self._scan_cache_directory()
                        self._last_cache_scan = now
                    # 让出事件循环
                    await asyncio.sleep(0.05)
                else:
                    # 队列有消息时，从队列获取消息（放到线程池，避免阻塞事件循环），超时1秒
                    message = await asyncio.to_thread(self.message_queue.get, 1.0)
                    
                    if message:
                        await self.process_message(message)
                    # 如果超时返回None，继续循环检查队列
            
            except Exception as e:
                logger.error(f"Error in task processor loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # 避免快速循环
    
    def stop(self):
        """停止任务处理"""
        self._running = False
        logger.info("Task processor stopped")

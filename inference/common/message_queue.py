"""
消息队列模块
基于 queue.Queue 实现的消息队列，用于存储从WebSocket接收的消息
"""
import queue
import json
from typing import Dict, Any, Optional
from datetime import datetime
from .logger import get_logger

logger = get_logger(__name__)


class MessageQueue:
    """消息队列类"""
    
    def __init__(self, maxsize: int = 1000):
        """
        初始化消息队列
        
        Args:
            maxsize: 队列最大容量
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def put_message(self, message: Dict[str, Any]):
        """
        放入原始消息（不做字段过滤/裁剪）。

        说明：
        - 该方法用于 WS -> Queue 的“透传”场景，避免新增字段时需要改中间层。
        - 现有 put_task_message/put_download_message 等仍保留，作为构造消息的便捷接口。
        """
        if self._closed:
            logger.warning("Message queue is closed, ignoring message")
            return
        if not isinstance(message, dict):
            logger.warning(f"Ignoring non-dict message: {type(message).__name__}")
            return
        try:
            # 浅拷贝：避免上游后续修改同一个 dict 造成队列内数据被“回写”
            self._queue.put_nowait(dict(message))
        except queue.Full:
            mtype = str(message.get("type") or "")
            mid = message.get("model_key") or message.get("task_id") or ""
            logger.error(f"Message queue is full, dropping message: type={mtype} id={mid}")
    
    def put_task_message(self, task_id: str, task_data: Dict[str, Any]):
        """
        放入任务消息（全量数据）
        
        Args:
            task_id: 任务ID
            task_data: 全量任务数据字典
        """
        if self._closed:
            logger.warning("Message queue is closed, ignoring task message")
            return
        
        message = {
            "type": "task",
            "task_id": task_id,
            "task_data": task_data  # 全量任务数据
        }
        self.put_message(message)
        logger.debug(f"Task message queued: task_id={task_id}")
    
    def put_cancel_message(self, task_id: str, timestamp: str):
        """
        放入取消消息
        
        Args:
            task_id: 任务ID
            timestamp: 时间戳
        """
        if self._closed:
            logger.warning("Message queue is closed, ignoring cancel message")
            return
        
        message = {
            "type": "cancel",
            "task_id": task_id,
            "timestamp": timestamp
        }
        self.put_message(message)
        logger.debug(f"Cancel message queued: task_id={task_id}")

    def put_download_message(
        self,
        model_key: str,
        source: Dict[str, Any],
        timestamp: Optional[str] = None,
        asset_type: Optional[str] = None,
    ):
        """
        放入下载消息
        """
        if self._closed:
            logger.warning("Message queue is closed, ignoring download message")
            return

        message = {
            "type": "download",
            "model_key": model_key,
            "source": dict(source or {}),
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
        # asset_type：用于下载器侧选择落盘目录（loras_dir vs models_dir）
        # 兼容：允许缺省（download worker 会默认使用 checkpoint）
        if asset_type is not None and str(asset_type).strip():
            message["asset_type"] = str(asset_type).strip()

        self.put_message(message)
        logger.debug(f"Download message queued: model_key={model_key}")

    def put_download_cancel_message(
        self,
        model_key: str,
        source: Dict[str, Any],
        timestamp: Optional[str] = None,
    ):
        """
        放入下载取消消息
        """
        if self._closed:
            logger.warning("Message queue is closed, ignoring download_cancel message")
            return

        message = {
            "type": "download_cancel",
            "model_key": model_key,
            "source": dict(source or {}),
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }

        self.put_message(message)
        logger.debug(f"Download cancel message queued: model_key={model_key}")
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        从队列获取消息（阻塞）
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
        
        Returns:
            消息字典，如果超时或队列关闭则返回None
        """
        if self._closed:
            return None
        
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_nowait(self) -> Optional[Dict[str, Any]]:
        """
        从队列获取消息（非阻塞）
        
        Returns:
            消息字典，如果队列为空则返回None
        """
        if self._closed:
            return None
        
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
    
    def qsize(self) -> int:
        """获取队列当前大小"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()
    
    def close(self):
        """关闭消息队列"""
        self._closed = True
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Message queue closed")


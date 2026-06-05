"""
信号处理和异常处理模块
实现优雅退出逻辑，确保推理器停止时调用 stop API
"""
import signal
import asyncio
import sys
from typing import Optional, Callable
from .logger import get_logger

logger = get_logger(__name__)


class SignalHandler:
    """信号处理器类"""
    
    def __init__(self, cleanup_callback: Optional[Callable] = None):
        """
        初始化信号处理器
        
        Args:
            cleanup_callback: 清理回调函数，在收到退出信号时调用
        """
        self.cleanup_callback = cleanup_callback
        self._shutdown_requested = False
        self._original_handlers = {}
    
    def is_shutdown_requested(self) -> bool:
        """检查是否收到关闭请求"""
        return self._shutdown_requested
    
    def _handle_signal(self, signum, frame):
        """处理信号"""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal: {signal_name}")
        self._shutdown_requested = True
        
        # 调用清理回调
        if self.cleanup_callback:
            try:
                if asyncio.iscoroutinefunction(self.cleanup_callback):
                    # 如果是协程函数，需要创建任务
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.cleanup_callback())
                else:
                    # 普通函数直接调用
                    self.cleanup_callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}", exc_info=True)
        
        # 恢复原始信号处理并退出
        signal.signal(signum, self._original_handlers.get(signum, signal.SIG_DFL))
        sys.exit(0)
    
    def register(self):
        """注册信号处理器"""
        # Windows不支持SIGTERM，只注册SIGINT
        if sys.platform != "win32":
            try:
                self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handle_signal)
            except (AttributeError, ValueError):
                pass
        
        # 注册SIGINT（Ctrl+C）
        try:
            self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._handle_signal)
        except (AttributeError, ValueError):
            pass
        
        logger.info("Signal handlers registered (SIGTERM, SIGINT)")
    
    def unregister(self):
        """取消注册信号处理器"""
        for signum, original_handler in self._original_handlers.items():
            try:
                signal.signal(signum, original_handler)
            except Exception as e:
                logger.warning(f"Failed to restore signal handler for {signum}: {e}")
        
        self._original_handlers.clear()
        logger.info("Signal handlers unregistered")


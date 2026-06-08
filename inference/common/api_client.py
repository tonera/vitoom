"""
API客户端模块
调用后端API上报推理器状态
"""
import aiohttp
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)


class APIClient:
    """API客户端类"""
    
    def __init__(self, base_url: str):
        """
        初始化API客户端
        
        Args:
            base_url: API后端基础URL，例如 "http://127.0.0.1:8000"
        """
        self.base_url = base_url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def notify_start(
        self,
        service_id: str,
        host: str,
        port: Optional[int],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        通知API后端推理器已启动
        
        Args:
            service_id: 服务ID
            host: 本机host
            port: 本机端口（仅text推理器需要）
            config: 服务配置（可选，包含已加载的模型等信息）
        
        Returns:
            是否成功
        """
        url = f"{self.base_url}/api/inference/services/{service_id}/start"
        
        payload = {
            "host": host,
            "port": port,
        }
        if config:
            payload["config"] = config
        
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Successfully notified API backend: service {service_id} started")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to notify API backend: status={response.status}, "
                        f"response={error_text}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error notifying API backend: {e}", exc_info=True)
            return False
    
    async def notify_stop(self, service_id: str) -> bool:
        """
        通知API后端推理器已停止
        
        Args:
            service_id: 服务ID
        
        Returns:
            是否成功
        """
        url = f"{self.base_url}/api/inference/services/{service_id}/stop"
        
        try:
            session = await self._get_session()
            async with session.post(url) as response:
                if response.status == 200:
                    logger.info(f"Successfully notified API backend: service {service_id} stopped")
                    return True
                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Failed to notify API backend: status={response.status}, "
                        f"response={error_text}"
                    )
                    return False
        except Exception as e:
            logger.warning(f"Error notifying API backend: {e}", exc_info=True)
            return False


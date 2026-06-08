"""
HTTP客户端工具
"""
import httpx
from typing import TYPE_CHECKING, Optional, Dict, Any, Union

if TYPE_CHECKING:
    from starlette.requests import Request
from pathlib import Path
import asyncio
from functools import lru_cache


def resolve_client_ip(request: "Request") -> Optional[str]:
    """解析 HTTP 请求来源 IP（优先 X-Forwarded-For / X-Real-IP，否则取直连地址）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
        if ip:
            return ip

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        ip = real_ip.strip()
        if ip:
            return ip

    if request.client and request.client.host:
        return request.client.host
    return None


class HTTPClient:
    """HTTP客户端类"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        verify: bool = True
    ):
        """
        初始化HTTP客户端
        
        Args:
            base_url: 基础URL
            timeout: 超时时间（秒）
            headers: 默认请求头
            verify: 是否验证SSL证书
        """
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers or {}
        self.verify = verify
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def start(self):
        """启动客户端"""
        if self._client is None:
            # httpx 的 base_url 需要是 str/httpx.URL；传 None 会抛 TypeError
            kwargs: Dict[str, Any] = {
                "timeout": self.timeout,
                "headers": self.headers,
                "verify": self.verify,
            }
            if isinstance(self.base_url, str) and self.base_url.strip():
                kwargs["base_url"] = self.base_url
            self._client = httpx.AsyncClient(**kwargs)
    
    async def close(self):
        """关闭客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法（GET, POST, PUT, DELETE等）
            url: 请求URL
            **kwargs: 其他httpx请求参数
        
        Returns:
            httpx.Response对象
        """
        if self._client is None:
            await self.start()
        
        return await self._client.request(method, url, **kwargs)
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """发送GET请求"""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """发送POST请求"""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> httpx.Response:
        """发送PUT请求"""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """发送DELETE请求"""
        return await self.request("DELETE", url, **kwargs)
    
    async def download_file(
        self,
        url: str,
        save_path: Union[str, Path],
        chunk_size: int = 8192
    ) -> Path:
        """
        下载文件
        
        Args:
            url: 文件URL
            save_path: 保存路径
            chunk_size: 块大小（字节）
        
        Returns:
            保存文件的Path对象
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with self.get(url) as response:
            response.raise_for_status()
            with open(save_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size):
                    f.write(chunk)
        
        return save_path


# 全局HTTP客户端实例
_global_client: Optional[HTTPClient] = None


@lru_cache(maxsize=1)
def get_http_client(
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None
) -> HTTPClient:
    """
    获取全局HTTP客户端实例（单例模式）
    
    Args:
        base_url: 基础URL
        timeout: 超时时间（秒）
        headers: 默认请求头
    
    Returns:
        HTTPClient实例
    """
    global _global_client
    if _global_client is None:
        _global_client = HTTPClient(base_url=base_url, timeout=timeout, headers=headers)
    return _global_client


async def request(
    method: str,
    url: str,
    **kwargs
) -> httpx.Response:
    """
    发送HTTP请求（便捷函数）
    
    Args:
        method: HTTP方法
        url: 请求URL
        **kwargs: 其他httpx请求参数
    
    Returns:
        httpx.Response对象
    """
    client = get_http_client()
    await client.start()
    return await client.request(method, url, **kwargs)


async def get(url: str, **kwargs) -> httpx.Response:
    """发送GET请求（便捷函数）"""
    return await request("GET", url, **kwargs)


async def post(url: str, **kwargs) -> httpx.Response:
    """发送POST请求（便捷函数）"""
    return await request("POST", url, **kwargs)


async def put(url: str, **kwargs) -> httpx.Response:
    """发送PUT请求（便捷函数）"""
    return await request("PUT", url, **kwargs)


async def delete(url: str, **kwargs) -> httpx.Response:
    """发送DELETE请求（便捷函数）"""
    return await request("DELETE", url, **kwargs)


async def download_file(
    url: str,
    save_path: Union[str, Path],
    chunk_size: int = 8192
) -> Path:
    """
    下载文件（便捷函数）
    
    Args:
        url: 文件URL
        save_path: 保存路径
        chunk_size: 块大小（字节）
    
    Returns:
        保存文件的Path对象
    """
    client = get_http_client()
    await client.start()
    return await client.download_file(url, save_path, chunk_size)


"""
推理器基类
抽象所有公共功能：启动流程、停止流程、多线程架构、取消任务处理
"""
import asyncio
import os
from typing import Optional, Callable, Awaitable, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
from .logger import get_logger
from .config_loader import StartupConfig, load_startup_config, load_inference_config
from .system_monitor import SystemMonitor
from .api_client import APIClient
from .ws_client import WebSocketClient
from .message_queue import MessageQueue
from .message_cache import MessageCache
from .task_processor import TaskProcessor
from .signal_handler import SignalHandler
from .egress_fanout import FanoutEgress, EgressClient
from .redis_list_transport import RedisListConfig, RedisListIngress, RedisListEgress
import sys
from pathlib import Path

# 导入InferenceRequestParams
sys.path.insert(0, str(Path(__file__).parent.parent))
from schemas import InferenceRequestParams

logger = get_logger(__name__)


class BaseInferrer:
    """推理器基类"""
    
    def __init__(self, service_id: str):
        """
        初始化推理器
        
        Args:
            service_id: 服务ID
        """
        self.service_id = service_id
        # 让未显式传 service_id 的配置读取也能获得“当前服务”视角（单进程单服务场景）
        try:
            if service_id and not os.environ.get("VITOOM_SERVICE_ID"):
                os.environ["VITOOM_SERVICE_ID"] = str(service_id)
        except Exception:
            pass
        self.config: Optional[StartupConfig] = None
        self.system_monitor = SystemMonitor()
        self.api_client: Optional[APIClient] = None
        # 兼容历史命名：对上层（ResultHandler/Inferrer）仍暴露 ws_client，
        # 但其真实含义是“Egress client”（可能是 WS / Redis / fanout）。
        self.ws_client: Optional[Any] = None
        self.message_queue: Optional[MessageQueue] = None
        self.message_cache: Optional[MessageCache] = None
        self.task_processor: Optional[TaskProcessor] = None
        self.signal_handler: Optional[SignalHandler] = None
        self._ingresses: List[Any] = []
        self._ws_transport: Optional[WebSocketClient] = None
        self._redis_transports: List[Any] = []
        self._has_ws_ingress: bool = False
        # 单线程执行器，用于将阻塞/重型推理放到线程，保持事件循环畅通
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{service_id}-worker")
        self._api_ready = False
        
        self._running = False
        self._task_processor_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """初始化推理器（加载配置、初始化组件）"""
        # 1. 加载启动配置
        logger.info(f"Loading startup config for service_id: {self.service_id}")
        self.config = load_startup_config(self.service_id)

        # 3. 初始化消息队列
        self.message_queue = MessageQueue(maxsize=1000)
        
        # 4. 初始化消息缓存
        # 尽量使用 startup 内部的 merged inference_config（包含 service 覆盖）
        inference_config = getattr(self.config, "inference_config", None) or load_inference_config(service_id=self.service_id)
        cache_dir = Path("resources/cache/messages").resolve()
        self.message_cache = MessageCache(str(cache_dir))

        # 5. 初始化传输层（Ingress/Egress）
        self._ingresses = []
        self._redis_transports = []
        self._ws_transport = None

        transport_cfg: Dict[str, Any] = {}
        try:
            transport_cfg = getattr(inference_config, "transport", {}) or {}
        except Exception:
            transport_cfg = {}

        ingresses_cfg = transport_cfg.get("ingresses")
        egresses_cfg = transport_cfg.get("egresses")
        if not isinstance(ingresses_cfg, list):
            ingresses_cfg = []
        if not isinstance(egresses_cfg, list):
            egresses_cfg = []

        # 默认：未配置 transport 时保持历史行为（WS ingress + WS egress）
        if not ingresses_cfg and not egresses_cfg:
            ingresses_cfg = [{"type": "ws"}]
            egresses_cfg = [{"type": "ws"}]

        # 2. 初始化 API 客户端（仅 WS ingress 场景）
        # 说明：
        # - WS ingress 代表走“后端 API/WS 体系”，需要上报 start/stop 以便服务注册、展示状态等
        # - redis_list ingress 属于“外部队列模式”，不依赖后端 API，避免因后端不可达刷错误日志
        self._has_ws_ingress = any(
            isinstance(c, dict) and str(c.get("type") or "").strip().lower() in ("ws", "websocket")
            for c in ingresses_cfg
        )
        has_redis_ingress = any(
            isinstance(c, dict) and str(c.get("type") or "").strip().lower() in ("redis_list", "redis", "list")
            for c in ingresses_cfg
        )
        # 仅当“确实配置了 WS ingress”时才启用 APIClient
        # 若只有 redis ingress（无 ws ingress），则彻底跳过上报逻辑
        self.api_client = APIClient(self.config.api_base_url) if self._has_ws_ingress else None

        egress_clients: List[EgressClient] = []

        def _ensure_ws_transport() -> WebSocketClient:
            if self._ws_transport is None:
                self._ws_transport = WebSocketClient(
                    ws_url=self.config.ws_url,
                    message_queue=self.message_queue,
                    service_id=self.service_id,
                    message_cache=self.message_cache,
                    on_reconnect=self._on_ws_reconnect,
                    on_disconnect=self._on_ws_disconnect,
                    on_session_message=self._on_session_message,
                    on_cancel_message=self._on_cancel_message,
                )
            return self._ws_transport

        def _build_redis_cfg(raw: Dict[str, Any], *, want_channel: bool, want_res: bool) -> RedisListConfig:
            r = raw.get("redis", raw) if isinstance(raw, dict) else {}
            if not isinstance(r, dict):
                r = {}
            host = str(r.get("host") or "127.0.0.1")
            port = int(r.get("port") or 6379)
            password = str(r.get("pwd") or r.get("password") or "")
            db = int(r.get("db") or 0)
            channel = str(r.get("channel") or "") if want_channel else ""
            reschannel = str(r.get("reschannle") or r.get("reschannel") or "") if want_res else ""
            push = str(r.get("push") or "lpush")
            brpop_timeout = int(r.get("brpop_timeout") or 5)
            return RedisListConfig(
                host=host,
                port=port,
                password=password,
                db=db,
                channel=channel,
                reschannel=reschannel,
                push=push,
                brpop_timeout=brpop_timeout,
            )

        # 先构建 egress（让 ingress 可以发 queued）
        for ecfg in egresses_cfg:
            if not isinstance(ecfg, dict):
                continue
            et = str(ecfg.get("type") or "").strip().lower()
            if et in ("ws", "websocket"):
                egress_clients.append(_ensure_ws_transport())
            elif et in ("redis_list", "redis", "list"):
                cfg = _build_redis_cfg(ecfg, want_channel=False, want_res=True)
                e = RedisListEgress(cfg)
                self._redis_transports.append(e)
                egress_clients.append(e)

        if not egress_clients:
            # 兜底：至少要有一个 egress，否则结果无法回传
            egress_clients.append(_ensure_ws_transport())

        # 对外暴露 ws_client（实际是 egress）
        if len(egress_clients) == 1:
            self.ws_client = egress_clients[0]
        else:
            self.ws_client = FanoutEgress(egress_clients)

        # 再构建 ingresses
        for icfg in ingresses_cfg:
            if not isinstance(icfg, dict):
                continue
            it = str(icfg.get("type") or "").strip().lower()
            if it in ("ws", "websocket"):
                _ensure_ws_transport()
            elif it in ("redis_list", "redis", "list"):
                cfg = _build_redis_cfg(icfg, want_channel=True, want_res=False)
                ingress = RedisListIngress(
                    cfg,
                    message_queue=self.message_queue,
                    egress=self.ws_client,
                    on_cancel_message=self._on_cancel_message,
                )
                self._ingresses.append(ingress)
                self._redis_transports.append(ingress)

        # 6. 初始化任务处理器
        self.task_processor = TaskProcessor(
            message_queue=self.message_queue,
            message_cache=self.message_cache,
            ws_client=self.ws_client,
            service_type=self.config.service_type,
            inference_callback=self.inference_callback
        )
        
        # 7. 注册信号处理器
        self.signal_handler = SignalHandler(cleanup_callback=self.cleanup)
        self.signal_handler.register()
        
        logger.info("Inferrer initialized successfully")
    
    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        """
        推理回调函数（子类需要实现）
        
        Args:
            params: 推理请求参数
        
        Returns:
            推理结果
        """
        raise NotImplementedError("Subclasses must implement inference_callback")
    
    async def start(self):
        """启动推理器"""
        if self._running:
            logger.warning("Inferrer is already running")
            return
        
        logger.info(f"Starting inferrer: service_id={self.service_id}")
        
        # 1. 初始化
        await self.initialize()
        # 标记为运行中，确保“早期启动失败”也能走完整的 stop/cleanup 逻辑
        self._running = True

        await self._before_backend_registration()
        
        # 2. 先通知 API 后端已启动（仅 WS ingress 场景）
        # 这样首次启动时，后端可以自动 upsert service 记录，避免随后连接 /ws/inference/{service_id}
        # 因“service 不存在”而在握手前被拒绝（HTTP 403）。
        if self._has_ws_ingress:
            await self._notify_api_start()

        # 3. 启动/连接 Ingress
        # WS：需要 connect()；Redis：启动 BRPOP loop
        if self._ws_transport is not None:
            logger.info("Connecting to WebSocket Server...")
            connected = await self._ws_transport.connect()
            if not connected:
                logger.error("Failed to connect to WebSocket Server")
                await self.stop()
                return
            await self._after_ws_connected()

        for ingress in self._ingresses:
            try:
                await ingress.start()
            except Exception as e:
                logger.error(f"Failed to start ingress: {e}", exc_info=True)
                await self.stop()
                return

        # 4. 启动任务处理循环
        self._task_processor_task = asyncio.create_task(self.task_processor.process_loop())
        
        logger.info("Inferrer started successfully")
    
    async def stop(self):
        """停止推理器"""
        if not self._running:
            return
        
        logger.info("Stopping inferrer...")
        self._running = False
        
        # 1. 停止任务处理
        if self.task_processor:
            self.task_processor.stop()
        
        # 2. 取消任务处理循环
        if self._task_processor_task:
            self._task_processor_task.cancel()
            try:
                await self._task_processor_task
            except asyncio.CancelledError:
                pass
        
        # 3. 停止 ingresses / transports
        for ingress in self._ingresses:
            try:
                await ingress.stop()
            except Exception:
                pass
        self._ingresses = []

        if self._ws_transport is not None:
            try:
                await self._ws_transport.disconnect()
            except Exception:
                pass
            self._ws_transport = None

        for t in self._redis_transports:
            # RedisListEgress/Ingress 都有 close/stop，已在上面 stop 处理一部分，这里做 best-effort close
            try:
                if hasattr(t, "close"):
                    await t.close()
            except Exception:
                pass
        self._redis_transports = []
        
        # 4. 通知API后端已停止（仅 WS ingress 场景）
        if self._has_ws_ingress and self.api_client:
            await self.api_client.notify_stop(self.service_id)
            await self.api_client.close()
        
        # 5. 关闭消息队列
        if self.message_queue:
            self.message_queue.close()
        
        # 6. 取消注册信号处理器
        if self.signal_handler:
            self.signal_handler.unregister()

        # 关闭线程池
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        
        logger.info("Inferrer stopped")
    
    async def cleanup(self):
        """清理资源（信号处理器回调）"""
        logger.info("Cleaning up resources...")
        await self.stop()

    async def _on_cancel_message(self, message: Dict[str, Any]) -> None:
        """收到取消消息时，旁路打标，避免依赖串行任务队列消费。"""
        task_id = str((message or {}).get("task_id") or "").strip()
        if not task_id or not self.task_processor:
            return
        await self.task_processor.handle_cancel_message(task_id)

    async def run_blocking(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        将阻塞/CPU/GPU 密集型函数放入线程执行，避免阻塞事件循环
        可被子类复用
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def _notify_api_start(self):
        """向后端上报服务启动/重连状态"""
        if not self.api_client or not self.config:
            return
        system_info = self.system_monitor.get_all_info()
        gpu_info = system_info.get("gpu", {})
        memory_info = system_info.get("memory", {})

        config_dict = {
            "gpu_available_memory": gpu_info.get("gpu_available_memory", 0),
            "gpu_total_memory": gpu_info.get("gpu_total_memory", 0),
            "system_load": system_info.get("system_load", 0.0),
            "memory": memory_info,
            "service_type": self.config.service_type,
            "program_name": self.service_id,
        }
        if self.config.inference_config.supervisor_url:
            config_dict["supervisor_url"] = self.config.inference_config.supervisor_url
        if self.config.config:
            config_dict.update(self.config.config)

        logger.info("Notifying API backend (startup/reconnect)...")
        success = await self.api_client.notify_start(
            service_id=self.service_id,
            host=self.config.host,
            port=self.config.port,
            config=config_dict,
        )
        if success:
            self._api_ready = True
        else:
            logger.warning("Failed to notify API backend, but continuing...")

    async def _on_ws_reconnect(self):
        """WS 重连后回调：重新上报服务启动状态"""
        try:
            await self._notify_api_start()
            await self._after_ws_connected()
        except Exception as e:
            logger.error(f"Failed to notify start after WS reconnect: {e}", exc_info=True)

    async def _on_ws_disconnect(self, reason: str):
        """WS 断开时回调；子类可重写。"""
        logger.warning(f"WS disconnected for service {self.service_id}: {reason}")

    async def _after_ws_connected(self):
        """WS 建连后的钩子；子类可重写。"""
        return

    async def _before_backend_registration(self):
        """后端启动上报/WS 注册前的钩子；子类可重写。"""
        return

    async def _on_session_message(self, message: Dict[str, Any]) -> bool:
        """处理 session 消息；子类可重写。"""
        return False
    
    async def run(self):
        """运行推理器（主循环）"""
        try:
            await self.start()
            
            # 保持运行直到收到停止信号
            while self._running and not self.signal_handler.is_shutdown_requested():
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt")
        except Exception as e:
            logger.error(f"Error in inferrer run loop: {e}", exc_info=True)
        finally:
            await self.stop()


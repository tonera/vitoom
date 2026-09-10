"""
WebSocket客户端模块
连接WS Server，发送心跳，接收消息。

保活只走应用层 ``heartbeat``（约 20s，并携带 queue_length）。
不启用 websockets 协议层 ping，也不回复网关 JSON ping/pong。
"""
import asyncio
import json
import websockets
from urllib.parse import urlencode
from websockets.protocol import State as WsState
from typing import Optional, Callable, Awaitable, Dict, Any
from datetime import datetime
from .logger import get_logger, summarize_tpl_list_for_log
from .inference_token import resolve_inference_token
from .message_queue import MessageQueue
from .message_cache import MessageCache

logger = get_logger(__name__)

_HEARTBEAT_INTERVAL = 20  # 秒，应用层 heartbeat → 网关
_WATCHDOG_INTERVAL = 5  # 秒，检查是否需要重连
_WS_OPEN_TIMEOUT = 30  # 秒，跨机房/高负载时默认 10s 易 handshake timeout


class WebSocketClient:
    """WebSocket客户端类"""
    
    def __init__(
        self,
        ws_url: str,
        message_queue: MessageQueue,
        service_id: str,
        message_cache: Optional[MessageCache] = None,
        on_reconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[str], Awaitable[None]]] = None,
        on_session_message: Optional[Callable[[Dict[str, Any]], Awaitable[bool]]] = None,
        on_cancel_message: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        inference_token: Optional[str] = None,
        queue_length_provider: Optional[Callable[[], int]] = None,
        service_type: str = "",
    ):
        """
        初始化WebSocket客户端
        
        Args:
            ws_url: WebSocket Server URL，例如 "ws://127.0.0.1:8000"
            message_queue: 消息队列，用于存储接收到的消息
            service_id: 服务ID
            message_cache: 消息缓存（可选）
        """
        self.ws_url = ws_url.rstrip('/')
        self.service_id = service_id
        self.message_queue = message_queue
        self.message_cache = message_cache
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_lock = asyncio.Lock()
        self._reconnect_attempts = 0
        self._max_reconnect_interval = 10
        self._on_reconnect = on_reconnect
        self._on_disconnect = on_disconnect
        self._on_session_message = on_session_message
        self._on_cancel_message = on_cancel_message
        self._inference_token_explicit = inference_token
        self._queue_length_provider = queue_length_provider
        self.service_type = str(service_type or "").strip()
        self._last_reported_queue_length: Optional[int] = None
        self._queue_notify_task: Optional[asyncio.Task] = None
        self._last_disconnect_reason = ""
        self._session_message_tasks: set[asyncio.Task] = set()

    def _resolve_queue_length(self) -> int:
        if self._queue_length_provider is None:
            return 0
        try:
            return max(0, int(self._queue_length_provider()))
        except Exception:
            return 0

    def notify_queue_length(self) -> None:
        """队列长度变化时补发一帧 heartbeat。

        网关按 ``queue_length`` 选实例，只靠 20s 周期心跳的话，两次上报之间
        新派发的请求会基于过期负载做决策，慢实例会持续被选中。
        """
        if not self.is_connected():
            return
        if self._queue_notify_task is not None and not self._queue_notify_task.done():
            return
        value = self._resolve_queue_length()
        if value == self._last_reported_queue_length:
            return
        try:
            self._queue_notify_task = asyncio.create_task(self._send_heartbeat(value))
        except RuntimeError:
            pass

    async def _send_heartbeat(self, queue_length: int) -> None:
        payload = {
            "type": "heartbeat",
            "queue_length": queue_length,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        self._last_reported_queue_length = queue_length

    def _track_session_message_task(self, task: asyncio.Task) -> None:
        self._session_message_tasks.add(task)

        def _cleanup(done_task: asyncio.Task) -> None:
            self._session_message_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Session message background task failed: {e}", exc_info=True)

        task.add_done_callback(_cleanup)

    async def _cancel_session_message_tasks(self) -> None:
        tasks = list(self._session_message_tasks)
        self._session_message_tasks.clear()
        current = asyncio.current_task()
        for task in tasks:
            if task is current:
                continue
            task.cancel()
        for task in tasks:
            if task is current:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    def _is_ws_open(self) -> bool:
        return (
            self.websocket is not None
            and getattr(self.websocket, "state", None) == WsState.OPEN
        )

    def _needs_reconnect(self) -> bool:
        if not self._connected or not self._is_ws_open():
            return True
        if self._heartbeat_task is None or self._heartbeat_task.done():
            return True
        if self._receive_task is None or self._receive_task.done():
            return True
        return False

    def _io_tasks_running(self) -> bool:
        return (
            self._heartbeat_task is not None
            and not self._heartbeat_task.done()
            and self._receive_task is not None
            and not self._receive_task.done()
        )

    async def _close_websocket_best_effort(self) -> None:
        ws = self.websocket
        self.websocket = None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    async def _mark_disconnected(self, reason: str) -> None:
        normalized_reason = str(reason or "websocket disconnected")
        self._connected = False
        self._last_reported_queue_length = None
        await self._cancel_session_message_tasks()
        if normalized_reason == self._last_disconnect_reason:
            return
        self._last_disconnect_reason = normalized_reason
        logger.warning("WebSocket disconnected: %s", normalized_reason)
        if self._on_disconnect:
            try:
                await self._on_disconnect(normalized_reason)
            except Exception as e:
                logger.error(f"on_disconnect callback failed: {e}", exc_info=True)

    async def _reconnect(self) -> None:
        async with self._reconnect_lock:
            if not self._needs_reconnect():
                return
            backoff = min(2 ** self._reconnect_attempts, self._max_reconnect_interval)
            self._reconnect_attempts += 1
            logger.warning(
                "WebSocket reconnect in %ss (service_id=%s)",
                backoff,
                self.service_id,
            )
            await self._close_websocket_best_effort()
            await asyncio.sleep(backoff)
            try:
                self.websocket = await self._open_websocket()
            except Exception as e:
                logger.warning(f"WebSocket reconnect failed: {e}")
                await self._close_websocket_best_effort()
                self._connected = False
                return
            self._last_disconnect_reason = ""
            await self._restart_io_tasks()
            if not self._io_tasks_running() or not self._is_ws_open():
                logger.warning("WebSocket reconnect: IO tasks not running after connect, retry later")
                await self._close_websocket_best_effort()
                self._connected = False
                return
            self._connected = True
            self._reconnect_attempts = 0
            logger.info("WebSocket reconnected successfully")
        if self._on_reconnect:
            try:
                await self._on_reconnect()
            except Exception as e:
                logger.error(f"on_reconnect callback failed: {e}", exc_info=True)

    def _schedule_reconnect(self) -> None:
        if not self._running:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        if self._reconnect_lock.locked():
            return
        self._reconnect_task = asyncio.create_task(
            self._reconnect(),
            name=f"ws-reconnect:{self.service_id}",
        )
    
    def _build_connect_target(self) -> tuple[str, Optional[Dict[str, str]]]:
        endpoint = f"{self.ws_url}/ws/inference/{self.service_id}"
        params: Dict[str, str] = {}
        if self.service_type:
            params["service_type"] = self.service_type
        token = resolve_inference_token(self._inference_token_explicit)
        headers: Optional[Dict[str, str]] = None
        if token:
            params["token"] = token
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Inference-Token": token,
            }
        if params:
            endpoint = f"{endpoint}?{urlencode(params)}"
        return endpoint, headers

    async def _open_websocket(self):
        endpoint, headers = self._build_connect_target()
        connect_kwargs = {
            # 协议层 ping 会与网关 uvicorn ping、应用层 heartbeat 叠加重杀连接。
            "ping_interval": None,
            "ping_timeout": None,
            "close_timeout": 10,
            "open_timeout": _WS_OPEN_TIMEOUT,
            # 默认 1MiB；session_text_input 带图（data URL）会超限，对端关连接 1009。
            "max_size": 5 * 1024 * 1024,
        }
        if headers:
            connect_kwargs["additional_headers"] = headers
        return await websockets.connect(endpoint, **connect_kwargs)

    async def connect(self) -> bool:
        """
        连接WebSocket Server
        
        Returns:
            是否成功连接
        """
        log_endpoint = f"{self.ws_url}/ws/inference/{self.service_id}"
        auth_suffix = " (with inference token)" if resolve_inference_token(self._inference_token_explicit) else ""
        logger.info(f"Connecting to WebSocket Server: {log_endpoint}{auth_suffix}")
        
        try:
            self.websocket = await self._open_websocket()
            self._connected = True
            self._running = True
            self._last_disconnect_reason = ""
            logger.info(f"WebSocket connected successfully: {log_endpoint}")
            
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket Server: {e}", exc_info=True)
            await self._close_websocket_best_effort()
            self._connected = False
            return False

    async def _cancel_io_tasks(self):
        """取消心跳与接收任务，不影响 watchdog"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

    async def _restart_io_tasks(self):
        """在 websocket 已连接的前提下重启心跳/接收任务"""
        await self._cancel_io_tasks()
        if not self._is_ws_open():
            logger.warning("Cannot restart IO tasks: websocket not open after reconnect")
            self._connected = False
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("IO tasks restarted after reconnection")
    
    async def disconnect(self):
        """断开WebSocket连接"""
        self._running = False
        await self._cancel_session_message_tasks()
        
        # 取消任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        # 关闭连接
        await self._close_websocket_best_effort()
        
        self._connected = False
        logger.info("WebSocket disconnected")
    
    async def _heartbeat_loop(self):
        """应用层心跳：失败即标记断开，由 watchdog 重连。"""
        logger.info("Heartbeat loop started (interval=%ss)", _HEARTBEAT_INTERVAL)
        while self._running and self._connected:
            try:
                await self._send_heartbeat(self._resolve_queue_length())
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed as e:
                await self._mark_disconnected(
                    f"heartbeat connection closed: code={e.code}, reason={e.reason}"
                )
                break
            except Exception as e:
                await self._mark_disconnected(f"heartbeat send failed: {e}")
                break
        logger.info("Heartbeat loop stopped")
    
    async def _receive_loop(self):
        """接收消息；阻塞 recv，连接关闭时 websockets 会抛 ConnectionClosed。"""
        logger.info("Receive loop started")
        
        while self._running:
            try:
                if not self._is_ws_open():
                    await self._mark_disconnected("receive loop: websocket not open")
                    break
                
                try:
                    message_str = await self.websocket.recv()
                except websockets.exceptions.ConnectionClosed as e:
                    await self._mark_disconnected(
                        f"recv connection closed: code={e.code}, reason={e.reason}"
                    )
                    break
                except Exception as e:
                    await self._mark_disconnected(f"recv failed: {e}")
                    break

                if isinstance(message_str, bytes):
                    logger.warning(
                        "Received orphan binary frame (%d bytes), expected JSON text first; dropping",
                        len(message_str),
                    )
                    continue

                try:
                    message = json.loads(message_str)
                    message_type = message.get("type")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {e}, message: {message_str[:100]}")
                    continue

                if str(message_type) == "session.asr.chunk":
                    try:
                        need = int(message.get("bytes_len") or 0)
                    except (TypeError, ValueError):
                        need = 0
                    if need > 0:
                        try:
                            bin_payload = await self.websocket.recv()
                        except websockets.exceptions.ConnectionClosed as e:
                            await self._mark_disconnected(
                                f"recv binary closed: code={e.code}, reason={e.reason}"
                            )
                            break
                        if not isinstance(bin_payload, bytes):
                            logger.error(
                                "session.asr.chunk expected binary frame, got %s",
                                type(bin_payload).__name__,
                            )
                            continue
                        message["binary_bytes"] = bin_payload

                if message_type == "task":
                    # 任务消息（全量数据）
                    task_id = message.get("task_id")
                    task_data = message.get("task_data", message)  # 全量任务数据
                    logger.info("="*60)
                    logger.info(f"✓✓✓ RECEIVED TASK MESSAGE ✓✓✓")
                    logger.info(f"  task_id: {task_id}")
                    logger.info("="*60)
                    
                    # 1. 将全量数据放入队列
                    msg_to_queue: Dict[str, Any] = dict(message)
                    msg_to_queue["task_id"] = task_id
                    msg_to_queue["task_data"] = task_data

                    # 直接输出 task.params.tpl_list 快照（对比 RAW_WS_INGRESS 更方便）
                    try:
                        params = task_data.get("params") if isinstance(task_data, dict) else None
                        tpl = params.get("tpl_list") if isinstance(params, dict) else None
                        logger.info(
                            f"[WS_TASK_PARAMS] task_id={task_id} "
                            f"tpl_list={summarize_tpl_list_for_log(tpl)}"
                        )
                    except Exception:
                        logger.info(f"[WS_TASK_PARAMS] task_id={task_id} tpl_list=<unavailable>")
                    self.message_queue.put_message(msg_to_queue)
                    self.notify_queue_length()
                    
                    # 2. 发送已入队状态消息到WS Server
                    logger.info(f"收到任务 {task_id} 已入队列")
                    await self.send_task_status(task_id, "queued")
                    
                    # 3. 异步写入缓存文件
                    if self.message_cache:
                        cache_file = await self.message_cache.save_message(task_id, task_data)
                        if cache_file:
                            logger.debug(f"Task message cached: {cache_file}")
                    
                    logger.info(f"Task message queued successfully: task_id={task_id}")
                
                elif message_type == "cancel":
                    # 取消消息
                    task_id = message.get("task_id")
                    timestamp = message.get("timestamp")
                    logger.info("="*60)
                    logger.info(f"✓✓✓ RECEIVED CANCEL MESSAGE ✓✓✓")
                    logger.info(f"  task_id: {task_id}")
                    logger.info(f"  timestamp: {timestamp}")
                    logger.info("="*60)
                    msg_to_queue: Dict[str, Any] = dict(message)
                    if timestamp is None:
                        msg_to_queue["timestamp"] = datetime.utcnow().isoformat()
                    handled_immediately = False
                    if self._on_cancel_message is not None:
                        try:
                            await self._on_cancel_message(msg_to_queue)
                            handled_immediately = True
                        except Exception as e:
                            logger.error(f"Immediate cancel handling failed: {e}", exc_info=True)
                    if not handled_immediately:
                        self.message_queue.put_message(msg_to_queue)
                        logger.info(f"Cancel message queued successfully: task_id={task_id}")
                    else:
                        logger.info(f"Cancel message handled immediately: task_id={task_id}")

                elif message_type == "download":
                    # 下载消息（下载服务）
                    model_key = message.get("model_key")
                    source = message.get("source") if isinstance(message.get("source"), dict) else {}
                    provider = str(source.get("provider") or "").strip()
                    repo_id = str(source.get("repo_id") or "").strip()
                    asset_type = message.get("asset_type")
                    timestamp = message.get("timestamp")
                    logger.info("=" * 60)
                    logger.info("✓✓✓ RECEIVED DOWNLOAD MESSAGE ✓✓✓")
                    logger.info(f"  model_key: {model_key}")
                    logger.info(f"  source.provider: {provider}")
                    logger.info(f"  source.repo_id: {repo_id}")
                    logger.info(f"  asset_type: {asset_type}")
                    logger.info("=" * 60)
                    if model_key and provider and repo_id:
                        msg_to_queue: Dict[str, Any] = dict(message)
                        # 兜底：缺少 timestamp 时补一个
                        if not msg_to_queue.get("timestamp"):
                            msg_to_queue["timestamp"] = datetime.utcnow().isoformat()
                        self.message_queue.put_message(msg_to_queue)
                    else:
                        logger.warning(f"Invalid download message: {message}")

                elif message_type == "download_cancel":
                    # 下载取消消息（避免与推理任务 cancel 混淆）
                    model_key = message.get("model_key")
                    source = message.get("source") if isinstance(message.get("source"), dict) else {}
                    provider = str(source.get("provider") or "").strip()
                    repo_id = str(source.get("repo_id") or "").strip()
                    timestamp = message.get("timestamp")
                    logger.info("=" * 60)
                    logger.info("✓✓✓ RECEIVED DOWNLOAD_CANCEL MESSAGE ✓✓✓")
                    logger.info(f"  model_key: {model_key}")
                    logger.info(f"  timestamp: {timestamp}")
                    logger.info("=" * 60)
                    if model_key and provider and repo_id:
                        msg_to_queue: Dict[str, Any] = dict(message)
                        if not msg_to_queue.get("timestamp"):
                            msg_to_queue["timestamp"] = datetime.utcnow().isoformat()
                        self.message_queue.put_message(msg_to_queue)
                    else:
                        logger.warning(f"Invalid download_cancel message: {message}")
                
                elif message_type and (
                    str(message_type).startswith("session_")
                    or str(message_type).startswith("session.")
                ):
                    if self._on_session_message is not None:
                        async def _dispatch_session_message(payload: Dict[str, Any], payload_type: str) -> None:
                            handled = False
                            try:
                                handled = bool(await self._on_session_message(payload))
                            except Exception as e:
                                logger.error(f"Session message callback failed: {e}", exc_info=True)
                            if not handled:
                                logger.warning(f"Unhandled session message type: {payload_type}")

                        self._track_session_message_task(
                            asyncio.create_task(_dispatch_session_message(dict(message), str(message_type)))
                        )
                    else:
                        logger.warning(f"Unhandled session message type: {message_type}")

                elif message_type in {"ping", "pong"}:
                    # 旧网关仍可能发 JSON ping。保活只靠 heartbeat，这里不回 pong，
                    # 更不能把 pong 发送失败当成断线（连接关闭时会误触发重连）。
                    continue

                elif message_type == "service_registered":
                    registered_service_id = message.get("service_id") or self.service_id
                    logger.info(f"Service registration acknowledged: service_id={registered_service_id}")

                elif message_type == "service_error":
                    error_service_id = message.get("service_id") or self.service_id
                    error_text = str(message.get("error") or "unknown service error")
                    logger.error(
                        f"Service control message failed: service_id={error_service_id}, error={error_text}"
                    )
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
            
            except websockets.exceptions.ConnectionClosed as e:
                await self._mark_disconnected(
                    f"receive loop closed: code={e.code}, reason={e.reason}"
                )
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}", exc_info=True)
                await self._mark_disconnected(f"receive loop error: {e}")
                break
        logger.info("Receive loop stopped")

    async def _watchdog_loop(self):
        """未连接或 IO 任务退出时指数退避重连。"""
        while self._running:
            try:
                await asyncio.sleep(_WATCHDOG_INTERVAL)
                if self._needs_reconnect():
                    self._schedule_reconnect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error (will retry): {e}", exc_info=True)
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected and self._is_ws_open()
    
    async def send_result(self, result_message: dict) -> bool:
        """
        发送推理结果消息到WS Server
        
        Args:
            result_message: 结果消息字典
        
        Returns:
            是否成功发送
        """
        if not self.is_connected():
            logger.warning("WebSocket not connected, cannot send result")
            # WS断开时，写入状态结果缓存文件
            if self.message_cache:
                task_id = result_message.get('task_id')
                status = result_message.get('status', 'failed')
                await self.message_cache.save_status_result(task_id, status, result_message)
            return False
        
        try:
            message_json = json.dumps(result_message, ensure_ascii=False)
            await self.websocket.send(message_json)
            logger.info(f"Result message sent: task_id={result_message.get('task_id')}")
            return True
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Failed to send result message, connection closed: code={e.code}, reason={e.reason}")
            if self.message_cache:
                task_id = result_message.get("task_id")
                status = result_message.get("status", "failed")
                await self.message_cache.save_status_result(task_id, status, result_message)
            await self._mark_disconnected(f"send_result closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send result message: {e}", exc_info=True)
            # 发送失败时，也写入状态结果缓存文件
            if self.message_cache:
                task_id = result_message.get('task_id')
                status = result_message.get('status', 'failed')
                await self.message_cache.save_status_result(task_id, status, result_message)
            await self._mark_disconnected(f"send_result failed: {e}")
            return False
    
    async def send_task_status(self, task_id: str, status: str, error: Optional[str] = None, **kwargs) -> bool:
        """
        发送任务状态更新到WS Server
        
        Args:
            task_id: 任务ID
            status: 任务状态（processing/completed/failed/cancelled）
            error: 错误信息（可选）
            **kwargs: 其他状态字段（如started_at, completed_at等）
        
        Returns:
            是否成功发送
        """
        status_message = {
            "type": "task_status",
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            status_message["error"] = error
        
        status_message.update(kwargs)
        
        if not self.is_connected():
            logger.warning("WebSocket not connected, cannot send task status")
            # WS断开时：终态 completed 的结果消息（type=result）已由 send_result() 负责落盘，
            # 这里不再重复写缓存，避免同一任务完成时写两次 status_result。
            if status != "completed":
                if self.message_cache:
                    await self.message_cache.save_status_result(task_id, status, status_message)
            return False

        try:
            message_json = json.dumps(status_message, ensure_ascii=False)
            await self.websocket.send(message_json)
            logger.info(f"Task status sent: task_id={task_id}, status={status}")
            return True
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Failed to send task status, connection closed: code={e.code}, reason={e.reason}")
            await self._mark_disconnected(f"send_task_status closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send task status: {e}", exc_info=True)
            # 发送失败时也写缓存；但 completed 终态仍跳过（由 send_result 落盘）
            if status != "completed":
                if self.message_cache:
                    await self.message_cache.save_status_result(task_id, status, status_message)
            await self._mark_disconnected(f"send_task_status failed: {e}")
            return False

    async def send_stream_event(self, message: dict) -> bool:
        """
        发送音频/文本流式事件。
        约定：流式消息只做实时透传，不写缓存文件。
        """
        if not self.is_connected():
            logger.warning("WebSocket not connected, cannot send stream event")
            return False
        try:
            message_json = json.dumps(message, ensure_ascii=False)
            await self.websocket.send(message_json)
            logger.debug(f"Stream event sent: type={message.get('type')}, task_id={message.get('task_id')}")
            return True
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Failed to send stream event, connection closed: code={e.code}, reason={e.reason}")
            await self._mark_disconnected(f"send_stream_event closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send stream event: {e}", exc_info=True)
            await self._mark_disconnected(f"send_stream_event failed: {e}")
            return False

    async def send_message(self, message: dict, *, binary: Optional[bytes] = None) -> bool:
        """
        通用发送：下载服务等非 task/result 场景复用。

        若 ``binary`` 非空：先发送 JSON text，再发送一帧裸 bytes（与 meta 中
        ``bytes_len`` 对齐；用于 ``session.audio.chunk`` 等）。
        """
        if not self.is_connected():
            logger.warning("WebSocket not connected, cannot send message")
            return False
        try:
            message_json = json.dumps(message, ensure_ascii=False)
            await self.websocket.send(message_json)
            if binary:
                await self.websocket.send(binary)
            logger.debug(f"Message sent: type={message.get('type')}")
            return True
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Failed to send message, connection closed: code={e.code}, reason={e.reason}")
            await self._mark_disconnected(f"send_message closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
            await self._mark_disconnected(f"send_message failed: {e}")
            return False


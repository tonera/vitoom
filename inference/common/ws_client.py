"""
WebSocket客户端模块
连接WS Server，发送心跳，接收消息
"""
import asyncio
import json
import websockets
from websockets.protocol import State as WsState
from typing import Optional, Callable, Awaitable, Dict, Any
from datetime import datetime
from .logger import get_logger
from .message_queue import MessageQueue
from .message_cache import MessageCache

logger = get_logger(__name__)


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
        self._reconnect_lock = asyncio.Lock()
        self._reconnect_attempts = 0
        self._max_reconnect_interval = 10
        self._on_reconnect = on_reconnect
        self._on_disconnect = on_disconnect
        self._on_session_message = on_session_message
        self._on_cancel_message = on_cancel_message
        self._last_disconnect_reason = ""
        self._session_message_tasks: set[asyncio.Task] = set()

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
        """内部检查 websocket 是否处于 OPEN 状态"""
        return (
            self.websocket is not None
            and getattr(self.websocket, "state", None) == WsState.OPEN
        )
    
    async def connect(self) -> bool:
        """
        连接WebSocket Server
        
        Returns:
            是否成功连接
        """
        ws_endpoint = f"{self.ws_url}/ws/inference/{self.service_id}"
        logger.info(f"Connecting to WebSocket Server: {ws_endpoint}")
        
        try:
            # 注意：我们不使用websockets库的底层ping/pong机制，因为FastAPI的WebSocket
            # 可能无法正确处理websockets库的底层ping/pong帧。我们只使用应用层的ping/pong。
            # close_timeout: 关闭连接的超时时间（秒）
            self.websocket = await websockets.connect(
                ws_endpoint,
                ping_interval=None,  # 禁用底层ping，只使用应用层ping/pong
                ping_timeout=None,  # 禁用底层ping超时
                close_timeout=10
            )
            self._connected = True
            self._running = True
            self._last_disconnect_reason = ""
            logger.info(f"WebSocket connected successfully: {ws_endpoint}")
            
            # 启动心跳任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info(f"Heartbeat task started: {self._heartbeat_task}")
            
            # 启动接收任务
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info(f"Receive task started: {self._receive_task}")

            # 启动状态监控任务
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            logger.info(f"Watchdog task started: {self._watchdog_task}")
            
            # 等待一小段时间，检查任务是否正常运行
            await asyncio.sleep(0.5)
            if self._heartbeat_task.done():
                try:
                    result = await self._heartbeat_task
                    logger.warning(f"Heartbeat task completed unexpectedly: {result}")
                except Exception as e:
                    logger.error(f"Heartbeat task failed immediately: {e}", exc_info=True)
            else:
                logger.info(f"Heartbeat task is running: {self._heartbeat_task}")
            
            if self._receive_task.done():
                try:
                    result = await self._receive_task
                    logger.warning(f"Receive task completed unexpectedly: {result}")
                except Exception as e:
                    logger.error(f"Receive task failed immediately: {e}", exc_info=True)
            else:
                logger.info(f"Receive task is running: {self._receive_task}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket Server: {e}", exc_info=True)
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

    async def _notify_disconnect(self, reason: str) -> None:
        normalized_reason = str(reason or "websocket disconnected")
        self._connected = False
        await self._cancel_session_message_tasks()
        if normalized_reason == self._last_disconnect_reason:
            return
        self._last_disconnect_reason = normalized_reason
        if self._on_disconnect:
            try:
                await self._on_disconnect(normalized_reason)
            except Exception as e:
                logger.error(f"on_disconnect callback failed: {e}", exc_info=True)

    async def _restart_io_tasks(self):
        """在 websocket 已连接的前提下重启心跳/接收任务"""
        await self._cancel_io_tasks()
        if not self._is_ws_open():
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
        
        # 关闭连接
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        self._connected = False
        logger.info("WebSocket disconnected")
    
    async def _heartbeat_loop(self):
        """心跳循环（每20秒发送一次）"""
        HEARTBEAT_INTERVAL = 20  # 秒
        
        logger.info("Heartbeat loop started, will send heartbeat every 20 seconds")
        
        # 立即发送第一条heartbeat，然后每20秒发送一次
        first_heartbeat = True
        heartbeat_count = 0
        
        while self._running:
            try:
                if not first_heartbeat:
                    # logger.info(f"Waiting {HEARTBEAT_INTERVAL} seconds before next heartbeat...")
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                    # logger.info(f"Sleep completed, preparing to send heartbeat #{heartbeat_count + 1}")
                else:
                    first_heartbeat = False
                    logger.info("Sending first heartbeat immediately")
                
                heartbeat_count += 1
                # logger.info(f"Preparing heartbeat #{heartbeat_count}, _running={self._running}, _connected={self._connected}")
                
                if not self._running or not self._connected:
                    logger.info(f"Heartbeat loop stopping: _running={self._running}, _connected={self._connected}")
                    break
                
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if self._is_ws_open():
                    try:
                        await self.websocket.send(json.dumps(heartbeat_message, ensure_ascii=False))
                        # logger.info(f"Heartbeat sent: {heartbeat_message['timestamp']}")  # 改为info级别以便调试
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"Heartbeat stopped, connection closed: code={e.code}, reason={e.reason}")
                        await self._notify_disconnect(f"heartbeat connection closed: code={e.code}, reason={e.reason}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to send heartbeat: {e}", exc_info=True)
                        await self._notify_disconnect(f"heartbeat send failed: {e}")
                        break
                else:
                    logger.warning("WebSocket not open, stopping heartbeat loop")
                    await self._notify_disconnect("heartbeat loop detected websocket not open")
                    break
            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}", exc_info=True)
                await self._notify_disconnect(f"heartbeat loop error: {e}")
                break
    
    async def _receive_loop(self):
        """接收消息循环"""
        logger.info("Receive loop started, waiting for messages from server")
        
        last_log_time = datetime.utcnow()
        LOG_INTERVAL = 30  # 每30秒记录一次状态
        
        while self._running:
            try:
                if not self._is_ws_open():
                    logger.warning("WebSocket not open in receive loop, breaking")
                    await self._notify_disconnect("receive loop detected websocket not open")
                    break
                
                # 定期记录receive_loop还在运行
                now = datetime.utcnow()
                if (now - last_log_time).total_seconds() >= LOG_INTERVAL:
                    # logger.info(
                    #     f"Receive loop still running, waiting for messages... "
                    #     f"(websocket state: {self.websocket.state if hasattr(self.websocket, 'state') else 'unknown'})"
                    # )
                    last_log_time = now
                
                # logger.debug("Waiting for message from server...")
                try:
                    # 检查websocket状态
                    # if hasattr(self.websocket, 'state'):
                    #     ws_state = self.websocket.state
                    #     logger.debug(f"WebSocket state before recv: {ws_state}")
                    
                    # 使用asyncio.wait_for添加超时，以便检测连接问题
                    # websockets库的recv()会自动处理文本和二进制消息
                    message_str = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=10.0  # 10秒超时，更快感知连接状态
                    )

                    # 二进制帧：与上一条 JSON 配对（协议约定 text meta 后紧跟 binary PCM）
                    if isinstance(message_str, bytes):
                        logger.warning(
                            "Received orphan binary frame (%d bytes), expected JSON text first; dropping",
                            len(message_str),
                        )
                        continue

                    # 直接输出 WS 收到的原始消息（用于定位 tpl_list 在链路中何处丢失）
                    # try:
                    #     raw_s = message_str if isinstance(message_str, str) else str(message_str)
                    #     max_len = 8000
                    #     if len(raw_s) <= max_len:
                    #         logger.info(f"[RAW_WS_INGRESS] {raw_s}")
                    #     else:
                    #         head = 4000
                    #         tail = 3500
                    #         logger.info(
                    #             f"[RAW_WS_INGRESS] {raw_s[:head]}...[truncated {len(raw_s) - head - tail} chars]...{raw_s[-tail:]}"
                    #         )
                    # except Exception:
                    #     logger.info("[RAW_WS_INGRESS] <unavailable>")
                except asyncio.TimeoutError:
                    # logger.warning("Receive timeout (35s), checking connection state...")
                    # 检查连接状态
                    if hasattr(self.websocket, 'state'):
                        # logger.warning(f"WebSocket state after timeout: {self.websocket.state}")
                        pass
                    # 超时后继续等待，不退出循环
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"WebSocket connection closed during recv: code={e.code}, reason={e.reason}")
                    await self._notify_disconnect(f"recv connection closed: code={e.code}, reason={e.reason}")
                    break
                except Exception as recv_error:
                    logger.error(f"Error receiving message: {recv_error}", exc_info=True)
                    await self._notify_disconnect(f"recv failed: {recv_error}")
                    break
                
                try:
                    message = json.loads(message_str)
                    message_type = message.get("type")
                    # logger.debug(f"Parsed message type: {message_type}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {e}, message: {message_str[:100]}")
                    continue

                # session.asr.chunk：meta 后紧跟一帧 PCM binary（见 chat_ws_protocol / plan）
                if str(message_type) == "session.asr.chunk":
                    try:
                        need = int(message.get("bytes_len") or 0)
                    except (TypeError, ValueError):
                        need = 0
                    if need > 0:
                        bin_payload = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
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
                        logger.info(f"[WS_TASK_PARAMS] task_id={task_id} tpl_list={tpl!r}")
                    except Exception:
                        logger.info(f"[WS_TASK_PARAMS] task_id={task_id} tpl_list=<unavailable>")
                    self.message_queue.put_message(msg_to_queue)
                    
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

                elif message_type == "ping":
                    # 服务器发送ping，回复pong
                    # logger.info("Received ping from server, sending pong")  # 改为info级别以便调试
                    pong_message = {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    try:
                        await self.websocket.send(json.dumps(pong_message, ensure_ascii=False))
                    except Exception as e:
                        logger.warning(f"Failed to send pong: {e}")
                        await self._notify_disconnect(f"pong send failed: {e}")
                        break
                
                elif message_type == "pong":
                    # 服务器响应应用层心跳；收到即可，避免刷 Unknown message type 警告。
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
                logger.warning(f"WebSocket connection closed: code={e.code}, reason={e.reason}")
                await self._notify_disconnect(f"receive loop closed: code={e.code}, reason={e.reason}")
                break
            except asyncio.CancelledError:
                logger.info("Receive loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}", exc_info=True)
                await self._notify_disconnect(f"receive loop error: {e}")
                break

    async def _watchdog_loop(self):
        """定期输出连接状态，便于诊断"""
        LOG_INTERVAL = 5  # 秒
        while self._running:
            try:
                await asyncio.sleep(LOG_INTERVAL)
                ws_state = getattr(self.websocket, "state", "unknown")

                if not self._connected or ws_state != WsState.OPEN:
                    # 触发自动重连，避免长时间任务导致底层连接被超时关闭
                    async with self._reconnect_lock:
                        backoff = min(2 ** self._reconnect_attempts, self._max_reconnect_interval)
                        self._reconnect_attempts += 1
                        logger.warning(
                            f"Watchdog detected websocket not open (state={ws_state}), "
                            f"attempting reconnect after {backoff}s"
                        )
                        await asyncio.sleep(backoff)
                        try:
                            # 尝试重新建立 websocket
                            ws_endpoint = f"{self.ws_url}/ws/inference/{self.service_id}"
                            self.websocket = await websockets.connect(
                                ws_endpoint,
                                ping_interval=None,
                                ping_timeout=None,
                                close_timeout=10,
                            )
                            self._connected = True
                            self._last_disconnect_reason = ""
                            logger.info("WebSocket reconnected successfully")
                            self._reconnect_attempts = 0
                            await self._restart_io_tasks()
                            if self._on_reconnect:
                                try:
                                    await self._on_reconnect()
                                except Exception as cb_e:
                                    logger.error(f"on_reconnect callback failed: {cb_e}", exc_info=True)
                            continue
                        except Exception as e:
                            # 连接拒绝等常见错误无需异常栈，按重试策略继续
                            logger.warning(f"WebSocket reconnect failed: {e}")
                            await self._notify_disconnect(f"watchdog reconnect failed: {e}")
                            continue
                else:
                    # 正常时重置重连计数
                    self._reconnect_attempts = 0
            except asyncio.CancelledError:
                logger.info("Watchdog loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}", exc_info=True)
                break
    
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
            await self._notify_disconnect(f"send_result closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send result message: {e}", exc_info=True)
            # 发送失败时，也写入状态结果缓存文件
            if self.message_cache:
                task_id = result_message.get('task_id')
                status = result_message.get('status', 'failed')
                await self.message_cache.save_status_result(task_id, status, result_message)
            await self._notify_disconnect(f"send_result failed: {e}")
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
            await self._notify_disconnect(f"send_task_status closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send task status: {e}", exc_info=True)
            # 发送失败时也写缓存；但 completed 终态仍跳过（由 send_result 落盘）
            if status != "completed":
                if self.message_cache:
                    await self.message_cache.save_status_result(task_id, status, status_message)
            await self._notify_disconnect(f"send_task_status failed: {e}")
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
            await self._notify_disconnect(f"send_stream_event closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send stream event: {e}", exc_info=True)
            await self._notify_disconnect(f"send_stream_event failed: {e}")
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
            await self._notify_disconnect(f"send_message closed: code={e.code}, reason={e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
            await self._notify_disconnect(f"send_message failed: {e}")
            return False


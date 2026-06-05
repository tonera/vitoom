"""
WebSocket连接管理器
管理WebSocket连接，提供任务进度推送功能
支持用户前端连接和推理器连接
"""
import json
import asyncio
from copy import deepcopy
from typing import Dict, Set, Optional, Tuple, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.core.logger import get_app_logger
from backend.utils import utc_now
from backend.services.chat.dispatch_feedback import build_dispatch_unavailable_feedback
from backend.i18n.ws_messages import enrich_task_ws_message

logger = get_app_logger(__name__)


def _resolve_requested_load_name(task_dict: Dict[str, Any]) -> str:
    params = task_dict.get("params") if isinstance(task_dict.get("params"), dict) else {}
    model = task_dict.get("model") if isinstance(task_dict.get("model"), dict) else {}
    candidates = [
        model.get("load_name"),
        params.get("load_name"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _resolve_audio_dispatch_capability(task_dict: Dict[str, Any]) -> str:
    """Infer audio sub-capability for task dispatch.

    Empty load_name audio tasks are routed to pinned audio services. Without this
    capability hint, TTS and ASR pinned services are indistinguishable to dispatch.
    """
    task_type = str(task_dict.get("type") or "").strip().lower()
    if task_type != "audio":
        return ""
    params = task_dict.get("params") if isinstance(task_dict.get("params"), dict) else {}
    audio_mode = str(params.get("audio_mode") or "").strip().lower()
    job_type = str(params.get("job_type") or task_dict.get("job_type") or "").strip().upper()
    if audio_mode == "asr" or job_type == "ASR":
        return "asr"
    if audio_mode in {"tts", "realtime_tts"} or job_type in {"TTS", "REALTIME_TTS", "RTT", "RTTS"}:
        return "tts"
    return ""


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        """初始化WebSocket管理器"""
        # 用户前端连接：任务ID到WebSocket连接的映射
        # {task_id: Set[WebSocket]}
        self._task_connections: Dict[str, Set[WebSocket]] = {}
        
        # 用户前端连接：WebSocket到任务ID和用户ID的映射（用于清理）
        # {WebSocket: (task_id, user_id)}
        self._connection_tasks: Dict[WebSocket, Tuple[str, str]] = {}

        # 用户前端连接：会话ID到 WebSocket 连接的映射
        # {session_id: Set[WebSocket]}
        self._session_connections: Dict[str, Set[WebSocket]] = {}

        # 用户前端连接：WebSocket 到会话ID和用户ID的映射（用于清理）
        # {WebSocket: (session_id, user_id)}
        self._connection_sessions: Dict[WebSocket, Tuple[str, str]] = {}

        # 后端内部订阅：session_id 到 asyncio.Queue 集合
        self._session_subscribers: Dict[str, Set[asyncio.Queue]] = {}

        # 后端内部订阅：task_id 到 asyncio.Queue 集合（用于 Agent 工具等同步代码
        # 通过 `register_task_subscriber + run_coroutine_threadsafe` 等推理器回调）
        self._task_subscribers: Dict[str, Set[asyncio.Queue]] = {}

        # 运行时事件循环引用：由 FastAPI 启动钩子注入，供不在 event loop
        # 里运行的代码（例如 CrewAI 线程里的工具实现）使用
        # `asyncio.run_coroutine_threadsafe(coro, loop)` 访问本管理器。
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 推理器连接：服务ID到WebSocket连接的映射
        # {service_id: WebSocket}
        self._inference_connections: Dict[str, WebSocket] = {}
        
        # 推理器连接：WebSocket到服务ID的映射（用于清理）
        # {WebSocket: service_id}
        self._connection_services: Dict[WebSocket, str] = {}

        # 任务派发记录：task_id -> service_id
        self._task_dispatch_services: Dict[str, str] = {}

        # 模型下载订阅：model_key 到 WebSocket 连接集合
        # {model_key: Set[WebSocket]}
        self._model_connections: Dict[str, Set[WebSocket]] = {}
        # 模型下载订阅：WebSocket 到 (model_key, user_id)（用于清理）
        # {WebSocket: (model_key, user_id)}
        self._connection_models: Dict[WebSocket, Tuple[str, str]] = {}
        
        # 连接锁
        self._lock = asyncio.Lock()
        
        logger.info("WebSocketManager initialized")

    async def notify_inference_services_changed(
        self,
        *,
        service_id: str = "",
        reason: str = "updated",
    ) -> None:
        """向所有活动 chat session runtime 广播推理服务拓扑变更。"""
        event = {
            "type": "inference.services_changed",
            "payload": {
                "service_id": str(service_id or "").strip(),
                "reason": str(reason or "updated").strip() or "updated",
            },
        }
        async with self._lock:
            subscribers_by_session = {
                session_id: queues.copy()
                for session_id, queues in self._session_subscribers.items()
                if queues
            }

        for session_id, subscribers in subscribers_by_session.items():
            for queue in subscribers:
                try:
                    msg_copy = deepcopy(event)
                    msg_copy["session_id"] = session_id
                    queue.put_nowait(msg_copy)
                except Exception as e:
                    logger.warning(f"Failed to notify session subscriber about inference service change: {e}")
    
    async def connect_user(self, websocket: WebSocket, task_id: str, user_id: str):
        """
        建立用户前端WebSocket连接
        
        Args:
            websocket: WebSocket连接
            task_id: 任务ID
            user_id: 用户ID
        """
        await websocket.accept()
        
        async with self._lock:
            # 添加到任务连接集合
            if task_id not in self._task_connections:
                self._task_connections[task_id] = set()
            self._task_connections[task_id].add(websocket)
            
            # 记录连接对应的任务和用户
            self._connection_tasks[websocket] = (task_id, user_id)
        
        logger.info(f"User WebSocket connected for task: {task_id} (user: {user_id})")

    async def connect_session(self, websocket: WebSocket, session_id: str, user_id: str):
        """
        建立用户前端会话 WebSocket 连接。
        """
        await websocket.accept()

        async with self._lock:
            if session_id not in self._session_connections:
                self._session_connections[session_id] = set()
            self._session_connections[session_id].add(websocket)
            self._connection_sessions[websocket] = (session_id, user_id)

        logger.info(f"User WebSocket connected for session: {session_id} (user: {user_id})")
    
    async def connect_inference_service(self, websocket: WebSocket, service_id: str):
        """
        建立推理器WebSocket连接
        
        Args:
            websocket: WebSocket连接
            service_id: 推理服务ID
        """
        await websocket.accept()
        
        async with self._lock:
            # 如果该服务已有连接，先断开旧连接
            if service_id in self._inference_connections:
                old_ws = self._inference_connections[service_id]
                try:
                    await old_ws.close()
                except:
                    pass
                self._connection_services.pop(old_ws, None)
            
            # 添加新连接
            self._inference_connections[service_id] = websocket
            self._connection_services[websocket] = service_id
        
        logger.info(f"Inference service WebSocket connected: {service_id}")
        await self.notify_inference_services_changed(service_id=service_id, reason="connected")

    async def register_session_subscriber(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if session_id not in self._session_subscribers:
                self._session_subscribers[session_id] = set()
            self._session_subscribers[session_id].add(queue)
        return queue

    async def unregister_session_subscriber(self, session_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subscribers = self._session_subscribers.get(session_id)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                del self._session_subscribers[session_id]

    async def register_task_subscriber(self, task_id: str) -> asyncio.Queue:
        """为指定 task_id 注册一个进程内订阅队列。

        用途：Agent 工具等同步代码希望监听推理器回传的 `task_status` / `result`
        消息时，可以通过该队列按顺序获得每一条消息（与前端 WS 转发同源）。
        """
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if task_id not in self._task_subscribers:
                self._task_subscribers[task_id] = set()
            self._task_subscribers[task_id].add(queue)
        return queue

    async def clear_task_dispatch_service(self, task_id: str) -> None:
        async with self._lock:
            self._task_dispatch_services.pop(str(task_id or "").strip(), None)

    async def unregister_task_subscriber(self, task_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subscribers = self._task_subscribers.get(task_id)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                del self._task_subscribers[task_id]

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """由 FastAPI startup 钩子调用，注入当前运行事件循环。"""
        self._event_loop = loop

    def get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._event_loop
    
    async def disconnect(self, websocket: WebSocket):
        """
        断开WebSocket连接（用户前端或推理器）
        
        Args:
            websocket: WebSocket连接
        """
        disconnected_inference_service_id: Optional[str] = None

        async with self._lock:
            # 检查是否是用户前端连接
            if websocket in self._connection_tasks:
                task_id, user_id = self._connection_tasks.pop(websocket)
                
                # 从任务连接集合中移除
                if task_id in self._task_connections:
                    self._task_connections[task_id].discard(websocket)
                    
                    # 如果该任务没有连接了，删除任务键
                    if not self._task_connections[task_id]:
                        del self._task_connections[task_id]
                
                logger.info(f"User WebSocket disconnected for task: {task_id} (user: {user_id})")

            # 检查是否是会话连接
            elif websocket in self._connection_sessions:
                session_id, user_id = self._connection_sessions.pop(websocket)

                if session_id in self._session_connections:
                    self._session_connections[session_id].discard(websocket)
                    if not self._session_connections[session_id]:
                        del self._session_connections[session_id]

                logger.info(f"User WebSocket disconnected for session: {session_id} (user: {user_id})")
            
            # 检查是否是推理器连接
            elif websocket in self._connection_services:
                service_id = self._connection_services.pop(websocket)
                self._inference_connections.pop(service_id, None)
                stale_task_ids = [
                    task_id for task_id, dispatched_service_id in self._task_dispatch_services.items()
                    if dispatched_service_id == service_id
                ]
                for task_id in stale_task_ids:
                    self._task_dispatch_services.pop(task_id, None)
                logger.info(f"Inference service WebSocket disconnected: {service_id}")
                disconnected_inference_service_id = service_id

            # 检查是否是模型下载订阅连接
            elif websocket in self._connection_models:
                model_key, user_id = self._connection_models.pop(websocket)
                if model_key in self._model_connections:
                    self._model_connections[model_key].discard(websocket)
                    if not self._model_connections[model_key]:
                        del self._model_connections[model_key]
                logger.info(f"Model WebSocket disconnected: model_key={model_key} (user: {user_id})")

        if disconnected_inference_service_id:
            await self.notify_inference_services_changed(
                service_id=disconnected_inference_service_id,
                reason="disconnected",
            )

    async def connect_model(self, websocket: WebSocket, model_key: str, user_id: str):
        """
        建立模型下载订阅 WebSocket 连接（前端使用）
        """
        await websocket.accept()
        async with self._lock:
            if model_key not in self._model_connections:
                self._model_connections[model_key] = set()
            self._model_connections[model_key].add(websocket)
            self._connection_models[websocket] = (model_key, user_id)
        logger.info(f"Model WebSocket connected: model_key={model_key} (user: {user_id})")

    async def forward_model_message(self, model_key: str, message: Dict[str, Any]):
        """
        转发模型下载相关消息到订阅该 model_key 的前端连接
        """
        message_json = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            connections = self._model_connections.get(model_key, set()).copy()

        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to forward message to model WebSocket: {e}")
                disconnected.add(websocket)

        if disconnected:
            for websocket in disconnected:
                await self.disconnect(websocket)

    async def broadcast_to_download_services(self, message: Dict[str, Any]) -> int:
        """
        广播消息给所有运行中的 download 服务（status=running && service_type=download）。

        返回成功发送的连接数。
        """
        from backend.database import InferenceService

        all_services = InferenceService.list_all() or []
        matching = [
            s for s in all_services
            if s.get("status") == "running" and str(s.get("service_type") or "").strip().lower() == "download"
        ]

        message_json = json.dumps(message, ensure_ascii=False)
        sent = 0

        # 注意：避免在持有 self._lock 时 await send_text()/disconnect()
        async with self._lock:
            targets = [self._inference_connections.get(str(s.get("id"))) for s in matching]

        disconnected = set()
        for websocket in targets:
            if not websocket:
                continue
            try:
                await websocket.send_text(message_json)
                sent += 1
            except Exception as e:
                logger.warning(f"Failed to broadcast to download service: {e}")
                disconnected.add(websocket)

        for websocket in disconnected:
            try:
                await self.disconnect(websocket)
            except Exception:
                pass

        return sent
    
    async def send_progress(
        self,
        task_id: str,
        progress: int,
        message: Optional[str] = None,
        status: Optional[str] = None
    ):
        """
        发送任务进度更新
        
        Args:
            task_id: 任务ID
            progress: 进度（0-100）
            message: 进度消息（可选）
            status: 任务状态（可选）
        """
        # 构建消息
        data = {
            "task_id": task_id,
            "progress": max(0, min(100, progress)),
            "timestamp": utc_now().isoformat()
        }
        
        if message:
            data["message"] = message
        if status:
            data["status"] = status
        
        message_json = json.dumps(data, ensure_ascii=False)
        
        # 获取该任务的所有连接
        async with self._lock:
            connections = self._task_connections.get(task_id, set()).copy()
        
        # 发送消息到所有连接
        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        if disconnected:
            for websocket in disconnected:
                await self.disconnect(websocket)
        
        if connections:
            logger.debug(f"Progress sent to {len(connections)} connections for task: {task_id}")
    
    async def send_task_update(
        self,
        task_id: str,
        status: str,
        message: Optional[str] = None,
        result: Optional[str] = None,  # 已废弃，保留以保持向后兼容
        error: Optional[str] = None
    ):
        """
        发送任务状态更新
        
        Args:
            task_id: 任务ID
            status: 任务状态
            message: 消息（可选）
            result: 结果（已废弃，不再使用，保留以保持向后兼容）
            error: 错误信息（可选）
        
        注意：result字段已废弃，任务结果文件应通过files表查询
        """
        # 构建消息
        data = {
            "type": "task_status",
            "task_id": task_id,
            "status": status,
            "timestamp": utc_now().isoformat()
        }
        
        if message:
            data["message"] = message
        # result字段已废弃，不再添加到消息中
        # 如果需要结果文件，客户端应通过files表查询
        if error:
            data["error"] = error
        
        # 根据状态设置进度
        if status == "completed":
            data["progress"] = 100
        elif status == "failed":
            data["progress"] = 0

        data = enrich_task_ws_message(data)
        
        message_json = json.dumps(data, ensure_ascii=False)
        
        # 获取该任务的所有连接
        async with self._lock:
            connections = self._task_connections.get(task_id, set()).copy()
        
        # 发送消息到所有连接
        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        if disconnected:
            for websocket in disconnected:
                await self.disconnect(websocket)
        
        if connections:
            logger.info(f"Task update sent to {len(connections)} connections for task: {task_id}")
    
    async def get_connection_count(self, task_id: str) -> int:
        """
        获取任务的连接数
        
        Args:
            task_id: 任务ID
        
        Returns:
            连接数
        """
        async with self._lock:
            return len(self._task_connections.get(task_id, set()))
    
    async def get_total_connections(self) -> int:
        """
        获取总连接数（用户前端连接数）
        
        Returns:
            总连接数
        """
        async with self._lock:
            return len(self._connection_tasks) + len(self._connection_sessions)

    async def get_session_connection_count(self, session_id: str) -> int:
        """
        获取会话的连接数。
        """
        async with self._lock:
            return len(self._session_connections.get(session_id, set()))
    
    async def forward_inference_message(self, task_id: str, message: Dict[str, Any]):
        """
        转发推理器消息到用户前端以及后端内部订阅者。

        Args:
            task_id: 任务ID
            message: 消息字典（JSON格式）

        注意：此方法由推理器 WS 路由调用，除了将消息转发给所有监听该任务的用户前端
        连接外，还会把消息副本送给所有通过 `register_task_subscriber` 注册的
        进程内订阅队列（供 Agent 工具等同步代码等待）。
        """
        if message.get("type") == "task_status" or message.get("status"):
            message = enrich_task_ws_message(message)

        message_json = json.dumps(message, ensure_ascii=False)
        
        # 获取该任务的所有用户前端连接 + 进程内订阅者
        async with self._lock:
            connections = self._task_connections.get(task_id, set()).copy()
            subscribers = self._task_subscribers.get(task_id, set()).copy()
        
        # 发送消息到所有连接
        disconnected = set()
        for websocket in connections:
            if (
                getattr(websocket, "application_state", None) != WebSocketState.CONNECTED
                or getattr(websocket, "client_state", None) != WebSocketState.CONNECTED
            ):
                disconnected.add(websocket)
                continue
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to forward message to user WebSocket: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        if disconnected:
            for websocket in disconnected:
                await self.disconnect(websocket)
        
        if connections:
            logger.debug(f"Inference message forwarded to {len(connections)} user connections for task: {task_id}")

        # 进程内订阅者（例如 Agent 工具）按顺序收到每条消息副本
        for queue in subscribers:
            try:
                queue.put_nowait(deepcopy(message))
            except Exception as e:
                logger.warning(f"Failed to deliver task message to in-process subscriber: {e}")

    async def forward_session_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        binary: Optional[bytes] = None,
    ):
        """
        转发会话消息到用户前端。

        ``binary`` 非空时：在 JSON text 之后对同一批连接再发送一帧 binary
       （与 ``audio_delta`` 等协议配对）。
        """
        message_json = json.dumps(message, ensure_ascii=False)

        async with self._lock:
            connections = self._session_connections.get(session_id, set()).copy()
            subscribers = self._session_subscribers.get(session_id, set()).copy()

        disconnected = set()
        for websocket in connections:
            # 客户端已经开始/完成关闭握手后再 send_text 会触发 Starlette 的
            # "Unexpected ASGI message 'websocket.send', after sending
            # 'websocket.close'" 报错。这里先看状态：只要 application 或
            # client 任一端不是 CONNECTED，就把连接当作已断开、丢弃本条消息。
            if (
                getattr(websocket, "application_state", None) != WebSocketState.CONNECTED
                or getattr(websocket, "client_state", None) != WebSocketState.CONNECTED
            ):
                disconnected.add(websocket)
                continue
            try:
                await websocket.send_text(message_json)
                if binary:
                    await websocket.send_bytes(binary)
            except Exception as e:
                logger.warning(f"Failed to forward message to session WebSocket: {e}")
                disconnected.add(websocket)

        if disconnected:
            for websocket in disconnected:
                await self.disconnect(websocket)

        for queue in subscribers:
            try:
                msg_copy = deepcopy(message)
                if binary is not None:
                    msg_copy["binary_bytes"] = binary
                queue.put_nowait(msg_copy)
            except Exception as e:
                logger.warning(f"Failed to deliver session message to in-process subscriber: {e}")

        if connections:
            logger.debug(
                f"Session message forwarded to {len(connections)} user connections for session: {session_id}"
            )

    async def publish_session_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        binary: Optional[bytes] = None,
    ):
        """
        仅将会话消息投递给进程内订阅者，不直接转发给前端 WebSocket。

        用途：
        - 推理器回流的原始协议事件（如 transcript_partial / llm_text_delta）先进入
          后端编排层做统一协议映射；
        - 前端最终只接收 /ws/chat 的规范化事件，而不是推理侧原始事件名。

        ``binary`` 非空时：在入队 dict 上挂 ``binary_bytes``，供 SessionRuntime 消费。
        """
        async with self._lock:
            subscribers = self._session_subscribers.get(session_id, set()).copy()

        for queue in subscribers:
            try:
                msg_copy = deepcopy(message)
                if binary is not None:
                    msg_copy["binary_bytes"] = binary
                queue.put_nowait(msg_copy)
            except Exception as e:
                logger.warning(f"Failed to publish session message to in-process subscriber: {e}")

    async def send_message_to_inference_service(
        self,
        service_id: str,
        message: Dict[str, Any],
        binary: Optional[bytes] = None,
    ) -> bool:
        """
        向指定推理服务发送消息。
        """
        message_json = json.dumps(message, ensure_ascii=False)
        message_type = str(message.get("type") or "").strip()
        session_id = str(message.get("session_id") or "").strip()

        async with self._lock:
            websocket = self._inference_connections.get(service_id)

        if not websocket:
            logger.warning(f"Inference service {service_id} is not connected via WebSocket")
            return False

        try:
            await websocket.send_text(message_json)
            if binary:
                await websocket.send_bytes(binary)
            logger.info(
                "Sent inference message type=%s service_id=%s session_id=%s",
                message_type or "<empty>",
                service_id,
                session_id or "<empty>",
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to inference service {service_id}: {e}")
            try:
                await self.disconnect(websocket)
            except Exception:
                pass
            return False

    async def get_connected_inference_service_ids(self) -> set[str]:
        """返回当前已通过 WebSocket 建立连接的推理服务 ID 集合。"""
        async with self._lock:
            return {str(service_id) for service_id in self._inference_connections.keys() if str(service_id)}

    async def _mark_task_failed_and_notify(
        self,
        task_id: str,
        reason: str,
        *,
        display_reason: Optional[str] = None,
        message_code: Optional[str] = None,
        message_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        将任务标记为失败，并向监听该 task_id 的前端 ws 推送失败状态。

        说明：
        - 会尽量更新 tasks 表中的状态（失败不应悄无声息地停留在 pending）
        - 推送消息使用 task_status 协议，并附带 error 字段便于前端展示原因
        """
        started_at = utc_now().isoformat()
        timestamp = utc_now().isoformat()
        user_reason = str(display_reason or reason or "").strip() or "任务处理失败"

        # 1) 更新数据库任务状态为 failed（便于后续用户重新连接 ws 时拿到最终状态）
        try:
            from backend.database import Task

            Task.update(
                task_id,
                status="failed",
                started_at=started_at,
                completed_at=timestamp,
                progress=0,
                error=user_reason,
            )
        except Exception as e:
            logger.error(
                f"Failed to update task status to failed: task_id={task_id}, error={e}",
                exc_info=True,
            )

        # 2) 给监听该 task_id 的前端连接推送失败消息（格式按协议 task_status）
        await self.forward_inference_message(
            task_id,
            {
                "type": "task_status",
                "task_id": task_id,
                "status": "failed",
                "timestamp": timestamp,
                "started_at": started_at,
                "message": user_reason,
                "error": user_reason,
                **({"message_code": message_code} if message_code else {}),
                **({"message_params": message_params} if message_params else {}),
            },
        )
    
    async def send_task_to_inference_service(self, task_id: str, task_type: str) -> bool:
        """
        向推理器发送任务（全量数据包，包含任务和模型信息）
        
        Args:
            task_id: 任务ID
            task_type: 任务类型（image/video/audio/text）
        
        注意：此方法由API后端调用，当创建本地模型任务时，向对应的推理器发送任务
        现在发送全量任务数据和模型信息，推理器不再需要查询数据库
        """
        from backend.database import Task, Model
        from backend.services.chat.router import (
            DispatchSelectionError,
            DispatchSpec,
            get_dispatch_router,
        )
        
        # 从数据库获取全量任务数据
        task_dict = Task.get_by_id(task_id)
        if not task_dict:
            logger.error(f"Task not found: {task_id}")
            return False
        # 使用任务创建时写入的 storage（local | server | s3 | oss），由推理 ResultHandler 分流落盘。
        raw_storage = str(task_dict.get("storage") or "").strip().lower()
        if not raw_storage:
            from backend.core.config import get_config

            raw_storage = str(get_config("storage.default", "server") or "server").strip().lower()
        task_dict["storage"] = raw_storage

        # 如果任务有 model_key，获取模型信息并添加到 task_data 中
        model_key = task_dict.get("model_key")
        if model_key:
            model_dict = Model.get_by_model_key(model_key)
            if model_dict:
                explicit_fps = task_dict.get("params", {}).get("fps")
                merged_model_dict = deepcopy(model_dict)
                base_runtime_config = merged_model_dict.get("runtime_config")
                if isinstance(base_runtime_config, dict):
                    merged_runtime_config = dict(base_runtime_config)
                else:
                    merged_runtime_config = {}
                if explicit_fps is not None:
                    try:
                        fps_value = int(explicit_fps)
                    except Exception:
                        fps_value = None
                    if fps_value is not None:
                        merged_runtime_config["fps"] = fps_value
                merged_model_dict["runtime_config"] = merged_runtime_config
                # 将模型信息添加到任务数据中
                task_dict["model"] = merged_model_dict
                logger.debug(f"Model info included in task data: {model_key}")
            else:
                logger.warning(f"Model not found: {model_key}, task will be sent without model info")

        requested_load_name = _resolve_requested_load_name(task_dict)
        dispatch_capability = _resolve_audio_dispatch_capability(task_dict)
        
        connected_service_ids = await self.get_connected_inference_service_ids()
        dispatch_spec = DispatchSpec(
            service_type=str(task_type or "").strip().lower(),
            require_supports_task=True,
            reason=f"task_type={task_type}",
            load_name=requested_load_name,
            capability=dispatch_capability,
        )
        try:
            selected_service = get_dispatch_router().pick_service(
                dispatch_spec,
                connected_service_ids=connected_service_ids,
            )
        except DispatchSelectionError as exc:
            reason = str(exc)
            feedback = build_dispatch_unavailable_feedback(
                service_type=str(task_type or ""),
                load_name=requested_load_name,
                capability=dispatch_capability,
            )
            logger.warning(f"{reason}, task_id={task_id}. Marking task as failed and notifying ws.")
            await self._mark_task_failed_and_notify(
                task_id,
                reason,
                message_code=str(feedback["message_code"]),
                message_params=dict(feedback.get("message_params") or {}),
            )
            return False
        service_id = selected_service["id"]
        
        # 构建任务消息（全量数据包）
        task_message = {
            "type": "task",
            "task_id": task_id,
            "task_data": task_dict  # 全量任务数据
        }
        message_json = json.dumps(task_message, ensure_ascii=False)
        
        # 发送任务到选中的推理服务
        # 注意：避免在持有 self._lock 时 await send_text()/disconnect()/forward_inference_message()，防止死锁
        async with self._lock:
            websocket = self._inference_connections.get(service_id)

        if not websocket:
            reason = f"Inference service {service_id} is not connected via WebSocket"
            logger.warning(f"{reason}, task_id={task_id}. Marking task as failed and notifying ws.")
            await self._mark_task_failed_and_notify(
                task_id,
                reason,
                message_code="inference.serviceNotRunning",
            )
            return False

        try:
            await websocket.send_text(message_json)
            async with self._lock:
                self._task_dispatch_services[task_id] = service_id
            logger.info(
                f"Task {task_id} sent to inference service: {service_id} "
                f"(type: {task_type}, load_name: {requested_load_name or '<empty>'}, "
                f"capability: {dispatch_capability or '<any>'}, selected by dispatch router, "
                f"full data packet)"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to send task to inference service {service_id}: {e}")
            # 尽量断开无效连接
            try:
                await self.disconnect(websocket)
            except Exception:
                pass

            # 发送失败状态到前端（视为推理器不可用）
            reason = f"Failed to send task to inference service {service_id}: {e}"
            await self._mark_task_failed_and_notify(
                task_id,
                reason,
                message_code="inference.serviceConnectionFailed",
            )
            return False

    async def send_cancel_signal_to_inference_service(self, task_id: str):
        """
        向推理器发送中断信号
        
        Args:
            task_id: 任务ID
        
        注意：此方法由API后端调用，当用户取消任务时，向推理器发送中断信号
        """
        # 根据task_id查找对应的推理服务
        from backend.database import Task
        task_dict = Task.get_by_id(task_id)
        
        if not task_dict:
            logger.warning(f"Task not found: {task_id}")
            return
        
        task_type = task_dict.get("type")
        requested_load_name = _resolve_requested_load_name(task_dict)
        
        from backend.services.chat.router import DispatchSpec, get_dispatch_router

        async with self._lock:
            dispatched_service_id = self._task_dispatch_services.get(task_id)

        matching_services = []
        if dispatched_service_id:
            async with self._lock:
                dispatched_websocket = self._inference_connections.get(dispatched_service_id)
            if dispatched_websocket is not None:
                matching_services = [{"id": dispatched_service_id}]

        if not matching_services:
            connected_service_ids = await self.get_connected_inference_service_ids()
            matching_services = get_dispatch_router().list_services(
                DispatchSpec(
                    service_type=str(task_type or "").strip().lower(),
                    require_supports_task=True,
                    reason=f"task_type={task_type}",
                    load_name=requested_load_name,
                ),
                connected_service_ids=connected_service_ids,
            )

        if not matching_services:
            logger.warning(
                f"No running inference service found for task_type={task_type}, "
                f"load_name={requested_load_name or '<empty>'}, "
                f"task_id={task_id}"
            )
            return
        
        # 构建中断消息
        cancel_message = {
            "type": "cancel",
            "task_id": task_id,
            "timestamp": utc_now().isoformat()
        }
        message_json = json.dumps(cancel_message, ensure_ascii=False)
        
        # 发送中断信号到所有匹配的推理服务
        async with self._lock:
            disconnected = set()
            sent_count = 0
            for service in matching_services:
                service_id = service["id"]
                websocket = self._inference_connections.get(service_id)
                if websocket:
                    try:
                        await websocket.send_text(message_json)
                        logger.info(f"Cancel signal sent to inference service: {service_id} for task: {task_id}")
                        sent_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to send cancel signal to inference service {service_id}: {e}")
                        disconnected.add(websocket)
            
            if sent_count > 0:
                logger.info(f"Cancel signal sent to {sent_count} inference service(s) for task: {task_id}")
            
            # 清理断开的连接
            for websocket in disconnected:
                await self.disconnect(websocket)


# 全局WebSocket管理器实例
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """
    获取全局WebSocket管理器实例（单例模式）
    
    Returns:
        WebSocketManager实例
    """
    global _websocket_manager
    
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    
    return _websocket_manager


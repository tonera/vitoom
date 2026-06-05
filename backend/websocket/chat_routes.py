"""``/ws/chat/{session_id}`` 统一会话实时通道。

对应重构计划：

    - 协议文档：``backend/websocket/chat_ws_protocol.md``
    - 状态机实现：``backend/services/chat/session.py``
    - 实施清单：``.cursor/plans/统一会话重构_实施与切换清单_63361fab.md``

职责：

    - WS 连接握手 + JWT 鉴权 + 会话归属校验；
    - 构造 ``SessionRuntime``（状态机宿主）并注入 ``MasterAgentRuntime.run``；
    - 循环接收客户端消息并委派给 ``SessionRuntime.on_client_message``；
    - 通过 ``WebSocketManager.forward_session_message`` 把服务端事件
      送给当前 WS 以及所有通过 ``register_session_subscriber`` 订阅该
      ``session_id`` 的进程内消费者（主要是 MasterAgentRuntime 自己）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.config import get_config
from backend.core.logger import get_app_logger
from backend.database import Conversation
from backend.services.chat import InputMode, MasterAgentRuntime, SessionRuntime
from backend.services.chat.inference_session import InferenceSessionManager
from backend.services.chat.router import get_dispatch_router, get_load_name_router
from backend.websocket.manager import get_websocket_manager

logger = get_app_logger(__name__)

router = APIRouter()


def _resolve_user_id(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    try:
        from backend.auth.jwt_utils import verify_token

        payload = verify_token(token)
        return payload.get("sub")
    except Exception as exc:
        logger.warning("chat ws token verification failed: %s", exc)
        return None


def _default_load_name_from_conversation(conv: Dict[str, Any]) -> str:
    meta = conv.get("metadata") if isinstance(conv.get("metadata"), dict) else {}
    if isinstance(meta, dict):
        load_name = str(meta.get("load_name") or "").strip()
        if load_name:
            return load_name
    load_name = str(get_config("agents.default_model", "") or "").strip()
    return load_name or "Qwen3.5-35B-A3B-GPTQ-Int4"


def _input_mode_from_conversation(conv: Dict[str, Any]) -> str:
    meta = conv.get("metadata") if isinstance(conv.get("metadata"), dict) else {}
    if isinstance(meta, dict):
        value = str(meta.get("input_mode") or "").strip()
        if value:
            return value
    return InputMode.TEXT


def _output_mode_from_conversation(conv: Dict[str, Any]) -> str:
    meta = conv.get("metadata") if isinstance(conv.get("metadata"), dict) else {}
    if isinstance(meta, dict):
        value = str(meta.get("output_mode") or "").strip()
        if value:
            return value
    return "text_stream"


def _conversation_metadata(conv: Dict[str, Any]) -> Dict[str, Any]:
    meta = conv.get("metadata") if isinstance(conv.get("metadata"), dict) else {}
    return dict(meta or {})


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat_session(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = None,
    locale: Optional[str] = None,
):
    """统一 chat WS 入口。"""
    manager = get_websocket_manager()
    subscriber_queue = await manager.register_session_subscriber(session_id)

    conv = Conversation.get_by_id(session_id)
    if not conv:
        await websocket.close(code=1008, reason="Session not found")
        return

    user_id = _resolve_user_id(token)
    if not user_id:
        await websocket.close(code=1008, reason="Token required or invalid")
        return
    if conv.get("user_id") != user_id:
        await websocket.close(code=1008, reason="Permission denied")
        return

    # 连接注册到 WebSocketManager（复用既有 session 连接表：它既负责把
    # 后端 forward_session_message 广播给前端，又负责进程内订阅分发）
    await manager.connect_session(websocket, session_id, str(user_id))
    conversation_metadata = _conversation_metadata(conv)

    async def _emit(event: Dict[str, Any], *, binary: Optional[bytes] = None) -> None:
        # 统一走 forward_session_message：前端 WS 与推理器回流共享同一条管道；
        # 这样 MasterAgentRuntime 里注册 subscriber 时，既能收到自己下发的事件，
        # 也能收到推理侧经由 manager.forward_session_message 广播的事件。
        await manager.forward_session_message(session_id, event, binary=binary)

    async def _send_to_service(
        service_id: str, message: Dict[str, Any], binary: Optional[bytes] = None
    ) -> bool:
        return await manager.send_message_to_inference_service(service_id, message, binary)

    async def _connected_service_ids() -> set:
        return await manager.get_connected_inference_service_ids()

    inference_session = InferenceSessionManager(
        chat_session_id=session_id,
        send_to_service=_send_to_service,
        get_connected_service_ids=_connected_service_ids,
        dispatch_router=get_dispatch_router(),
    )

    master = MasterAgentRuntime(
        ws_manager=manager,
        router=get_load_name_router(),
        default_load_name=_default_load_name_from_conversation(conv),
    )
    runtime = SessionRuntime(
        session_id=session_id,
        user_id=str(user_id),
        emit=_emit,
        master_run=master.run,
        input_mode=_input_mode_from_conversation(conv),
        output_mode=_output_mode_from_conversation(conv),
        agent_id=conv.get("agent_id"),
        metadata=conversation_metadata,
        inference_session=inference_session,
    )

    async def _subscriber_loop() -> None:
        while True:
            event = await subscriber_queue.get()
            etype = str(event.get("type") or "").strip()
            if etype == "inference.services_changed":
                await runtime.on_inference_services_changed(event)
                continue
            # PR2 起，推理侧统一以 ``session.`` 前缀回流实时事件；任何带前缀的
            # 事件都交给 SessionRuntime 翻译成 chat 协议再下发前端。
            if etype.startswith("session."):
                await runtime.on_inference_session_event(event)

    try:
        subscriber_task = asyncio.create_task(_subscriber_loop())
        await runtime.open()
        while True:
            raw_in = await websocket.receive()
            if raw_in.get("type") == "websocket.disconnect":
                break
            if raw_in.get("type") != "websocket.receive" or "text" not in raw_in:
                continue
            raw = raw_in["text"]
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _emit(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "turn_id": None,
                        "run_id": None,
                        "payload": {
                            "code": "invalid_payload",
                            "message": "invalid json",
                            "recoverable": True,
                        },
                    }
                )
                continue
            if not isinstance(message, dict):
                await runtime.on_client_message({})
                if runtime.state == "closed":
                    break
                continue

            mtype = str(message.get("type") or "").strip()
            if mtype == "audio_chunk":
                payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
                try:
                    need = int(payload.get("bytes_len") or 0)
                except (TypeError, ValueError):
                    need = 0
                if need > 0:
                    fr2 = await websocket.receive()
                    if fr2.get("type") != "websocket.receive" or "bytes" not in fr2:
                        await _emit(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "turn_id": None,
                                "run_id": None,
                                "payload": {
                                    "code": "invalid_payload",
                                    "message": "audio_chunk requires binary frame after JSON meta",
                                    "recoverable": True,
                                },
                            }
                        )
                        continue
                    body = bytes(fr2["bytes"])
                    if len(body) != need:
                        logger.warning(
                            "chat ws audio_chunk bytes_len mismatch expected=%s actual=%s session=%s",
                            need,
                            len(body),
                            session_id,
                        )
                    message["binary_bytes"] = body

            await runtime.on_client_message(message)

            if runtime.state == "closed":
                break
    except WebSocketDisconnect:
        logger.info("chat ws disconnected session=%s", session_id)
    except Exception as exc:
        logger.error("chat ws error session=%s: %s", session_id, exc, exc_info=True)
    finally:
        try:
            subscriber_task.cancel()
        except Exception:
            pass
        try:
            await manager.unregister_session_subscriber(session_id, subscriber_queue)
        except Exception:
            pass
        try:
            if runtime.state != "closed":
                await runtime.close(reason="client_requested")
        except Exception:
            pass
        try:
            await manager.disconnect(websocket)
        except Exception:
            pass


__all__ = ["router"]

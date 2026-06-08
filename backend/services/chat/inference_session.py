"""`InferenceSessionManager`：chat session 与多 role inference session 的桥。

每条 ``/ws/chat/{session_id}`` 会创建一个 manager 实例，负责：

- 为每个 role（当前：``asr`` / ``tts``）维护一条独立的 inference session：
  ``<chat_sid>:<role>``
- 发送 ``session.*`` 协议请求（``session.open`` / ``session.close`` /
  ``session.asr.chunk`` / ``session.asr.commit`` / ``session.tts.request`` /
  ``session.tts.cancel``）
- 按 role 绑定/重绑 audio 推理服务（复用 `DispatchRouter`）
- 为 ``submit_tts`` 暴露 per-request_id 等待器（Future），供上层阻塞到对应
  ``session.audio.end`` / ``session.error`` 事件到达

本文件**不解析任何** inference → backend 的回流事件——回流事件的消费由
``SessionRuntime.on_inference_session_event`` 负责（它会在处理完
``session.audio.end`` 后调用 ``resolve_tts_waiter`` 唤醒等待器）。

设计约束：

- ``role`` 只在 chat session 内部区分角色，不进入 ``DispatchSpec``。
  底层仍用 ``service_type="audio"`` + ``load_name`` 选服务；单 audio service
  部署下 ASR / TTS 会绑定到同一个 ``service_id``。
- 所有出站消息统一带 ``session_id=<chat_sid>:<role>`` + ``role=<role>``；
  ``role`` 字段是冗余的，仅便于推理侧日志/路由排障，别把 role 当 session 主键。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from backend.core.logger import get_app_logger
from backend.services.chat.router import (
    DispatchRouter,
    DispatchSelectionError,
    DispatchSpec,
    get_dispatch_router,
)

logger = get_app_logger(__name__)


# ---------------------------------------------------------------------------
# 依赖回调签名
# ---------------------------------------------------------------------------


SendToService = Callable[..., Awaitable[bool]]
"""向指定 service_id 投递 JSON（及可选 binary）消息。通常来自 WebSocketManager。"""

GetConnectedServiceIds = Callable[[], Awaitable[set]]
"""返回当前已连接的 audio inference service id 集合。"""


# ---------------------------------------------------------------------------
# Role spec / 内部状态
# ---------------------------------------------------------------------------


@dataclass
class RoleSpec:
    """open 一个 role 的 inference session 所需的最小描述。"""

    role: str
    load_name: str
    family: str = ""
    runtime_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _RoleState:
    role: str
    spec: RoleSpec
    service_id: Optional[str] = None
    opened: bool = False


# ---------------------------------------------------------------------------
# InferenceSessionManager
# ---------------------------------------------------------------------------


class InferenceSessionManager:
    """为一条 chat session 承载多 role inference session 的协调器。"""

    def __init__(
        self,
        *,
        chat_session_id: str,
        send_to_service: SendToService,
        get_connected_service_ids: GetConnectedServiceIds,
        dispatch_router: Optional[DispatchRouter] = None,
    ) -> None:
        self._chat_session_id = chat_session_id
        self._send_to_service = send_to_service
        self._get_connected_service_ids = get_connected_service_ids
        self._dispatch = dispatch_router or get_dispatch_router()

        self._roles: Dict[str, _RoleState] = {}

        # per-request_id TTS 等待器
        self._tts_waiters: Dict[str, asyncio.Future[None]] = {}
        self._active_tts_request_id: Optional[str] = None

        self._seq = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    def chat_session_id(self) -> str:
        return self._chat_session_id

    @property
    def active_tts_request_id(self) -> Optional[str]:
        return self._active_tts_request_id

    def inference_session_id(self, role: str) -> str:
        return f"{self._chat_session_id}:{role}"

    def is_role_opened(self, role: str) -> bool:
        state = self._roles.get(role)
        return bool(state and state.opened)

    def has_role(self, role: str) -> bool:
        return role in self._roles

    async def probe_role_available(self, role: str, *, load_name: str = "") -> bool:
        role_name = str(role or "").strip().lower()
        if not role_name:
            return False
        connected = await self._get_connected_service_ids()
        try:
            self._dispatch.pick_service(
                DispatchSpec(
                    service_type="audio",
                    reason=f"chat audio capability probe role={role_name}",
                    load_name=str(load_name or "").strip(),
                    capability=role_name,
                ),
                connected_service_ids=connected,
            )
            return True
        except DispatchSelectionError:
            return False

    def invalidate_all_roles(self, *, reason: str = "service topology changed") -> None:
        for role, state in self._roles.items():
            if state.opened or state.service_id:
                logger.info(
                    "invalidate inference role chat_session=%s role=%s reason=%s service_id=%s",
                    self._chat_session_id,
                    role,
                    reason,
                    state.service_id or "<unset>",
                )
            state.opened = False
            state.service_id = None

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def open(self, spec: RoleSpec) -> bool:
        """打开一条 role inference session（幂等：已 open 立即返回 True）。

        ``spec.load_name`` 允许为空串：当音频推理服务声明了 ``fixed_model``
        时，dispatch 会把"空 load_name"任务路由到该 pinned 服务，由推理
        器用 fixed_model / fixed_family 自我纠正。
        """
        role = str(spec.role or "").strip().lower()
        if not role:
            raise ValueError("RoleSpec.role is required")

        existing = self._roles.get(role)
        if existing and existing.opened:
            return True

        state = existing or _RoleState(role=role, spec=spec)
        state.spec = spec
        self._roles[role] = state

        message = {
            "type": "session.open",
            "session_id": self.inference_session_id(role),
            "role": role,
            "seq": self._next_seq(),
            "load_name": spec.load_name,
            "family": spec.family or "",
            "runtime_config": dict(spec.runtime_config or {}),
            "timestamp": _utc_iso(),
        }
        sent = await self._send_to_role(role, message, desired_load_name=spec.load_name)
        state.opened = bool(sent)
        if not sent:
            logger.warning(
                "inference session open failed chat_session=%s role=%s load_name=%s",
                self._chat_session_id,
                role,
                spec.load_name,
            )
        return bool(sent)

    async def close_all(self) -> None:
        """关闭所有已打开的 role session，并拒绝未完成的 TTS 等待器。"""
        for role, state in list(self._roles.items()):
            if not state.opened:
                continue
            try:
                await self._send_to_role(
                    role,
                    {
                        "type": "session.close",
                        "session_id": self.inference_session_id(role),
                        "role": role,
                        "seq": self._next_seq(),
                        "timestamp": _utc_iso(),
                    },
                )
            except Exception as exc:
                logger.debug(
                    "inference session close failed chat_session=%s role=%s err=%s",
                    self._chat_session_id,
                    role,
                    exc,
                )
            state.opened = False
            state.service_id = None

        for request_id, fut in list(self._tts_waiters.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("inference session closed"))
        self._tts_waiters.clear()
        self._active_tts_request_id = None

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    async def asr_chunk(self, pcm_bytes: bytes, seq: Any) -> bool:
        if not pcm_bytes:
            return False
        role = "asr"
        return await self._send_to_role(
            role,
            {
                "type": "session.asr.chunk",
                "session_id": self.inference_session_id(role),
                "role": role,
                "seq": seq,
                "bytes_len": len(pcm_bytes),
                "timestamp": _utc_iso(),
            },
            binary=pcm_bytes,
        )

    async def asr_commit(self, seq: Any = None) -> bool:
        role = "asr"
        return await self._send_to_role(
            role,
            {
                "type": "session.asr.commit",
                "session_id": self.inference_session_id(role),
                "role": role,
                "seq": seq if seq is not None else self._next_seq(),
                "timestamp": _utc_iso(),
            },
        )

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def tts_request(
        self,
        *,
        text: str,
        voice_cfg: Dict[str, Any],
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """发出 ``session.tts.request``。

        - 若已存在活跃 request，先对其发送 ``session.tts.cancel``（barge-in）；
        - 在本地登记该 request_id 的 Future 等待器（由
          ``resolve_tts_waiter`` / ``close_all`` 解除）；
        - 返回最终采用的 request_id。
        """
        role = "tts"
        rid = (request_id or "").strip() or uuid.uuid4().hex

        async with self._lock:
            prev_rid = self._active_tts_request_id
            self._active_tts_request_id = rid
            fut = asyncio.get_event_loop().create_future()
            self._tts_waiters[rid] = fut

        if prev_rid and prev_rid != rid:
            # barge-in：上一条还没来 end，先 cancel
            try:
                await self._send_tts_cancel(prev_rid)
            except Exception:
                pass
            prev_fut = self._tts_waiters.pop(prev_rid, None)
            if prev_fut and not prev_fut.done():
                prev_fut.set_exception(RuntimeError("superseded by new session.tts.request"))

        message = {
            "type": "session.tts.request",
            "session_id": self.inference_session_id(role),
            "role": role,
            "seq": self._next_seq(),
            "request_id": rid,
            "text": str(text or ""),
            "voice_cfg": dict(voice_cfg or {}),
            "metadata": dict(metadata or {}),
            "timestamp": _utc_iso(),
        }
        desired = str((voice_cfg or {}).get("load_name") or "").strip() or None
        sent = await self._send_to_role(role, message, desired_load_name=desired)
        if not sent:
            pending = self._tts_waiters.pop(rid, None)
            if pending and not pending.done():
                pending.set_exception(RuntimeError("failed to send session.tts.request"))
            if self._active_tts_request_id == rid:
                self._active_tts_request_id = None
            raise RuntimeError("failed to send session.tts.request to any audio inference service")
        return rid

    async def tts_cancel(self, request_id: Optional[str] = None) -> bool:
        rid = str(request_id or "").strip() or self._active_tts_request_id
        if not rid:
            return False
        sent = await self._send_tts_cancel(rid)
        # 注意：最终的等待器解除要等 `session.audio.end{cancelled:true}` 到达后在
        # SessionRuntime 里 resolve，这里不提前 set_result，避免抢先唤醒导致
        # chat 协议里 final audio_delta 晚于 message_completed。
        if self._active_tts_request_id == rid:
            self._active_tts_request_id = None
        return sent

    async def _send_tts_cancel(self, request_id: str) -> bool:
        role = "tts"
        return await self._send_to_role(
            role,
            {
                "type": "session.tts.cancel",
                "session_id": self.inference_session_id(role),
                "role": role,
                "seq": self._next_seq(),
                "request_id": request_id,
                "timestamp": _utc_iso(),
            },
        )

    async def await_tts_finish(self, request_id: str, *, timeout: float = 240.0) -> None:
        """阻塞到 ``session.audio.end`` / ``session.error`` 到达。

        超时 / 异常时会自动发 ``session.tts.cancel``，但不会重复发送（幂等）。
        """
        fut = self._tts_waiters.get(request_id)
        if fut is None:
            return
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            try:
                await self._send_tts_cancel(request_id)
            except Exception:
                pass
            if not fut.done():
                fut.set_exception(asyncio.TimeoutError(f"tts request {request_id} timed out after {timeout}s"))
            raise
        finally:
            self._tts_waiters.pop(request_id, None)
            if self._active_tts_request_id == request_id:
                self._active_tts_request_id = None

    def resolve_tts_waiter(
        self,
        request_id: str,
        *,
        error: Optional[str] = None,
    ) -> None:
        """由 ``SessionRuntime`` 在处理完 ``session.audio.end`` 后调用。"""
        fut = self._tts_waiters.get(request_id)
        if fut is None or fut.done():
            return
        if error:
            fut.set_exception(RuntimeError(error))
        else:
            fut.set_result(None)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    async def _send_to_role(
        self,
        role: str,
        message: Dict[str, Any],
        *,
        desired_load_name: Optional[str] = None,
        binary: Optional[bytes] = None,
    ) -> bool:
        state = self._roles.get(role)
        bound_service_id = state.service_id if state else None
        message_type = str(message.get("type") or "")

        if bound_service_id:
            sent = await self._send_to_service(bound_service_id, message, binary)
            if sent:
                return True
            logger.warning(
                "inference session lost bound service chat_session=%s role=%s service_id=%s",
                self._chat_session_id,
                role,
                bound_service_id,
            )
            if state is not None:
                state.service_id = None

        spec_load_name = desired_load_name or (state.spec.load_name if state else "")
        spec_load_name = str(spec_load_name or "").strip()

        connected = await self._get_connected_service_ids()
        try:
            # role 直接作为 dispatch 的 capability 过滤键：chat 侧 role ∈ {tts, asr}
            # 与配置侧 capabilities 字面一致，无需再引入映射表。
            service = self._dispatch.pick_service(
                DispatchSpec(
                    service_type="audio",
                    reason=f"chat audio session role={role}",
                    load_name=spec_load_name,
                    capability=role,
                ),
                connected_service_ids=connected,
            )
        except DispatchSelectionError as exc:
            logger.warning(
                "inference session route failed chat_session=%s role=%s load_name=%s type=%s err=%s",
                self._chat_session_id,
                role,
                spec_load_name or "<unset>",
                message_type,
                exc,
            )
            return False

        service_id_value = str(service.get("id") or "").strip()
        if not service_id_value:
            logger.warning(
                "inference session selected service without id chat_session=%s role=%s load_name=%s",
                self._chat_session_id,
                role,
                spec_load_name,
            )
            return False

        sent = await self._send_to_service(service_id_value, message, binary)
        if sent and state is not None:
            state.service_id = service_id_value
            logger.info(
                "inference session bound service chat_session=%s role=%s type=%s service_id=%s",
                self._chat_session_id,
                role,
                message_type,
                service_id_value,
            )
        return sent


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


__all__ = [
    "InferenceSessionManager",
    "RoleSpec",
]

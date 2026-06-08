"""LiveTalking sidecar 推流 client（后端 → sidecar 单向 PCM 镜像）。

底层不变量（详见 ``.cursor/plans/livetalking_装饰接入_*.plan.md``）：

1. **对调用方完全 non-blocking**：``push_pcm`` 是同步方法，内部仅做"resample
   + bounded queue 入队"两步纯内存操作。``TtsCoordinator.handle_audio_out()``
   单次调用必须在亚毫秒级返回，sidecar 慢 / 卡死 / 网络丢包都不能反向阻塞
   ``audio_delta`` emit 链路。
2. **sidecar 入口契约严守 16k mono pcm_s16le**：所有采样率 / 声道转换在本
   client 内通过 ``Resampler`` 完成，按 ``(session_id, request_id)`` 维度持有跨
   chunk state；切换 request_id / flush / interrupt 都要 reset。

关键设计选择：

* **lazy WS 连接**：第一次有真实 PCM 要发时才创建 WS + consumer task，
  避免文本会话 / 数字人未启用的 session 也开 sidecar 资源。
* **bounded queue + drop oldest**：每个 session 一个 ``asyncio.Queue(maxsize)``，
  满了 drop oldest 同 request_id 的 chunk + 限频 log warn。绝不 ``await put``。
* **不重连**：sidecar 挂掉后 session 标记为 failed，后续 ``push_pcm`` 全部
  no-op + 限频 log。前端 panel 进入 ``error`` 由用户手动重试（重启服务时
  cache 自然清掉）。
* **interrupt 200ms 超时**：``InterruptCoordinator`` 调用本类 ``interrupt()`` 是
  低频操作（用户主动停 / barge-in），允许 ``await`` 但有短超时 + swallow，
  绝不影响主链路 interrupt 的后续步骤。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from backend.services.chat.avatar._resampler import Resampler, TARGET_SAMPLE_RATE
from backend.services.chat.avatar.livetalking_config import (
    LiveTalkingSettings,
    get_livetalking_settings,
)

logger = logging.getLogger(__name__)

# Bounded queue 上限：缓存 ≤ ~2s 16k mono PCM，按 20ms chunk 计 ≈ 100 条
# 超过即 drop oldest（同 request_id 优先），保护内存
_QUEUE_MAXSIZE = 128

# 单次 WS send 超时：sidecar 默认在内网，1s 足够；超时计入 failed 路径
_WS_SEND_TIMEOUT_SECONDS = 1.0

# WS 建连超时
_WS_CONNECT_TIMEOUT_SECONDS = 3.0

# interrupt 超时（plan 不变量第 1 条配套：interrupt 是低频允许 await 但有短超时）
_INTERRUPT_TIMEOUT_SECONDS = 0.2

# drop log 限频窗口：避免 backpressure 时刷屏
_DROP_LOG_THROTTLE_SECONDS = 1.0

# 兜底假设的源采样率（chunk 元数据缺 sample_rate 时用）
_FALLBACK_SOURCE_SAMPLE_RATE = 24000

# Consumer 任务 graceful shutdown 信号
_SHUTDOWN_SENTINEL: Dict[str, Any] = {"__shutdown__": True}


@dataclass
class _OutboundMessage:
    """consumer 待发送消息：JSON meta + 可选 binary frame。"""

    meta: Dict[str, Any]
    binary: Optional[bytes] = None
    request_id: str = ""  # 用于 drop oldest 优先匹配


@dataclass
class _SessionState:
    session_id: str
    enabled: bool = False
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=_QUEUE_MAXSIZE))
    consumer_task: Optional[asyncio.Task] = None
    ws: Optional[Any] = None  # websockets.WebSocketClientProtocol
    open_sent: bool = False
    active_request_id: Optional[str] = None
    resampler: Resampler = field(default_factory=Resampler)
    failed: bool = False  # sidecar 不可达后置 True，全 no-op
    last_drop_log_ts: float = 0.0


class LiveTalkingClient:
    """单例 client。生命周期跟随 backend 进程。

    线程模型假设：所有公共方法在主 asyncio event loop 上下文调用（``push_pcm``
    虽然是同步签名，但内部只做 ``asyncio.Queue.put_nowait``，必须在 loop 上）。
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionState] = {}
        self._lock = asyncio.Lock()  # 仅 close/teardown 整体清理时用，不用于 hot path

    # ------------------------------------------------------------------
    # 状态控制（同步）
    # ------------------------------------------------------------------

    def set_enabled(self, session_id: str, enabled: bool) -> None:
        """前端 ``avatar_toggle`` 路由到 ``runtime`` 后调本方法更新状态。

        ``enabled=False`` 时同步入队 close 控制 + 标记 disabled，后续 push_pcm
        全部 no-op；下次 ``set_enabled(True)`` 时重新创建 consumer/queue。
        """
        if not get_livetalking_settings().enabled:
            return  # 总开关关，全程 no-op
        existing = self._sessions.get(session_id)
        if enabled:
            if existing is None:
                self._sessions[session_id] = _SessionState(session_id=session_id, enabled=True)
            else:
                existing.enabled = True
                if existing.failed:
                    # 用户手动重试：清掉旧 failed 状态 + 旧资源，下次 push 重连
                    self._teardown_state(existing, schedule_close=False)
                    self._sessions[session_id] = _SessionState(session_id=session_id, enabled=True)
        else:
            if existing is not None:
                existing.enabled = False
                # 安排 close 消息 + cleanup
                asyncio.create_task(
                    self._async_cleanup_state(session_id, reason="disabled"),
                    name=f"livetalking-cleanup-{session_id[:8]}",
                )

    # ------------------------------------------------------------------
    # 推流入口（同步！绝不阻塞主链路）
    # ------------------------------------------------------------------

    def push_pcm(
        self,
        session_id: str,
        request_id: str,
        chunk: bytes,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> None:
        """非阻塞旁路 entry。所有失败路径 swallow + log。

        前置守卫：总开关 / session enabled / failed 任一为假即立即返回。
        正常路径：resample → 入队（或 drop oldest）。consumer task lazy 起。
        """
        settings = get_livetalking_settings()
        if not settings.enabled:
            return
        if not chunk:
            return
        state = self._sessions.get(session_id)
        if state is None or not state.enabled or state.failed:
            return

        # 切换 request_id：reset resample state，避免段间相位跳变
        if state.active_request_id != request_id:
            state.resampler.reset()
            state.active_request_id = request_id

        try:
            sr = int(sample_rate) if sample_rate else _FALLBACK_SOURCE_SAMPLE_RATE
        except (TypeError, ValueError):
            sr = _FALLBACK_SOURCE_SAMPLE_RATE
        try:
            ch = int(channels) if channels else 1
        except (TypeError, ValueError):
            ch = 1

        try:
            resampled = state.resampler.process(chunk, source_sr=sr, source_channels=ch)
        except Exception as exc:
            logger.warning(
                "resample failed session=%s request=%s err=%s",
                session_id, request_id, exc,
            )
            return
        if not resampled:
            return

        msg = _OutboundMessage(
            meta={"type": "audio_chunk", "request_id": request_id},
            binary=resampled,
            request_id=request_id,
        )
        self._enqueue_or_drop(state, msg, settings)
        self._ensure_consumer(state, settings)

    def flush(self, session_id: str, request_id: str) -> None:
        """``is_final=True`` 路径调用，标记本 request 结束。同步入队。"""
        if not get_livetalking_settings().enabled:
            return
        state = self._sessions.get(session_id)
        if state is None or not state.enabled or state.failed:
            return
        # flush 自身就是段结束，reset resample state 留给下次 push 切 request_id 时再做
        msg = _OutboundMessage(
            meta={"type": "audio_flush", "request_id": request_id},
            request_id=request_id,
        )
        self._enqueue_or_drop(state, msg, get_livetalking_settings())
        self._ensure_consumer(state, get_livetalking_settings())

    # ------------------------------------------------------------------
    # 控制流（async，低频）
    # ------------------------------------------------------------------

    async def interrupt(self, session_id: str) -> None:
        """打断：清空本 session pending PCM + 通知 sidecar flush_talk。

        plan 不变量：低频允许 await 但 200ms 超时 + swallow，绝不影响主链路。
        """
        if not get_livetalking_settings().enabled:
            return
        state = self._sessions.get(session_id)
        if state is None or not state.enabled or state.failed:
            return

        # 1. 立即清空 pending queue + reset resampler，防止 stale PCM 继续漏出
        _drain_queue(state.queue)
        state.resampler.reset()
        state.active_request_id = None

        # 2. 直接经现有 ws 发 interrupt（不入队 → 不被 drained）
        ws = state.ws
        if ws is None:
            return
        try:
            await asyncio.wait_for(
                ws.send(json.dumps({"type": "interrupt", "session_id": session_id})),
                timeout=_INTERRUPT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "livetalking interrupt timed out (>%.0fms) session=%s; main interrupt path continues",
                _INTERRUPT_TIMEOUT_SECONDS * 1000, session_id,
            )
        except Exception as exc:
            logger.warning("livetalking interrupt failed session=%s err=%s", session_id, exc)

    async def close(self, session_id: str) -> None:
        """会话结束：发送 close 控制 + 关闭 WS + 移除 state。"""
        await self._async_cleanup_state(session_id, reason="close")

    async def shutdown(self) -> None:
        """进程退出时调用：清掉所有 session。"""
        async with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                await self._async_cleanup_state(sid, reason="shutdown")
            except Exception as exc:
                logger.debug("shutdown cleanup failed session=%s err=%s", sid, exc)

    # ------------------------------------------------------------------
    # 内部：consumer / WS 生命周期
    # ------------------------------------------------------------------

    def _enqueue_or_drop(
        self, state: _SessionState, msg: _OutboundMessage, settings: LiveTalkingSettings,
    ) -> None:
        """bounded queue 满 → drop oldest 同 request_id 的 chunk → 限频 log。"""
        try:
            state.queue.put_nowait(msg)
            return
        except asyncio.QueueFull:
            pass
        # 容量满：尽量丢一个老的同 request_id 的 chunk
        dropped = _drop_oldest_matching(state.queue, request_id=msg.request_id)
        try:
            state.queue.put_nowait(msg)
        except asyncio.QueueFull:
            dropped = True
        if dropped:
            now = time.monotonic()
            if now - state.last_drop_log_ts >= _DROP_LOG_THROTTLE_SECONDS:
                logger.warning(
                    "livetalking queue backpressure session=%s rid=%s; dropping oldest chunk",
                    state.session_id, msg.request_id,
                )
                state.last_drop_log_ts = now

    def _ensure_consumer(self, state: _SessionState, settings: LiveTalkingSettings) -> None:
        """lazy 起 consumer task；已有 + 未结束就什么都不做。"""
        if state.consumer_task is not None and not state.consumer_task.done():
            return
        state.consumer_task = asyncio.create_task(
            self._consumer_loop(state, settings),
            name=f"livetalking-consumer-{state.session_id[:8]}",
        )

    async def _consumer_loop(
        self, state: _SessionState, settings: LiveTalkingSettings,
    ) -> None:
        """单 session consumer：取队列 → 发 WS。失败 → 标 failed + 退出。"""
        try:
            while True:
                msg = await state.queue.get()
                if isinstance(msg, dict) and msg.get("__shutdown__"):
                    return
                if not isinstance(msg, _OutboundMessage):
                    continue
                if state.failed:
                    return
                try:
                    await self._send_message(state, msg, settings)
                except Exception as exc:
                    logger.warning(
                        "livetalking ws send failed session=%s err=%s; marking session failed",
                        state.session_id, exc,
                    )
                    state.failed = True
                    await self._safe_close_ws(state)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "livetalking consumer loop crashed session=%s err=%s",
                state.session_id, exc, exc_info=True,
            )
            state.failed = True
            await self._safe_close_ws(state)

    async def _send_message(
        self,
        state: _SessionState,
        msg: _OutboundMessage,
        settings: LiveTalkingSettings,
    ) -> None:
        ws = await self._ensure_ws(state, settings)
        await asyncio.wait_for(
            ws.send(json.dumps(msg.meta)), timeout=_WS_SEND_TIMEOUT_SECONDS,
        )
        if msg.binary:
            await asyncio.wait_for(
                ws.send(msg.binary), timeout=_WS_SEND_TIMEOUT_SECONDS,
            )

    async def _ensure_ws(self, state: _SessionState, settings: LiveTalkingSettings) -> Any:
        if state.ws is not None:
            return state.ws
        url = settings.avatar_stream_url
        try:
            ws = await asyncio.wait_for(
                websockets.connect(url, max_size=4 * 1024 * 1024),
                timeout=_WS_CONNECT_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, OSError, WebSocketException) as exc:
            raise RuntimeError(f"sidecar ws connect failed: {exc}") from exc
        state.ws = ws
        # 首条 open 消息：sidecar 用它绑 AvatarSession + 校验 sample_rate / format / channels
        # 这里 request_id 用一个会话级占位，后续每条 audio_chunk 再带具体 rid
        open_msg = {
            "type": "open",
            "session_id": state.session_id,
            "request_id": f"session-init-{state.session_id[:8]}",
            "sample_rate": TARGET_SAMPLE_RATE,
            "format": "pcm_s16le",
            "channels": 1,
        }
        try:
            await asyncio.wait_for(
                ws.send(json.dumps(open_msg)), timeout=_WS_SEND_TIMEOUT_SECONDS,
            )
            state.open_sent = True
        except Exception as exc:
            await self._safe_close_ws(state)
            raise RuntimeError(f"sidecar ws open failed: {exc}") from exc
        return ws

    async def _async_cleanup_state(self, session_id: str, *, reason: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return
        # 给 consumer 一个 close 控制（best-effort），再 cancel + close ws
        try:
            await asyncio.wait_for(
                state.queue.put({"__shutdown__": True}), timeout=0.05,
            )
        except (asyncio.TimeoutError, asyncio.QueueFull):
            pass
        if state.ws is not None:
            try:
                await asyncio.wait_for(
                    state.ws.send(json.dumps({"type": "close"})),
                    timeout=_WS_SEND_TIMEOUT_SECONDS,
                )
            except Exception:
                pass
        await self._safe_close_ws(state)
        if state.consumer_task is not None and not state.consumer_task.done():
            state.consumer_task.cancel()
            try:
                await state.consumer_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.debug("livetalking session cleaned up session=%s reason=%s", session_id, reason)

    def _teardown_state(self, state: _SessionState, *, schedule_close: bool) -> None:
        """同步清理（用于 set_enabled 路径，不能 await）。"""
        if state.consumer_task is not None and not state.consumer_task.done():
            state.consumer_task.cancel()
        if state.ws is not None:
            asyncio.create_task(_close_ws_safely(state.ws))
            state.ws = None

    async def _safe_close_ws(self, state: _SessionState) -> None:
        ws = state.ws
        state.ws = None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass


def _drain_queue(queue: asyncio.Queue) -> int:
    """非阻塞清空 queue。返回丢弃数量（仅日志用）。"""
    dropped = 0
    while True:
        try:
            queue.get_nowait()
            dropped += 1
        except asyncio.QueueEmpty:
            return dropped


def _drop_oldest_matching(queue: asyncio.Queue, *, request_id: str) -> bool:
    """从 queue 头丢一个，优先丢同 request_id 的（asyncio.Queue 内部是 deque，
    不支持按内容查找，这里只能 pop 头）。返回是否丢成功。"""
    try:
        queue.get_nowait()
        return True
    except asyncio.QueueEmpty:
        return False


async def _close_ws_safely(ws: Any) -> None:
    try:
        await ws.close()
    except Exception:
        pass


# ----------------------------------------------------------------------
# 单例访问入口
# ----------------------------------------------------------------------

_client_singleton: Optional[LiveTalkingClient] = None


def get_livetalking_client() -> LiveTalkingClient:
    """进程级单例。``backend`` 主线程的 FastAPI event loop 上调用。"""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LiveTalkingClient()
    return _client_singleton


def reset_client_for_test() -> None:
    """测试用：清掉单例，强制下次 ``get_livetalking_client`` 重建。"""
    global _client_singleton
    _client_singleton = None


__all__ = [
    "LiveTalkingClient",
    "get_livetalking_client",
    "reset_client_for_test",
]

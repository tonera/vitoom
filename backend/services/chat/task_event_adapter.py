from __future__ import annotations

import base64
from typing import Any, Dict

from backend.services.chat.session import SessionRuntime, Turn


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


async def handle_task_event(
    *,
    runtime: SessionRuntime,
    turn: Turn,
    event: Dict[str, Any],
) -> bool:
    if not isinstance(event, dict):
        return True

    # 漂移事件守卫：master_runtime 用 ``asyncio.run_coroutine_threadsafe`` 把工作线程
    # 的 task event 投递到主 loop，已 schedule 但未执行的协程**不会随 master run
    # cancel 一起被取消**。interrupt 路径下 ``_finalize_turn`` 已把 ``current_turn``
    # 清空、状态切到 READY 后，这些漂回来的 ``task_bound`` 仍会调
    # ``enter_waiting_task()`` 把 state 反向改回 WAITING_TASK，导致用户接下来发
    # ``audio_chunk`` 拿到 ``state=waiting_task refuses audio_chunk`` 的 busy 报错。
    #
    # 双守卫覆盖竞态全窗口：
    #   - identity 守卫挡 ``_finalize_turn`` 之后的 stale 事件（current_turn 已 None
    #     或已切到新 turn）；
    #   - interrupt_requested 守卫挡 ``_finalize_turn`` 之前、interrupt 已发起的
    #     窗口期（同一 turn 但本轮已被用户取消）。
    if runtime.current_turn is not turn or runtime.interrupt_requested:
        return True

    event_type = str(event.get("type") or "").strip().lower()
    task_id = str(event.get("task_id") or "").strip()
    if task_id:
        turn.bind_task_id(task_id)

    if event_type == "task_bound":
        await runtime.enter_waiting_task()

    if event_type in {"task_bound", "task_status", "task_result"}:
        await runtime.emit_task_status(
            payload={
                "task_id": task_id or None,
                "task_status": str(event.get("status") or "processing"),
                "progress": event.get("progress"),
                "error": event.get("error"),
                "task_kind": event.get("task_kind"),
                "files_count": event.get("files_count"),
                "total": event.get("total"),
            }
        )
        return True

    if event_type == "audio_stream_start":
        await runtime.enter_streaming_output()
        return True

    if event_type == "audio_stream_chunk":
        await runtime.enter_streaming_output()
        raw_b64 = str(event.get("data") or "")
        try:
            pcm = base64.b64decode(raw_b64) if raw_b64 else b""
        except Exception:
            pcm = b""
        await runtime.emit_audio_delta(
            pcm_bytes=pcm,
            mime=str(event.get("mime") or event.get("mime_type") or "audio/wav"),
            is_final=False,
            sample_rate=_safe_int(event.get("sample_rate")),
        )
        return True

    if event_type == "audio_stream_end":
        await runtime.enter_streaming_output()
        await runtime.emit_audio_delta(
            pcm_bytes=b"",
            mime=str(event.get("mime") or event.get("mime_type") or "audio/wav"),
            is_final=True,
            sample_rate=_safe_int(event.get("sample_rate")),
        )
        return True

    return False

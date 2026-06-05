from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional


PROGRESS_ESTIMATE_MIN_CHARS = 50
PROGRESS_CALIBRATION_TEXT = "这是用于估算语音生成速度的固定测试文本。"
PROGRESS_ESTIMATE_SAFETY_FACTOR = 1.2
PROGRESS_ESTIMATE_ELAPSED_MULTIPLIER = 2.0

SendProgress = Callable[[str, int, str], Awaitable[None]]
CancelCheck = Callable[[], Awaitable[bool]]


def effective_text_length(text: str) -> int:
    return len("".join(str(text or "").split()))


async def estimate_tts_generation_seconds(
    *,
    engine: Any,
    task_id: str,
    text: str,
    voice_cfg: Any,
    send_progress: SendProgress,
    cancel_check: Optional[CancelCheck],
    logger: Any,
    log_prefix: str,
) -> Optional[float]:
    text_len = effective_text_length(text)
    if text_len <= PROGRESS_ESTIMATE_MIN_CHARS:
        return None

    await send_progress(task_id, 18, "正在测速生成速度")
    started = time.monotonic()
    async for _chunk in engine.synthesize_stream(
        text=PROGRESS_CALIBRATION_TEXT,
        voice_cfg=voice_cfg,
        cancel_check=cancel_check,
        stream_mode=False,
    ):
        pass
    elapsed = max(0.1, time.monotonic() - started)
    calibration_len = max(1, effective_text_length(PROGRESS_CALIBRATION_TEXT))
    estimated = (elapsed / calibration_len) * text_len * PROGRESS_ESTIMATE_SAFETY_FACTOR
    logger.info(
        "[%s] task_id=%s calibrated progress estimate: calibration=%.3fs chars=%d target_chars=%d estimate=%.3fs",
        log_prefix,
        task_id,
        elapsed,
        calibration_len,
        text_len,
        estimated,
    )
    return max(1.0, estimated)


async def tick_estimated_progress(
    *,
    task_id: str,
    send_progress: SendProgress,
    start: int,
    end: int,
    estimated_seconds: float,
    message: str,
    interval_seconds: float = 2.0,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    progress = max(0, min(100, int(start)))
    end = max(progress, min(100, int(end)))
    tick_count = 0
    while progress < end:
        if stop_event is not None:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                return
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(interval_seconds)

        if stop_event is not None and stop_event.is_set():
            return

        tick_count += 1
        elapsed = tick_count * interval_seconds * PROGRESS_ESTIMATE_ELAPSED_MULTIPLIER
        ratio = min(1.0, elapsed / max(0.1, estimated_seconds))
        next_progress = min(end, max(progress, start + int(round((end - start) * ratio))))
        if next_progress > progress:
            progress = next_progress
            await send_progress(task_id, progress, message)

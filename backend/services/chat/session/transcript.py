"""ASR final transcript 处理：噪声过滤、barge-in 控制词识别、turn 收尾。

barge-in 控制词列表参见 docs/实时语音和文本聊天全生命周期流程.md §3.3。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from backend.core.logger import get_app_logger
from backend.services.chat.session._models import SessionState
from backend.services.conversation import append_message

if TYPE_CHECKING:
    from backend.services.chat.session.runtime import SessionRuntime

logger = get_app_logger(__name__)


_BARGE_IN_CONTROL_EXACT_TEXTS = {
    # "停止/打断"
    "停", "停止", "停下", "停一下", "别说", "别说了", "不要说", "不要说了", "闭嘴",
    # 附和 / 同意
    "行", "行了", "可以", "可以了", "好了", "好", "好的",
    "对", "对的", "是", "是的", "明白", "明白了", "知道", "知道了", "ok", "okay",
    # 喉音附和（仅在 barge-in turn 上走这条路；normal turn 由噪声过滤兜底）
    "嗯", "嗯嗯", "哦", "啊", "不",
}

_BARGE_IN_CONTROL_PREFIXES = ("停", "别说", "不要说", "别讲", "不要讲", "闭嘴")

# barge-in 控制词 / 噪声过滤的"剥噪标点"集合（中英标点 + ASR 噪声常凑出的装饰符号）
_BARGE_IN_NOISE_PUNCT_RE = re.compile(
    r"[\s,，.。!！?？、~～…‼⁉《》〈〉「」『』【】〖〗()（）\[\]<>"
    r"\u00a1\u00bf\u00ab\u00bb\u2018\u2019\u201c\u201d"
    r"\-\u2013\u2014_/\\:;|]+"
)

# ASR 在环境噪声上的常见误识别短词，下游 LLM 拿到也只会回"我不太理解"。
_NOISE_FILLER_TOKENS = {
    "嗯", "啊", "呃", "哦", "唉", "诶", "嘿", "啦", "嘞", "呵", "哈", "喂", "呀",
    "el", "uh", "um", "ah", "oh",
}


def _normalize(text: str) -> str:
    return _BARGE_IN_NOISE_PUNCT_RE.sub("", str(text or "")).strip().lower()


def _has_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\uf900" <= ch <= "\ufaff"
        for ch in text
    )


def _barge_in_control_text(text: str) -> Optional[str]:
    n = _normalize(text)
    if not n:
        return None
    if n in _BARGE_IN_CONTROL_EXACT_TEXTS:
        return n
    for prefix in _BARGE_IN_CONTROL_PREFIXES:
        if n.startswith(prefix):
            return prefix
    return None


def _is_barge_in_control_text(text: str) -> bool:
    return _barge_in_control_text(text) is not None


def _is_likely_noise_transcript(text: str) -> bool:
    """noise 二级兜底（VAD 漏过 → ASR 凑词）。详见文档 §3.6。"""
    n = _normalize(text)
    if not n:
        return True
    if len(n) == 1:
        if n in _NOISE_FILLER_TOKENS:
            return True
        return n.isascii() and n.isalpha()  # 单拉丁字母视为噪声；单字汉字保留
    if n in _NOISE_FILLER_TOKENS:
        return True
    if len(set(n)) == 1 and n[0] in _NOISE_FILLER_TOKENS:
        return True
    if len(n) <= 4 and not _has_cjk(n):
        return True
    return False


class TranscriptProcessor:
    def __init__(self, runtime: "SessionRuntime") -> None:
        self._runtime = runtime

    async def handle_final(self, event: Dict[str, Any]) -> None:
        rt = self._runtime
        text = str(event.get("text") or "")
        turn = rt.current_turn
        committed = bool(turn and rt._audio_turn_committed)

        # barge-in 控制词捷径：cancel TTS 但不开 LLM 新一轮
        if committed and turn and turn.barge_in:
            ctrl = _barge_in_control_text(text)
            if ctrl is not None:
                await self._emit_and_persist(turn, text, ctrl_text=ctrl)
                logger.info(
                    "barge-in control turn completed without LLM session=%s turn=%s ctrl=%s",
                    rt.session_id, turn.turn_id, ctrl,
                )
                rt._finalize_turn()
                await rt._emitter.set_state(rt, SessionState.READY)
                return

        text_stripped = text.strip()
        is_noise = bool(text_stripped) and _is_likely_noise_transcript(text)
        meaningful = bool(text_stripped) and not is_noise

        if is_noise:
            logger.info(
                "transcript dropped as likely noise session=%s turn=%s text=%r",
                rt.session_id, turn.turn_id if turn else None, text_stripped,
            )
        if meaningful:
            await rt.emit_transcript_delta(text=text, is_final=True)

        if turn:
            turn.user_text = text if meaningful else ""
            if meaningful:
                await self._persist_user_audio_text(turn, text)

        if committed and meaningful:
            rt._audio.reset_vad_detector()
            await rt._start_run_for_current_turn()
            return
        if committed and not meaningful:
            await self._cancel_empty_turn(turn, text_stripped, is_noise)

        rt._finalize_turn()
        await rt._emitter.set_state(rt, SessionState.READY)

    async def _emit_and_persist(self, turn, text: str, *, ctrl_text: str) -> None:
        rt = self._runtime
        await rt.emit_transcript_delta(text=text, is_final=True)
        turn.user_text = text
        try:
            append_message(
                conversation_id=rt.session_id,
                role="user",
                content=text,
                turn_id=turn.turn_id,
                metadata={
                    "audio_chunks": rt.current_assembler.audio_count() if rt.current_assembler else 0,
                    "barge_in_control": True,
                    "barge_in_control_text": ctrl_text,
                },
                user_id=rt.user_id,
            )
        except Exception as exc:
            logger.warning("persist control transcript failed session=%s err=%s", rt.session_id, exc)

    async def _persist_user_audio_text(self, turn, text: str) -> None:
        rt = self._runtime
        try:
            append_message(
                conversation_id=rt.session_id,
                role="user",
                content=text,
                turn_id=turn.turn_id,
                metadata={
                    "audio_chunks": rt.current_assembler.audio_count() if rt.current_assembler else 0,
                },
                user_id=rt.user_id,
            )
        except Exception as exc:
            logger.warning("persist transcript failed session=%s err=%s", rt.session_id, exc)

    async def _cancel_empty_turn(self, turn, text_stripped: str, is_noise: bool) -> None:
        rt = self._runtime
        if turn is None:
            return
        reason = "noise" if is_noise else "empty"
        await rt._emitter.send(
            "transcript_canceled",
            payload={"reason": reason, "text": text_stripped},
            turn=turn,
        )
        msg = (
            f"audio turn committed but transcript is likely noise: {text_stripped!r}"
            if is_noise
            else "audio turn committed but final transcript is empty"
        )
        await rt._emitter.error(rt, "empty_transcript", msg)


__all__ = [
    "TranscriptProcessor",
    "_barge_in_control_text",
    "_is_likely_noise_transcript",
]

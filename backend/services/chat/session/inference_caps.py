"""ASR / TTS 推理 session 开闭、能力探测与广播。"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from backend.core.logger import get_app_logger
from backend.services.chat.inference_session import RoleSpec
from backend.services.chat.session._models import InputMode

if TYPE_CHECKING:
    from backend.services.chat.session.runtime import SessionRuntime

logger = get_app_logger(__name__)

_AUDIO_INPUT_MODES = {InputMode.AUDIO_ONCE, InputMode.AUDIO_STREAM, InputMode.MIXED}
_AUDIO_OUTPUT_MODES = {"audio_once", "audio_stream", "multimodal", "multimodal_result"}


class InferenceCoordinator:
    def __init__(self, runtime: "SessionRuntime") -> None:
        self._runtime = runtime
        self.asr_session_opened = False
        self.tts_session_opened = False
        self.supports_audio_input = False
        self.supports_audio_output = False

    def is_audio_input_mode(self) -> bool:
        return self._runtime.input_mode in _AUDIO_INPUT_MODES

    def is_audio_output_mode(self) -> bool:
        return str(self._runtime.output_mode or "").strip().lower() in _AUDIO_OUTPUT_MODES

    def voice_output_config(self) -> Dict[str, Any]:
        meta = self._runtime.metadata
        for key in ("audio_output", "tts"):
            raw = meta.get(key)
            if isinstance(raw, dict):
                return dict(raw)
        return {}

    def capabilities_payload(self) -> Dict[str, Any]:
        return {
            "supports_audio_input": bool(self.supports_audio_input),
            "supports_audio_output": bool(self.supports_audio_output),
            "supports_tool_artifacts": True,
        }

    async def set_audio_capabilities(
        self,
        *,
        supports_audio_input: bool,
        supports_audio_output: bool,
        emit_event: Optional[str] = None,
    ) -> None:
        next_in, next_out = bool(supports_audio_input), bool(supports_audio_output)
        changed = next_in != self.supports_audio_input or next_out != self.supports_audio_output
        self.supports_audio_input = next_in
        self.supports_audio_output = next_out
        if emit_event and changed:
            await self._runtime._emitter.send(
                emit_event,
                payload={"capabilities": self.capabilities_payload()},
            )

    async def ensure_asr_session_opened(self) -> bool:
        if not self.is_audio_input_mode() or self.asr_session_opened:
            return self.asr_session_opened or False
        return await self._open_role("asr", self._runtime.metadata.get("audio_input"))

    async def ensure_tts_session_opened(self) -> bool:
        if not self.is_audio_output_mode() or self.tts_session_opened:
            return self.tts_session_opened or False
        cfg = self.voice_output_config()
        if not cfg:
            logger.debug(
                "skip tts session open session=%s output_mode=%s (no audio_output config)",
                self._runtime.session_id,
                self._runtime.output_mode,
            )
            return False
        return await self._open_role("tts", cfg)

    async def _open_role(self, role: str, raw_cfg: Any) -> bool:
        rt = self._runtime
        if rt._inference_session is None:
            return False
        cfg = dict(raw_cfg or {})
        spec = RoleSpec(
            role=role,
            load_name=str(cfg.get("load_name") or "").strip(),
            family=str(cfg.get("family") or "").strip(),
            runtime_config=dict(cfg.get("runtime_config") or {}),
        )
        opened = bool(await rt._inference_session.open(spec))
        if role == "asr":
            self.asr_session_opened = opened
        else:
            self.tts_session_opened = opened
        return opened

    async def probe_audio_capabilities(self) -> tuple[bool, bool]:
        rt = self._runtime
        if rt._inference_session is None:
            return False, False
        asr_av = tts_av = False
        if self.is_audio_input_mode():
            asr_av = await rt._inference_session.probe_role_available(
                "asr",
                load_name=str((rt.metadata.get("audio_input") or {}).get("load_name") or "").strip(),
            )
        if self.is_audio_output_mode():
            tts_av = await rt._inference_session.probe_role_available(
                "tts",
                load_name=str(self.voice_output_config().get("load_name") or "").strip(),
            )
        return asr_av, tts_av

    async def handle_services_changed(self, event: Optional[Dict[str, Any]] = None) -> None:
        rt = self._runtime
        if rt._inference_session is None:
            await self.set_audio_capabilities(
                supports_audio_input=False,
                supports_audio_output=False,
                emit_event="capabilities_changed",
            )
            return
        payload = event.get("payload") if isinstance(event, dict) else None
        reason = str((payload or {}).get("reason") or "service topology changed").strip() or "service topology changed"
        rt._inference_session.invalidate_all_roles(reason=reason)
        self.asr_session_opened = False
        self.tts_session_opened = False
        asr_av, tts_av = await self.probe_audio_capabilities()
        await self.set_audio_capabilities(
            supports_audio_input=asr_av,
            supports_audio_output=tts_av,
            emit_event="capabilities_changed",
        )

    def reset_opened_flags(self) -> None:
        self.asr_session_opened = False
        self.tts_session_opened = False


__all__ = ["InferenceCoordinator"]

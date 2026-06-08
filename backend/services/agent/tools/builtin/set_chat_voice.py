"""set_chat_voice 工具：修改当前实时语音聊天的会话级音色配置。"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Optional

from backend.database import Conversation
from backend.services.agent.tools.builtin._arg_utils import clean_optional_str
from backend.services.agent.tools.registry import register_tool
from backend.services.conversation import get_conversation
from backend.services.tts_speakers import list_speaker_options

SET_CHAT_VOICE_TOOL_NAME = "set_chat_voice"

SET_CHAT_VOICE_DESCRIPTION = (
    "切换当前会话后续回复的音色（写操作）。"
    "仅当 query 含切换动词（换/改/切/不喜欢这个声音/以后用…对话）时调用。"
    "询问类（『是什么声音/什么音色/which/what voice』）和撤销类（『我没让你换』）一律不调用，直接用文字回答 speaker_name/instruct。"
    "朗读用 audio_tts，多角色配音用 audio_drama_tts。"
    "\n用法：(1) 只说换一个 → mode=random_custom_voice；"
    "(2) 描述声线 → mode=voice_design + instruct；"
    "(3) 点名 speaker → mode=custom_voice + speaker_name（仅做大小写规范化，不替换成其他 speaker）。"
)

SET_CHAT_VOICE_DOCSTRING = (
    "Set the persistent voice profile for the current realtime chat session. "
    "Use audio_tts for standalone text-to-speech tasks."
)

_MODE_ALIASES = {
    "": "random_custom_voice",
    "random": "random_custom_voice",
    "change": "random_custom_voice",
    "switch": "random_custom_voice",
    "random_custom_voice": "random_custom_voice",
    "custom_voice": "custom_voice",
    "speaker": "custom_voice",
    "voice_design": "voice_design",
    "design": "voice_design",
}


def _normalize_mode(raw: Optional[str]) -> str:
    key = str(raw or "").strip().lower().replace("-", "_")
    return _MODE_ALIASES.get(key, "")


def _current_audio_output(metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("audio_output")
    if isinstance(raw, dict):
        return dict(raw)
    raw = metadata.get("tts")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _speaker_names(family: str) -> list[str]:
    return [
        str(item.get("name") or "").strip()
        for item in list_speaker_options(family)
        if str(item.get("name") or "").strip()
    ]


def _pick_random_speaker(family: str, current: str = "") -> str:
    speakers = _speaker_names(family)
    if not speakers:
        raise ValueError(f"no speakers configured for family={family}")
    current_norm = current.strip().lower()
    candidates = [name for name in speakers if name.strip().lower() != current_norm]
    return random.choice(candidates or speakers)


def update_chat_voice_profile(
    *,
    conversation_id: str,
    user_id: str,
    runtime: Any = None,
    mode: Optional[str] = None,
    tts_mode: Optional[str] = None,
    speaker_name: Optional[str] = None,
    instruct: Optional[str] = None,
    family: str = "voxcpm",
) -> Dict[str, Any]:
    normalized_conversation_id = clean_optional_str(conversation_id) or ""
    normalized_user_id = clean_optional_str(user_id) or ""
    if not normalized_conversation_id:
        raise ValueError("set_chat_voice requires conversation_id")
    if not normalized_user_id:
        raise ValueError("set_chat_voice requires user_id")

    effective_mode = _normalize_mode(mode) or _normalize_mode(tts_mode)
    if not effective_mode:
        raise ValueError("unsupported chat voice mode")

    family = (clean_optional_str(family) or "voxcpm").lower()
    instruct = clean_optional_str(instruct) or ""
    requested_speaker = clean_optional_str(speaker_name) or ""

    conv = get_conversation(normalized_conversation_id, user_id=normalized_user_id)
    current_metadata = dict(getattr(runtime, "metadata", {}) or {})
    if not current_metadata:
        current_metadata = dict(conv.get("metadata") or {})

    current_voice = _current_audio_output(current_metadata)
    next_voice = dict(current_voice)

    if effective_mode == "voice_design":
        if not instruct:
            raise ValueError("voice_design requires instruct")
        next_voice.update(
            {
                "tts_mode": "voice_design",
                "instruct": instruct,
                "design_instruct": instruct,
            }
        )
        for key in ("speaker_name", "voice_preset", "ref_audio", "ref_text", "prompt_wav_path", "prompt_text"):
            next_voice.pop(key, None)
    else:
        selected_speaker = requested_speaker
        if not selected_speaker or effective_mode == "random_custom_voice":
            selected_speaker = _pick_random_speaker(
                family, current=str(current_voice.get("speaker_name") or current_voice.get("voice_preset") or "")
            )
        next_voice.update(
            {
                "tts_mode": "custom_voice",
                "speaker_name": selected_speaker,
            }
        )
        if instruct:
            next_voice["instruct"] = instruct
        else:
            next_voice.pop("instruct", None)
        for key in ("voice_preset", "design_instruct", "ref_audio", "ref_text", "prompt_wav_path", "prompt_text"):
            next_voice.pop(key, None)

    next_voice["updated_by"] = SET_CHAT_VOICE_TOOL_NAME
    next_voice["voice_family"] = family

    next_metadata = dict(current_metadata)
    next_metadata["audio_output"] = next_voice

    if runtime is not None and hasattr(runtime, "metadata"):
        runtime.metadata = dict(next_metadata)

    updated = Conversation.update(normalized_conversation_id, metadata=next_metadata)
    if not updated:
        raise RuntimeError("failed to persist chat voice profile")

    return {
        "tool": SET_CHAT_VOICE_TOOL_NAME,
        "status": "ok",
        "conversation_id": normalized_conversation_id,
        "voice": next_voice,
        "message": "当前会话后续语音回复将使用新的音色配置。",
    }


@register_tool(
    name=SET_CHAT_VOICE_TOOL_NAME,
    description=SET_CHAT_VOICE_DESCRIPTION,
    tags=["audio", "tts", "voice", "chat voice", "换声音", "音色", "语音设计", "speaker"],
    provider="local",
    enabled=True,
)
def build_set_chat_voice_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()
    bound_conversation_id = str(ctx.get("conversation_id") or "").strip()
    bound_runtime = ctx.get("session_runtime")

    try:
        from crewai.tools import BaseTool
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, Field
    except Exception as exc:
        raise RuntimeError("pydantic is required to build set_chat_voice tool") from exc

    class SetChatVoiceArgs(BaseModel):
        mode: str = Field(
            default="random_custom_voice",
            description=(
                "random_custom_voice / custom_voice / voice_design. "
                "Use custom_voice when user names a speaker; voice_design when describing voice only; "
                "random_custom_voice when user just asks to change voice."
            ),
        )
        speaker_name: Optional[str] = Field(
            default=None,
            description=(
                "Speaker name under custom_voice. Fill only when user explicitly names a speaker; "
                "use the user's exact speaker name with only case/whitespace normalization; "
                "do not pick a different speaker based on gender/tone descriptions. "
                "Leave empty when user only asks to change voice or describes voice traits."
            ),
        )
        instruct: Optional[str] = Field(
            default=None,
            description="Required for voice_design: user-described voice/age/gender/texture, e.g. 'gruff husky male voice'.",
        )
        family: str = Field(
            default="voxcpm",
            description="Speaker config family; defaults to voxcpm for realtime voice chat.",
        )

    class SetChatVoiceTool(BaseTool):
        name: str = SET_CHAT_VOICE_TOOL_NAME
        description: str = SET_CHAT_VOICE_DESCRIPTION
        args_schema: type = SetChatVoiceArgs

        def _run(self, **kwargs: Any) -> str:
            try:
                payload = update_chat_voice_profile(
                    conversation_id=bound_conversation_id,
                    user_id=bound_user_id,
                    runtime=bound_runtime,
                    **kwargs,
                )
            except Exception as exc:
                payload = {
                    "tool": SET_CHAT_VOICE_TOOL_NAME,
                    "status": "failed",
                    "error": str(exc),
                }
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = SetChatVoiceTool()
    tool_instance.__doc__ = SET_CHAT_VOICE_DOCSTRING
    return tool_instance

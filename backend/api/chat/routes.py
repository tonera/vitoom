"""``/v1/chat/*`` HTTP 路由。

统一会话重构里，前端的交互流程归并为：

    1. ``POST /v1/chat/sessions`` 创建一次 ChatSession（对应 conversations 行）；
    2. ``GET /v1/chat/sessions`` 分页列出当前用户的会话（供侧栏历史等）；
    3. 客户端连接 ``WS /ws/chat/{session_id}`` 做实时交互；
    4. 需要回看历史时调用 ``GET /v1/chat/sessions/{session_id}/messages``。

本模块是当前唯一的会话 HTTP 入口；会话列表与会话详情、消息拉取均收敛在此文件。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import get_current_user_id
from backend.core.config import get_config
from backend.core.response import ok
from backend.database import Model
from backend.services.agent.settings import get_master_preset_agent_id
from backend.services.conversation import (
    ConversationValidationError,
    create_conversation,
    get_conversation,
    list_conversations,
    list_messages,
)


def _resolve_default_load_name() -> str:
    """聊天会话未显式指定时的兜底模型加载名。

    优先 ``agents.default_model``（config/default.yaml），留空才回退到写死的
    hint。推理侧会把这个字符串当 LLM 真实 load_name 使用。
    """
    value = str(get_config("agents.default_model", "") or "").strip()
    if value:
        return value
    return "Qwen3.5-35B-A3B-GPTQ-Int4"

router = APIRouter(prefix="/v1/chat", tags=["Chat"])


def _to_http_exception(exc: Exception) -> HTTPException:
    detail = str(exc) or exc.__class__.__name__
    if isinstance(exc, ConversationValidationError):
        status_code = 400
        if "not found" in detail.lower():
            status_code = 404
        elif "permission denied" in detail.lower():
            status_code = 403
        return HTTPException(status_code=status_code, detail=detail)
    return HTTPException(status_code=500, detail=detail)


class ChatSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_id: Optional[str] = Field(
        default=None,
        description=(
            "[debug] Optional agent_id override. Normally omit this field; sessions attach to "
            "Master Agent automatically and routing is decided by the LLM."
        ),
    )
    title: Optional[str] = Field(default=None, description="Session title; auto-generated from first message if omitted")
    input_mode: str = Field(default="text", description="Supported: text / audio_once / audio_stream")
    output_mode: str = Field(default="text_stream", description="Preferred output modality")
    load_name: Optional[str] = Field(
        default=None,
        description="Model load name. For text sessions: main chat model; for audio_* sessions: ASR load_name.",
    )
    audio_output: Optional["ChatSessionAudioOutputRequest"] = Field(
        default=None,
        description="Voice reply preferences; used for assistant TTS configuration",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSessionAudioOutputRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    load_name: Optional[str] = Field(default=None, description="TTS load_name; falls back to server default if omitted")
    tts_mode: Optional[str] = Field(
        default=None,
        description="TTS mode: custom_voice / voice_design / voice_clone",
    )
    speaker_name: Optional[str] = Field(default=None, description="Speaker name under custom_voice mode")
    voice_preset: Optional[str] = Field(default=None, description="Compatibility alias for speaker_name")
    instruct: Optional[str] = Field(default=None, description="Natural-language style description for TTS")
    language: Optional[str] = Field(default=None, description="TTS output language")
    sample_rate: Optional[int] = Field(default=None, description="Target sample rate")
    file_type: Optional[str] = Field(default=None, description="Audio format, e.g. wav/mp3/ogg/flac")


def _normalize_audio_output_config(request: ChatSessionCreateRequest) -> Optional[Dict[str, Any]]:
    """把请求里的 audio_output 偏好归一化成 metadata.audio_output。

    新协议下：
      - ``load_name`` 为空是合法的（推理侧 ``fixed_model`` 已声明时，dispatch 会
        走"空 load_name -> pinned 服务"分支）。这里不再兜底到任何默认模型；
      - 当整体偏好都为空（load_name / tts_mode / speaker / instruct / ...）
        时返回 None，表示调用方并未开启 voice reply。
    """
    config = request.audio_output
    if config is None:
        return None

    load_name = (config.load_name or "").strip()
    tts_mode = (config.tts_mode or "").strip()
    speaker_name = (config.speaker_name or "").strip()
    voice_preset = (config.voice_preset or "").strip()
    instruct = (config.instruct or "").strip()
    language = (config.language or "").strip()
    file_type = (config.file_type or "").strip()

    has_tts_intent = any(
        [load_name, tts_mode, speaker_name, voice_preset, instruct, language, file_type,
         config.sample_rate is not None]
    )
    if not has_tts_intent:
        return None

    payload = {
        "load_name": load_name,
        "tts_mode": tts_mode,
        "speaker_name": speaker_name,
        "voice_preset": voice_preset,
        "instruct": instruct,
        "language": language,
        "file_type": file_type,
    }
    normalized: Dict[str, Any] = {
        key: value for key, value in payload.items() if value
    }
    if config.sample_rate is not None:
        normalized["sample_rate"] = int(config.sample_rate)

    if load_name:
        resolved = _maybe_resolve_audio_model_meta(load_name)
        if resolved is not None:
            normalized["load_name"] = resolved["load_name"]
            normalized["family"] = resolved["family"]
            normalized["runtime_config"] = resolved["runtime_config"]
    return normalized or None


def _maybe_resolve_audio_model_meta(load_name: str) -> Optional[Dict[str, Any]]:
    """显式 load_name 时尝试查 DB 拿 family / runtime_config。

    查不到或名称为空时返回 None，把原 load_name 原样透传；由推理侧在 pin
    模式下根据 ``fixed_family`` 兜底，非 pin 模式下报错即可。
    """
    normalized = str(load_name or "").strip()
    if not normalized:
        return None

    model_dict = Model.get_by_load_name(normalized)
    if not model_dict:
        return None

    routed_load_name = str(model_dict.get("load_name") or normalized).strip()
    return {
        "load_name": routed_load_name,
        "family": str(model_dict.get("family") or "").strip(),
        "runtime_config": dict(model_dict.get("runtime_config") or {}),
    }


def _build_audio_input_metadata(load_name: str) -> Dict[str, Any]:
    """语音输入场景的 audio_input 元数据。

    允许 ``load_name`` 为空：dispatch 会走 pinned 服务分支；非空时尽量补齐
    family / runtime_config（便于推理侧做精细化 runtime），补不到也不阻塞。
    """
    normalized = str(load_name or "").strip()
    if not normalized:
        return {"load_name": "", "family": "", "runtime_config": {}}
    resolved = _maybe_resolve_audio_model_meta(normalized)
    if resolved is not None:
        return resolved
    return {"load_name": normalized, "family": "", "runtime_config": {}}


def _build_session_metadata(request: ChatSessionCreateRequest) -> Dict[str, Any]:
    metadata = dict(request.metadata or {})
    metadata.setdefault("input_mode", request.input_mode)
    metadata.setdefault("output_mode", request.output_mode)

    if request.input_mode in {"audio_once", "audio_stream", "mixed"}:
        metadata["audio_input"] = _build_audio_input_metadata(request.load_name or "")
    else:
        resolved_load_name = (request.load_name or "").strip() or _resolve_default_load_name()
        metadata["load_name"] = resolved_load_name

    audio_output = _normalize_audio_output_config(request)
    if audio_output:
        metadata["audio_output"] = audio_output
    return metadata


@router.post("/sessions")
async def create_chat_session(
    request: ChatSessionCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    try:
        agent_id = (request.agent_id or "").strip() or get_master_preset_agent_id()
        metadata = _build_session_metadata(request)

        session = create_conversation(
            user_id=user_id,
            agent_id=agent_id,
            title=request.title,
            metadata=metadata,
        )
        return ok(data=session)
    except Exception as exc:
        raise _to_http_exception(exc)


@router.get("/sessions")
async def list_chat_sessions(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    q: Optional[str] = Query(
        default=None,
        max_length=200,
        description="Filter by session title substring (case-insensitive); returns all when omitted (paginated)",
    ),
    user_id: str = Depends(get_current_user_id),
):
    """分页返回当前用户的 Chat 会话，按最近活动时间倒序；可选标题关键字搜索。"""
    try:
        title_query = (q or "").strip() or None
        items = list_conversations(
            user_id, limit=limit, offset=offset, title_query=title_query
        )
        return ok(data={"items": items, "count": len(items)})
    except Exception as exc:
        raise _to_http_exception(exc)


@router.get("/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    try:
        session = get_conversation(session_id, user_id=user_id)
        return ok(data=session)
    except Exception as exc:
        raise _to_http_exception(exc)


@router.get("/sessions/{session_id}/messages")
async def list_chat_session_messages(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user_id: str = Depends(get_current_user_id),
):
    try:
        items = list_messages(session_id, user_id=user_id, limit=limit, offset=offset)
        return ok(data={"items": items, "count": len(items)})
    except Exception as exc:
        raise _to_http_exception(exc)


__all__ = [
    "router",
    "ChatSessionAudioOutputRequest",
    "ChatSessionCreateRequest",
    "_build_session_metadata",
]

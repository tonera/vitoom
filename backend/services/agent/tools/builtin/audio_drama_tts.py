"""audio_drama_tts 工具：多角色对白/广播剧 TTS，输出一个合成音频文件。"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.services.agent.tools.registry import register_tool
from backend.services.tts_speakers import list_speaker_options

from ._arg_utils import clean_optional_str
from ._arg_utils import coerce_optional
from ._media_common import audio_file_fields, submit_and_collect

logger = logging.getLogger(__name__)

AUDIO_DRAMA_TTS_TOOL_NAME = "audio_drama_tts"

# 已落地的 drama 后端集合：用于按 model_name 推断走 VoxCPM 还是 Qwen-TTS 链路。
# 命中 "qwen" 前缀走 qwen-tts（design+clone），其余视作 voxcpm 链路。
_QWEN_MODEL_PREFIXES: Tuple[str, ...] = ("qwen3-tts", "qwen-tts")
_VOXCPM_MODEL_PREFIXES: Tuple[str, ...] = ("voxcpm",)

AUDIO_DRAMA_TTS_DESCRIPTION = (
    "把多角色对白/广播剧脚本合成为一个可下载的音频文件。适合用户要求『写一段三人对白并生成语音』、"
    "『A/B/C 分别是什么声音，按故事情节生成对话音频』这类任务。"
    "\n\n调用方式：LLM 先设计角色表和对白，再把结构化 characters/dialogues 传给本工具。"
    "生成 characters 时必须为每个角色定义明确声音来源：用户未指定音色时，根据角色名、性别、年龄、"
    "人物关系和剧情情绪推断并填写 voice_mode='voice_design' + instruct；用户明确指定系统预置音色时，"
    "填写 voice_mode='custom_voice' + speaker_name。不要让角色缺少声音定义，也不要依赖后端默认音色。"
    "每个角色会先生成一段 voice seed audio，后续该角色台词使用 seed audio 做 controllable clone，"
    "以保持同一角色音色一致。"
    "\n\n后端选择（一般不需要 LLM 操心）：默认不要传 model_name 和 clone_base_model_name，"
    "由推理侧根据当前在线的 TTS 服务和 character 的 voice_mode 自动决定加载哪份权重——"
    "voice_design 角色会走 Qwen3-TTS 的 VoiceDesign 权重，custom_voice 角色会走 CustomVoice 权重，"
    "clone 阶段统一切到 Base 权重。仅当用户明确点名一个后端家族时再填 model_name："
    "VoxCPM 系列填 'VoxCPM2' 之类；Qwen3-TTS 系列填 'Qwen3-TTS-12Hz-1.7B-VoiceDesign'。"
    "speaker_name 取值空间随后端不同：voxcpm 用 linda/luoli/...，qwen 用 Vivian/Serena/Uncle_Fu/...。"
    "\n\n不要用本工具修改实时聊天助手自己的声音；那是 set_chat_voice。不要用本工具处理单段朗读；那是 audio_tts。"
)

AUDIO_DRAMA_TTS_DOCSTRING = (
    "Generate a multi-character dialogue/drama audio file from structured characters and dialogue lines."
)


def _jsonish(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def _resolve_target_family(model_name: Optional[str]) -> str:
    """根据 model_name 推断目标后端 family：``"qwen"`` / ``"voxcpm"`` / ``""``。

    返回空串表示无法判定（model_name 缺省走 pin 服务）；上层应按 voxcpm 兼容
    路径处理（保留旧的 qwen-preset → voice_design 转换，避免预置音色直接落到
    voxcpm 后端时退到默认音色）。
    """
    name = (model_name or "").strip().lower()
    if not name:
        return ""
    if any(name.startswith(prefix) for prefix in _QWEN_MODEL_PREFIXES):
        return "qwen"
    if any(name.startswith(prefix) for prefix in _VOXCPM_MODEL_PREFIXES):
        return "voxcpm"
    return ""


def _normalize_characters(
    value: Any,
    *,
    target_family: str = "",
) -> List[Dict[str, Any]]:
    """规范化 LLM 给的角色列表。

    target_family：
      - ``"qwen"``：按 qwen 预置校验，``custom_voice + speaker_name`` 原样保留
        （由 qwen-tts handler 自行去 design 那段 seed），不再静默转 voice_design；
      - ``"voxcpm"``、``""``（旧默认）：按 voxcpm 预置校验；如 LLM 误填 qwen 预置，
        转成 voice_design + 由 speaker meta 拼出来的 instruct，避免落到默认音色。
    """
    raw = _jsonish(value)
    if isinstance(raw, dict):
        raw = raw.get("characters") or raw.get("items") or []
    if not isinstance(raw, list):
        return []
    voxcpm_speakers = {
        str(item.get("name") or "").strip().lower(): dict(item)
        for item in list_speaker_options("voxcpm")
        if str(item.get("name") or "").strip()
    }
    qwen_speakers = {
        str(item.get("name") or "").strip().lower(): dict(item)
        for item in list_speaker_options("qwen")
        if str(item.get("name") or "").strip()
    }
    use_qwen = target_family == "qwen"
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = clean_optional_str(item.get("id"))
        name = clean_optional_str(item.get("name"))
        voice_mode = clean_optional_str(item.get("voice_mode"))
        speaker_name = clean_optional_str(item.get("speaker_name"))
        instruct = clean_optional_str(item.get("instruct"))
        if not cid or not name or not voice_mode:
            continue

        if use_qwen:
            # qwen 后端：保留 custom_voice + qwen 预置；若 LLM 给了 voxcpm 预置
            # （或其它未知名字），转成 voice_design + 拼出来的 instruct，让 handler
            # 走 voice_design → voice_clone 链路兜底。
            if speaker_name and speaker_name.lower() not in qwen_speakers:
                vox_meta = voxcpm_speakers.get(speaker_name.lower())
                if vox_meta:
                    description = clean_optional_str(vox_meta.get("description"))
                    language_meta = clean_optional_str(vox_meta.get("language"))
                    instruct = instruct or "，".join(
                        s for s in [speaker_name, description, language_meta] if s
                    )
                voice_mode = "voice_design"
                speaker_name = None
        else:
            # voxcpm 链路（默认）：若 LLM 给的是 Qwen 预置（如 Vivian），先转成
            # voice_design 描述，避免 VoxCPM speaker preset 未命中后退到默认音色。
            if speaker_name and speaker_name.lower() not in voxcpm_speakers:
                qwen_meta = qwen_speakers.get(speaker_name.lower())
                if qwen_meta:
                    description = clean_optional_str(qwen_meta.get("description"))
                    language_meta = clean_optional_str(qwen_meta.get("language"))
                    instruct = instruct or "，".join(
                        s for s in [speaker_name, description, language_meta] if s
                    )
                    voice_mode = "voice_design"
                    speaker_name = None
        out.append(
            {
                "id": cid,
                "name": name,
                "voice_mode": voice_mode,
                "speaker_name": speaker_name,
                "instruct": instruct,
                "seed_text": clean_optional_str(item.get("seed_text")),
                "language": clean_optional_str(item.get("language")) or "Chinese",
            }
        )
    return out


def _character_voice_source_error(
    characters: List[Dict[str, Any]],
    *,
    target_family: str = "",
) -> Optional[str]:
    """按目标 family 校验角色声音定义。

    target_family ``"qwen"`` 时只接受 qwen 预置；其余按 voxcpm 预置校验。
    """
    if target_family == "qwen":
        valid_speakers = {
            str(item.get("name") or "").strip().lower()
            for item in list_speaker_options("qwen")
            if str(item.get("name") or "").strip()
        }
        family_label = "qwen"
    else:
        valid_speakers = {
            str(item.get("name") or "").strip().lower()
            for item in list_speaker_options("voxcpm")
            if str(item.get("name") or "").strip()
        }
        family_label = "voxcpm"
    for item in characters:
        cid = str(item.get("id") or "").strip()
        voice_mode = str(item.get("voice_mode") or "").strip()
        speaker_name = str(item.get("speaker_name") or "").strip()
        instruct = str(item.get("instruct") or "").strip()
        if voice_mode == "voice_design":
            if not instruct:
                return f"character {cid} requires instruct when voice_mode=voice_design"
            continue
        if voice_mode == "custom_voice":
            if not speaker_name:
                return f"character {cid} requires speaker_name when voice_mode=custom_voice"
            if speaker_name.lower() not in valid_speakers:
                return (
                    f"character {cid} uses unknown speaker_name: {speaker_name} "
                    f"(target backend family={family_label})"
                )
            continue
        return f"character {cid} has unsupported voice_mode; use voice_design or custom_voice"
    return None


def _normalize_dialogues(value: Any) -> List[Dict[str, Any]]:
    raw = _jsonish(value)
    if isinstance(raw, dict):
        raw = raw.get("dialogues") or []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker_id = clean_optional_str(item.get("speaker_id"))
        text = clean_optional_str(item.get("text"))
        if not speaker_id or not text:
            continue
        out.append(
            {
                "speaker_id": speaker_id,
                "text": text,
                "emotion": clean_optional_str(item.get("emotion")),
                "instruct": clean_optional_str(item.get("instruct")),
                "pause_after_ms": item.get("pause_after_ms"),
            }
        )
    return out


def synthesize_drama_audio_sync(
    *,
    user_id: str,
    agent_run_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    title: Optional[str] = None,
    synopsis: Optional[str] = None,
    characters: Any,
    dialogues: Any,
    model_name: Optional[str] = None,
    clone_base_model_name: Optional[str] = None,
    file_type: str = "wav",
    storage: str = "local",
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "tool": AUDIO_DRAMA_TTS_TOOL_NAME,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
    }
    if not str(user_id or "").strip():
        result["error"] = "audio_drama_tts requires a user_id (bound via agent context)"
        return result

    cleaned_model_name = clean_optional_str(model_name)
    cleaned_clone_base = clean_optional_str(clone_base_model_name)
    target_family = _resolve_target_family(cleaned_model_name)
    # 工具层不再补 clone_base_model_name 默认值——把"无 model_name 时的兜底"
    # 全部下沉到推理侧 handler（qwen-tts handler 会按 character.voice_mode 决定
    # design/clone 阶段加载哪份权重）。这条边界保持"前端传了用，没传留空"的承诺。

    normalized_characters = _normalize_characters(characters, target_family=target_family)
    normalized_dialogues = _normalize_dialogues(dialogues)
    if not normalized_characters:
        result["error"] = "characters is required and must contain at least one character"
        return result
    if not normalized_dialogues:
        result["error"] = "dialogues is required and must contain at least one dialogue line"
        return result

    voice_source_error = _character_voice_source_error(
        normalized_characters, target_family=target_family
    )
    if voice_source_error:
        result["error"] = voice_source_error
        return result

    character_ids = {item["id"] for item in normalized_characters}
    missing = [line["speaker_id"] for line in normalized_dialogues if line["speaker_id"] not in character_ids]
    if missing:
        result["error"] = f"dialogues contain unknown speaker_id: {missing[0]}"
        return result

    from backend.api.tasks.routes import TaskCreateRequest

    prompt = clean_optional_str(synopsis) or clean_optional_str(title) or "multi-character audio drama"
    request_kwargs: Dict[str, Any] = {
        "task_type": "audio",
        "job_type": "TTS",
        "audio_mode": "tts",
        "tts_mode": "custom_voice",
        "prompt": prompt,
        "response_format": "audio_file",
        "file_type": str(file_type or "wav"),
        "storage": str(storage or "local"),
        "load_name": cleaned_model_name or "",
        "family": "",
        "drama": {
            "title": clean_optional_str(title) or "",
            "synopsis": clean_optional_str(synopsis) or "",
            "characters": normalized_characters,
            "dialogues": normalized_dialogues,
        },
    }
    if str(agent_run_id or "").strip():
        request_kwargs["agent_run_id"] = str(agent_run_id).strip()
    if cleaned_clone_base:
        request_kwargs["clone_base_model_name"] = cleaned_clone_base

    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        result["error"] = f"invalid request params: {exc}"
        return result

    timeout_value = coerce_optional(timeout)
    try:
        effective_timeout = float(timeout_value) if timeout_value is not None else 600.0
    except (TypeError, ValueError):
        effective_timeout = 600.0
    if effective_timeout <= 0:
        effective_timeout = 600.0

    logger.info(
        "audio_drama_tts submitting task: user=%s characters=%d dialogues=%d model=%s family=%s clone_base=%s",
        user_id,
        len(normalized_characters),
        len(normalized_dialogues),
        request_kwargs.get("load_name") or "<pinned>",
        target_family or "<auto>",
        cleaned_clone_base or "<unset>",
    )

    return submit_and_collect(
        tool_name=AUDIO_DRAMA_TTS_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=1,
        effective_timeout=effective_timeout,
        task_kind_label="audio-drama-tts",
        file_fields=audio_file_fields(),
        extra_result_fields=("duration", "sample_rate"),
        task_event_callback=task_event_callback,
    )


@register_tool(
    name=AUDIO_DRAMA_TTS_TOOL_NAME,
    description=AUDIO_DRAMA_TTS_DESCRIPTION,
    tags=["audio", "tts", "dialogue", "drama", "广播剧", "多角色", "对白", "配音"],
    provider="local",
    enabled=True,
)
def build_audio_drama_tts_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()
    bound_agent_run_id = str(ctx.get("agent_run_id") or "").strip()
    bound_task_event_callback = ctx.get("task_event_callback")

    try:
        from crewai.tools import BaseTool
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, Field
    except Exception as exc:
        raise RuntimeError("pydantic is required to build audio_drama_tts tool") from exc

    class AudioDramaTtsArgs(BaseModel):
        title: Optional[str] = Field(default=None, description="Script title.")
        synopsis: Optional[str] = Field(default=None, description="Scene/plot synopsis.")
        characters: List[Dict[str, Any]] = Field(
            ...,
            description=(
                "Character list. Each item includes id/name/voice_mode/speaker_name/instruct/seed_text/language. "
                "voice_mode may be voice_design or custom_voice. When no preset voice is specified, infer and fill "
                "voice_design + instruct from character context; when a preset is specified, use custom_voice + "
                "speaker_name. VoxCPM presets: voxcpm family (linda/luoli/...); Qwen presets: qwen family "
                "(Vivian/Serena/Uncle_Fu/...). Do not leave voice definition empty."
            ),
        )
        dialogues: List[Dict[str, Any]] = Field(
            ...,
            description="Dialogue lines. Each item includes speaker_id/text; optional emotion/instruct/pause_after_ms.",
        )
        model_name: Optional[str] = Field(
            default=None,
            description=(
                "Usually leave empty. Fill only when user explicitly names a TTS backend: "
                "VoxCPM series: 'VoxCPM2' / 'VoxCPM1.5' / 'VoxCPM-0.5B'; "
                "Qwen3-TTS series: 'Qwen3-TTS-12Hz-1.7B-VoiceDesign'. "
                "When empty, backend dispatch routes to any online TTS service; "
                "inference side picks weights by character.voice_mode."
            ),
        )
        clone_base_model_name: Optional[str] = Field(
            default=None,
            description=(
                "Usually leave empty. Qwen3-TTS Base weight name for voice_clone stage. "
                "Defaults to 'Qwen3-TTS-12Hz-1.7B-Base' when empty. Not needed for VoxCPM."
            ),
        )
        timeout: Optional[float] = Field(default=None, description="Task wait timeout in seconds, default 600.")

    class AudioDramaTtsTool(BaseTool):
        name: str = AUDIO_DRAMA_TTS_TOOL_NAME
        description: str = AUDIO_DRAMA_TTS_DESCRIPTION
        args_schema: type = AudioDramaTtsArgs

        def _run(self, **kwargs: Any) -> str:
            payload = synthesize_drama_audio_sync(
                user_id=bound_user_id,
                agent_run_id=bound_agent_run_id or None,
                task_event_callback=bound_task_event_callback if callable(bound_task_event_callback) else None,
                **kwargs,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = AudioDramaTtsTool()
    tool_instance.__doc__ = AUDIO_DRAMA_TTS_DOCSTRING
    return tool_instance

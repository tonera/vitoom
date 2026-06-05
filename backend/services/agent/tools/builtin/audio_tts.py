"""audio_tts 工具：文字转语音（Qwen3-TTS 四种模式合一）。

本工具把 Qwen3-TTS 的 4 种玩法通过一个 `tts_mode` 字段统一入口：

- `custom_voice`（默认）：使用 CustomVoice 权重 + 9 个预置 speaker（可叠 instruct 做情感/风格控制）
- `voice_design`：使用 VoiceDesign 权重 + 自然语言 instruct 描述声线（没有 speaker）
- `voice_clone`：使用 Base 权重 + 用户录音 ref_audio + 参考文本 ref_text 零样本克隆
- `voice_design_then_clone`：两步工作流（VoiceDesign 合成参考音 → Base 克隆）；
  **仅供 slash command 调用**，不暴露给 Master Agent（LLM）

LLM 工具层面只开放前 3 种模式；DtC 仅由 slash_commands_audio_tts 通过 `_allow_design_then_clone=True`
参数显式触发。

工具成功时 JSON 的 ``files[]`` 含 **url**（完整可播放地址）、可选 **thumb_url**（图/视频才有，TTS 音频一般无）及 ``storage_path``。
向用户写 Markdown 链接时：**有 ``thumb_url`` 则用 ``thumb_url``，否则用 ``url``**；须整段原样复制，**禁止**用 ``storage_path`` 自拼域名。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import clean_optional_str as _clean_optional_str
from ._arg_utils import coerce_optional as _coerce_optional
from ._media_common import audio_file_fields, submit_and_collect

logger = logging.getLogger(__name__)

AUDIO_TTS_TOOL_NAME = "audio_tts"

_TTS_MODE_ALIASES = {
    "": "custom_voice",
    "custom_voice": "custom_voice",
    "customvoice": "custom_voice",
    "custom": "custom_voice",
    "voice_design": "voice_design",
    "voicedesign": "voice_design",
    "design": "voice_design",
    "voice_clone": "voice_clone",
    "voiceclone": "voice_clone",
    "clone": "voice_clone",
    "voice_design_then_clone": "voice_design_then_clone",
    "voicedesignthenclone": "voice_design_then_clone",
    "design_then_clone": "voice_design_then_clone",
    "designclone": "voice_design_then_clone",
    "dtc": "voice_design_then_clone",
}


def _normalize_tts_mode(raw: Optional[str]) -> str:
    key = str(raw or "").strip().lower().replace("-", "_")
    return _TTS_MODE_ALIASES.get(key, "")


AUDIO_TTS_DESCRIPTION = (
    "把一段文本合成为语音（TTS / text-to-speech）。仅当用户明确要求『把这段文字转成语音 / 念出来 / "
    "用某个音色说出来 / 模仿这段录音说出来』时使用；不要用来生成背景音乐、音效或改写文本。"
    "\n\n支持三种合成模式（字段 tts_mode）："
    "\n1. custom_voice（默认）：在预置 9 个说话人（aiden/dylan/eric/ono_anna/ryan/serena/sohee/"
    "uncle_fu/vivian）里选一个；可用 speaker_name 指定，用 instruct 控制情感/风格（沙哑/撒娇/愤怒等）。"
    "\n2. voice_design：无预置说话人，完全由 instruct 描述声线，如『磁性的低沉中年男声，语速偏慢』；"
    "此模式下不要传 speaker_name。"
    "\n3. voice_clone：零样本克隆用户提供的音频。必须同时提供 ref_audio（音频 URL 或本地路径）"
    "和 ref_text（该音频里所念的文字）。用户典型表述：『用这段录音里的声音念……』、『模仿这段音频说……』。"
    "\n\n工具调用要点：要合成的目标文本必须写进 prompt；不要自己编造 model_name（不接受该参数）；"
    "根据用户意图选择 tts_mode；speaker_name 只在 custom_voice 下生效。"
    "\n\n返回后写给用户时：``files[]`` 中有``url``则用 ``url``**"
    "作为试听链接（须整段原样复制）；**禁止**用 ``storage_path`` 拼 ``https://…`` 类自造 URL。"
)

AUDIO_TTS_DOCSTRING = (
    "Synthesize speech audio from a text prompt. Returns JSON files[] with url."
    "For user-facing markdown links: use ``url``; copy verbatim."
)


def _default_timeout() -> float:
    try:
        from backend.services.agent.settings import get_audio_tts_default_timeout

        return float(get_audio_tts_default_timeout())
    except Exception:
        return 240.0


def _optional_float(value: Any) -> Optional[float]:
    coerced = _coerce_optional(value)
    if coerced is None:
        return None
    try:
        return float(coerced)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    coerced = _coerce_optional(value)
    if coerced is None:
        return None
    try:
        return int(coerced)
    except (TypeError, ValueError):
        return None


def synthesize_audio_sync(
    *,
    user_id: str,
    agent_run_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    prompt: str,
    tts_mode: Optional[str] = None,
    model_name: Optional[str] = None,
    clone_base_model_name: Optional[str] = None,
    instruct: Optional[str] = None,
    voice_preset: Optional[str] = None,
    speaker_name: Optional[str] = None,
    # voice_clone 专用（prompt_wav_path / prompt_text 作为兼容别名）
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    x_vector_only: bool = False,
    # voice_design_then_clone 专用（仅 slash 触发时才会传）
    design_seed_text: Optional[str] = None,
    design_instruct: Optional[str] = None,
    # 其它
    language: Optional[str] = None,
    sample_rate: Optional[int] = None,
    file_type: str = "wav",
    storage: str = "local",
    stream: bool = False,
    timeout: Optional[float] = None,
    job_type: str = "TTS",
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    # 内部开关：只有 slash command 会把它置 True；LLM 走 registry 永远是 False
    _allow_design_then_clone: bool = False,
) -> Dict[str, Any]:
    """同步触发一次 TTS 任务并等待完成，返回结构化结果。

    tts_mode 路由与必填字段：
      - custom_voice: prompt（speaker_name/instruct 选填）
      - voice_design: prompt + (instruct 或 design_instruct)
      - voice_clone : prompt + ref_audio + (ref_text 或 x_vector_only=True)
      - voice_design_then_clone: prompt + design_seed_text + (design_instruct 或 instruct)
        仅在 _allow_design_then_clone=True 时可用（slash command 专用）
    """
    tool_result: Dict[str, Any] = {
        "tool": AUDIO_TTS_TOOL_NAME,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
    }

    if not str(user_id or "").strip():
        tool_result["error"] = "audio_tts requires a user_id (bound via agent context)"
        return tool_result

    prompt_value = str(prompt or "").strip()
    if not prompt_value:
        tool_result["error"] = "prompt (text to synthesize) is required"
        return tool_result

    model_name = _clean_optional_str(model_name)
    instruct = _clean_optional_str(instruct)
    voice_preset = _clean_optional_str(voice_preset)
    speaker_name = _clean_optional_str(speaker_name)
    ref_audio = _clean_optional_str(ref_audio)
    ref_text = _clean_optional_str(ref_text)
    prompt_wav_path = _clean_optional_str(prompt_wav_path)
    prompt_text = _clean_optional_str(prompt_text)
    # 向后兼容：LLM/老协议里常把参考音写 prompt_wav_path / prompt_text，
    # 这里仅在 voice_clone 场景下把它们视作 ref_audio / ref_text 的别名。
    # 注意：Qwen-TTS handler 内部 `prompt_text` 语义是"instruct 兜底"，
    # 这里合并到 ref_text 后就不会再被下发为 prompt_text，避免跟 instruct 打架。
    if not ref_audio and prompt_wav_path:
        ref_audio = prompt_wav_path
    if not ref_text and prompt_text:
        ref_text = prompt_text
    design_seed_text = _clean_optional_str(design_seed_text)
    design_instruct = _clean_optional_str(design_instruct)
    language = _clean_optional_str(language)
    timeout = _optional_float(timeout)
    guidance_scale = _optional_float(guidance_scale)
    num_inference_steps = _optional_int(num_inference_steps)

    mode = _normalize_tts_mode(tts_mode)
    if not mode:
        tool_result["error"] = (
            "tts_mode must be one of: custom_voice, voice_design, voice_clone"
        )
        return tool_result

    # DtC 只允许 slash command 内部走；LLM 误传时忽略并回退到 voice_design
    if mode == "voice_design_then_clone" and not _allow_design_then_clone:
        logger.warning(
            "audio_tts: tts_mode=voice_design_then_clone ignored because this path is slash-only; "
            "falling back to voice_design"
        )
        mode = "voice_design"

    # 新协议：不再对 model_name 做默认回填。
    # - 推理侧若声明 config.fixed_model / fixed_family（pin 模式），允许 model_name 为空，
    #   由 dispatch 路由到 pinned 服务，推理侧用 fixed_model 覆盖；
    # - voice_design_then_clone 两步流必须由调用方显式给出两阶段权重
    #   （``model_name`` 指向 VoiceDesign，``clone_base_model_name`` 指向 Base）。
    effective_model_name = (model_name or "").strip()
    effective_clone_base = (clone_base_model_name or "").strip() or None

    if mode == "voice_design_then_clone":
        if not effective_model_name:
            tool_result["error"] = (
                "voice_design_then_clone requires explicit model_name (VoiceDesign stage weight)"
            )
            return tool_result
        if not effective_clone_base:
            tool_result["error"] = (
                "voice_design_then_clone requires explicit clone_base_model_name "
                "(Qwen3-TTS Base stage weight)"
            )
            return tool_result

    # 字段必填校验
    if mode == "voice_design":
        if not (instruct or design_instruct):
            tool_result["error"] = "tts_mode=voice_design requires 'instruct' (声线描述)"
            return tool_result
        speaker_name = None
        voice_preset = None
    elif mode == "voice_clone":
        if not ref_audio:
            tool_result["error"] = (
                "tts_mode=voice_clone requires 'ref_audio' (URL 或本地路径)"
            )
            return tool_result
        if not ref_text and not x_vector_only:
            tool_result["error"] = (
                "tts_mode=voice_clone requires 'ref_text' unless x_vector_only=True"
            )
            return tool_result
        speaker_name = None
        voice_preset = None
    elif mode == "voice_design_then_clone":
        if not design_seed_text:
            tool_result["error"] = (
                "tts_mode=voice_design_then_clone requires 'design_seed_text'"
            )
            return tool_result
        if not (design_instruct or instruct):
            tool_result["error"] = (
                "tts_mode=voice_design_then_clone requires 'design_instruct' (or instruct)"
            )
            return tool_result
        speaker_name = None
        voice_preset = None

    from backend.api.tasks.routes import TaskCreateRequest

    request_kwargs: Dict[str, Any] = {
        "task_type": "audio",
        "job_type": str(job_type or "TTS").upper(),
        "prompt": prompt_value,
        "audio_mode": "tts",
        "tts_mode": mode,
        "response_format": "audio_file",
        "file_type": str(file_type or "wav"),
        "stream": bool(stream),
        "storage": str(storage or "local"),
        "load_name": effective_model_name,
        "family": "",
    }
    if str(agent_run_id or "").strip():
        request_kwargs["agent_run_id"] = str(agent_run_id).strip()
    if instruct:
        request_kwargs["instruct"] = instruct
    if voice_preset:
        request_kwargs["voice_preset"] = voice_preset
    if speaker_name:
        request_kwargs["speaker_name"] = speaker_name
    if ref_audio:
        request_kwargs["ref_audio"] = ref_audio
    if ref_text:
        request_kwargs["ref_text"] = ref_text
    if x_vector_only:
        request_kwargs["x_vector_only"] = True
    if effective_clone_base:
        request_kwargs["clone_base_model_name"] = effective_clone_base
    if design_seed_text:
        request_kwargs["design_seed_text"] = design_seed_text
    if design_instruct or (mode == "voice_design_then_clone" and instruct):
        request_kwargs["design_instruct"] = design_instruct or instruct
    if language:
        request_kwargs["language"] = language
    if sample_rate is not None:
        try:
            request_kwargs["sample_rate"] = max(8000, min(96000, int(sample_rate)))
        except (TypeError, ValueError):
            pass
    if guidance_scale is not None:
        try:
            request_kwargs["guidance_scale"] = max(0.0, min(20.0, float(guidance_scale)))
        except (TypeError, ValueError):
            pass
    if num_inference_steps is not None:
        try:
            request_kwargs["num_inference_steps"] = max(1, min(100, int(num_inference_steps)))
        except (TypeError, ValueError):
            pass

    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        tool_result["error"] = f"invalid request params: {exc}"
        return tool_result

    effective_timeout = timeout if timeout is not None else _default_timeout()
    if effective_timeout <= 0:
        effective_timeout = 240.0

    logger.info(
        "audio_tts submitting task: user=%s tts_mode=%s model=%s "
        "speaker=%s has_ref_audio=%s has_ref_text=%s x_vector_only=%s stream=%s",
        user_id,
        mode,
        request_kwargs.get("load_name"),
        request_kwargs.get("speaker_name") or request_kwargs.get("voice_preset"),
        bool(ref_audio),
        bool(ref_text),
        x_vector_only,
        bool(stream),
    )

    return submit_and_collect(
        tool_name=AUDIO_TTS_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=1,
        effective_timeout=effective_timeout,
        task_kind_label="audio-tts",
        file_fields=audio_file_fields(),
        extra_result_fields=("duration", "sample_rate"),
        task_event_callback=task_event_callback,
    )


# ----------------------- CrewAI 工具包装 -----------------------


@register_tool(
    name=AUDIO_TTS_TOOL_NAME,
    description=AUDIO_TTS_DESCRIPTION,
    tags=[
        "audio", "tts", "text-to-speech",
        "文字转语音", "语音合成", "念出来", "朗读",
        "voice_clone", "克隆", "模仿声音",
    ],
    provider="local",
    enabled=True,
)
def build_audio_tts_tool(*, context: Optional[Dict[str, Any]] = None):
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
        raise RuntimeError("pydantic is required to build audio_tts tool") from exc

    class AudioTtsArgs(BaseModel):
        prompt: str = Field(..., description="Text to read/synthesize (any language).")
        tts_mode: Optional[str] = Field(
            default=None,
            description=(
                "Synthesis mode: custom_voice (default) / voice_design / voice_clone. "
                "Use custom_voice when user specifies a preset speaker; voice_design when describing "
                "a voice without a specific speaker; voice_clone when mimicking reference audio."
            ),
        )
        instruct: Optional[str] = Field(
            default=None,
            description=(
                "Natural-language style/emotion/voice description. Under custom_voice, controls emotion; "
                "under voice_design, describes the entire voice. Generally not needed for voice_clone."
            ),
        )
        speaker_name: Optional[str] = Field(
            default=None,
            description=(
                "Preset speaker under custom_voice mode; one of: "
                "aiden / dylan / eric / ono_anna / ryan / serena / sohee / uncle_fu / vivian. "
                "Do not pass for voice_design or voice_clone modes."
            ),
        )
        voice_preset: Optional[str] = Field(
            default=None,
            description="Compatibility alias for speaker_name; speaker_name takes precedence.",
        )
        ref_audio: Optional[str] = Field(
            default=None,
            description=(
                "Required for voice_clone: reference audio (voice source), http(s) URL or local absolute path. "
                "Do not pass for other modes."
            ),
        )
        ref_text: Optional[str] = Field(
            default=None,
            description=(
                "Required for voice_clone: transcript of ref_audio (must match exactly) to avoid accent drift. "
                "May be omitted if x_vector_only=True, but clone quality degrades."
            ),
        )
        x_vector_only: bool = Field(
            default=False,
            description="voice_clone option: use speaker embedding only, allows omitting ref_text.",
        )
        language: Optional[str] = Field(
            default=None, description="Language code (zh/en/...); uses model default/auto if omitted.",
        )
        sample_rate: Optional[int] = Field(
            default=None, description="Sample rate (8000-96000); uses model default if omitted.",
        )
        file_type: str = Field(
            default="wav", description="Output audio format: wav/mp3/ogg/flac.",
        )
        timeout: Optional[float] = Field(
            default=None, description="Synthesis timeout in seconds, default 240s.",
        )

    class AudioTtsTool(BaseTool):
        name: str = AUDIO_TTS_TOOL_NAME
        description: str = AUDIO_TTS_DESCRIPTION
        args_schema: type = AudioTtsArgs

        def _run(self, **kwargs: Any) -> str:
            # LLM 通道永远不开 DtC
            kwargs.pop("_allow_design_then_clone", None)
            payload = synthesize_audio_sync(
                user_id=bound_user_id,
                agent_run_id=bound_agent_run_id or None,
                task_event_callback=bound_task_event_callback if callable(bound_task_event_callback) else None,
                _allow_design_then_clone=False,
                **kwargs,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = AudioTtsTool()
    tool_instance.__doc__ = AUDIO_TTS_DOCSTRING
    return tool_instance

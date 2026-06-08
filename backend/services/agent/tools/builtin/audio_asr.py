"""audio_asr 工具：语音转文字（ASR）。

与 `image_generator` / `video_generator` 同构：共享
`_media_common.submit_and_collect` 负责"提交任务 + 订阅 WS + 聚合结果"。
特化点：

1. `TaskCreateRequest.task_type="audio"`, `audio_mode="asr"`
2. 必须传 `input_audio_url`（或 `prompt_wav_path` 兼容）作为输入音频
3. 最终产物是 `.txt`（`response_format="text_file"` 由后端自动强制）
4. **文本返回走流式**：`stream=True`，工具侧捕获推理器发来的
   `text_stream_delta` 并累加到返回结果的 `text` 字段
5. `model_name` 不再有配置默认：若目标推理器声明了 ``fixed_model``（pin 模式），
   允许调用方不传，dispatch 会把任务路由到 pinned 服务；否则必须显式传入

返回结构与 image/video 对齐，且多出 `text / stream_deltas` 两个字段：
    {
      "tool": "audio_asr",
      "task_id": "...",
      "status": "completed",
      "total": 1,
      "files": [{file_id, url, thumb_url?, ..., mime_type}],
      "text": "这是识别出的完整文本...",
      "stream_deltas": [{type, sequence, delta, ...}, ...],
      "error"?: "..."
    }

调用者（Master Agent / `/audio-asr` 斜杠命令）识别正文看 `text`；若要在回复里附下载链接：
``files[]`` 有 ``thumb_url`` 则用 ``thumb_url``，否则用 ``url``（``.txt`` 的完整地址）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import clean_optional_str as _clean_optional_str
from ._media_common import audio_file_fields, submit_and_collect

logger = logging.getLogger(__name__)

AUDIO_ASR_TOOL_NAME = "audio_asr"


AUDIO_ASR_DESCRIPTION = (
    "把一段音频转写成文字（ASR / speech-to-text）。"
    "仅当用户明确要求『把这段音频转成文字 / 识别这段语音 / 帮我听写』且"
    "提供了音频 URL 时使用；不要用来给图片/视频加字幕或做其他通用分析。"
    "不要自己编造模型名；本工具不接受 model_name 参数，后端会自动选用合适的 ASR 模型。"
)

AUDIO_ASR_DOCSTRING = (
    "Transcribe an audio URL into Chinese/English text. Returns a JSON "
    "code block containing the recognized text plus the saved .txt file URL."
)


def _default_timeout() -> float:
    try:
        from backend.services.agent.settings import get_audio_asr_default_timeout

        return float(get_audio_asr_default_timeout())
    except Exception:
        return 180.0


def transcribe_audio_sync(
    *,
    user_id: str,
    agent_run_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    url: Optional[str] = None,
    input_audio_url: Optional[str] = None,
    prompt_wav_path: Optional[str] = None,
    model_name: Optional[str] = None,
    family: Optional[str] = None,
    prompt_text: Optional[str] = None,
    language: Optional[str] = None,
    timestamps: bool = False,
    speaker_diarization: bool = False,
    sample_rate: Optional[int] = None,
    stream: bool = True,
    storage: str = "local",
    timeout: Optional[float] = None,
    job_type: str = "ASR",
) -> Dict[str, Any]:
    """同步触发一次 ASR 任务并等待完成，返回结构化结果。

    Args:
        url / input_audio_url / prompt_wav_path: 三者任选其一作为输入音频；
            优先级 `input_audio_url > url > prompt_wav_path`。
        language: 语言代码（zh/en/ja...）；不传走自动识别。
        timestamps: 是否要求 forced-aligner 生成时间戳。
        stream: 是否走流式文本增量；默认 True，聚合到 `text` 字段。
    """
    tool_result: Dict[str, Any] = {
        "tool": AUDIO_ASR_TOOL_NAME,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
        "text": "",
    }

    if not str(user_id or "").strip():
        tool_result["error"] = "audio_asr requires a user_id (bound via agent context)"
        return tool_result

    url = _clean_optional_str(url)
    input_audio_url = _clean_optional_str(input_audio_url)
    prompt_wav_path = _clean_optional_str(prompt_wav_path)
    model_name = _clean_optional_str(model_name)
    family = _clean_optional_str(family)
    prompt_text = _clean_optional_str(prompt_text)
    language = _clean_optional_str(language)

    effective_input = str(
        input_audio_url or url or prompt_wav_path or ""
    ).strip()
    if not effective_input:
        tool_result["error"] = (
            "input audio url is required (pass url / input_audio_url / prompt_wav_path)"
        )
        return tool_result

    from backend.api.tasks.routes import TaskCreateRequest

    request_kwargs: Dict[str, Any] = {
        "task_type": "audio",
        "job_type": str(job_type or "ASR").upper(),
        "prompt": "",
        "audio_mode": "asr",
        "input_audio_url": effective_input,
        "response_format": "text_file",
        "file_type": "txt",
        "stream": bool(stream),
        "storage": str(storage or "local"),
        "timestamps": bool(timestamps),
        "speaker_diarization": bool(speaker_diarization),
    }
    if str(agent_run_id or "").strip():
        request_kwargs["agent_run_id"] = str(agent_run_id).strip()

    # 新协议：不再对 model_name 做默认回填。
    # - 推理侧声明 fixed_model 的 pinned 服务允许 model_name 为空，dispatch 会路由到它；
    # - 非 pin 模式下必须显式传 load_name，否则 dispatch 会直接失败。
    effective_model_name = str(model_name or "").strip()
    if effective_model_name:
        request_kwargs["load_name"] = effective_model_name
    if family:
        request_kwargs["family"] = str(family).strip()
    if prompt_text:
        request_kwargs["prompt_text"] = str(prompt_text)
    if language:
        request_kwargs["language"] = str(language).strip()
    if sample_rate is not None:
        try:
            request_kwargs["sample_rate"] = max(8000, min(96000, int(sample_rate)))
        except (TypeError, ValueError):
            pass

    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        tool_result["error"] = f"invalid request params: {exc}"
        return tool_result

    effective_timeout = float(timeout) if timeout else _default_timeout()
    if effective_timeout <= 0:
        effective_timeout = 180.0

    logger.info(
        "audio_asr submitting task: user=%s input=%s model=%s stream=%s language=%s",
        user_id,
        effective_input,
        request_kwargs.get("load_name"),
        bool(stream),
        language,
    )

    stream_types = ("text_stream_delta", "transcript_segment") if stream else ()
    extra_result_fields = ("language", "segments", "duration")

    return submit_and_collect(
        tool_name=AUDIO_ASR_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=1,
        effective_timeout=effective_timeout,
        task_kind_label="audio-asr",
        file_fields=audio_file_fields(),
        stream_text_types=stream_types,
        extra_result_fields=extra_result_fields,
        task_event_callback=task_event_callback,
    )


# ----------------------- CrewAI 工具包装 -----------------------


@register_tool(
    name=AUDIO_ASR_TOOL_NAME,
    description=AUDIO_ASR_DESCRIPTION,
    tags=[
        "audio", "asr", "speech-to-text", "stt",
        "语音转文字", "语音识别", "听写", "转录",
    ],
    provider="local",
    enabled=True,
)
def build_audio_asr_tool(*, context: Optional[Dict[str, Any]] = None):
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
        raise RuntimeError("pydantic is required to build audio_asr tool") from exc

    class AudioAsrArgs(BaseModel):
        url: Optional[str] = Field(
            default=None,
            description="Audio URL to transcribe (recommended: pass audio link directly, equivalent to input_audio_url).",
        )
        input_audio_url: Optional[str] = Field(
            default=None,
            description="Audio URL to transcribe; interchangeable with `url`.",
        )
        language: Optional[str] = Field(
            default=None,
            description="Language code, e.g. `zh` / `en` / `ja`; auto-detect if omitted.",
        )
        prompt_text: Optional[str] = Field(
            default=None,
            description="Recognition context/hint text (e.g. domain terms) to help the model.",
        )
        timestamps: bool = Field(
            default=False,
            description="Whether to return segment timestamps (requires forced aligner).",
        )
        speaker_diarization: bool = Field(
            default=False, description="Whether to return speaker diarization (not supported by all models).",
        )
        stream: bool = Field(
            default=True,
            description="Whether to stream text output. Enabled by default; when disabled, returns final full text only.",
        )
        timeout: Optional[float] = Field(
            default=None, description="Recognition timeout in seconds, default 180s.",
        )

    class AudioAsrTool(BaseTool):
        name: str = AUDIO_ASR_TOOL_NAME
        description: str = AUDIO_ASR_DESCRIPTION
        args_schema: type = AudioAsrArgs

        def _run(self, **kwargs: Any) -> str:
            payload = transcribe_audio_sync(
                user_id=bound_user_id,
                agent_run_id=bound_agent_run_id or None,
                task_event_callback=bound_task_event_callback if callable(bound_task_event_callback) else None,
                **kwargs,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = AudioAsrTool()
    tool_instance.__doc__ = AUDIO_ASR_DOCSTRING
    return tool_instance

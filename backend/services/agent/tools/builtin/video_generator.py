"""video_generator 工具：文生视频 / 图生视频。

与 `image_generator` 同构：共享 `_media_common.submit_and_collect` 负责
"提交任务 + 订阅 WS + 聚合结果"；本文件只负责：

1. 把工具参数映射到 `TaskCreateRequest(task_type="video")`
2. 尺寸派生：优先 width/height；其次 resolution/size；再退化到 aspect_ratio
3. 以 `@register_tool` 注册到 Master Agent
4. 对外暴露 `generate_video_sync`，供 `/video` 斜杠命令复用

关键约束：
- `model_name` 不传时使用 `config/default.yaml::agents.tools.video_generator.default_model_name`
  兜底；都没有再让 `TaskCreateRequest` 校验失败报错
- 默认超时更长（默认 600s；视频任务更重）
- 返回结构与 image 一致：`{tool, task_id, status, total, files[], error?}`
  其中 video 的每个 file 额外携带 `duration / fps / resolution / aspect_ratio /
  mime_type / file_size`（若推理器在消息里提供）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import clean_optional_str
from ._media_common import submit_and_collect, video_file_fields

logger = logging.getLogger(__name__)

VIDEO_GENERATOR_TOOL_NAME = "video_generator"

# 视频推理器支持的 job_type（与 inference/video/inferrer.py 一致）
VALID_VIDEO_JOB_TYPES = frozenset({"MKV", "S2V", "INP", "CCV"})
VIDEO_DEFAULT_JOB_TYPE = "MKV"

VIDEO_GENERATOR_DESCRIPTION = (
    "根据中文文本描述或参考图/视频生成新的视频（文生视频/图生视频/视频编辑）。"
    "仅当用户明确提出『生成一段视频/做个动画/合成视频』等视频生成需求时使用；"
    "不要用于分析已有 URL 的视频（那是 analyze_media 的职责），也不要与"
    " image_generator 混用。"
    "model_name 可选：不传时使用后端配置的默认视频模型；仅当用户明确指定某个"
    "模型时才需要填写。"
    "返回的 ``files[]``：向用户写 Markdown 预览/链接时，**有 ``thumb_url`` 则用 ``thumb_url``，否则用 ``url``**；须原样复制，禁止用 ``storage_path`` 自拼域名。"
)

VIDEO_GENERATOR_DOCSTRING = (
    "Generate new videos from a Chinese text prompt or reference media. "
    "Returns JSON with task_id and files[]; for markdown use thumb_url if present else url."
)


# ----------------------- 尺寸 / 比例工具 -----------------------
# 说明：视频尺寸与 image 的对齐要求不完全一致：
# - 很多视频模型要求 16 的倍数或特定分辨率（如 720p/1080p），这里仅做"8 的倍数对齐"
#   保证不会落到非法尺寸；具体模型约束交给推理器校验。

VIDEO_ALIGNMENT = 8
VIDEO_MIN_EDGE = 64
VIDEO_MAX_EDGE = 4096
DEFAULT_VIDEO_SHORT_EDGE = 720


def _align(value: int, alignment: int = VIDEO_ALIGNMENT) -> int:
    if value <= 0:
        return 0
    return max(alignment, int(round(value / alignment) * alignment))


def _clamp(value: int) -> int:
    return max(VIDEO_MIN_EDGE, min(VIDEO_MAX_EDGE, value))


def _parse_two_number_string(text: str) -> Optional[tuple[float, float]]:
    if not isinstance(text, str):
        return None
    s = text.strip().lower()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
    elif "x" in s:
        parts = s.split("x")
    elif "*" in s:
        parts = s.split("*")
    else:
        return None
    if len(parts) != 2:
        return None
    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError:
        return None
    if a <= 0 or b <= 0:
        return None
    return (a, b)


def _resolve_video_size(
    *,
    width: Optional[int],
    height: Optional[int],
    aspect_ratio: Optional[str],
    resolution: Optional[str],
    short_edge: int = DEFAULT_VIDEO_SHORT_EDGE,
) -> Optional[tuple[int, int]]:
    """
    解析视频尺寸：
      - width/height 同时给则优先
      - resolution 支持 '1280x720' 这类；也支持纯数字（作为短边基准）
      - aspect_ratio 需要配合 short_edge

    Returns:
      (width, height) 或 None（表示"让推理器按模型默认处理"）
    """
    if width and height and width > 0 and height > 0:
        return _clamp(_align(int(width))), _clamp(_align(int(height)))

    if resolution:
        parsed = _parse_two_number_string(resolution)
        if parsed is not None:
            w, h = parsed
            return _clamp(_align(int(w))), _clamp(_align(int(h)))
        try:
            edge = int(str(resolution).strip().rstrip("p"))
            if edge > 0:
                short_edge = edge
        except ValueError:
            raise ValueError(
                f"invalid --resolution value: {resolution!r}, expected e.g. '1280x720' or '720'"
            )

    if aspect_ratio:
        parsed = _parse_two_number_string(aspect_ratio)
        if parsed is None:
            raise ValueError(
                f"invalid aspect_ratio: {aspect_ratio!r}, expected e.g. '16:9'"
            )
        a, b = parsed
        scale = short_edge / min(a, b)
        return _clamp(_align(int(round(a * scale)))), _clamp(_align(int(round(b * scale))))

    return None


# ----------------------- loras 规范化（和 image 完全一致，暂不共享以减少耦合）-----------------------


def _normalize_loras(value: Any) -> Optional[Any]:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                try:
                    weight_f = float(item.get("weight", 1.0))
                except (TypeError, ValueError):
                    weight_f = 1.0
                normalized.append({"name": str(item["name"]).strip(), "weight": weight_f})
                continue
            if isinstance(item, str) and item.strip():
                text = item.strip()
                if ":" in text:
                    name_part, weight_part = text.rsplit(":", 1)
                    try:
                        weight_f = float(weight_part)
                    except ValueError:
                        weight_f = 1.0
                    normalized.append({"name": name_part.strip(), "weight": weight_f})
                else:
                    normalized.append({"name": text, "weight": 1.0})
        return normalized or None
    if isinstance(value, dict) and value.get("name"):
        try:
            weight_f = float(value.get("weight", 1.0))
        except (TypeError, ValueError):
            weight_f = 1.0
        return [{"name": str(value["name"]).strip(), "weight": weight_f}]
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _resolve_video_job_type(
    *,
    job_type: Optional[str],
    url: Optional[str] = None,
    ref_video: Optional[str] = None,
    face_video: Optional[str] = None,
    image_file2: str = "",
    prompt_wav_path: Optional[str] = None,
    direction: Optional[str] = None,
) -> str:
    """解析视频 job_type：纠正误用的 MK，并按输入自动推断 S2V/INP/CCV/MKV。"""
    jt = str(job_type or "").strip().upper()
    if jt == "MK":
        logger.info(
            "video_generator: job_type=MK is for image tasks; using %s instead",
            VIDEO_DEFAULT_JOB_TYPE,
        )
        jt = VIDEO_DEFAULT_JOB_TYPE
    if jt in VALID_VIDEO_JOB_TYPES:
        return jt

    if str(prompt_wav_path or "").strip() and str(url or "").strip():
        return "S2V"
    if str(direction or "").strip() and str(url or "").strip():
        return "CCV"
    if str(url or "").strip() and str(image_file2 or "").strip() and not str(ref_video or "").strip():
        return "INP"
    # MKV 内部再按 url/ref_video/face_video 路由 t2v/i2v/ti2v/vicv/ivv2v
    return VIDEO_DEFAULT_JOB_TYPE


# ----------------------- 核心同步函数 -----------------------


def _default_timeout() -> float:
    try:
        from backend.services.agent.settings import get_video_generator_default_timeout

        return float(get_video_generator_default_timeout())
    except Exception:
        return 600.0


def generate_video_sync(
    *,
    user_id: str,
    agent_run_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    prompt: str,
    negative_prompt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    model_name: Optional[str] = None,
    family: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    generate_num: int = 1,
    duration: Optional[int] = None,
    fps: Optional[int] = None,
    ref_video: Optional[str] = None,
    face_video: Optional[str] = None,
    url: Optional[str] = None,
    image_file2: str = "",
    edit_act: str = "",
    tpl_list: Optional[List[str]] = None,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    direction: Optional[str] = None,
    speed: Optional[float] = None,
    fast_mode: bool = True,
    loras: Any = None,
    job_type: str = VIDEO_DEFAULT_JOB_TYPE,
    file_type: str = "mp4",
    storage: str = "local",
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """同步触发一次视频生成任务并等待完成，返回结构化结果。

    见模块 docstring。
    """
    tool_result: Dict[str, Any] = {
        "tool": VIDEO_GENERATOR_TOOL_NAME,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
    }

    if not str(user_id or "").strip():
        tool_result["error"] = "video_generator requires a user_id (bound via agent context)"
        return tool_result

    prompt_text_value = str(prompt or "").strip()
    if not prompt_text_value:
        tool_result["error"] = "prompt is required"
        return tool_result

    try:
        size_pair = _resolve_video_size(
            width=width, height=height, aspect_ratio=aspect_ratio, resolution=resolution,
        )
    except ValueError as exc:
        tool_result["error"] = str(exc)
        return tool_result

    try:
        final_generate_num = max(1, min(10, int(generate_num or 1)))
    except (TypeError, ValueError):
        final_generate_num = 1

    normalized_loras = _normalize_loras(loras)

    # 延迟导入，避免模块加载期硬依赖 FastAPI 路由层
    from backend.api.tasks.routes import TaskCreateRequest

    effective_job_type = _resolve_video_job_type(
        job_type=job_type,
        url=url,
        ref_video=ref_video,
        face_video=face_video,
        image_file2=image_file2,
        prompt_wav_path=prompt_wav_path,
        direction=direction,
    )

    request_kwargs: Dict[str, Any] = {
        "task_type": "video",
        "job_type": effective_job_type,
        "prompt": prompt_text_value,
        "negative_prompt": negative_prompt or "",
        "generate_num": final_generate_num,
        "storage": str(storage or "local"),
        "fast_mode": bool(fast_mode),
        "file_type": str(file_type or "mp4"),
        "image_file2": str(image_file2 or ""),
        "edit_act": str(edit_act or ""),
        "tpl_list": list(tpl_list or []),
    }
    if size_pair is not None:
        request_kwargs["width"], request_kwargs["height"] = size_pair
    if aspect_ratio:
        request_kwargs["aspect_ratio"] = str(aspect_ratio)
    if resolution:
        request_kwargs["resolution"] = str(resolution)
    effective_model_name = clean_optional_str(model_name) or ""
    if not effective_model_name:
        try:
            from backend.services.agent.settings import (
                get_video_generator_default_model_name,
            )

            effective_model_name = get_video_generator_default_model_name()
        except Exception:
            effective_model_name = ""
        if effective_model_name:
            logger.info(
                "video_generator: model_name not provided, falling back to default=%s",
                effective_model_name,
            )
    if effective_model_name:
        request_kwargs["load_name"] = effective_model_name
    if str(agent_run_id or "").strip():
        request_kwargs["agent_run_id"] = str(agent_run_id).strip()
    if family:
        request_kwargs["family"] = str(family).strip()
    if num_inference_steps is not None:
        try:
            request_kwargs["num_inference_steps"] = max(1, min(100, int(num_inference_steps)))
        except (TypeError, ValueError):
            pass
    if guidance_scale is not None:
        try:
            request_kwargs["guidance_scale"] = max(0.0, min(20.0, float(guidance_scale)))
        except (TypeError, ValueError):
            pass
    if seed is not None:
        try:
            request_kwargs["seed"] = int(seed)
        except (TypeError, ValueError):
            pass
    if duration is not None:
        try:
            request_kwargs["duration"] = max(1, min(60, int(duration)))
        except (TypeError, ValueError):
            pass
    if fps is not None:
        try:
            request_kwargs["fps"] = max(1, min(60, int(fps)))
        except (TypeError, ValueError):
            pass
    if ref_video:
        request_kwargs["ref_video"] = str(ref_video)
    if face_video:
        request_kwargs["face_video"] = str(face_video)
    if url:
        request_kwargs["url"] = str(url)
    if prompt_wav_path:
        request_kwargs["prompt_wav_path"] = str(prompt_wav_path)
    if prompt_text is not None:
        request_kwargs["prompt_text"] = str(prompt_text)
    if direction:
        request_kwargs["direction"] = str(direction)
    if speed is not None:
        try:
            request_kwargs["speed"] = float(speed)
        except (TypeError, ValueError):
            pass
    if normalized_loras is not None:
        request_kwargs["loras"] = normalized_loras

    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        tool_result["error"] = f"invalid request params: {exc}"
        return tool_result

    effective_timeout = float(timeout) if timeout else _default_timeout()
    if effective_timeout <= 0:
        effective_timeout = 600.0

    logger.info(
        "video_generator submitting task: user=%s size=%s duration=%s model=%s num=%d",
        user_id,
        size_pair,
        request_kwargs.get("duration"),
        request_kwargs.get("load_name"),
        final_generate_num,
    )
    return submit_and_collect(
        tool_name=VIDEO_GENERATOR_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=final_generate_num,
        effective_timeout=effective_timeout,
        task_kind_label="video",
        file_fields=video_file_fields(),
        task_event_callback=task_event_callback,
    )


# ----------------------- CrewAI 工具包装 -----------------------


@register_tool(
    name=VIDEO_GENERATOR_TOOL_NAME,
    description=VIDEO_GENERATOR_DESCRIPTION,
    tags=["video", "generate", "文生视频", "图生视频", "视频生成", "txt2video", "AI视频"],
    provider="local",
    enabled=True,
)
def build_video_generator_tool(*, context: Optional[Dict[str, Any]] = None):
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
        raise RuntimeError("pydantic is required to build video_generator tool") from exc

    class VideoGeneratorArgs(BaseModel):
        prompt: str = Field(..., description="Video generation prompt (positive prompt).")
        negative_prompt: str = Field(default="", description="Negative prompt.")
        aspect_ratio: Optional[str] = Field(
            default=None, description="Aspect ratio, e.g. `16:9` / `9:16` / `1:1`.",
        )
        resolution: Optional[str] = Field(
            default=None, description="Resolution, e.g. `1280x720` / `720` (short side).",
        )
        width: Optional[int] = Field(default=None, description="Video width in pixels, 64-4096.")
        height: Optional[int] = Field(default=None, description="Video height in pixels, 64-4096.")
        duration: Optional[int] = Field(
            default=None, ge=1, le=60, description="Video duration in seconds, 1-60; uses model default if omitted.",
        )
        fps: Optional[int] = Field(
            default=None, ge=1, le=60, description="Frame rate, 1-60; uses model default if omitted.",
        )
        num_inference_steps: Optional[int] = Field(
            default=None, description="Sampling steps; uses model default if omitted.",
        )
        guidance_scale: Optional[float] = Field(
            default=None, description="CFG guidance scale (0-20).",
        )
        seed: Optional[int] = Field(default=None, description="Random seed; negative values mean random.")
        generate_num: int = Field(default=1, ge=1, le=10, description="Number of videos to generate.")
        url: Optional[str] = Field(default=None, description="Reference first-frame image URL (image-to-video).")
        ref_video: Optional[str] = Field(default=None, description="Reference/control video URL.")
        face_video: Optional[str] = Field(default=None, description="Face/expression driving video URL.")
        file_type: str = Field(default="mp4", description="Output format: mp4, etc.")
        loras: Optional[str] = Field(
            default=None,
            description='LoRA JSON string, e.g. `[{"name":"x.safetensors","weight":0.8}]`.',
        )
        timeout: Optional[float] = Field(
            default=None, description="Generation timeout in seconds, default 600s.",
        )

    class VideoGeneratorTool(BaseTool):
        name: str = VIDEO_GENERATOR_TOOL_NAME
        description: str = VIDEO_GENERATOR_DESCRIPTION
        args_schema: type = VideoGeneratorArgs

        def _run(self, **kwargs: Any) -> str:
            # LLM 不应选择模型/job_type；统一走后端默认与输入推断，避免误传 MK 等无效值。
            kwargs.pop("model_name", None)
            kwargs.pop("job_type", None)
            payload = generate_video_sync(
                user_id=bound_user_id,
                agent_run_id=bound_agent_run_id or None,
                task_event_callback=bound_task_event_callback if callable(bound_task_event_callback) else None,
                **kwargs,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = VideoGeneratorTool()
    tool_instance.__doc__ = VIDEO_GENERATOR_DOCSTRING
    return tool_instance

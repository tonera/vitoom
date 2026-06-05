"""image_generator 工具：文生图。

职责：把一段中文描述转成后端任务（`POST /v1/tasks` 等价调用），订阅推理器
经由 `WebSocketManager` 回传的 `task_status` / `result` 消息，聚合多张图片
后把 `task_id + files[]` 返回给调用方。

工具与服务端 `/image` 斜杠命令共用 `generate_image_sync` 纯函数，便于维护。

设计取舍：
- `model_name` 不传时使用 `config/default.yaml::agents.tools.image_generator.default_model_name`
  兜底；都没有再让 `_create_task` 抛 400
- 超时默认 200 秒，可按配置覆盖，也允许单次调用 `timeout` 参数覆盖
- 等待实现：在 FastAPI event loop 内 `register_task_subscriber`，工具侧同步
  使用 `asyncio.run_coroutine_threadsafe` 依次 `queue.get()`；这样无需启动
  额外的 WS 客户端，也不给 DB 增加轮询压力
- 终止条件：任一条件命中退出循环
    1. `type="result"` 且 `status="completed"` 且 `progress=100`
    2. `type="task_status"` 且 `status` in {failed, cancelled}
    3. 已聚合文件数 >= `expected_total`（`expected_total` 从 `generate_num`
       或 `result.total` 获取）
    4. 全局超时
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import clean_optional_str, coerce_optional
from ._media_common import submit_and_collect

logger = logging.getLogger(__name__)

IMAGE_GENERATOR_TOOL_NAME = "image_generator"

IMAGE_GENERATOR_DESCRIPTION = (
    "根据中文文本描述生成新的图片（文生图 / text-to-image）。"
    "仅当用户明确提出『画一张/生成一张/帮我画』等图片生成需求时使用；"
    "不要用于分析或描述已有 URL 的图片——那是 analyze_media 的职责。"
    "重要：调用本工具时只需描述『要画什么』，其他参数（宽高比、模型名、"
    "采样步数等）按用户实际要求填入对应字段。"
    "返回的 ``files[]`` 仅供客户端附件/下载区展示；最终文字回答只需简短总结生成结果，"
    "不要内嵌 Markdown 图片、URL 或文件路径。"
)

IMAGE_GENERATOR_DOCSTRING = (
    "Generate new images from a Chinese text prompt. Returns JSON with task_id and files[]; "
    "do not include image URLs or markdown image embeds in the final text response."
)

# 短边基准，用于 aspect_ratio -> width/height 的换算（8 的倍数对齐）
DEFAULT_SHORT_EDGE = 1024
SIZE_ALIGNMENT = 8
MIN_EDGE = 64
MAX_EDGE = 4096

VALID_UPSCALE_VALUES = {0, 1, 2, 4}


# ----------------------- 尺寸 / 比例工具 -----------------------


def _align_to(value: int, *, alignment: int = SIZE_ALIGNMENT) -> int:
    if value <= 0:
        return 0
    return max(alignment, int(round(value / alignment) * alignment))


def _clamp_edge(value: int) -> int:
    return max(MIN_EDGE, min(MAX_EDGE, value))


def _parse_aspect_ratio(aspect_ratio: str) -> Optional[tuple[float, float]]:
    """解析形如 '2:3' / '16:9' 的比例字符串，返回 (w, h)。"""
    if not isinstance(aspect_ratio, str):
        return None
    text = aspect_ratio.strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
    elif "x" in text.lower():
        parts = text.lower().split("x")
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


def _resolve_size(
    *,
    width: Optional[int],
    height: Optional[int],
    aspect_ratio: Optional[str],
    size: Optional[str],
    short_edge: int = DEFAULT_SHORT_EDGE,
) -> tuple[int, int]:
    """解析最终使用的 width/height。

    优先级：显式 width/height > size（形如 '1024x1536'） > aspect_ratio。
    `aspect_ratio` 时以 short_edge 作为短边基准换算。
    未给任何尺寸参数时，返回 (short_edge, short_edge)。
    """
    if width and height and width > 0 and height > 0:
        return _clamp_edge(_align_to(int(width))), _clamp_edge(_align_to(int(height)))

    if size:
        parsed = _parse_aspect_ratio(size)
        if parsed is None:
            raise ValueError(f"invalid --size value: {size!r}, expected e.g. '1024x1536'")
        w, h = parsed
        return _clamp_edge(_align_to(int(w))), _clamp_edge(_align_to(int(h)))

    if aspect_ratio:
        parsed = _parse_aspect_ratio(aspect_ratio)
        if parsed is None:
            raise ValueError(f"invalid aspect_ratio: {aspect_ratio!r}, expected e.g. '2:3'")
        a, b = parsed
        shorter = min(a, b)
        scale = short_edge / shorter
        w = _align_to(int(round(a * scale)))
        h = _align_to(int(round(b * scale)))
        return _clamp_edge(w), _clamp_edge(h)

    return short_edge, short_edge


# ----------------------- loras 规范化 -----------------------


def _normalize_loras(value: Any) -> Optional[Any]:
    """接受 list[dict] / JSON 字符串 / 形如 'name:weight' 的列表；输出符合
    `InferenceRequestParams.parse_loras` 语义的值（list[dict] 或 JSON 字符串）。
    """
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                weight = item.get("weight", 1.0)
                try:
                    weight_f = float(weight)
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
        weight = value.get("weight", 1.0)
        try:
            weight_f = float(weight)
        except (TypeError, ValueError):
            weight_f = 1.0
        return [{"name": str(value["name"]).strip(), "weight": weight_f}]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return text
    return None


# ----------------------- 核心同步函数 -----------------------


def _default_timeout() -> float:
    """从配置读取默认超时秒数；`settings.py` 里有专门的取值入口。"""
    try:
        from backend.services.agent.settings import get_image_generator_default_timeout

        return float(get_image_generator_default_timeout())
    except Exception:
        return 200.0


def _default_num_inference_steps() -> int:
    """从配置读取用户未显式指定时的默认采样步数。"""
    try:
        from backend.services.agent.settings import get_image_generator_default_num_inference_steps

        return int(get_image_generator_default_num_inference_steps())
    except Exception:
        logger.exception(
            "image_generator: failed to read default_num_inference_steps from config; falling back to 30"
        )
        return 30


def _default_guidance_scale() -> float:
    """从配置读取用户未显式指定时的默认 CFG 引导强度。"""
    try:
        from backend.services.agent.settings import get_image_generator_default_guidance_scale

        return float(get_image_generator_default_guidance_scale())
    except Exception:
        logger.exception(
            "image_generator: failed to read default_guidance_scale from config; falling back to 7.5"
        )
        return 7.5


def generate_image_sync(
    *,
    user_id: str,
    agent_run_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    prompt: str,
    negative_prompt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    size: Optional[str] = None,
    model_name: Optional[str] = None,
    family: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    generate_num: int = 1,
    scheduler: Optional[str] = None,
    upscale: int = 0,
    face_enhance: bool = False,
    remove_bg: bool = False,
    fast_mode: bool = True,
    low_vram: bool = False,
    file_type: str = "jpeg",
    strength: Optional[float] = None,
    url: Optional[str] = None,
    image_file2: str = "",
    edit_act: str = "",
    tpl_list: Optional[List[str]] = None,
    loras: Any = None,
    job_type: str = "MK",
    keep_size: str = "user",
    timeout: Optional[float] = None,
    storage: str = "local",
) -> Dict[str, Any]:
    """同步触发一次文生图任务并等待完成，返回结构化结果。

    Returns:
        dict: 永远返回（即便失败），包含 `tool / task_id / status / total / files / error?` 字段。
    """
    tool_result: Dict[str, Any] = {
        "tool": IMAGE_GENERATOR_TOOL_NAME,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
    }

    if not str(user_id or "").strip():
        tool_result["error"] = "image_generator requires a user_id (bound via agent context)"
        return tool_result

    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        tool_result["error"] = "prompt is required"
        return tool_result

    try:
        final_width, final_height = _resolve_size(
            width=width, height=height, aspect_ratio=aspect_ratio, size=size
        )
    except ValueError as exc:
        tool_result["error"] = str(exc)
        return tool_result

    try:
        final_upscale = int(upscale or 0)
    except (TypeError, ValueError):
        final_upscale = 0
    if final_upscale not in VALID_UPSCALE_VALUES:
        tool_result["error"] = f"upscale must be one of {sorted(VALID_UPSCALE_VALUES)}, got {upscale!r}"
        return tool_result

    try:
        final_generate_num = max(1, min(10, int(generate_num or 1)))
    except (TypeError, ValueError):
        final_generate_num = 1

    normalized_loras = _normalize_loras(loras)

    # 延迟导入，避免模块加载期建立对 FastAPI 路由层的硬依赖
    from backend.api.tasks.routes import TaskCreateRequest

    request_kwargs: Dict[str, Any] = {
        "task_type": "image",
        "job_type": str(job_type or "MK").upper(),
        "prompt": prompt_text,
        "negative_prompt": negative_prompt or "",
        "width": final_width,
        "height": final_height,
        "generate_num": final_generate_num,
        "storage": str(storage or "local"),
        "fast_mode": bool(fast_mode),
        "remove_bg": bool(remove_bg),
        "face_enhance": bool(face_enhance),
        "upscale": final_upscale,
        # 0/1 表示不超分，2/4 表示超分；推理侧会按同一语义校验。
        "file_type": str(file_type or "jpeg"),
        "keep_size": str(keep_size or "user"),
        "image_file2": str(image_file2 or ""),
        "edit_act": str(edit_act or ""),
        "tpl_list": list(tpl_list or []),
    }
    effective_model_name = clean_optional_str(model_name) or ""
    if not effective_model_name:
        try:
            from backend.services.agent.settings import (
                get_image_generator_default_model_name,
            )

            effective_model_name = get_image_generator_default_model_name()
        except Exception:
            effective_model_name = ""
        if effective_model_name:
            logger.info(
                "image_generator: model_name not provided, falling back to default=%s",
                effective_model_name,
            )
    if effective_model_name:
        request_kwargs["load_name"] = effective_model_name
    if str(agent_run_id or "").strip():
        request_kwargs["agent_run_id"] = str(agent_run_id).strip()
    if family:
        request_kwargs["family"] = str(family).strip()
    effective_num_inference_steps = coerce_optional(num_inference_steps)
    steps_source = "explicit"
    if effective_num_inference_steps is None:
        effective_num_inference_steps = _default_num_inference_steps()
        steps_source = "config"
    try:
        request_kwargs["num_inference_steps"] = max(1, min(100, int(effective_num_inference_steps)))
    except (TypeError, ValueError):
        request_kwargs["num_inference_steps"] = _default_num_inference_steps()
        steps_source = "fallback"
    effective_guidance_scale = coerce_optional(guidance_scale)
    guidance_source = "explicit"
    if effective_guidance_scale is None:
        effective_guidance_scale = _default_guidance_scale()
        guidance_source = "config"
    try:
        request_kwargs["guidance_scale"] = max(0.0, min(20.0, float(effective_guidance_scale)))
    except (TypeError, ValueError):
        request_kwargs["guidance_scale"] = _default_guidance_scale()
        guidance_source = "fallback"
    if seed is not None:
        try:
            request_kwargs["seed"] = int(seed)
        except (TypeError, ValueError):
            pass
    if scheduler:
        request_kwargs["schedulerName"] = str(scheduler)
    if strength is not None:
        try:
            request_kwargs["strength"] = max(0.0, min(1.0, float(strength)))
        except (TypeError, ValueError):
            pass
    if url:
        request_kwargs["url"] = str(url)
    if normalized_loras is not None:
        # TaskCreateRequest.loras 是 Optional[Any]，_extract_params_from_request 会原样塞回 params。
        # 推理器 InferenceRequestParams.parse_loras 兼容 list/dict/JSON 字符串。
        request_kwargs["loras"] = normalized_loras
    # low_vram 透传到 params（TaskCreateRequest 没定义该字段时，pydantic 会被过滤；
    # 通过 loras 字段旁路承载？——不合适。简单起见：当前 /v1/tasks 不支持 low_vram，
    # 待后续前端/后端协议补齐后再开放此开关。
    # 这里保留 low_vram 参数但暂不写入 request（避免静默丢失误导调用方）。
    if low_vram:
        logger.info("image_generator: low_vram=True requested but /v1/tasks does not accept it yet; ignoring")

    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        tool_result["error"] = f"invalid request params: {exc}"
        return tool_result

    effective_timeout = float(timeout) if timeout else _default_timeout()
    if effective_timeout <= 0:
        effective_timeout = 200.0

    logger.info(
        "image_generator submitting task: user=%s size=%sx%s num=%d model=%s steps=%s steps_source=%s guidance=%s guidance_source=%s",
        user_id, final_width, final_height, final_generate_num,
        request_kwargs.get("load_name"), request_kwargs.get("num_inference_steps"), steps_source,
        request_kwargs.get("guidance_scale"), guidance_source,
    )
    return submit_and_collect(
        tool_name=IMAGE_GENERATOR_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=final_generate_num,
        effective_timeout=effective_timeout,
        task_kind_label="image",
        task_event_callback=task_event_callback,
    )


# ----------------------- CrewAI 工具包装 -----------------------


@register_tool(
    name=IMAGE_GENERATOR_TOOL_NAME,
    description=IMAGE_GENERATOR_DESCRIPTION,
    tags=["image", "generate", "文生图", "画图", "生成图片", "txt2img", "AI绘画"],
    provider="local",
    enabled=True,
)
def build_image_generator_tool(*, context: Optional[Dict[str, Any]] = None):
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
        raise RuntimeError("pydantic is required to build image_generator tool") from exc

    class ImageGeneratorArgs(BaseModel):
        prompt: str = Field(..., description="Image generation prompt (positive prompt).")
        negative_prompt: str = Field(default="", description="Negative prompt (content to avoid).")
        aspect_ratio: Optional[str] = Field(
            default=None,
            description="Aspect ratio, e.g. `2:3` / `16:9` / `1:1`. Mutually exclusive with width/height.",
        )
        width: Optional[int] = Field(default=None, description="Image width in pixels, 64-4096.")
        height: Optional[int] = Field(default=None, description="Image height in pixels, 64-4096.")
        generate_num: int = Field(default=1, ge=1, le=10, description="Number of images to generate per request, 1-10.")
        seed: Optional[int] = Field(
            default=None, description="Random seed; negative or omitted means random."
        )
        scheduler: Optional[str] = Field(default=None, description="Scheduler name (e.g. euler_a).")
        file_type: str = Field(default="jpeg", description="Output format: jpeg/png/webp.")
        loras: Optional[str] = Field(
            default=None,
            description=(
                'LoRA list as JSON string, e.g. `[{"name":"xxx.safetensors","weight":0.8}]`.'
            ),
        )
        timeout: Optional[float] = Field(
            default=None, description="Generation timeout in seconds, default 200s."
        )

    class ImageGeneratorTool(BaseTool):
        name: str = IMAGE_GENERATOR_TOOL_NAME
        description: str = IMAGE_GENERATOR_DESCRIPTION
        args_schema: type = ImageGeneratorArgs

        def _run(self, **kwargs: Any) -> str:
            # LLM 不应选择模型、采样步数、CFG、超分或增强开关；统一走后端默认配置。
            llm_has_steps = "num_inference_steps" in kwargs
            llm_raw_steps = kwargs.get("num_inference_steps")
            llm_has_guidance = "guidance_scale" in kwargs
            llm_raw_guidance = kwargs.get("guidance_scale")
            llm_has_remove_bg = "remove_bg" in kwargs
            llm_raw_remove_bg = kwargs.get("remove_bg")
            llm_has_face_enhance = "face_enhance" in kwargs
            llm_raw_face_enhance = kwargs.get("face_enhance")
            llm_has_upscale = "upscale" in kwargs
            llm_raw_upscale = kwargs.get("upscale")
            kwargs.pop("model_name", None)
            kwargs.pop("num_inference_steps", None)
            kwargs.pop("guidance_scale", None)
            kwargs.pop("remove_bg", None)
            kwargs.pop("face_enhance", None)
            kwargs.pop("upscale", None)
            logger.info(
                "image_generator tool call args: has_steps=%s raw_steps=%r ignored_steps=%s has_guidance=%s raw_guidance=%r ignored_guidance=%s has_remove_bg=%s raw_remove_bg=%r ignored_remove_bg=%s has_face_enhance=%s raw_face_enhance=%r ignored_face_enhance=%s has_upscale=%s raw_upscale=%r ignored_upscale=%s",
                llm_has_steps,
                llm_raw_steps,
                llm_has_steps,
                llm_has_guidance,
                llm_raw_guidance,
                llm_has_guidance,
                llm_has_remove_bg,
                llm_raw_remove_bg,
                llm_has_remove_bg,
                llm_has_face_enhance,
                llm_raw_face_enhance,
                llm_has_face_enhance,
                llm_has_upscale,
                llm_raw_upscale,
                llm_has_upscale,
            )
            payload = generate_image_sync(
                user_id=bound_user_id,
                agent_run_id=bound_agent_run_id or None,
                task_event_callback=bound_task_event_callback if callable(bound_task_event_callback) else None,
                **kwargs,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = ImageGeneratorTool()
    tool_instance.__doc__ = IMAGE_GENERATOR_DOCSTRING
    return tool_instance

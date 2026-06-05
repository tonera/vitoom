"""image_editor 工具：图片编辑（img2img / edit）。

与 `image_generator` 的唯一区别是**针对已有图片做编辑**：
- `job_type` 强制 `ED`（不对 LLM 暴露）
- `tpl_list` 必填，必须至少含一个图片 URL（LLM 只要知道"要改哪张图"）
- 默认 `model_name` 走 FLUX 系编辑权重（`FLUX.2-klein-9b`）

不复用 `image_generator` 是因为这两种能力对 LLM 是两个"选择题"，
混在一个工具里参数必填条件化、description 语义撕裂，容易让 LLM
给错 `job_type` 或漏掉 `tpl_list`。独立工具 + 写死关键字段能显著
降低 LLM 误用率（同 `audio_tts` 的 `_allow_design_then_clone` 思路）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import clean_optional_str
from ._media_common import submit_and_collect
from .image_generator import (
    VALID_UPSCALE_VALUES,
    _normalize_loras,
    _resolve_size,
)

logger = logging.getLogger(__name__)

IMAGE_EDITOR_TOOL_NAME = "image_editor"
IMAGE_EDITOR_JOB_TYPE = "ED"

IMAGE_EDITOR_DESCRIPTION = (
    "对用户提供的**已有图片**进行编辑（修图 / 图生图编辑 / img2img edit）。"
    "典型触发场景：『把这张图 X 换成 Y』『给这张图换个背景』『去掉图里的某个元素』"
    "『把 A 图的风格迁到 B 图』『对这张图做局部重绘』『把这张图改成水彩/赛博朋克风』等；"
    "**必须同时提供至少一张原图 URL**（放进 tpl_list 字段）。"
    "\n\n严禁用于：(1) 从零生成新图（那是 image_generator 的职责，区别是无原图 URL）；"
    "(2) 描述/理解图片内容（那是 analyze_media）。"
    "\n\n参数要点："
    "\n- prompt：用一句话说清楚『希望把图改成什么样』，这是必填。"
    "\n- tpl_list：要编辑的原图 URL 列表。允许是单个字符串或一个数组；至少要 1 条 http(s) URL。"
    "\n- edit_act（可选）：自由文本的编辑动作提示，例如『换背景为海边日落』『抠图』"
    "『去水印』『局部重绘头发为粉色』等；帮助模型把 prompt 聚焦到具体动作上。"
    "\n- 不要自己编造 model_name（本工具不接受该参数）。"
    "\n\n返回 ``files[]``：向用户写 Markdown 图片/链接时，**有 ``thumb_url`` 则用 ``thumb_url``，否则用 ``url``**；须原样复制，禁止用 ``storage_path`` 自拼域名。"
)

IMAGE_EDITOR_DOCSTRING = (
    "Edit existing image(s) whose URLs are passed via `tpl_list`. "
    "Returns JSON with task_id and files[]; for markdown use thumb_url if present else url."
)


_URL_PREFIXES = ("http://", "https://")


def _normalize_tpl_list(value: Any) -> List[str]:
    """把各种形态的 tpl_list 归一成 List[str]：

    - list/tuple：逐项 strip
    - str：若是 JSON 数组字符串，解析为 list；否则拆逗号/换行/空格；
      不适用时就当单 URL 处理
    - 其它类型：返回空
    """
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        # 尝试拆分：逗号 / 空格 / 换行
        for sep in (",", "\n", "\t", ";"):
            if sep in text:
                parts = [p.strip() for p in text.split(sep)]
                return [p for p in parts if p]
        return [text]
    return []


def _default_timeout() -> float:
    try:
        from backend.services.agent.settings import get_image_editor_default_timeout

        return float(get_image_editor_default_timeout())
    except Exception:
        return 240.0


def _resolve_default_model_name() -> str:
    try:
        from backend.services.agent.settings import get_image_editor_default_model_name

        return get_image_editor_default_model_name()
    except Exception:
        return ""


def edit_image_sync(
    *,
    user_id: str,
    agent_run_id: Optional[str] = None,
    task_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    prompt: str,
    tpl_list: Any,
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
    edit_act: str = "",
    loras: Any = None,
    keep_size: str = "user",
    # 允许内部调用时覆盖（例如测试）；LLM 通道永远被工具层强制覆盖
    job_type: str = IMAGE_EDITOR_JOB_TYPE,
    timeout: Optional[float] = None,
    storage: str = "local",
) -> Dict[str, Any]:
    """同步触发一次图片编辑任务并等待完成，返回结构化结果。"""
    tool_result: Dict[str, Any] = {
        "tool": IMAGE_EDITOR_TOOL_NAME,
        "task_id": None,
        "status": "failed",
        "total": 0,
        "files": [],
    }

    if not str(user_id or "").strip():
        tool_result["error"] = "image_editor requires a user_id (bound via agent context)"
        return tool_result

    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        tool_result["error"] = "prompt is required (tell model what to change)"
        return tool_result

    normalized_tpl = _normalize_tpl_list(tpl_list)
    if not normalized_tpl:
        tool_result["error"] = (
            "tpl_list is required: pass at least one source image URL "
            "(list[str] or a single URL string). image_editor 只做编辑，"
            "如果你是想从零生成图片，请改用 image_generator。"
        )
        return tool_result
    # 过滤掉明显非 URL 的条目（允许本地路径也可能有效，但至少要有一条 http(s)）
    has_url = any(item.lower().startswith(_URL_PREFIXES) for item in normalized_tpl)
    if not has_url:
        tool_result["error"] = (
            "tpl_list must contain at least one http(s) image URL; "
            f"got {normalized_tpl!r}"
        )
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
        tool_result["error"] = (
            f"upscale must be one of {sorted(VALID_UPSCALE_VALUES)}, got {upscale!r}"
        )
        return tool_result

    try:
        final_generate_num = max(1, min(10, int(generate_num or 1)))
    except (TypeError, ValueError):
        final_generate_num = 1

    normalized_loras = _normalize_loras(loras)

    from backend.api.tasks.routes import TaskCreateRequest

    # 强制 job_type=ED；LLM 就算传了别的也覆盖掉
    effective_job_type = str(job_type or IMAGE_EDITOR_JOB_TYPE).strip().upper()
    if effective_job_type != IMAGE_EDITOR_JOB_TYPE:
        logger.warning(
            "image_editor: job_type=%r overridden to %s",
            effective_job_type, IMAGE_EDITOR_JOB_TYPE,
        )
        effective_job_type = IMAGE_EDITOR_JOB_TYPE

    request_kwargs: Dict[str, Any] = {
        "task_type": "image",
        "job_type": effective_job_type,
        "prompt": prompt_text,
        "negative_prompt": negative_prompt or "",
        "width": final_width,
        "height": final_height,
        "generate_num": final_generate_num,
        "storage": str(storage or "local"),
        "fast_mode": bool(fast_mode),
        "remove_bg": bool(remove_bg),
        "face_enhance": bool(face_enhance),
        "upscale": max(1, final_upscale) if final_upscale in (2, 4) else 1,
        "file_type": str(file_type or "jpeg"),
        "keep_size": str(keep_size or "user"),
        "edit_act": str(edit_act or ""),
        "tpl_list": normalized_tpl,
    }

    effective_model_name = clean_optional_str(model_name) or ""
    if not effective_model_name:
        effective_model_name = _resolve_default_model_name()
        if effective_model_name:
            logger.info(
                "image_editor: model_name not provided, falling back to default=%s",
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
    if scheduler:
        request_kwargs["schedulerName"] = str(scheduler)
    if strength is not None:
        try:
            request_kwargs["strength"] = max(0.0, min(1.0, float(strength)))
        except (TypeError, ValueError):
            pass
    if normalized_loras is not None:
        request_kwargs["loras"] = normalized_loras
    if low_vram:
        logger.info(
            "image_editor: low_vram=True requested but /v1/tasks does not accept it yet; ignoring"
        )

    try:
        request = TaskCreateRequest(**request_kwargs)
    except Exception as exc:
        tool_result["error"] = f"invalid request params: {exc}"
        return tool_result

    effective_timeout = float(timeout) if timeout else _default_timeout()
    if effective_timeout <= 0:
        effective_timeout = 240.0

    logger.info(
        "image_editor submitting task: user=%s size=%sx%s num=%d model=%s tpl_count=%d act=%r",
        user_id, final_width, final_height, final_generate_num,
        request_kwargs.get("load_name"), len(normalized_tpl), edit_act,
    )
    return submit_and_collect(
        tool_name=IMAGE_EDITOR_TOOL_NAME,
        user_id=user_id,
        request_obj=request,
        expected_total=final_generate_num,
        effective_timeout=effective_timeout,
        task_kind_label="image-edit",
        task_event_callback=task_event_callback,
    )


# ----------------------- CrewAI 工具包装 -----------------------


@register_tool(
    name=IMAGE_EDITOR_TOOL_NAME,
    description=IMAGE_EDITOR_DESCRIPTION,
    tags=[
        "image", "edit", "image_edit", "img2img",
        "修图", "编辑图片", "改图", "换背景", "抠图",
        "去水印", "局部重绘", "风格迁移",
    ],
    provider="local",
    enabled=True,
)
def build_image_editor_tool(*, context: Optional[Dict[str, Any]] = None):
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
        raise RuntimeError("pydantic is required to build image_editor tool") from exc

    class ImageEditorArgs(BaseModel):
        prompt: str = Field(
            ...,
            description="Edit intent description (what the image should become), required.",
        )
        tpl_list: List[str] = Field(
            ...,
            description=(
                "Source image URL list to edit (at least one http(s) URL). For a single image, "
                'still use array form, e.g. ["https://example.com/a.jpg"]. This field is the '
                "input image for this tool; do not confuse it with prompt (target description)."
            ),
            min_length=1,
        )
        edit_act: Optional[str] = Field(
            default=None,
            description=(
                "Free-text edit action hint, optional but strongly recommended, e.g.: "
                "'change background to seaside sunset', 'remove background', 'remove watermark', "
                "'change hair to pink', 'style transfer to ink painting'."
            ),
        )
        negative_prompt: str = Field(default="", description="Content to avoid.")
        aspect_ratio: Optional[str] = Field(
            default=None,
            description="Output aspect ratio, e.g. `2:3` / `16:9`; keeps default if omitted. Mutually exclusive with width/height.",
        )
        width: Optional[int] = Field(default=None, description="Output width in pixels, 64-4096.")
        height: Optional[int] = Field(default=None, description="Output height in pixels, 64-4096.")
        generate_num: int = Field(default=1, ge=1, le=10, description="Number of images to generate, 1-10.")
        num_inference_steps: Optional[int] = Field(
            default=None, description="Sampling steps (typically 20-50); uses model default if omitted."
        )
        guidance_scale: Optional[float] = Field(
            default=None, description="CFG guidance scale (0-20); uses model default if omitted."
        )
        seed: Optional[int] = Field(
            default=None, description="Random seed; negative or omitted means random."
        )
        strength: Optional[float] = Field(
            default=None,
            description="Edit strength (0-1; higher = more change, lower = closer to original).",
        )
        file_type: str = Field(default="jpeg", description="Output format: jpeg/png/webp.")
        timeout: Optional[float] = Field(
            default=None, description="Edit timeout in seconds, default 240s."
        )

    class ImageEditorTool(BaseTool):
        name: str = IMAGE_EDITOR_TOOL_NAME
        description: str = IMAGE_EDITOR_DESCRIPTION
        args_schema: type = ImageEditorArgs

        def _run(self, **kwargs: Any) -> str:
            # 防御：LLM 漏传或塞了内部字段都清掉
            kwargs.pop("job_type", None)
            kwargs.pop("model_name", None)
            kwargs.pop("keep_size", None)
            payload = edit_image_sync(
                user_id=bound_user_id,
                agent_run_id=bound_agent_run_id or None,
                task_event_callback=bound_task_event_callback if callable(bound_task_event_callback) else None,
                **kwargs,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
            return f"```json\n{rendered}\n```"

    tool_instance = ImageEditorTool()
    tool_instance.__doc__ = IMAGE_EDITOR_DOCSTRING
    return tool_instance

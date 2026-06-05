"""analyze_media 工具：按需把图片/视频 URL 送入多模态 LLM 生成文本结果。

设计原则：
- 只有当用户**明确要求**对媒体 URL 做分析（描述、总结、问答）时，Master LLM 才应调用本工具。
- URL 识别与 HEAD 探测一律不做，类型简单依据 URL 扩展名（image vs video），猜不准时按 image 兜底。
- 工具内部复用 ``backend.services.llm.run_multimodal_completion``：走本机 HTTP 调用
  ``/v1/chat/completions``，把对应内容以 ``image_url`` / ``video_url`` 形式送入。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from backend.services.agent.tools.registry import register_tool
from backend.services.llm.multimodal import (
    MultimodalCompletionError,
    run_multimodal_completion,
)

logger = logging.getLogger(__name__)

ANALYZE_MEDIA_TOOL_NAME = "analyze_media"

ANALYZE_MEDIA_DESCRIPTION = (
    "当用户明确希望对某一张图片或某一段视频的 URL 进行理解/描述/问答/摘要时调用。"
    "典型场景：'这张图是什么/帮我描述这个视频/视频里讲了什么'。"
    "非媒体内容相关的任务（例如只是整理链接、生成文案、与画面无关的提问）请勿调用。"
    "重要：使用本工具时，请直接把 URL 传入，不要先基于文件名/路径进行任何推测；"
    "真正的描述以本工具返回的 Observation 为准。"
)

ANALYZE_MEDIA_DOCSTRING = (
    "Analyze an image or video URL with the multimodal LLM. "
    "Invoke with: `url` (single URL) or `urls` (JSON array string of URLs) "
    'and an optional `question` (in Chinese). Example Action Input: '
    '{"url": "https://.../a.jpg", "question": "图里有什么?"}. '
    "Do NOT prepend any explanatory text before invoking; just call it."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".m4v"}


def _guess_media_type(url: str) -> str:
    try:
        path = urlparse(url).path or ""
    except Exception:
        path = url
    lower = path.lower()
    for ext in VIDEO_EXTS:
        if lower.endswith(ext):
            return "video"
    for ext in IMAGE_EXTS:
        if lower.endswith(ext):
            return "image"
    return "image"


def _build_messages(*, urls: List[str], question: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    text = question.strip() or "请用中文详细描述这些媒体内容的要点。"
    content.append({"type": "text", "text": text})
    for url in urls:
        media_type = _guess_media_type(url)
        if media_type == "video":
            content.append({"type": "video_url", "video_url": {"url": url}})
        else:
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def _coerce_url_input(value: Any) -> List[str]:
    """把 LLM 传进来的 url/urls 参数归一化成 list[str]。

    LLM 经常把单 URL 传成 list、把 urls 传成真 JSON 数组而非字符串，
    这里一律兼容：str / list / JSON-encoded list 都接受。
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except Exception:
                return [s]
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, (str, int, float)) and str(x).strip()]
        return [s]
    return [str(value).strip()] if str(value).strip() else []


def _do_analyze_media(
    *,
    effective_user_id: str,
    preferred_model_name: str = "",
    url: Any = "",
    urls_json: Any = "",
    question: str = "",
) -> str:
    collected: List[str] = []
    collected.extend(_coerce_url_input(url))
    collected.extend(_coerce_url_input(urls_json))

    seen = set()
    final_urls: List[str] = []
    for u in collected:
        if u in seen:
            continue
        seen.add(u)
        final_urls.append(u)

    if not final_urls:
        raise RuntimeError(
            "analyze_media requires at least one image/video URL via `url` or `urls`."
        )
    for u in final_urls:
        if not (u.startswith("http://") or u.startswith("https://")):
            raise RuntimeError(
                f"analyze_media only supports http(s) URLs, got: {u}"
            )

    if not effective_user_id:
        raise RuntimeError(
            "analyze_media could not determine the effective user id; "
            "ensure the tool is invoked from an agent run with a user context."
        )

    messages = _build_messages(urls=final_urls, question=question or "")
    logger.info(
        "analyze_media invoking multimodal completion urls=%d user=%s question_chars=%d model=%s",
        len(final_urls),
        effective_user_id,
        len(question or ""),
        preferred_model_name or "-",
    )
    try:
        result = run_multimodal_completion(
            user_id=effective_user_id,
            messages=messages,
            model=preferred_model_name or None,
        )
    except MultimodalCompletionError as exc:
        raise RuntimeError(f"analyze_media failed: {exc}") from exc

    content = result.get("content") or ""
    if not content:
        raise RuntimeError("analyze_media got empty content from the multimodal model")
    return content


@register_tool(
    name=ANALYZE_MEDIA_TOOL_NAME,
    description=ANALYZE_MEDIA_DESCRIPTION,
    tags=["multimodal", "vision", "image", "video", "media", "媒体分析", "图片", "视频"],
    provider="local",
    enabled=True,
)
def build_analyze_media_tool(*, context: Optional[Dict[str, Any]] = None):
    ctx = dict(context or {})
    bound_user_id = str(ctx.get("user_id") or "").strip()
    runtime_config = dict(ctx.get("runtime_config") or {})
    bound_model_name = str(
        runtime_config.get("load_name") or ctx.get("load_name") or ""
    ).strip()

    try:
        from crewai.tools import BaseTool
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    try:
        from pydantic import BaseModel, Field
    except Exception as e:
        raise RuntimeError("pydantic is required to build analyze_media tool") from e

    class AnalyzeMediaArgs(BaseModel):
        url: Any = Field(
            default="",
            description=(
                "Image or video URL to analyze (http/https). "
                "Pass a string for a single item; for multiple items, pass a string array "
                "or a JSON array string in `urls`."
            ),
        )
        urls: Any = Field(
            default="",
            description=(
                "Alternative way to pass multiple URLs: string array or JSON array string, e.g. "
                '\'["https://a.jpg","https://b.mp4"]\'.'
            ),
        )
        question: str = Field(
            default="",
            description="Question about the media or description points to extract.",
        )

    class AnalyzeMediaTool(BaseTool):
        name: str = ANALYZE_MEDIA_TOOL_NAME
        description: str = ANALYZE_MEDIA_DESCRIPTION
        args_schema: type = AnalyzeMediaArgs

        def _run(
            self,
            url: Any = "",
            urls: Any = "",
            question: str = "",
            **_ignored: Any,
        ) -> str:
            return _do_analyze_media(
                effective_user_id=bound_user_id,
                preferred_model_name=bound_model_name,
                url=url,
                urls_json=urls,
                question=question or "",
            )

    return AnalyzeMediaTool()

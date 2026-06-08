"""web_page_reader 工具：读取普通网页正文，供 Agent 基于页面内容回答问题。"""

from __future__ import annotations

import json
import logging
from pathlib import PurePosixPath
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from backend.services.agent.tools.registry import register_tool

from ._arg_utils import coerce_optional

logger = logging.getLogger(__name__)

WEB_PAGE_READER_TOOL_NAME = "web_page_reader"

WEB_PAGE_READER_DESCRIPTION = (
    "读取用户给出的普通网页 URL 内容，适合『总结这个网页/这篇文章讲什么/这个页面主要内容是什么』"
    "这类需要打开单个网页并理解正文的请求。不要用于搜索互联网多个结果；搜索类问题使用 tavily_search。"
    "不要用于图片、视频、PDF 或 Office 文档；这些应交给对应媒体/文档工具。"
)

WEB_PAGE_READER_DOCSTRING = (
    "Read a single public web page URL using CrewAI ScrapeWebsiteTool. "
    "Invoke with `url` and optional `question`/`max_chars`. Return JSON containing "
    "the extracted page content for the agent to summarize or answer from."
)

DEFAULT_MAX_CHARS = 12000
MIN_MAX_CHARS = 1000
MAX_MAX_CHARS = 30000

_NON_HTML_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".mkv",
    ".flv",
    ".m4v",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".7z",
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _clean_str(value: Any) -> str:
    normalized = coerce_optional(value)
    return str(normalized).strip() if normalized is not None else ""


def _coerce_max_chars(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_CHARS
    return max(MIN_MAX_CHARS, min(MAX_MAX_CHARS, parsed))


def _validate_web_page_url(url: Any) -> str:
    text = _clean_str(url)
    if not text:
        raise RuntimeError("web_page_reader requires a non-empty `url`.")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"web_page_reader only supports absolute http(s) URLs, got: {text}")
    suffix = PurePosixPath(parsed.path or "").suffix.lower()
    if suffix in _NON_HTML_EXTS:
        raise RuntimeError(
            f"web_page_reader only supports ordinary HTML web pages, got non-page URL: {text}"
        )
    return text


def _truncate_content(content: str, max_chars: int) -> tuple[str, bool, int]:
    original_length = len(content)
    if original_length <= max_chars:
        return content, False, original_length
    return content[:max_chars].rstrip(), True, original_length


def read_web_page_sync(
    *,
    url: Any,
    question: Any = "",
    max_chars: Any = None,
) -> Dict[str, Any]:
    page_url = _validate_web_page_url(url)
    limit = _coerce_max_chars(max_chars)
    try:
        from crewai_tools import ScrapeWebsiteTool  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("crewai-tools is required to use web_page_reader") from exc

    logger.info("web_page_reader scraping url=%s max_chars=%d", page_url, limit)
    scraper = ScrapeWebsiteTool()
    try:
        raw_content = scraper.run(website_url=page_url)
    except TypeError:
        raw_content = scraper.run(page_url)
    except Exception as exc:
        raise RuntimeError(f"web_page_reader failed to read page: {exc}") from exc

    content = _clean_str(raw_content)
    if not content:
        raise RuntimeError("web_page_reader got empty content from the web page")

    clipped_content, truncated, original_length = _truncate_content(content, limit)
    return {
        "tool": WEB_PAGE_READER_TOOL_NAME,
        "status": "completed",
        "url": page_url,
        "question": _clean_str(question),
        "content": clipped_content,
        "truncated": truncated,
        "original_length": original_length,
        "returned_length": len(clipped_content),
    }


@register_tool(
    name=WEB_PAGE_READER_TOOL_NAME,
    description=WEB_PAGE_READER_DESCRIPTION,
    tags=["web", "webpage", "url", "reader", "网页", "网页读取", "网页总结"],
    provider="local",
    enabled=True,
)
def build_web_page_reader_tool(*, context: Optional[Dict[str, Any]] = None):
    try:
        from crewai.tools import BaseTool  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("crewai is required to register native agent tools") from exc

    try:
        from pydantic import BaseModel, ConfigDict, Field, field_validator  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pydantic is required to build web_page_reader tool") from exc

    class WebPageReaderArgs(BaseModel):
        model_config = ConfigDict(extra="ignore")

        url: str = Field(default="", description="Web page URL to read; must be http/https.")
        question: Optional[str] = Field(
            default="",
            description="User question about the page, e.g. summarize main content; returned as-is for the agent.",
        )
        max_chars: Optional[int] = Field(
            default=DEFAULT_MAX_CHARS,
            description="Max page body characters to return; clamped to 1000-30000.",
        )

        @field_validator("url", "question", "max_chars", mode="before")
        @classmethod
        def _normalize_llm_string_nones(cls, value: Any) -> Any:
            return coerce_optional(value)

    class WebPageReaderTool(BaseTool):
        name: str = WEB_PAGE_READER_TOOL_NAME
        description: str = WEB_PAGE_READER_DESCRIPTION
        args_schema: type = WebPageReaderArgs

        def _run(
            self,
            url: str = "",
            question: Optional[str] = "",
            max_chars: Optional[int] = DEFAULT_MAX_CHARS,
            **_ignored: Any,
        ) -> str:
            try:
                args = WebPageReaderArgs.model_validate(
                    {"url": url, "question": question, "max_chars": max_chars}
                )
                payload = read_web_page_sync(**args.model_dump())
            except Exception as exc:
                payload = {
                    "tool": WEB_PAGE_READER_TOOL_NAME,
                    "status": "failed",
                    "error": str(exc),
                }
            return f"```json\n{_json_dumps(payload)}\n```"

    tool_instance = WebPageReaderTool()
    tool_instance.__doc__ = WEB_PAGE_READER_DOCSTRING
    return tool_instance

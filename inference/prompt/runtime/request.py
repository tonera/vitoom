"""把网关任务或 OpenAI messages 收成扩写输入。"""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_NEW_TOKENS = 3584
MAX_IMAGES = 10
MAX_SYSTEM_PROMPT_CHARS = 8000


class PromptRequestError(ValueError):
    """调用方请求不合法。"""


def resolve_max_new_tokens(raw: Any, *, default: int = DEFAULT_MAX_NEW_TOKENS, cap: int = DEFAULT_MAX_NEW_TOKENS) -> int:
    limit = max(1, int(cap or DEFAULT_MAX_NEW_TOKENS))
    fallback = max(1, min(int(default or limit), limit))
    if raw is None or raw == "":
        return fallback
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise PromptRequestError("max_tokens must be an integer") from exc
    if value < 1:
        raise PromptRequestError("max_tokens must be at least 1")
    return min(value, limit)


def resolve_system_prompt(raw: Any, *, max_chars: int = MAX_SYSTEM_PROMPT_CHARS) -> str | None:
    """只认调用方传入的系统提示词字符串。空值表示继续用内置 t2i / i2i。"""
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise PromptRequestError("system_prompt must be a string")
    text = raw.strip()
    if not text:
        return None
    limit = max(1, int(max_chars or MAX_SYSTEM_PROMPT_CHARS))
    if len(text) > limit:
        raise PromptRequestError(f"system_prompt must be at most {limit} characters")
    return text


def resolve_upsample_input(
    *,
    prompt: str,
    image_sources: list[Any] | None,
    messages: list[Any] | None,
    max_images: int = MAX_IMAGES,
) -> tuple[str, list[str]]:
    """返回 (用户文本, 参考图来源)。有 messages 时只读最后一条 user 消息。"""
    image_limit = max(1, int(max_images or MAX_IMAGES))
    if isinstance(messages, list) and messages:
        return _from_messages(messages, max_images=image_limit)
    text = str(prompt or "").strip()
    sources = _normalize_sources(image_sources)
    if len(sources) > image_limit:
        raise PromptRequestError(f"At most {image_limit} reference images")
    if not text:
        raise PromptRequestError("Text is required")
    return text, sources


def _from_messages(messages: list[Any], *, max_images: int) -> tuple[str, list[str]]:
    user = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            user = message
    if user is None:
        raise PromptRequestError("messages must include a user message")

    content = user.get("content")
    texts: list[str] = []
    sources: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue
            if not isinstance(part, dict):
                raise PromptRequestError("message.content contains an unrecognized part")
            kind = part.get("type")
            if kind == "text":
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif kind == "image_url":
                if len(sources) >= max_images:
                    raise PromptRequestError(f"At most {max_images} reference images")
                sources.append(_image_url(part))
            else:
                raise PromptRequestError(f"Unsupported content type: {kind}")
    else:
        raise PromptRequestError("user.content must be a string, or an array of text / image_url parts")

    prompt = "\n".join(text.strip() for text in texts if str(text or "").strip())
    if not prompt:
        raise PromptRequestError("Text is required")
    return prompt, sources


def _normalize_sources(image_sources: list[Any] | None) -> list[str]:
    sources: list[str] = []
    for item in image_sources or []:
        text = str(item or "").strip()
        if text:
            sources.append(text)
    return sources


def _image_url(part: dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        url = image_url.strip()
        if url:
            return url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    raise PromptRequestError("image_url is missing url")

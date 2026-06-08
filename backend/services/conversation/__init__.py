"""多轮会话服务 - Conversation / ConversationMessage 的业务封装。

提供：
- create_conversation / get_conversation / list_conversations
- append_message / list_messages
- build_prompt_with_history：供 Master Agent 在新一轮 Run 时把历史消息拼到 prompt
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.database import Conversation, ConversationMessage
from backend.utils import generate_uuid

MAX_HISTORY_TURNS = 6  # build_prompt_with_history 默认回看的轮数
RECENT_VERBATIM_TURNS = 1  # 最近多少轮保留原文，其余压缩成摘要


class ConversationValidationError(ValueError):
    """会话相关校验异常。"""


def _require_conversation(conversation_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    conv = Conversation.get_by_id(conversation_id)
    if not conv:
        raise ConversationValidationError("Conversation not found")
    if user_id and conv.get("user_id") != user_id:
        raise ConversationValidationError("Permission denied")
    return conv


def create_conversation(
    *,
    user_id: str,
    agent_id: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建一个新会话。"""
    if not user_id:
        raise ConversationValidationError("user_id is required")
    conversation_id = generate_uuid()
    record = Conversation.create(
        id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        title=title,
        status="active",
        metadata=metadata,
    )
    if not record:
        raise RuntimeError("Failed to create conversation")
    return record


def get_conversation(conversation_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    return _require_conversation(conversation_id, user_id)


def list_conversations(
    user_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    title_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return Conversation.list_by_user(user_id, limit=limit, offset=offset, title_query=title_query)


def update_conversation_title(conversation_id: str, title: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    conv = _require_conversation(conversation_id, user_id)
    updated = Conversation.update(conv["id"], title=title)
    return updated or conv


def append_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    agent_run_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """追加一条对话消息。role 应为 user / assistant / system / tool。"""
    conv = _require_conversation(conversation_id, user_id)
    normalized_role = str(role or "").strip().lower() or "user"
    if normalized_role not in {"user", "assistant", "system", "tool"}:
        raise ConversationValidationError(f"Unsupported role: {role}")
    message_id = generate_uuid()
    record = ConversationMessage.create(
        id=message_id,
        conversation_id=conv["id"],
        role=normalized_role,
        content=str(content or ""),
        agent_run_id=agent_run_id,
        turn_id=turn_id,
        metadata=metadata,
    )
    if not record:
        raise RuntimeError("Failed to append conversation message")

    # 若会话尚未有标题，用首条 user 消息截取 32 字做标题（非阻塞性，失败忽略）。
    if normalized_role == "user" and not (conv.get("title") or "").strip():
        auto_title = _derive_title(content)
        if auto_title:
            try:
                Conversation.update(conv["id"], title=auto_title)
            except Exception:
                pass

    return record


def update_message_run_binding(message_id: str, agent_run_id: str) -> Optional[Dict[str, Any]]:
    """把先前插入的消息与新生成的 AgentRun 绑定。"""
    if not message_id or not agent_run_id:
        return None
    return ConversationMessage.update(message_id, agent_run_id=agent_run_id)


def list_messages(
    conversation_id: str,
    *,
    user_id: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    _require_conversation(conversation_id, user_id)
    return ConversationMessage.list_by_conversation(conversation_id, limit=limit, offset=offset)


_SUMMARY_CHAR_CAP = 80  # 老消息摘要里每条保留的字符数
_URL_RE_FOR_SUMMARY = re.compile(r"https?://\S+", re.IGNORECASE)


def _truncate_for_summary(text: str, *, cap: int = _SUMMARY_CHAR_CAP) -> str:
    """把单条消息压成一行短摘要。

    - 首行 cap 内：整行保留；
    - 超出 cap：截断前抽出其中的 URL 单独追加，避免老消息里的图片/视频
      链接被 80 字 cap 截掉半截，导致多轮上下文无法回溯。
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    first_line = raw.splitlines()[0].strip()
    if len(first_line) <= cap:
        return first_line

    urls = _URL_RE_FOR_SUMMARY.findall(raw)
    truncated = first_line[: cap - 1] + "…"
    if urls:
        seen = set()
        kept: List[str] = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            kept.append(u)
        return truncated + " [urls: " + " ".join(kept) + "]"
    return truncated


def build_prompt_with_history(
    *,
    conversation_id: str,
    new_message: str,
    max_turns: int = MAX_HISTORY_TURNS,
    recent_verbatim_turns: int = RECENT_VERBATIM_TURNS,
) -> str:
    """把最近 max_turns 轮对话 + 新消息拼接成单段 prompt。

    压缩策略：
    - 最近 ``recent_verbatim_turns`` 轮（一"轮"= user+assistant 两条）保留原文。
    - 更老的保留为单行摘要，每条截到 ``_SUMMARY_CHAR_CAP`` 字。
    - 超出 ``max_turns`` 的历史直接丢弃。

    输出格式：

        [历史摘要]           # 可选，若有被压缩的老对话
        - user: …
        - assistant: …

        [过去对话]            # 可选，recent_verbatim_turns 轮原文
        user: …
        assistant: …

        [本轮输入]
        user: <new_message>
    """
    normalized_new = str(new_message or "").strip()
    total_messages = max_turns * 2
    history = ConversationMessage.list_by_conversation(
        conversation_id, limit=total_messages * 2, offset=0, ascending=False
    )
    ordered = list(reversed(history))[-total_messages:]

    non_empty = [item for item in ordered if str(item.get("content") or "").strip()]
    verbatim_slots = max(0, recent_verbatim_turns) * 2
    split_at = max(0, len(non_empty) - verbatim_slots)
    older = non_empty[:split_at]
    recent = non_empty[split_at:]

    lines: List[str] = []

    if older:
        lines.append("[历史摘要]")
        for item in older:
            role = str(item.get("role") or "").lower() or "unknown"
            summary = _truncate_for_summary(str(item.get("content") or ""))
            if summary:
                lines.append(f"- {role}: {summary}")
        lines.append("")

    if recent:
        lines.append("[过去对话]")
        for item in recent:
            role = str(item.get("role") or "").lower() or "unknown"
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        lines.append("")

    lines.append("[本轮输入]")
    lines.append(f"user: {normalized_new}")
    return "\n".join(lines).strip()


def _derive_title(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    if len(text) > 32:
        text = text[:29] + "..."
    return text


__all__ = [
    "ConversationValidationError",
    "MAX_HISTORY_TURNS",
    "RECENT_VERBATIM_TURNS",
    "append_message",
    "build_prompt_with_history",
    "create_conversation",
    "get_conversation",
    "list_conversations",
    "list_messages",
    "update_conversation_title",
    "update_message_run_binding",
]

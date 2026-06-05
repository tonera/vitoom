"""ConversationService.build_prompt_with_history 单元测试。"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.conversation import build_prompt_with_history


FAKE_HISTORY_DESC = [
    {"role": "assistant", "content": "你好，有什么可以帮您？", "created_at": "2026-04-01T00:00:02"},
    {"role": "user", "content": "你好", "created_at": "2026-04-01T00:00:01"},
]


def test_build_prompt_concatenates_history_and_new_message():
    with patch(
        "backend.services.conversation.ConversationMessage.list_by_conversation",
        return_value=list(FAKE_HISTORY_DESC),
    ):
        prompt = build_prompt_with_history(
            conversation_id="conv-1",
            new_message="帮我规划东京3日游",
            max_turns=5,
        )

    assert "[过去对话]" in prompt
    assert "user: 你好" in prompt
    assert "assistant: 你好，有什么可以帮您？" in prompt
    assert "[本轮输入]" in prompt
    assert "user: 帮我规划东京3日游" in prompt
    # 过去对话应在本轮之前
    assert prompt.index("[过去对话]") < prompt.index("[本轮输入]")


def test_build_prompt_without_history_still_has_new_message():
    with patch(
        "backend.services.conversation.ConversationMessage.list_by_conversation",
        return_value=[],
    ):
        prompt = build_prompt_with_history(
            conversation_id="conv-1",
            new_message="hi",
            max_turns=5,
        )

    assert "[过去对话]" not in prompt
    assert "[本轮输入]" in prompt
    assert "user: hi" in prompt


def test_build_prompt_keeps_only_latest_turn_verbatim():
    history_desc = [
        {"role": "assistant", "content": "最后一轮助手回复", "created_at": "2026-04-01T00:00:06"},
        {"role": "user", "content": "最后一轮用户消息", "created_at": "2026-04-01T00:00:05"},
        {"role": "assistant", "content": "第二轮助手回复\n还有补充说明", "created_at": "2026-04-01T00:00:04"},
        {"role": "user", "content": "第二轮用户消息", "created_at": "2026-04-01T00:00:03"},
        {"role": "assistant", "content": "第一轮助手回复", "created_at": "2026-04-01T00:00:02"},
        {"role": "user", "content": "第一轮用户消息", "created_at": "2026-04-01T00:00:01"},
    ]

    with patch(
        "backend.services.conversation.ConversationMessage.list_by_conversation",
        return_value=list(history_desc),
    ):
        prompt = build_prompt_with_history(
            conversation_id="conv-1",
            new_message="继续",
            max_turns=6,
        )

    summary_part, recent_and_input = prompt.split("[过去对话]", 1)
    recent_part, _input_part = recent_and_input.split("[本轮输入]", 1)

    assert "[历史摘要]" in prompt
    assert "- user: 第一轮用户消息" in prompt
    assert "- assistant: 第二轮助手回复" in prompt
    assert "user: 最后一轮用户消息" in prompt
    assert "assistant: 最后一轮助手回复" in prompt
    assert "- user: 第二轮用户消息" in summary_part
    assert "user: 第二轮用户消息" not in recent_part
    assert "assistant: 第二轮助手回复\n还有补充说明" not in recent_part

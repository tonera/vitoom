import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.no_tool_runner import NoToolLLMResult, build_no_tool_messages
from backend.services.agent.specs import AgentSpec, TaskSpec
from backend.services.agent.types import AgentCommand
from backend.workers.agent_worker import _result_to_text


def test_build_no_tool_messages_preserves_agent_and_task_prompt():
    command = AgentCommand(
        user_id="u1",
        agent_id="a1",
        message="解释一下 Python 闭包",
        context={"original_user_message": "解释一下 Python 闭包"},
    )
    messages = build_no_tool_messages(
        agent_specs=[
            AgentSpec(
                name="primary",
                role="Helpful assistant",
                goal="Answer accurately",
                backstory="You are concise.",
            )
        ],
        task_specs=[
            TaskSpec(
                task_id="main",
                description="Complete the request: {message}",
                expected_output="Markdown answer.",
            )
        ],
        command=command,
    )

    assert messages[0]["role"] == "system"
    assert "Helpful assistant" in messages[0]["content"]
    assert "Answer accurately" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "解释一下 Python 闭包" in messages[1]["content"]
    assert "Markdown answer." in messages[1]["content"]


def test_worker_result_to_text_accepts_no_tool_result_raw():
    result = NoToolLLMResult(raw=" 直接回答 ", token_usage={"total_tokens": 3})

    assert _result_to_text(result) == "直接回答"

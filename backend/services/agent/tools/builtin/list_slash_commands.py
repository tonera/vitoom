"""列出当前支持的 slash command。"""

from __future__ import annotations

from typing import Any

from backend.services.agent.tools.registry import register_tool

LIST_SLASH_COMMANDS_NAME = "list_slash_commands"

LIST_SLASH_COMMANDS_DESCRIPTION = (
    "列出当前系统支持的 slash command 命令列表。仅当用户询问『你支持哪些命令』"
    "『你有哪些可用命令』『/help』或其他 slash command 帮助问题时使用；"
    "不要用于介绍 Agent 工具能力，也不要用于执行具体命令。"
)

LIST_SLASH_COMMANDS_DOCSTRING = (
    "Return the current slash command help text. Use only for questions about "
    "supported commands, available slash commands, or /help."
)


@register_tool(
    name=LIST_SLASH_COMMANDS_NAME,
    description=LIST_SLASH_COMMANDS_DESCRIPTION,
    tags=["slash", "command", "help", "命令列表", "可用命令", "slash命令", "帮助"],
    provider="local",
    enabled=True,
)
def build_list_slash_commands_tool():
    try:
        from crewai.tools import BaseTool
    except Exception as e:
        raise RuntimeError("crewai is required to register native agent tools") from e

    try:
        from pydantic import BaseModel, Field
    except Exception as e:
        raise RuntimeError("pydantic is required to build list_slash_commands tool") from e

    class ListSlashCommandsArgs(BaseModel):
        include_examples: bool = Field(
            default=True,
            description="Whether to include help hints. Usually keep default true.",
        )

    class ListSlashCommandsTool(BaseTool):
        name: str = LIST_SLASH_COMMANDS_NAME
        description: str = LIST_SLASH_COMMANDS_DESCRIPTION
        args_schema: type = ListSlashCommandsArgs

        def _run(self, include_examples: bool = True, **_ignored: Any) -> str:
            del include_examples
            from backend.services.chat.slash_commands import (
                build_slash_command_help_text,
                ensure_slash_commands_registered,
            )

            ensure_slash_commands_registered()
            return build_slash_command_help_text()

    ListSlashCommandsTool.__doc__ = LIST_SLASH_COMMANDS_DOCSTRING
    return ListSlashCommandsTool()

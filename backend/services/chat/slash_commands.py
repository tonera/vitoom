"""新的 chat 体系下的 slash command 分发器。

注意：这里只承接命令解析与执行；统一会话的入口已经收敛到 chat runtime。
真正入口在 `MasterAgentRuntime.run()`：当本轮 user_text 以 `/` 开头时短路到这里。
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from backend.services.chat.artifacts import build_chat_files_from_tool_result

logger = logging.getLogger(__name__)


@dataclass
class ChatSlashCommandResult:
    handled: bool
    status: str = "completed"
    assistant_text: str = ""
    task_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)


class ChatSlashCommand(Protocol):
    name: str

    async def handle(
        self,
        *,
        user_id: str,
        argv: List[str],
        raw_message: str,
    ) -> ChatSlashCommandResult:
        ...


_REGISTRY: Dict[str, ChatSlashCommand] = {}

_SLASH_COMMAND_ARGUMENTS: Dict[str, Dict[str, List[str]]] = {
    "audio-asr": {
        "positionals": ["audio_url"],
        "options": [
            "--model",
            "--model-version",
            "--language",
            "--prompt-text",
            "--timestamps",
            "--no-timestamps",
            "--diarize",
            "--no-diarize",
            "--stream",
            "--no-stream",
            "--sample-rate",
            "--job-type",
            "--timeout",
        ],
    },
    "audio-tts": {
        "positionals": ["prompt"],
        "options": [
            "--mode",
            "--model",
            "--clone-base-model",
            "--model-version",
            "--instruct",
            "--speaker",
            "--voice",
            "--ref-audio",
            "--ref-text",
            "--x-vector-only",
            "--design-from",
            "--design-seed-text",
            "--language",
            "--sample-rate",
            "--file-type",
            "--cfg",
            "--steps",
            "--job-type",
            "--timeout",
        ],
    },
    "doc-to-md": {
        "positionals": ["instruction"],
        "options": ["--url", "--timeout"],
    },
    "help": {
        "positionals": [],
        "options": [],
    },
    "image": {
        "positionals": ["prompt"],
        "options": [
            "-n",
            "--negative",
            "--model",
            "--model-version",
            "--ratio",
            "--size",
            "--width",
            "--height",
            "--steps",
            "--cfg",
            "--seed",
            "--num",
            "--scheduler",
            "--upscale",
            "--face-enhance",
            "--no-face-enhance",
            "--remove-bg",
            "--no-remove-bg",
            "--fast",
            "--no-fast",
            "--low-vram",
            "--no-low-vram",
            "--file-type",
            "--lora",
            "--url",
            "--strength",
            "--image2",
            "--edit-act",
            "--job-type",
            "--keep-size",
            "--tpl",
            "--timeout",
        ],
    },
    "image-edit": {
        "positionals": ["prompt"],
        "options": [
            "--tpl",
            "--url",
            "--edit-act",
            "-n",
            "--negative",
            "--ratio",
            "--size",
            "--width",
            "--height",
            "--num",
            "--steps",
            "--cfg",
            "--seed",
            "--strength",
            "--upscale",
            "--face-enhance",
            "--remove-bg",
            "--fast",
            "--no-fast",
            "--low-vram",
            "--file-type",
            "--lora",
            "--model",
            "--model-version",
            "--scheduler",
            "--keep-size",
            "--timeout",
        ],
    },
    "video": {
        "positionals": ["prompt"],
        "options": [
            "-n",
            "--negative",
            "--model",
            "--model-version",
            "--ratio",
            "--resolution",
            "--width",
            "--height",
            "--duration",
            "--fps",
            "--steps",
            "--cfg",
            "--seed",
            "--num",
            "--url",
            "--ref-video",
            "--face-video",
            "--direction",
            "--speed",
            "--prompt-wav",
            "--prompt-text",
            "--fast",
            "--no-fast",
            "--file-type",
            "--lora",
            "--image2",
            "--edit-act",
            "--job-type",
            "--tpl",
            "--timeout",
        ],
    },
}


def register_slash_command(cmd: ChatSlashCommand) -> None:
    name = str(getattr(cmd, "name", "") or "").strip().lstrip("/").lower()
    if not name:
        raise ValueError("slash command must have a non-empty name")
    _REGISTRY[name] = cmd


def list_slash_command_names() -> List[str]:
    return sorted(_REGISTRY.keys())


def build_slash_command_help_text() -> str:
    if not _REGISTRY:
        return "当前没有已注册的 slash command。"

    lines = [
        "```md",
        "# 当前支持的 slash command",
    ]
    for name in list_slash_command_names():
        args = _SLASH_COMMAND_ARGUMENTS.get(name, {})
        positionals = args.get("positionals") or []
        options = args.get("options") or []
        signature = " ".join([f"/{name}", *[f"<{item}>" for item in positionals]])
        lines.append("")
        lines.append(f"## {signature}")
        lines.append(f"- 位置参数：{', '.join(positionals) if positionals else '无'}")
        lines.append(f"- 可用选项：{', '.join(options) if options else '无'}")
    lines.append("```")
    return "\n".join(lines).strip()


def _parse_slash_message(raw: str) -> Optional[tuple[str, List[str]]]:
    text = str(raw or "").lstrip()
    if not text.startswith("/"):
        return None
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        logger.debug("slash parsing failed for %r: %s", text, exc)
        return None
    if not tokens:
        return None
    first = str(tokens[0] or "").strip()
    if not first.startswith("/"):
        return None
    return first.lstrip("/").lower(), tokens[1:]


async def try_dispatch(
    raw_message: str,
    *,
    user_id: str,
) -> ChatSlashCommandResult:
    parsed = _parse_slash_message(raw_message)
    if parsed is None:
        return ChatSlashCommandResult(handled=False)
    name, argv = parsed
    command = _REGISTRY.get(name)
    if command is None:
        return ChatSlashCommandResult(handled=False)
    try:
        return await command.handle(user_id=user_id, argv=argv, raw_message=raw_message)
    except Exception as exc:
        logger.exception("slash command /%s failed with unexpected error", name)
        return ChatSlashCommandResult(
            handled=True,
            status="failed",
            assistant_text=f"`/{name}` 执行失败：{exc}",
            error=str(exc),
        )


class HelpSlashCommand:
    name = "help"

    async def handle(
        self,
        *,
        user_id: str,
        argv: List[str],
        raw_message: str,
    ) -> ChatSlashCommandResult:
        del user_id, argv, raw_message
        return ChatSlashCommandResult(
            handled=True,
            status="usage",
            assistant_text=build_slash_command_help_text(),
        )


def build_artifacts_from_tool_result(
    payload: Dict[str, Any],
    *,
    default_category: str = "text",
) -> List[Dict[str, Any]]:
    return build_chat_files_from_tool_result(
        payload,
        default_category=default_category,
        source_tool=str(payload.get("tool") or "").strip() or None,
    )


def ensure_slash_commands_registered() -> None:
    from . import slash_commands_audio_asr  # noqa: F401
    from . import slash_commands_audio_tts  # noqa: F401
    from . import slash_commands_image  # noqa: F401
    from . import slash_commands_image_editor  # noqa: F401
    from . import slash_commands_video  # noqa: F401
    from . import slash_commands_doc_to_md  # noqa: F401

    register_slash_command(HelpSlashCommand())


import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.chat.slash_commands import (  # noqa: E402
    build_slash_command_help_text,
    ensure_slash_commands_registered,
    try_dispatch,
)


def test_slash_help_lists_registered_commands():
    ensure_slash_commands_registered()

    text = build_slash_command_help_text()

    assert text.startswith("```md")
    assert text.endswith("```")
    assert "当前支持的 slash command" in text
    assert "## /help" in text
    assert "## /image <prompt>" in text
    assert "## /audio-asr <audio_url>" in text
    assert "## /doc-to-md <instruction>" in text
    assert "--model" in text
    assert "--url" in text


def test_help_slash_command_dispatches_help_text():
    async def _run():
        ensure_slash_commands_registered()
        return await try_dispatch("/help", user_id="u1")

    result = asyncio.run(_run())

    assert result.handled is True
    assert result.status == "usage"
    assert result.assistant_text.startswith("```md")
    assert "## /help" in result.assistant_text
    assert "## /image <prompt>" in result.assistant_text

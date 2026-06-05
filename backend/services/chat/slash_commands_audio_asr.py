import argparse
import asyncio
import json
from typing import Any, Dict, List

from backend.services.chat.slash_commands import (
    ChatSlashCommandResult,
    build_artifacts_from_tool_result,
    register_slash_command,
)

_USAGE_TEXT = (
    "用法：`/audio-asr <音频URL> [选项]`\n\n"
    "常用示例：\n"
    "- `/audio-asr https://example.com/demo.wav`\n"
    "- `/audio-asr https://example.com/demo.wav --language zh --timestamps`\n"
    "- `/audio-asr https://example.com/demo.wav --model Qwen3-ASR-1.7B --no-stream`\n\n"
    "常用选项：\n"
    "- `--model <name>` ASR 模型名；不传使用后端默认模型\n"
    "- `--language zh|en|ja` 指定语言；不填走自动识别\n"
    "- `--prompt-text \"...\"` 识别上下文/提示文本\n"
    "- `--timestamps` 返回时间戳\n"
    "- `--diarize` 开启说话人分离\n"
    "- `--no-stream` 关闭流式文本返回\n"
    "- `--timeout 180` 超时秒数"
)


class _ArgparseError(Exception):
    pass


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgparseError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(prog="/audio-asr", add_help=False)
    parser.add_argument("audio_url", nargs="+")
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--model-version", dest="family", default=None)
    parser.add_argument("--language", dest="language", default=None)
    parser.add_argument("--prompt-text", dest="prompt_text", default=None)
    parser.add_argument("--timestamps", dest="timestamps", action="store_true")
    parser.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    parser.set_defaults(timestamps=False)
    parser.add_argument("--diarize", dest="speaker_diarization", action="store_true")
    parser.add_argument("--no-diarize", dest="speaker_diarization", action="store_false")
    parser.set_defaults(speaker_diarization=False)
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=None)
    parser.add_argument("--job-type", dest="job_type", default="ASR")
    parser.add_argument("--timeout", dest="timeout", type=float, default=None)
    return parser


def _render_result_markdown(payload: dict) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


class AudioAsrSlashCommand:
    name = "audio-asr"

    async def handle(
        self,
        *,
        user_id: str,
        argv: List[str],
        raw_message: str,
    ) -> ChatSlashCommandResult:
        if not argv or any(tok in {"--help", "-h", "/help", "/?"} for tok in argv):
            return ChatSlashCommandResult(
                handled=True,
                status="usage",
                assistant_text=_USAGE_TEXT,
            )
        parser = _build_parser()
        try:
            ns = parser.parse_args(argv)
        except _ArgparseError as exc:
            usage = parser.format_usage().strip()
            return ChatSlashCommandResult(
                handled=True,
                status="failed",
                assistant_text=_render_result_markdown(
                    {
                        "tool": "audio_asr",
                        "task_id": None,
                        "status": "failed",
                        "total": 0,
                        "files": [],
                        "text": "",
                        "error": f"参数错误: {exc}\n{usage}",
                    }
                ),
                error=str(exc),
            )

        audio_url = " ".join(list(ns.audio_url or [])).strip()
        if not audio_url:
            return ChatSlashCommandResult(
                handled=True,
                status="failed",
                assistant_text=_render_result_markdown(
                    {
                        "tool": "audio_asr",
                        "task_id": None,
                        "status": "failed",
                        "total": 0,
                        "files": [],
                        "text": "",
                        "error": "audio_url is required",
                    }
                ),
                error="audio_url is required",
            )

        kwargs: Dict[str, Any] = dict(
            user_id=user_id,
            input_audio_url=audio_url,
            model_name=ns.model_name,
            family=ns.family,
            language=ns.language,
            prompt_text=ns.prompt_text,
            timestamps=ns.timestamps,
            speaker_diarization=ns.speaker_diarization,
            stream=ns.stream,
            sample_rate=ns.sample_rate,
            job_type=ns.job_type,
            timeout=ns.timeout,
        )

        from backend.services.agent.tools.builtin.audio_asr import transcribe_audio_sync

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: transcribe_audio_sync(**kwargs),
        )
        task_id = str(payload.get("task_id") or "").strip()
        return ChatSlashCommandResult(
            handled=True,
            status=str(payload.get("status") or "failed"),
            assistant_text=_render_result_markdown(payload),
            task_ids=[task_id] if task_id else [],
            error=str(payload.get("error") or "") or None,
            artifacts=build_artifacts_from_tool_result(payload, default_category="text"),
        )


register_slash_command(AudioAsrSlashCommand())

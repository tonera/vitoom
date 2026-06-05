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
    "用法：`/audio-tts <要朗读的文本> [选项]`\n\n"
    "4 种合成模式（`--mode` 不传时按其他参数自动推断）：\n"
    "1. `custom_voice`：`/audio-tts 你好呀 --speaker vivian --instruct \"撒娇萝莉音\"`\n"
    "2. `voice_design`：`/audio-tts 今天天气真好 --mode voice_design --instruct \"磁性的低沉中年男声\"`\n"
    "3. `voice_clone`：`/audio-tts 这是要念的内容 --ref-audio https://x.wav --ref-text \"参考音频里念的原文\"`\n"
    "4. `voice_design_then_clone`：`/audio-tts 我要念的内容 --design-from \"体现撒娇稚嫩的萝莉女声\" --design-seed-text \"哥哥，你回来啦\"`\n\n"
    "常用选项：\n"
    "- `--mode <name>` custom_voice / voice_design / voice_clone / voice_design_then_clone\n"
    "- `--speaker <name>` custom_voice 下的预置说话人\n"
    "- `--instruct \"...\"` 风格/情感/声线指令\n"
    "- `--ref-audio <url|path>` voice_clone：参考音频\n"
    "- `--ref-text \"...\"` voice_clone：参考音频原文\n"
    "- `--x-vector-only` voice_clone：允许省略 ref_text\n"
    "- `--design-from \"...\"` voice_design_then_clone：第 1 阶段声线描述\n"
    "- `--design-seed-text \"...\"` voice_design_then_clone：第 1 阶段种子文本\n"
    "- `--model <name>` 显式权重名；推理侧未 pin（未设 fixed_model）时必传\n"
    "- `--clone-base-model <name>` voice_design_then_clone 第 2 阶段 Base 权重；该模式下必传\n"
    "- `--language zh|en|ja` 指定语言\n"
    "- `--sample-rate 24000` 采样率\n"
    "- `--file-type wav|mp3|...` 输出音频格式\n"
    "- `--timeout 240` 超时秒数"
)


class _ArgparseError(Exception):
    pass


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgparseError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(prog="/audio-tts", add_help=False)
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("--mode", dest="tts_mode", default=None)
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--clone-base-model", dest="clone_base_model_name", default=None)
    parser.add_argument("--model-version", dest="family", default=None)
    parser.add_argument("--instruct", dest="instruct", default=None)
    parser.add_argument("--speaker", dest="speaker_name", default=None)
    parser.add_argument("--voice", dest="voice_preset", default=None)
    parser.add_argument("--ref-audio", dest="ref_audio", default=None)
    parser.add_argument("--ref-text", dest="ref_text", default=None)
    parser.add_argument("--x-vector-only", dest="x_vector_only", action="store_true")
    parser.add_argument("--design-from", dest="design_instruct", default=None)
    parser.add_argument("--design-seed-text", dest="design_seed_text", default=None)
    parser.add_argument("--language", dest="language", default=None)
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=None)
    parser.add_argument("--file-type", dest="file_type", default="wav", choices=["wav", "mp3", "ogg", "flac"])
    parser.add_argument("--cfg", dest="guidance_scale", type=float, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--job-type", dest="job_type", default="TTS")
    parser.add_argument("--timeout", dest="timeout", type=float, default=None)
    return parser


def _render_result_markdown(payload: dict) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


class AudioTtsSlashCommand:
    name = "audio-tts"

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
                        "tool": "audio_tts",
                        "task_id": None,
                        "status": "failed",
                        "total": 0,
                        "files": [],
                        "error": f"参数错误: {exc}\n{usage}",
                    }
                ),
                error=str(exc),
            )

        prompt_text = " ".join(list(ns.prompt or [])).strip()
        if not prompt_text:
            return ChatSlashCommandResult(
                handled=True,
                status="failed",
                assistant_text=_render_result_markdown(
                    {
                        "tool": "audio_tts",
                        "task_id": None,
                        "status": "failed",
                        "total": 0,
                        "files": [],
                        "error": "prompt is required",
                    }
                ),
                error="prompt is required",
            )

        tts_mode = (ns.tts_mode or "").strip() or None
        has_design_from = bool((ns.design_instruct or "").strip())
        has_design_seed = bool((ns.design_seed_text or "").strip())
        if not tts_mode:
            if has_design_from and has_design_seed:
                tts_mode = "voice_design_then_clone"
            elif (ns.ref_audio or "").strip():
                tts_mode = "voice_clone"
            elif (ns.instruct or "").strip() and not (ns.speaker_name or ns.voice_preset):
                tts_mode = "voice_design"
            else:
                tts_mode = "custom_voice"

        if tts_mode == "voice_design_then_clone" and not (has_design_from and has_design_seed):
            error_msg = (
                "tts_mode=voice_design_then_clone requires both --design-from and "
                "--design-seed-text; only one was provided."
            )
            return ChatSlashCommandResult(
                handled=True,
                status="failed",
                assistant_text=_render_result_markdown(
                    {
                        "tool": "audio_tts",
                        "task_id": None,
                        "status": "failed",
                        "total": 0,
                        "files": [],
                        "error": error_msg,
                    }
                ),
                error=error_msg,
            )

        kwargs: Dict[str, Any] = dict(
            user_id=user_id,
            prompt=prompt_text,
            tts_mode=tts_mode,
            model_name=ns.model_name,
            clone_base_model_name=ns.clone_base_model_name,
            family=ns.family,
            instruct=ns.instruct,
            speaker_name=ns.speaker_name,
            voice_preset=ns.voice_preset,
            ref_audio=ns.ref_audio,
            ref_text=ns.ref_text,
            x_vector_only=bool(ns.x_vector_only),
            design_seed_text=ns.design_seed_text,
            design_instruct=ns.design_instruct,
            language=ns.language,
            sample_rate=ns.sample_rate,
            file_type=ns.file_type,
            guidance_scale=ns.guidance_scale,
            num_inference_steps=ns.num_inference_steps,
            job_type=ns.job_type,
            timeout=ns.timeout,
        )

        from backend.services.agent.tools.builtin.audio_tts import synthesize_audio_sync

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: synthesize_audio_sync(
                _allow_design_then_clone=True,
                **kwargs,
            ),
        )
        task_id = str(payload.get("task_id") or "").strip()
        return ChatSlashCommandResult(
            handled=True,
            status=str(payload.get("status") or "failed"),
            assistant_text=_render_result_markdown(payload),
            task_ids=[task_id] if task_id else [],
            error=str(payload.get("error") or "") or None,
            artifacts=build_artifacts_from_tool_result(payload, default_category="audio"),
        )


register_slash_command(AudioTtsSlashCommand())

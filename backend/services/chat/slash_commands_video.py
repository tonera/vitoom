import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional

from backend.services.chat.slash_commands import (
    ChatSlashCommandResult,
    build_artifacts_from_tool_result,
    register_slash_command,
)

_USAGE_TEXT = (
    "用法：`/video <描述> [选项]`\n\n"
    "常用示例：\n"
    "- `/video 一只会飞的猫咪，卡通风格`\n"
    "- `/video 海浪拍打沙滩，电影感 --ratio 16:9 --duration 5 --fps 24`\n"
    "- `/video 跳舞的机器人 --model Wan2.2-TI2V-5B-FP8 --resolution 1280x720`\n\n"
    "常用选项：\n"
    "- `--ratio 16:9` 宽高比\n"
    "- `--resolution 1280x720` 显式分辨率\n"
    "- `--duration 5` 视频时长（秒）\n"
    "- `--fps 24` 帧率\n"
    "- `--num 1` 生成条数\n"
    "- `--model <name>` 模型名；不传使用后端默认模型\n"
    "- `--url <image_url>` 参考首帧图（图生视频）\n"
    "- `--ref-video <url>` 参考/控制视频\n"
    "- `--seed 42` 随机种子\n"
    "- `--lora \"name:0.8\"` LoRA 规格，可重复"
)


class _ArgparseError(Exception):
    pass


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgparseError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(prog="/video", add_help=False)
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("-n", "--negative", dest="negative_prompt", default="")
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--model-version", dest="family", default=None)
    parser.add_argument("--ratio", dest="aspect_ratio", default=None)
    parser.add_argument("--resolution", dest="resolution", default=None)
    parser.add_argument("--width", dest="width", type=int, default=None)
    parser.add_argument("--height", dest="height", type=int, default=None)
    parser.add_argument("--duration", dest="duration", type=int, default=None)
    parser.add_argument("--fps", dest="fps", type=int, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg", dest="guidance_scale", type=float, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--num", dest="generate_num", type=int, default=1)
    parser.add_argument("--url", dest="url", default=None)
    parser.add_argument("--ref-video", dest="ref_video", default=None)
    parser.add_argument("--face-video", dest="face_video", default=None)
    parser.add_argument("--direction", dest="direction", default=None)
    parser.add_argument("--speed", dest="speed", type=float, default=None)
    parser.add_argument("--prompt-wav", dest="prompt_wav_path", default=None)
    parser.add_argument("--prompt-text", dest="prompt_text", default=None)
    parser.add_argument("--fast", dest="fast_mode", action="store_true")
    parser.add_argument("--no-fast", dest="fast_mode", action="store_false")
    parser.set_defaults(fast_mode=True)
    parser.add_argument("--file-type", dest="file_type", default="mp4")
    parser.add_argument("--lora", dest="loras", action="append", default=[])
    parser.add_argument("--image2", dest="image_file2", default="")
    parser.add_argument("--edit-act", dest="edit_act", default="")
    parser.add_argument("--job-type", dest="job_type", default="MKV")
    parser.add_argument("--tpl", dest="tpl_list", action="append", default=[])
    parser.add_argument("--timeout", dest="timeout", type=float, default=None)
    return parser


def _render_result_markdown(payload: dict) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _normalize_lora_args(values: List[str]) -> Optional[Any]:
    if not values:
        return None
    normalized: List[Dict[str, Any]] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("name"):
                        try:
                            weight = float(item.get("weight", 1.0))
                        except (TypeError, ValueError):
                            weight = 1.0
                        normalized.append({"name": str(item["name"]).strip(), "weight": weight})
            continue
        if ":" in text:
            name_part, weight_part = text.rsplit(":", 1)
            try:
                weight = float(weight_part)
            except ValueError:
                weight = 1.0
            normalized.append({"name": name_part.strip(), "weight": weight})
        else:
            normalized.append({"name": text, "weight": 1.0})
    return normalized or None


class VideoSlashCommand:
    name = "video"

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
                        "tool": "video_generator",
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
                        "tool": "video_generator",
                        "task_id": None,
                        "status": "failed",
                        "total": 0,
                        "files": [],
                        "error": "prompt is required",
                    }
                ),
                error="prompt is required",
            )

        kwargs: Dict[str, Any] = dict(
            user_id=user_id,
            prompt=prompt_text,
            negative_prompt=ns.negative_prompt,
            width=ns.width,
            height=ns.height,
            aspect_ratio=ns.aspect_ratio,
            resolution=ns.resolution,
            model_name=ns.model_name,
            family=ns.family,
            num_inference_steps=ns.num_inference_steps,
            guidance_scale=ns.guidance_scale,
            seed=ns.seed,
            generate_num=ns.generate_num,
            duration=ns.duration,
            fps=ns.fps,
            url=ns.url,
            ref_video=ns.ref_video,
            face_video=ns.face_video,
            direction=ns.direction,
            speed=ns.speed,
            prompt_wav_path=ns.prompt_wav_path,
            prompt_text=ns.prompt_text,
            fast_mode=ns.fast_mode,
            file_type=ns.file_type,
            image_file2=ns.image_file2,
            edit_act=ns.edit_act,
            tpl_list=list(ns.tpl_list or []),
            loras=_normalize_lora_args(ns.loras),
            job_type=ns.job_type,
            timeout=ns.timeout,
        )

        from backend.services.agent.tools.builtin.video_generator import generate_video_sync

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, lambda: generate_video_sync(**kwargs))
        task_id = str(payload.get("task_id") or "").strip()
        return ChatSlashCommandResult(
            handled=True,
            status=str(payload.get("status") or "failed"),
            assistant_text=_render_result_markdown(payload),
            task_ids=[task_id] if task_id else [],
            error=str(payload.get("error") or "") or None,
            artifacts=build_artifacts_from_tool_result(payload, default_category="video"),
        )


register_slash_command(VideoSlashCommand())

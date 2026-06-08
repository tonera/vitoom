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
    "用法：`/image <描述> [选项]`\n\n"
    "常用示例：\n"
    "- `/image 一只可爱的小狗`\n"
    "- `/image 赛博朋克风格的猫咪 --ratio 2:3 --num 2`\n"
    "- `/image 水墨风山水画 --model Z-Image-Turbo --upscale 4`\n\n"
    "常用选项：\n"
    "- `--ratio 2:3` 宽高比\n"
    "- `--size 1024x1536` 显式宽高\n"
    "- `--num 2` 一次生成张数\n"
    "- `--model <name>` 模型名；不传使用后端默认模型\n"
    "- `--upscale 2|4` 超分倍数\n"
    "- `--seed 42` 随机种子\n"
    "- `--cfg 7.5` CFG 引导强度\n"
    "- `--steps 30` 采样步数\n"
    "- `--lora \"name:0.8\"` LoRA 规格，可重复"
)


class _ArgparseError(Exception):
    pass


class _NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgparseError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _NonExitingArgumentParser(prog="/image", add_help=False)
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("-n", "--negative", dest="negative_prompt", default="")
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--model-version", dest="family", default=None)
    parser.add_argument("--ratio", dest="aspect_ratio", default=None)
    parser.add_argument("--size", dest="size", default=None)
    parser.add_argument("--width", dest="width", type=int, default=None)
    parser.add_argument("--height", dest="height", type=int, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg", dest="guidance_scale", type=float, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--num", dest="generate_num", type=int, default=1)
    parser.add_argument("--scheduler", dest="scheduler", default=None)
    parser.add_argument("--upscale", dest="upscale", type=int, default=0, choices=[0, 1, 2, 4])
    parser.add_argument("--face-enhance", dest="face_enhance", action="store_true")
    parser.add_argument("--no-face-enhance", dest="face_enhance", action="store_false")
    parser.set_defaults(face_enhance=False)
    parser.add_argument("--remove-bg", dest="remove_bg", action="store_true")
    parser.add_argument("--no-remove-bg", dest="remove_bg", action="store_false")
    parser.set_defaults(remove_bg=False)
    parser.add_argument("--fast", dest="fast_mode", action="store_true")
    parser.add_argument("--no-fast", dest="fast_mode", action="store_false")
    parser.set_defaults(fast_mode=True)
    parser.add_argument("--low-vram", dest="low_vram", action="store_true")
    parser.add_argument("--no-low-vram", dest="low_vram", action="store_false")
    parser.set_defaults(low_vram=False)
    parser.add_argument("--file-type", dest="file_type", default="jpeg", choices=["jpeg", "png", "webp"])
    parser.add_argument("--lora", dest="loras", action="append", default=[])
    parser.add_argument("--url", dest="url", default=None)
    parser.add_argument("--strength", dest="strength", type=float, default=None)
    parser.add_argument("--image2", dest="image_file2", default="")
    parser.add_argument("--edit-act", dest="edit_act", default="")
    parser.add_argument("--job-type", dest="job_type", default="MK")
    parser.add_argument("--keep-size", dest="keep_size", default="user")
    parser.add_argument("--tpl", dest="tpl_list", action="append", default=[])
    parser.add_argument("--timeout", dest="timeout", type=float, default=None)
    return parser


def _render_result_markdown(payload: dict) -> str:
    status = str(payload.get("status") or "").strip().lower()
    error = str(payload.get("error") or "").strip()
    if status == "completed":
        return ""

    if error:
        return error

    if status:
        return status

    return ""


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


class ImageSlashCommand:
    name = "image"

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
                        "tool": "image_generator",
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
                        "tool": "image_generator",
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
            size=ns.size,
            model_name=ns.model_name,
            family=ns.family,
            num_inference_steps=ns.num_inference_steps,
            guidance_scale=ns.guidance_scale,
            seed=ns.seed,
            generate_num=ns.generate_num,
            scheduler=ns.scheduler,
            upscale=ns.upscale,
            face_enhance=ns.face_enhance,
            remove_bg=ns.remove_bg,
            fast_mode=ns.fast_mode,
            low_vram=ns.low_vram,
            file_type=ns.file_type,
            strength=ns.strength,
            url=ns.url,
            image_file2=ns.image_file2,
            edit_act=ns.edit_act,
            tpl_list=list(ns.tpl_list or []),
            loras=_normalize_lora_args(ns.loras),
            job_type=ns.job_type,
            keep_size=ns.keep_size,
            timeout=ns.timeout,
        )

        from backend.services.agent.tools.builtin.image_generator import generate_image_sync

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: generate_image_sync(**kwargs),
        )
        task_id = str(payload.get("task_id") or "").strip()
        return ChatSlashCommandResult(
            handled=True,
            status=str(payload.get("status") or "failed"),
            assistant_text=_render_result_markdown(payload),
            task_ids=[task_id] if task_id else [],
            error=str(payload.get("error") or "") or None,
            artifacts=build_artifacts_from_tool_result(payload, default_category="image"),
        )


register_slash_command(ImageSlashCommand())

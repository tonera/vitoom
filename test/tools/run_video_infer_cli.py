"""
手动运行视频推理（不走前端/WS）：

用命令行参数构造 InferenceRequestParams，然后直接调用 inference/video 的 VideoInferrer.inference_callback()。

示例：
  python test/tools/run_video_infer_cli.py --job_type MKV --prompt "a cat" --duration 5 --width 832 --height 480

低显存模式（force_offload -> low_vram）：
  python test/tools/run_video_infer_cli.py --job_type MKV --prompt "a cat" --duration 5 --width 832 --height 480 --force-offload

更高 FPS（更顺滑，耗时/算力也会增加）：
  python test/tools/run_video_infer_cli.py --job_type MKV --prompt "a cat" --duration 5 --width 832 --height 480 --fps 24

I2V：
  python test/tools/run_video_infer_cli.py --job_type MKV --url "https://xxx/a.jpg" --prompt "" --duration 5 --width 832 --height 480 --lora my_high_noise.safetensors:1@dit 

S2V：
  python test/tools/run_video_infer_cli.py --job_type S2V --url "https://xxx/a.jpg" --prompt_wav_path "https://xxx/a.mp3" --prompt "a person is singing" --duration 10 --width 832 --height 448

python test/tools/run_video_infer_cli.py \
  --job_type MKV \
  --prompt "a cat" \
  --duration 5 --width 832 --height 480 \
  --model_name "/home/tonera/models/Wan2.2-TI2V-5B-FP8" \
  --force-offload \
  --no-fast-mode
  
python test/tools/run_video_infer_cli.py \
  --job_type MKV \
  --prompt "a cat" \
  --duration 5 --width 832 --height 480 \
  --model_name "/home/tonera/models/Turbo-Wan2.2-TI2V-5B-FP8" \
  --force-offload \
  --fast-mode
  

"""

from __future__ import annotations

import argparse
import asyncio
import uuid
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple

# 确保能从任意工作目录运行：把项目根目录加入 sys.path
# file: <repo>/test/tools/run_video_infer_cli.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_resolution(res: Optional[str]) -> Optional[Tuple[int, int]]:
    if not res:
        return None
    s = str(res).lower().strip()
    if "x" in s:
        a, b = s.split("x", 1)
        try:
            return int(a), int(b)
        except Exception:
            return None
    return None


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run video inferrer directly (no WS).")
    p.add_argument("--service_id", default="cli_video", help="service_id（仅用于日志/线程名前缀）")
    p.add_argument("--user_id", default="cli_user")
    p.add_argument("--task_id", default=None)
    p.add_argument("--job_type", required=True, help="MKV/S2V/INP/CCV")
    p.add_argument("--prompt", default="", help="允许为空；MKV 的 T2V/TI2V 需非空")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--url", default=None, help="参考图 URL/本地路径")
    p.add_argument("--image_file2", default=None, help="INP 尾帧图 URL/本地路径（可选）")
    p.add_argument("--ref_video", default=None, help="控制/pose 视频 URL/本地路径（S2V 复用为 pose_video）")
    p.add_argument("--face_video", default=None, help="face 驱动视频 URL/本地路径（IVV2V）")
    p.add_argument("--prompt_wav_path", default=None, help="S2V 音频 URL/本地路径")
    p.add_argument("--duration", type=int, default=5, help="视频时长（秒）")
    p.add_argument("--fps", type=int, default=None, help="输出视频帧率（可选，默认 24）")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--resolution", default=None, help="例如 832x480（若 width/height 未传则使用）")
    p.add_argument("--steps", type=int, default=30, help="num_inference_steps")
    p.add_argument("--cfg", type=float, default=7.5, help="guidance_scale/cfg_scale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_name", default=None, help="本地模型目录名或绝对路径（本项目仅使用该字段）")
    p.add_argument("--direction", default=None, help="CCV 镜头方向，例如 Left/Right/Up/Down")
    p.add_argument("--speed", type=float, default=None, help="CCV 镜头速度，例如 0.01")
    p.add_argument("--storage", default="local", choices=["local", "oss", "s3", "server"])
    p.add_argument(
        "--output",
        default=None,
        help="可选：推理完成后，将生成的本地 mp4 移动到该目录（仅对 --storage=local 生效）",
    )
    p.add_argument(
        "--force-offload",
        action="store_true",
        help="低显存模式：等价于请求参数 model_config.force_offload=true（视频侧将走 low_vram vram_config 或 OOM 回退）",
    )
    p.add_argument(
        "--fast-mode",
        dest="fast_mode",
        action="store_true",
        default=True,
        help="快速模式：用于路由 TurboDiffusion 等加速分支（默认开启）",
    )
    p.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="关闭快速模式（强制走常规分支）",
    )
    p.add_argument(
        "--lora",
        action="append",
        default=None,
        help="视频 LoRA（可多次）：格式 name[:weight][@target]，例如 my.safetensors:0.8@dit2 或 /abs/path/a.safetensors:1",
    )
    return p.parse_args()


async def main() -> None:
    args = build_args()
    task_id = args.task_id or f"cli_{uuid.uuid4().hex[:12]}"

    # 延迟导入项目依赖：保证 --help 在缺少部分依赖（如 Pillow/cv2）时也能正常显示
    from inference.common.logger import get_logger
    from inference.common.result_handler import ResultHandler
    from inference.common.config_loader import load_inference_config
    from inference.schemas import InferenceRequestParams
    from inference.video.inferrer import VideoInferrer

    global logger
    logger = get_logger(__name__)

    class CliResultHandler(ResultHandler):
        """
        CLI 专用 ResultHandler：
        - 复用原有保存逻辑
        - 额外记录每次 local 保存的绝对路径，便于推理结束后进行搬运/重命名
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved_local_paths = []  # List[Path]

        def save_file_local(self, file_data, file_name: str, subdir: Optional[str] = None) -> Optional[Path]:
            p = super().save_file_local(file_data, file_name, subdir=subdir)
            if p:
                try:
                    self.saved_local_paths.append(Path(p))
                except Exception:
                    pass
            return p

    # 分辨率：优先使用 width/height，否则尝试从 --resolution 解析
    wh = _parse_resolution(args.resolution)
    width = args.width or (wh[0] if wh else 832)
    height = args.height or (wh[1] if wh else 480)

    # 构造 request
    model_cfg = None
    if bool(getattr(args, "force_offload", False)):
        model_cfg = (model_cfg or {})
        model_cfg["force_offload"] = True
    if getattr(args, "fps", None) is not None:
        model_cfg = (model_cfg or {})
        model_cfg["fps"] = int(args.fps)

    # LoRA list -> req.loras (schema 支持 JSON 字符串 / list / dict)
    loras_payload = None
    if getattr(args, "lora", None):
        arr = []
        for spec in (args.lora or []):
            s = str(spec).strip()
            if not s:
                continue
            target = None
            if "@" in s:
                s, target = s.split("@", 1)
                target = target.strip() or None
            weight = None
            name = s
            if ":" in s:
                name, w = s.split(":", 1)
                name = name.strip()
                w = w.strip()
                if w:
                    try:
                        weight = float(w)
                    except Exception:
                        weight = None
            item = {"name": name}
            if weight is not None:
                item["weight"] = weight
            if target:
                item["target"] = target
            arr.append(item)
        loras_payload = arr if arr else None

    req = InferenceRequestParams(
        type="video",
        job_type=str(args.job_type).upper(),
        storage=args.storage,
        reference_id="",
        id=task_id,
        user_id=args.user_id,
        task_id=task_id,
        prompt=args.prompt or "",
        negative_prompt=args.negative_prompt or "",
        fast_mode=bool(getattr(args, "fast_mode", True)),
        width=width,
        height=height,
        guidance_scale=float(args.cfg),
        seed=int(args.seed),
        num_inference_steps=int(args.steps),
        file_type="mp4",
        url=args.url,
        image_file2=args.image_file2 or "",
        model_name=args.model_name,
        duration=int(args.duration),
        ref_video=args.ref_video,
        face_video=args.face_video,
        prompt_wav_path=args.prompt_wav_path,
        direction=args.direction,
        speed=args.speed,
        model_cfg=model_cfg,
        loras=loras_payload,
    )

    logger.info(f"Running video inference (no WS): task_id={task_id}, job_type={req.job_type}")

    # 直接创建 inferrer（不调用 BaseInferrer.initialize/start，不依赖 config/<service_id>.yaml）
    inferrer = VideoInferrer(service_id=args.service_id)
    cfg = load_inference_config()
    # models_dir 从配置文件读取（inference.yaml），不允许通过命令行覆盖
    rh = CliResultHandler(ws_client=None, storage_base_path=cfg.outputs_dir)
    inferrer.result_handler = rh

    await inferrer.inference_callback(req)

    # 可选：将输出文件移动到 --output 目录
    if args.output:
        if str(req.storage).lower() != "local":
            logger.warning(f"--output 仅对 --storage=local 生效；当前 storage={req.storage}，跳过搬运。")
        else:
            out_dir = Path(args.output).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            candidates = [p for p in (rh.saved_local_paths or []) if p and p.exists()]
            if not candidates:
                logger.warning("未找到可搬运的本地输出文件（可能保存失败或文件已被清理）。")
            else:
                src = candidates[-1]
                dst = out_dir / src.name
                # 避免覆盖：若已存在则自动加后缀
                if dst.exists():
                    stem, suf = src.stem, src.suffix
                    i = 1
                    while True:
                        alt = out_dir / f"{stem}_{i}{suf}"
                        if not alt.exists():
                            dst = alt
                            break
                        i += 1
                try:
                    shutil.move(str(src), str(dst))
                    logger.info(f"Moved output video to: {dst}")
                except Exception as e:
                    logger.error(f"Failed to move output video: src={src} dst={dst} err={e}", exc_info=True)
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())


import os

# 必须在导入 torch 之前设置。直接运行本文件，不要用 python -m，否则父包会先导入 torch。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from PIL import Image, ImageOps
from schemas import InferenceRequestParams

from video.handlers.minimax_h3.media import verify_mp4
from video.handlers.minimax_h3.models import H3Models
from video.handlers.minimax_h3.release import release_h3_models
from video.handlers.minimax_h3.references import load_references
from video.handlers.minimax_h3.request import H3Request, prepare_h3_request, reference_sources
from video.handlers.minimax_h3.runner import load_h3_models, reset_peak_stats, resize_keyframes, run_h3

# 同一种子两次运行：量化内核做不到逐字节一致，相对误差和画面 PSNR 用这两个上限。
LATENT_REL_MAX = 0.02
FRAME_PSNR_MIN = 30.0
# 关键帧锚在 t=0.999 且经过 VAE 往返（实测往返约 27.75 dB），首尾帧与输入图的 PSNR 下限。
KEYFRAME_PSNR_MIN = 22.0
ALLOC_SLACK_BYTES = 512 * 1024 * 1024
DEFAULT_MODEL_ROOT = Path("/home/tonera/models/MiniMax-H3")
FALLBACK_TOKENIZER = Path("/home/tonera/models/Qwen3-VL-8B-Instruct")


def main() -> None:
    args = _parse_args()
    model_root = Path(args.model_root).expanduser().resolve()
    tokenizer = _tokenizer_dir(model_root, args.tokenizer)
    reuse = int(args.ttl) > 0
    models: H3Models | None = None
    allocated: list[int] = []
    failures: list[str] = []
    try:
        for job_index in range(int(args.jobs)):
            reset_peak_stats()
            request = _request(args, model_root, tokenizer)
            if models is None:
                models = load_h3_models(request, log=print)
            videos, stats = run_h3(
                request,
                models,
                [_output_path(args.out, job_index, args.jobs)],
                task_id=f"cli-{job_index}",
                log=print,
                return_latents=int(args.repeat) > 1,
            )
            _check_job(args, request, videos, stats, failures)
            if int(args.repeat) > 1:
                _check_repeat(args, request, models, stats.latents, videos, failures)
            allocated.append(stats.allocated_bytes)
            print(
                f"[job {job_index + 1}] peak={stats.peak_bytes / 2**30:.2f} GiB "
                f"allocated={stats.allocated_bytes / 2**30:.2f} GiB -> {videos[0].path}"
            )
            if not reuse:
                release_h3_models(models)
                models = None
    finally:
        release_h3_models(models)
    if int(args.jobs) >= 2:
        base = allocated[0]
        for index, value in enumerate(allocated[1:], start=2):
            if abs(value - base) > ALLOC_SLACK_BYTES:
                failures.append(
                    f"第 {index} 个任务结束后显存 {value / 2**30:.2f} GiB，"
                    f"与第 1 个任务的 {base / 2**30:.2f} GiB 相差超过 512 MiB"
                )
    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)
    if args.check or int(args.jobs) >= 2 or int(args.repeat) > 1:
        print("PASS")


def _request(args, model_root: Path, tokenizer: Path | None) -> H3Request:
    params = _params(args, model_root)
    sources = reference_sources(params)
    return prepare_h3_request(
        params,
        models_dir=str(model_root.parent),
        image=_open_image(args.image),
        last_image=_open_image(args.last_image),
        references=asyncio.run(load_references(sources)) if sources else None,
        log=print,
        tokenizer_dir=str(tokenizer) if tokenizer else None,
    )


def _open_image(path: str | None) -> Image.Image | None:
    if not path:
        return None
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _check_job(args, request: H3Request, videos, stats, failures: list[str]) -> None:
    if not args.check:
        return
    limit = _peak_limit_gib(int(args.seconds), request.workflow)
    peak_gib = stats.peak_bytes / 2**30
    if limit is not None and peak_gib > limit:
        failures.append(f"峰值显存 {peak_gib:.2f} GiB，超过 {limit} GiB")
    for video in videos:
        try:
            verify_mp4(video.path, width=video.width, height=video.height, num_frames=video.num_frames)
        except Exception as exc:
            failures.append(str(exc))
        _check_keyframes(request, video.path, failures)


def _check_keyframes(request: H3Request, path: Path, failures: list[str]) -> None:
    """首帧、尾帧要和放到画布上的输入图一致。"""
    keyframes, anchors = resize_keyframes(request)
    if not keyframes:
        return
    frames = _read_frames(path)
    for keyframe, anchor in zip(keyframes, anchors):
        frame = frames[0] if anchor == "first" else frames[-1]
        psnr = _psnr(np.asarray(keyframe), np.asarray(frame))
        print(f"[keyframe] {anchor} psnr={psnr:.2f} dB")
        if psnr < KEYFRAME_PSNR_MIN:
            failures.append(f"{anchor} 关键帧 PSNR {psnr:.2f} dB，低于 {KEYFRAME_PSNR_MIN} dB")


def _check_repeat(args, request: H3Request, models: H3Models, first_latents, first_videos, failures) -> None:
    if not first_latents:
        failures.append("第一次运行没有返回 latent")
        return
    out = Path(args.out).with_name(Path(args.out).stem + "_repeat" + Path(args.out).suffix)
    videos, stats = run_h3(request, models, [out], task_id="cli-repeat", log=print, return_latents=True)
    second = stats.latents or []
    if len(second) != len(first_latents):
        failures.append("两次运行的 latent 条数不一致")
        return
    for index, ((video_a, audio_a), (video_b, audio_b)) in enumerate(zip(first_latents, second)):
        video_err = _relative(video_a, video_b)
        audio_err = _relative(audio_a, audio_b)
        print(f"[repeat] latent {index} video_rel={video_err:.4f} audio_rel={audio_err:.4f}")
        if video_err > LATENT_REL_MAX or audio_err > LATENT_REL_MAX:
            failures.append(f"latent 相对误差 {max(video_err, audio_err):.4f} 超过 {LATENT_REL_MAX}")
    if first_videos and videos:
        psnr = _video_psnr(first_videos[0].path, videos[0].path)
        print(f"[repeat] frame psnr={psnr:.2f} dB")
        if psnr < FRAME_PSNR_MIN:
            failures.append(f"解码 PSNR {psnr:.2f} dB，低于 {FRAME_PSNR_MIN} dB")


def _params(args, model_root: Path) -> InferenceRequestParams:
    return InferenceRequestParams(
        type="video",
        job_type=args.job_type,
        id="h3-cli",
        user_id="cli",
        task_id="h3-cli",
        prompt=args.prompt,
        url=args.image or "",
        image_file2=args.last_image or "",
        tpl_list=list(args.ref_image or []),
        ref_video=args.ref_video,
        ref_audio=args.ref_audio,
        duration=int(args.seconds),
        aspect_ratio=args.aspect_ratio,
        width=int(args.width or 1024),
        height=int(args.height or 1024),
        seed=args.seed,
        fast_mode=not args.no_turbo,
        num_inference_steps=int(args.steps),
        file_type="mp4",
        load_name=str(model_root),
        generate_num=1,
        resolution=args.resolution,
    )


def _tokenizer_dir(model_root: Path, override: str | None) -> Path | None:
    if override:
        return Path(override)
    bundled = model_root / "tokenizer"
    if (bundled / "tokenizer.json").is_file():
        return None
    if (FALLBACK_TOKENIZER / "tokenizer.json").is_file():
        print(f"[h3] 模型包里没有 tokenizer/，暂用 {FALLBACK_TOKENIZER}")
        return FALLBACK_TOKENIZER
    return None


def _output_path(raw: str, index: int, jobs: int) -> Path:
    path = Path(raw)
    if jobs <= 1 or index == 0:
        return path
    return path.with_name(f"{path.stem}_{index}{path.suffix}")


def _peak_limit_gib(seconds: int, workflow: str) -> float | None:
    if seconds >= 15 or workflow != "t2va":
        return 31.0
    if seconds <= 5:
        return 24.0
    return None


def _relative(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left.float()).clamp_min(1e-8)
    return float(torch.linalg.vector_norm(left.float() - right.float()) / denom)


def _video_psnr(left: Path, right: Path) -> float:
    import imageio

    a = _read_frames(left)
    b = _read_frames(right)
    count = min(len(a), len(b))
    if count == 0:
        return 0.0
    return _psnr(np.stack(a[:count]), np.stack(b[:count]))


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _read_frames(path: Path):
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 文生视频 / 首尾帧生视频")
    parser.add_argument("--prompt", default=(
        "Realistic live-action footage, golden hour on a quiet seaside pier. A middle-aged fisherman in a yellow raincoat "
        "casts his line. Seagulls call overhead. Audio: waves, the fishing reel, distant seagulls. No music, no subtitles."
    ))
    parser.add_argument("--image", default=None, help="首帧图片路径")
    parser.add_argument("--last-image", default=None, help="尾帧图片路径")
    parser.add_argument("--job-type", default="MKV", choices=["MKV", "INP"], help="INP 在有关键帧时允许空 prompt")
    parser.add_argument("--ref-image", action="append", help="参考图（可重复，最多 9 张，顺序即 <Picture i> 的编号）")
    parser.add_argument("--ref-video", default=None, help="参考视频，自带音轨时一并作为条件")
    parser.add_argument("--ref-audio", default=None, help="参考音频，不能单独使用")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--aspect-ratio", default=None)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30, help="非 Turbo 时的 DiT 前向次数；Turbo 的步数由 LoRA 决定")
    parser.add_argument("--no-turbo", action="store_true")
    parser.add_argument("--ttl", type=int, default=0, help="大于 0 时任务之间复用分词器，模拟 PipelineCache 的 TTL>0")
    parser.add_argument("--jobs", type=int, default=1, help="连续任务数，用来看显存是否回落")
    parser.add_argument("--repeat", type=int, default=1, help="同一种子再跑一次，比较 latent 和 PSNR")
    parser.add_argument("--check", action="store_true", help="校验分辨率、帧数、音轨和峰值显存")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--out", default="minimax_h3.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    main()

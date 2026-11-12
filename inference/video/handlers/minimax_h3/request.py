"""把 InferenceRequestParams 映射成 H3 推理参数：t2va（纯文本）或 fl2va（首帧 / 尾帧 / 首尾帧）。"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas import InferenceRequestParams

from video.handlers.minimax_h3.checkpoints import H3Checkpoints, resolve_checkpoints, resolve_model_root

H3_FAMILY = "minimax_h3"
H3_FPS = 24
H3_MIN_SECONDS = 5
H3_MAX_SECONDS = 15
H3_SHORT_EDGE = 768
# ComfyUI 尺寸对照表的最小档（0.2 MP，16:9 为 608x352）。
H3_MIN_SHORT_EDGE = 352
H3_MAX_PIXELS = 768 * 1344
H3_CANVAS_MULTIPLE = 32
H3_AUDIO_RATE = 32000
H3_DEFAULT_STEPS = 30
H3_MAX_STEPS = 50
H3_BASE_VIDEO_SHIFT = 12.0
H3_TURBO_768P_VIDEO_SHIFT = 6.0
H3_AUDIO_SHIFT = 3.0
H3_MAX_IMAGE_REFERENCES = 9
_DEFAULT_CANVAS = (1024, 1024)
_LORA_STEPS = re.compile(r"(\d+)step")
_LORA_TAG = re.compile(r"<\s*lora\b", re.IGNORECASE)


@dataclass(frozen=True)
class H3Request:
    workflow: str
    prompt: str
    height: int
    width: int
    num_frames: int
    aligned_frames: int
    duration_seconds: int
    seeds: list[int]
    turbo: bool
    steps: int
    video_shift: float
    checkpoints: H3Checkpoints
    image: Any = None
    last_image: Any = None
    references: tuple = ()

    @property
    def grid_points(self) -> int:
        """diffusers 的 num_inference_steps 数的是 sigma 网格点（含终点 0），比 DiT 前向次数多 1。"""
        return self.steps + 1

    @property
    def keyframes(self) -> dict[str, Any]:
        """传给 diffusers 关键帧步骤的 image / last_image，只含实际给出的那几张。"""
        return {name: frame for name, frame in (("image", self.image), ("last_image", self.last_image)) if frame is not None}


@dataclass(frozen=True)
class H3ReferenceSources:
    """Ref2VA 参考素材的来源，顺序即打包顺序：图片（按 tpl_list）、视频、音频。"""

    images: tuple[str, ...] = ()
    video: str | None = None
    audio: str | None = None

    def __bool__(self) -> bool:
        return bool(self.images or self.video or self.audio)

    def __len__(self) -> int:
        return len(self.images) + int(self.video is not None) + int(self.audio is not None)


def reference_sources(params) -> H3ReferenceSources:
    """读出并校验参考来源。只看字符串、不下载，校验不过就在加载任何东西之前失败。"""
    images = tuple(str(item).strip() for item in (getattr(params, "tpl_list", None) or []) if str(item or "").strip())
    video = str(getattr(params, "ref_video", None) or "").strip() or None
    audio = str(getattr(params, "ref_audio", None) or "").strip() or None
    if len(images) > H3_MAX_IMAGE_REFERENCES:
        raise ValueError(f"H3 最多 {H3_MAX_IMAGE_REFERENCES} 张参考图，收到 {len(images)} 张")
    if audio and not images and not video:
        raise ValueError("ref_audio 不能单独使用，至少还要有一张参考图（tpl_list）或一段参考视频（ref_video）")
    sources = H3ReferenceSources(images=images, video=video, audio=audio)
    if sources:
        if str(getattr(params, "job_type", "") or "").strip().upper() != "MKV":
            raise ValueError("参考生视频只支持 MKV 任务；INP 是首尾帧生视频")
        if str(getattr(params, "url", "") or "").strip() or str(getattr(params, "image_file2", "") or "").strip():
            raise ValueError("参考生视频（tpl_list / ref_video / ref_audio）不能和首尾帧（url / image_file2）同时使用")
    return sources


def is_minimax_h3_family(family: str | None) -> bool:
    return str(family or "").strip() == H3_FAMILY


def turbo_schedule(lora_name: str) -> tuple[int, float]:
    """Turbo LoRA 的 DiT 前向次数和视频偏移，必须与蒸馏时一致，从文件名读出。

    ModelTC 发布说明：768p 版（v1.0）训练偏移为视频 6、音频 3；544p 版为视频 12、音频 3。
    """
    name = lora_name.lower()
    match = _LORA_STEPS.search(name)
    if match is None:
        raise ValueError(f"无法从 Turbo LoRA 文件名读出步数（需要形如 4step）: {lora_name}")
    return int(match.group(1)), (H3_TURBO_768P_VIDEO_SHIFT if "768p" in name else H3_BASE_VIDEO_SHIFT)


def align_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    aligned = int(num_frames)
    if aligned < 1:
        raise ValueError(f"帧数必须为正，收到 {num_frames}")
    while aligned % frames_per_chunk != latents_per_chunk:
        aligned += 1
    return aligned


def parse_short_edge(resolution) -> int:
    """resolution 为短边像素（480、"720"、"544p" 均可），与前端约定一致；缺省为官方 768。"""
    text = str(resolution or "").strip().lower().removesuffix("p")
    if not text:
        return H3_SHORT_EDGE
    try:
        short_edge = int(float(text))
    except ValueError as exc:
        raise ValueError(f"resolution 应为短边像素（如 480、720、768），收到 {resolution}") from exc
    if not H3_MIN_SHORT_EDGE <= short_edge <= H3_SHORT_EDGE:
        raise ValueError(f"H3 的 resolution 必须在 {H3_MIN_SHORT_EDGE}–{H3_SHORT_EDGE} 之间，收到 {resolution}")
    return short_edge


def resolve_h3_canvas(
    aspect_ratio: str | None,
    width: int | None,
    height: int | None,
    keyframe_size: tuple[int, int] | None = None,
    short_edge: int = H3_SHORT_EDGE,
) -> tuple[int, int]:
    """返回 (height, width)。比例优先级：aspect_ratio，第一张关键帧，非默认宽高，否则 16:9。

    短边取最接近的 32 的倍数（恰在中间时取大），面积不超过官方 768x1344。
    """
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

    ratio_w, ratio_h = _aspect_pair(aspect_ratio, width, height, keyframe_size)
    snapped = max(H3_CANVAS_MULTIPLE, int(short_edge / H3_CANVAS_MULTIPLE + 0.5) * H3_CANVAS_MULTIPLE)
    canvas_h, canvas_w = resolve_canvas_size(ratio_w, ratio_h, H3_CANVAS_MULTIPLE, snapped, H3_MAX_PIXELS)
    return int(canvas_h), int(canvas_w)


def prepare_h3_request(
    params: InferenceRequestParams,
    *,
    models_dir: str,
    image=None,
    last_image=None,
    references=None,
    log=None,
    tokenizer_dir: str | None = None,
) -> H3Request:
    """校验并解析一条 H3 请求。

    image / last_image 是调用方按 url / image_file2 加载好的 PIL 图；references 是按 reference_sources 的顺序
    加载好的 diffusers 参考对象。通过后把整数秒和画布写回 params。
    """
    job_type = str(params.job_type or "").strip().upper()
    if job_type not in {"MKV", "INP"}:
        raise ValueError(f"H3 不支持该任务类型: {params.job_type}（仅支持 MKV 与 INP）")

    file_type = str(params.file_type or "mp4").strip().lower()
    if file_type != "mp4":
        raise ValueError(f"H3 只支持 mp4，收到 file_type={params.file_type}")

    seconds = int(params.duration)
    if seconds < H3_MIN_SECONDS or seconds > H3_MAX_SECONDS:
        raise ValueError(f"H3 的时长必须在 {H3_MIN_SECONDS}–{H3_MAX_SECONDS} 秒之间，收到 {seconds} 秒")

    for field, frame in (("url", image), ("image_file2", last_image)):
        if bool(str(getattr(params, field) or "").strip()) != (frame is not None):
            raise ValueError(f"{field} 与加载出的关键帧不一致，调用方应先按 {field} 加载图片")
    first_keyframe = image if image is not None else last_image
    sources = reference_sources(params)
    references = tuple(references or ())
    if len(references) != len(sources):
        raise ValueError(f"加载出的参考素材数 {len(references)} 与 tpl_list / ref_video / ref_audio 不一致（{len(sources)}）")
    if sources:
        workflow = "ref2va"
    else:
        workflow = "t2va" if first_keyframe is None else "fl2va"

    prompt = str(params.prompt or "")
    if not prompt.strip() and (job_type == "MKV" or workflow == "t2va"):
        raise ValueError("prompt 不能为空（只有带首帧或尾帧的 INP 任务允许空 prompt）")
    if _has_user_lora(prompt, getattr(params, "loras", None)):
        raise ValueError("H3 暂不支持额外 LoRA（loras 或 <lora:> 标签）。Turbo LoRA 由 fast_mode 控制")

    _warn_ignored_fields(params, log)
    height, width = resolve_h3_canvas(
        params.aspect_ratio,
        params.width,
        params.height,
        keyframe_size=None if first_keyframe is None else tuple(first_keyframe.size),
        short_edge=parse_short_edge(getattr(params, "resolution", None)),
    )
    raw_frames = int(round(seconds * H3_FPS))
    aligned = align_num_frames(raw_frames)
    turbo = bool(getattr(params, "fast_mode", True))
    root = resolve_model_root(models_dir, str(params.load_name or ""))
    checkpoints = resolve_checkpoints(
        root,
        dit_variant="ref2va" if workflow == "ref2va" else "fl2va",
        turbo=turbo,
        tokenizer_dir=Path(tokenizer_dir) if tokenizer_dir else None,
    )
    if turbo:
        steps, video_shift = turbo_schedule(checkpoints.lora.name)
    else:
        steps = int(params.num_inference_steps or H3_DEFAULT_STEPS)
        video_shift = H3_BASE_VIDEO_SHIFT
        if not 1 <= steps <= H3_MAX_STEPS:
            raise ValueError(f"H3 的 num_inference_steps 必须在 1–{H3_MAX_STEPS} 之间，收到 {steps}")
    seeds = _resolve_seeds(params.seed, int(params.generate_num or 1))

    params.duration = seconds
    params.file_type = "mp4"
    params.width = width
    params.height = height
    if log is not None:
        log(
            f"[h3] workflow={workflow} first={image is not None} last={last_image is not None} "
            f"references={[type(item).__name__ for item in references]} "
            f"canvas={height}x{width} frames={raw_frames}->{aligned} ({aligned / H3_FPS:.2f}s) "
            f"turbo={turbo} steps={steps} video_shift={video_shift:g} seeds={seeds} dit={checkpoints.dit.name}"
        )
    return H3Request(
        workflow=workflow,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=raw_frames,
        aligned_frames=aligned,
        duration_seconds=seconds,
        seeds=seeds,
        turbo=turbo,
        steps=steps,
        video_shift=video_shift,
        checkpoints=checkpoints,
        image=image,
        last_image=last_image,
        references=references,
    )


def _aspect_pair(
    aspect_ratio: str | None,
    width: int | None,
    height: int | None,
    keyframe_size: tuple[int, int] | None,
) -> tuple[float, float]:
    text = str(aspect_ratio or "").strip()
    if text:
        return _parse_ratio(text)
    if keyframe_size is not None:
        return float(keyframe_size[0]), float(keyframe_size[1])
    w = int(width or 0)
    h = int(height or 0)
    if (w, h) != _DEFAULT_CANVAS and w > 0 and h > 0:
        return float(w), float(h)
    return 16.0, 9.0


def _parse_ratio(text: str) -> tuple[float, float]:
    normalized = text.replace("：", ":").replace("/", ":")
    if ":" not in normalized:
        raise ValueError(f"aspect_ratio 应为 宽:高，收到 {text}")
    left, right = normalized.split(":", 1)
    try:
        width = float(left)
        height = float(right)
    except ValueError as exc:
        raise ValueError(f"aspect_ratio 应为 宽:高，收到 {text}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"aspect_ratio 必须为正，收到 {text}")
    return width, height


def _has_user_lora(prompt: str, loras) -> bool:
    if _LORA_TAG.search(prompt or ""):
        return True
    if loras is None:
        return False
    if isinstance(loras, str):
        text = loras.strip()
        return bool(text) and text not in {"[]", "null", "None"}
    if isinstance(loras, dict):
        return bool(loras)
    if isinstance(loras, (list, tuple)):
        return len(loras) > 0
    return True


def _warn_ignored_fields(params: InferenceRequestParams, log) -> None:
    if log is None:
        return
    if str(params.negative_prompt or "").strip():
        log("[h3] 忽略 negative_prompt：权重已做 CFG 蒸馏，没有负向分支")
    guidance = float(getattr(params, "guidance_scale", 7.5) or 0.0)
    if abs(guidance - 7.5) > 1e-6:
        log(f"[h3] 忽略 guidance_scale={guidance}：权重已做 CFG 蒸馏")
    if str(getattr(params, "schedulerName", "") or "").strip():
        log("[h3] 忽略 schedulerName：固定使用 MiniMaxH3Scheduler（视频 shift 12，音频 shift 3）")
    runtime = getattr(params, "runtime_config", None)
    if isinstance(runtime, dict) and runtime.get("fps") is not None:
        log(f"[h3] 忽略 runtime_config.fps={runtime.get('fps')}：H3 固定 {H3_FPS} fps")


def _resolve_seeds(seed, generate_num: int) -> list[int]:
    if seed is None or int(seed) < 0:
        base = random.SystemRandom().randint(0, 2**31 - 1)
    else:
        base = int(seed)
    count = max(1, int(generate_num))
    return [base + index for index in range(count)]

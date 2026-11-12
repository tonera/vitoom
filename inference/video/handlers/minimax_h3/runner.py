"""分阶段执行 H3：条件编码（文本 / 视觉塔、关键帧 VAE）、DiT 采样、VAE 解码，任一时刻只有一个大模型在卡上。"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep, MiniMaxH3ResizeStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3KeyframeVaeEncoderStep,
    MiniMaxH3Ref2VAReferenceEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3CoreDenoiseStep,
    MiniMaxH3DecodeStep,
    MiniMaxH3FL2VACoreDenoiseStep,
    MiniMaxH3Ref2VACoreDenoiseStep,
)
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
from PIL import Image

from common.task_cancel import TaskCancelledError
from video.handlers.minimax_h3.dit import load_pruned_dit
from video.handlers.minimax_h3.lora import attach_comfy_lora
from video.handlers.minimax_h3.media import save_mp4
from video.handlers.minimax_h3.models import H3Models
from video.handlers.minimax_h3.release import free_cuda
from video.handlers.minimax_h3.request import H3Request, H3_AUDIO_RATE, H3_AUDIO_SHIFT, H3_CANVAS_MULTIPLE, H3_FPS
from video.handlers.minimax_h3.text_encoder import (
    build_presentation,
    build_reference_presentation,
    encode_prompt,
    load_processor,
    load_text_encoder,
)
from video.handlers.minimax_h3.vae import load_audio_vae, load_video_vae

_CORE_DENOISE = {
    "t2va": MiniMaxH3CoreDenoiseStep,
    "fl2va": MiniMaxH3FL2VACoreDenoiseStep,
    "ref2va": MiniMaxH3Ref2VACoreDenoiseStep,
}
# ref2va 的 DiT 是另一份权重，diffusers 用单独的组件名。
_TRANSFORMER_COMPONENT = {"ref2va": "transformer_ref"}
# 单张 32 GB 卡的 DiT 序列上限。实测 15 秒首尾帧约 11.3 万行、峰值 25.3 GiB，每行约 75 KB 激活。
H3_MAX_SEQUENCE_ROWS = 170_000
_AUDIO_LATENTS_PER_SECOND = 40


@dataclass
class H3Output:
    path: Path
    seed: int
    num_frames: int
    height: int
    width: int


@dataclass
class H3RunStats:
    peak_bytes: int
    allocated_bytes: int
    latents: list[tuple[torch.Tensor, torch.Tensor]] | None = None


@dataclass
class H3Conditions:
    """采样所需的全部条件。extra 是工作流专属的去噪块输入：fl2va 为关键帧 latent 和锚点，ref2va 为参考三件套。"""

    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    extra: dict = field(default_factory=dict)

    def pipeline_inputs(self) -> dict:
        return {"prompt_embeds": self.prompt_embeds, "text_token_tags": self.text_token_tags, **self.extra}


def reset_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def memory_bytes() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.memory_allocated())


def load_h3_models(request: H3Request, log=print) -> H3Models:
    tokenizer_dir = str(request.checkpoints.tokenizer_dir)
    log(f"[h3] load processor {tokenizer_dir}")
    return H3Models(
        model_root=str(request.checkpoints.root),
        tokenizer_dir=tokenizer_dir,
        processor=load_processor(tokenizer_dir),
    )


def resize_keyframes(request: H3Request) -> tuple[list, tuple]:
    """把关键帧放到画布上：第一张拉伸，跟随的那张 cover-crop，规则与 diffusers 的 MiniMaxH3ResizeStep 一致。"""
    if not request.keyframes:
        return [], ()
    out = MiniMaxH3ResizeStep().init_pipeline()(
        **request.keyframes,
        height=request.height,
        width=request.width,
        output=["keyframes", "keyframe_anchors"],
    )
    return list(out["keyframes"]), tuple(out["keyframe_anchors"])


def normalize_references(request: H3Request) -> list:
    """参考视频重采样到 24 fps、截到生成时长并放到自身比例的画布上，音轨截到生成时长（diffusers 原样）。

    参考图改用 ComfyUI 默认的 match：按比例只缩不放，缩到生成画面的像素量。diffusers 会放大到短边 2048，
    一张 16:9 参考图约 7200 行序列，480p 生成时比生成画面本身每帧多十几倍，采样慢数倍。
    """
    with _allow_snapped_duration():
        out = MiniMaxH3Ref2VASetupStep().init_pipeline()(
            references=list(request.references),
            height=request.height,
            width=request.width,
            num_frames=request.aligned_frames,
            output="normalized_references",
        )
    area = request.height * request.width
    return [
        _match_image(original.image, area) if item.kind == "image" else item
        for original, item in zip(request.references, out)
    ]


def _match_image(image: Image.Image, area: int) -> MiniMaxH3ImageReference:
    width, height = image.size
    scale = min(1.0, (area / (width * height)) ** 0.5)
    size = (
        max(H3_CANVAS_MULTIPLE, round(width * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE),
        max(H3_CANVAS_MULTIPLE, round(height * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE),
    )
    return MiniMaxH3ImageReference(image=image if image.size == size else image.resize(size, Image.Resampling.LANCZOS))


def estimate_sequence_rows(request: H3Request, references: list, num_text_tokens: int) -> int:
    """DiT 打包序列的行数：文本、参考的视频行、生成的视频行，加上音频行。"""
    rows = num_text_tokens + _latent_frames(request.aligned_frames) * request.height * request.width // 1024
    rows += round(request.aligned_frames / H3_FPS * _AUDIO_LATENTS_PER_SECOND) * 2
    for reference in references:
        if reference.kind == "image":
            width, height = reference.image.size
            rows += height * width // 1024
        elif reference.kind == "video":
            frames, height, width = reference.frames.shape[:3]
            rows += _latent_frames(max(1, (frames - 5) // 17) * 17 + 5) * height * width // 1024
        if reference.has_audio:
            rows += round(reference.audio.shape[-1] / H3_AUDIO_RATE * _AUDIO_LATENTS_PER_SECOND) * 2
    return rows


def _latent_frames(num_frames: int) -> int:
    """视频 VAE 每 17 帧压成 5 个 latent 帧，17n+5 帧对应 5n+2 个。"""
    return (num_frames - 5) // 17 * 5 + 2


def run_h3(
    request: H3Request,
    models: H3Models,
    output_paths: list[Path],
    *,
    task_id: str = "",
    is_cancelled: Callable[[], bool] | None = None,
    log=print,
    return_latents: bool = False,
) -> tuple[list[H3Output], H3RunStats]:
    """同步执行一条 H3 任务。返回时卡上不留任何 H3 权重。"""
    if request.workflow not in _CORE_DENOISE:
        raise ValueError(f"H3 暂不支持 workflow={request.workflow}")
    check = _cancel_checker(is_cancelled, task_id)
    try:
        check("before encode")
        conditions = _encode_conditions(models, request, check, log)
        check("before sample")
        samples = _denoise(request, conditions, check, log)
        del conditions
        saved = [(v.detach().float().cpu(), a.detach().float().cpu()) for v, a in samples] if return_latents else []
        outputs = _decode_all(request, samples, output_paths, check, log)
    finally:
        free_cuda()
    peak, allocated = memory_bytes()
    return outputs, H3RunStats(peak_bytes=peak, allocated_bytes=allocated, latents=saved or None)


def _encode_conditions(models: H3Models, request: H3Request, check, log) -> H3Conditions:
    if request.workflow == "ref2va":
        return _encode_ref2va(models, request, check, log)
    keyframes, anchors = resize_keyframes(request)
    presentation = build_presentation(models.processor, request.prompt, keyframes)
    prompt_embeds, tags = _encode_text(models, request, presentation, log)
    if not keyframes:
        return H3Conditions(prompt_embeds, tags)
    check("before keyframe encode")
    latents = _encode_keyframes(request, keyframes, log)
    return H3Conditions(prompt_embeds, tags, {"condition_latents": latents, "keyframe_anchors": anchors})


def _encode_ref2va(models: H3Models, request: H3Request, check, log) -> H3Conditions:
    references = normalize_references(request)
    presentation = build_reference_presentation(models.processor, request.prompt, references)
    rows = estimate_sequence_rows(request, references, len(presentation[0]))
    log(f"[h3] ref2va references={[item.kind for item in references]} estimated_rows={rows}")
    if rows > H3_MAX_SEQUENCE_ROWS:
        raise ValueError(
            f"参考素材太多或太长：估算 DiT 序列 {rows} 行，超过单卡上限 {H3_MAX_SEQUENCE_ROWS}。"
            "请减少参考图、缩短参考视频或生成时长"
        )
    prompt_embeds, tags = _encode_text(models, request, presentation, log)
    check("before reference encode")
    latents, audio_latents = _encode_references(request, references, log)
    return H3Conditions(
        prompt_embeds,
        tags,
        {"normalized_references": references, "condition_latents": latents, "audio_condition_latents": audio_latents},
    )


def _encode_text(models: H3Models, request: H3Request, presentation, log) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids, token_tags, vision_inputs = presentation
    encoder = load_text_encoder(str(request.checkpoints.text_encoder), with_vision=bool(vision_inputs), log=log)
    try:
        embeds = encode_prompt(encoder, token_ids, processor=models.processor, vision_inputs=vision_inputs)
        log(f"[h3] presentation tokens={len(token_ids)} vision={sorted(vision_inputs)} embeds={tuple(embeds.shape)}")
        return embeds, torch.tensor(token_tags, dtype=torch.long)
    finally:
        encoder = None
        free_cuda()


def _encode_references(request: H3Request, references: list, log) -> tuple[list, list]:
    video_vae = audio_vae = pipe = None
    try:
        video_vae = load_video_vae(str(request.checkpoints.video_vae), log=log)
        audio_vae = load_audio_vae(str(request.checkpoints.audio_vae), log=log)
        pipe = MiniMaxH3Ref2VAReferenceEncoderStep().init_pipeline()
        pipe.update_components(vae=video_vae, audio_vae=audio_vae)
        out = pipe(normalized_references=references, output=["condition_latents", "audio_condition_latents"])
        latents, audio_latents = out["condition_latents"], out["audio_condition_latents"]
        log(
            f"[h3] reference latents={[tuple(item.shape) for item in latents]} "
            f"audio={[tuple(item.shape) for item in audio_latents]}"
        )
        return latents, audio_latents
    finally:
        if pipe is not None:
            pipe.update_components(vae=None, audio_vae=None)
        pipe = video_vae = audio_vae = None
        free_cuda()


def _encode_keyframes(request: H3Request, keyframes: list, log) -> list[torch.Tensor]:
    video_vae = pipe = None
    try:
        video_vae = load_video_vae(str(request.checkpoints.video_vae), log=log)
        pipe = MiniMaxH3KeyframeVaeEncoderStep().init_pipeline()
        pipe.update_components(vae=video_vae)
        latents = pipe(keyframes=keyframes, output="condition_latents")
        log(f"[h3] keyframe latents={[tuple(item.shape) for item in latents]}")
        return latents
    finally:
        if pipe is not None:
            pipe.update_components(vae=None)
        pipe = video_vae = None
        free_cuda()


def _denoise(request: H3Request, conditions: H3Conditions, check, log) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dit = load_pruned_dit(str(request.checkpoints.dit), log=log)
    component = _TRANSFORMER_COMPONENT.get(request.workflow, "transformer")
    pipe = None
    try:
        if request.turbo:
            attach_comfy_lora(dit, str(request.checkpoints.lora), device="cuda", log=log)
        dit.time_proj.on_step = lambda: check("dit step")
        pipe = _CORE_DENOISE[request.workflow]().init_pipeline()
        pipe.update_components(
            **{component: dit},
            scheduler=MiniMaxH3Scheduler(shift=request.video_shift),
            audio_scheduler=MiniMaxH3Scheduler(shift=H3_AUDIO_SHIFT),
        )
        samples = []
        with _allow_snapped_duration():
            for seed in request.seeds:
                check("before sample")
                log(
                    f"[h3] denoise workflow={request.workflow} seed={seed} steps={request.steps} "
                    f"video_shift={request.video_shift:g} turbo={request.turbo}"
                )
                state = pipe(
                    **conditions.pipeline_inputs(),
                    height=request.height,
                    width=request.width,
                    num_frames=request.aligned_frames,
                    num_inference_steps=request.grid_points,
                    generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                    output=["latents", "audio_latents"],
                )
                samples.append((state["latents"], state["audio_latents"]))
        return samples
    finally:
        # 异常时 traceback 仍引用本帧，必须显式断开，DiT 的显存才能在解码前还回来。
        if pipe is not None:
            pipe.update_components(**{component: None})
        pipe = dit = None
        free_cuda()


def _decode_all(request: H3Request, samples, output_paths: list[Path], check, log) -> list[H3Output]:
    check("before decode")
    video_vae = audio_vae = dpipe = None
    try:
        video_vae = load_video_vae(str(request.checkpoints.video_vae), log=log)
        audio_vae = load_audio_vae(str(request.checkpoints.audio_vae), log=log)
        dpipe = MiniMaxH3DecodeStep().init_pipeline()
        dpipe.update_components(vae=video_vae, audio_vae=audio_vae)
        outputs: list[H3Output] = []
        for index, ((latents, audio_latents), seed) in enumerate(zip(samples, request.seeds)):
            check("before decode")
            out = dpipe(
                latents=latents,
                audio_latents=audio_latents,
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
            )
            video = np.asarray(out["videos"][0])
            if video.dtype != np.uint8:
                video = (np.clip(video, 0, 1) * 255).astype(np.uint8)
            audio = out["audio"]
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            sample_rate = int(out["sampling_rate"] or H3_AUDIO_RATE)
            frames, height, width, _ = video.shape
            log(
                f"[h3] decoded seed={seed} frames={frames} canvas={height}x{width} "
                f"audio={tuple(audio.shape)} sr={sample_rate} aligned_seconds={frames / H3_FPS:.2f}"
            )
            check("before mux")
            path = _output_path(output_paths, index, seed)
            save_mp4(video, audio, sample_rate, path, fps=H3_FPS)
            outputs.append(H3Output(path=path, seed=int(seed), num_frames=int(frames), height=int(height), width=int(width)))
        return outputs
    finally:
        if dpipe is not None:
            dpipe.update_components(vae=None, audio_vae=None)
        dpipe = video_vae = audio_vae = None
        free_cuda()


def _output_path(paths: list[Path], index: int, seed: int) -> Path:
    if index < len(paths):
        return Path(paths[index])
    base = Path(paths[0])
    return base.with_name(f"{base.stem}_{index}_seed{seed}{base.suffix}")


def _cancel_checker(is_cancelled: Callable[[], bool] | None, task_id: str) -> Callable[[str], None]:
    def check(stage: str) -> None:
        if is_cancelled is not None and is_cancelled():
            raise TaskCancelledError(task_id, stage)

    return check


@contextlib.contextmanager
def _allow_snapped_duration():
    """15 秒对齐到 362 帧后是 15.083 秒，原上界 15.0 会拒绝这次合法请求。"""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    original = MiniMaxH3ModularPipeline.max_duration.fget

    def _max(self):
        return 15.0 + 3.0 / 24.0

    MiniMaxH3ModularPipeline.max_duration = property(_max)
    try:
        yield
    finally:
        MiniMaxH3ModularPipeline.max_duration = property(original)

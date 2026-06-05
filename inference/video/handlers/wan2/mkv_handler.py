"""
Wan2 专用：MKV（视频生成）Handler

按用户确认的规则做分支路由：
1) TI2V: prompt 非空 且 url 有值
2) IVV2V: url + ref_video + face_video
3) VICV: url + ref_video (且 face_video 为空)
4) T2V: prompt 非空 且 url 为空
5) I2V: url 有值 且 ref_video 为空（prompt 允许为空）

注意：当前只搭路由与参数校验框架，不实现具体推理。
"""

from __future__ import annotations

import asyncio
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from common.io_utils import download_url_to_tempfile
from common.config_loader import InferenceConfig
from common.result_handler import ResultHandler
from common.logger import get_logger
from common.Constant import VIDEO_SAVE_QUALITY, VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB
from common.task_cancel import TaskCancelledError
from diffsynth.utils.data import VideoData, save_video
from schemas import InferenceRequestParams

from .types import Wan2MkvMode
from video.runtime.io_utils import load_image_from_url_or_path
from common.pipeline_cache import PipelineCache
from video.runtime.wan2_pipeline_factory import is_oom, cleanup_after_oom
from video.runtime.wan2_pipeline_manager import acquire_wan2_pipe, finish_wan2_pipe_use
from video.runtime.wan2_call_utils import call_pipe, build_wan2_teacache_kwargs
from video.runtime.wan2_lora_manager import build_lora_list, append_trigger_words_to_prompt, apply_wan2_loras, clear_wan2_loras
from video.runtime.gpu_vram import decide_force_offload

logger = get_logger(__name__)


def _non_empty(s: Optional[str]) -> bool:
    return bool((s or "").strip())


def _pad_or_trim(frames: list, target_len: int) -> list:
    if target_len <= 0:
        return []
    if len(frames) >= target_len:
        return frames[:target_len]
    if not frames:
        raise ValueError("video has no frames")
    last = frames[-1]
    return frames + [last] * (target_len - len(frames))


class Wan2MkvHandler:
    def __init__(
        self,
        *,
        inference_config: InferenceConfig,
        result_handler: ResultHandler,
        service_id: str,
        logger=logger,
        run_blocking: Optional[Callable[..., Any]] = None,
        check_cancelled: Optional[Callable[[str], Any]] = None,
        is_task_cancelled: Optional[Callable[[], bool]] = None,
        pipeline_cache: Optional[PipelineCache] = None,
    ):
        self.inference_config = inference_config
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled
        self.is_task_cancelled = is_task_cancelled
        self.pipeline_cache = pipeline_cache

    def _raise_if_cancelled(self, task_id: str, stage: str) -> None:
        if callable(self.is_task_cancelled) and self.is_task_cancelled():
            raise TaskCancelledError(task_id, stage)

    def resolve_mode(self, req: InferenceRequestParams) -> Wan2MkvMode:
        prompt_ok = _non_empty(req.prompt)
        has_img = bool(req.url)
        has_ref_video = bool(req.ref_video)
        has_face_video = bool(req.face_video)

        # 优先级：条件更强的分支先判定，避免冲突
        if prompt_ok and has_img:
            return "ti2v"
        if has_img and has_ref_video and has_face_video:
            return "ivv2v"
        if has_img and has_ref_video and (not has_face_video):
            return "vicv"
        if prompt_ok and (not has_img):
            return "t2v"
        if has_img and (not has_ref_video):
            return "i2v"

        raise ValueError(
            "MKV routing failed: need one of "
            "TI2V(prompt+url), IVV2V(url+ref_video+face_video), VICV(url+ref_video), "
            "T2V(prompt), I2V(url)."
        )

    async def run(self, req: InferenceRequestParams, *, task_id: str) -> None:
        if req.type != "video":
            raise ValueError(f"Wan2MkvHandler expected req.type='video', got '{req.type}'")

        mode = self.resolve_mode(req)
        self.logger.info(f"[Wan2][MKV] task_id={task_id} resolved mode={mode}")

        if self.check_cancelled and await self.check_cancelled(f"before mkv.{mode}"):
            return

        # Wan2 输出固定为 mp4
        req.file_type = "mp4"
        # fps 可通过 model_config.fps 覆盖（用于更顺滑的输出；会同步影响 num_frames 以保持时长）
        model_cfg = getattr(req, "model_cfg", None)
        # 统一 seed 语义：seed=None -> 生成随机 seed，并写回 req.seed（便于日志/结果追溯）
        raw_seed = req.seed
        effective_seed = (
            int(raw_seed) if (raw_seed is not None and int(raw_seed) >= 0) else random.randint(0, 2**32 - 1)
        )
        try:
            req.seed = effective_seed
        except Exception:
            pass
        fps = 24
        try:
            if isinstance(model_cfg, dict) and model_cfg.get("fps") is not None:
                fps = int(model_cfg.get("fps"))
        except Exception:
            fps = 24
        fps = max(1, min(60, int(fps)))
        num_frames = int(req.duration * fps) + 1

        # LoRA（若有）：从 prompt 标签与 req.loras 合并；并将 trigger_word 追加到 prompt
        lora_list = build_lora_list(req.prompt or "", getattr(req, "loras", None))
        prompt = append_trigger_words_to_prompt((req.prompt or "").strip(), lora_list)
        negative_prompt = req.negative_prompt or ""

        # 预加载输入（I/O 走 asyncio，不阻塞推理线程）
        input_image = None
        end_image = None
        control_video_frames = None
        pose_video_frames = None
        face_video_frames = None
        reference_image = None

        if mode in ("i2v", "ti2v", "vicv", "ivv2v"):
            # url 作为参考图/输入图
            input_image = await load_image_from_url_or_path(req.url, size=(req.width, req.height))
            if self.check_cancelled and await self.check_cancelled(f"after mkv.{mode} input image"):
                return
        if mode == "vicv":
            # ref_video 作为 control_video；reference_image 来自 url
            ref_path = await download_url_to_tempfile(req.ref_video, default_suffix=".mp4", timeout_seconds=120.0, max_bytes=500 * 1024 * 1024)
            vd = VideoData(str(ref_path), height=req.height, width=req.width)
            frames = vd.raw_data()
            control_video_frames = _pad_or_trim(frames, num_frames)
            reference_image = input_image
            if self.check_cancelled and await self.check_cancelled("after mkv.vicv control video"):
                return
        if mode == "ivv2v":
            # ref_video 作为 animate_pose_video；face_video 作为 animate_face_video
            pose_path = await download_url_to_tempfile(req.ref_video, default_suffix=".mp4", timeout_seconds=120.0, max_bytes=500 * 1024 * 1024)
            face_path = await download_url_to_tempfile(req.face_video, default_suffix=".mp4", timeout_seconds=120.0, max_bytes=500 * 1024 * 1024)
            pose_vd = VideoData(str(pose_path), height=req.height, width=req.width)
            face_vd = VideoData(str(face_path), height=req.height, width=req.width)
            split_len = max(0, num_frames - 4)
            pose_video_frames = _pad_or_trim(pose_vd.raw_data(), split_len)
            face_video_frames = _pad_or_trim(face_vd.raw_data(), split_len)
            if self.check_cancelled and await self.check_cancelled("after mkv.ivv2v reference videos"):
                return

        # 本项目只认 load_name：本地模型目录名或绝对路径
        model_override = (req.load_name or "").strip() or None
        force_offload = bool(model_cfg.get("force_offload")) if isinstance(model_cfg, dict) else False
        # 低显存策略：推理前探测一次可用 VRAM；若不足则强制进入 low_vram(cpu offload)
        force_offload, _free_vram = decide_force_offload(
            requested=force_offload,
            threshold_gib=VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB,
            device=None,
            logger=self.logger,
            log_prefix=f"[Wan2][MKV] task_id={task_id} ",
        )
        # 为了避免 LoRA fuse 污染缓存权重：只允许在 low_vram(hotload) 模式启用 LoRA
        if lora_list and not force_offload:
            self.logger.info("[Wan2][MKV] loras specified -> enabling low_vram (force_offload) for reversible hotload")
            force_offload = True

        def _infer_and_save(pipe: Any, tmp_mp4: Path, teacache_kwargs: dict) -> None:
            t_pipe_start = time.time()
            try:
                self._raise_if_cancelled(task_id, f"before mkv.{mode} lora load")
                apply_wan2_loras(
                    pipe=pipe, lora_list=lora_list, loras_dir=self.inference_config.loras_dir, logger=self.logger
                )
                self._raise_if_cancelled(task_id, f"after mkv.{mode} lora load")
                if mode == "t2v":
                    video_frames = call_pipe(
                        pipe,
                        task_id=task_id,
                        stage=f"mkv.{mode} denoise",
                        is_task_cancelled=self.is_task_cancelled,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=effective_seed,
                        height=req.height,
                        width=req.width,
                        num_frames=num_frames,
                        num_inference_steps=req.num_inference_steps,
                        # cfg_scale=req.guidance_scale,
                        tiled=True,
                        **teacache_kwargs,
                    )
                elif mode == "i2v":
                    video_frames = call_pipe(
                        pipe,
                        task_id=task_id,
                        stage=f"mkv.{mode} denoise",
                        is_task_cancelled=self.is_task_cancelled,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        input_image=input_image,
                        seed=effective_seed,
                        height=req.height,
                        width=req.width,
                        num_frames=num_frames,
                        num_inference_steps=req.num_inference_steps,
                        # cfg_scale=req.guidance_scale,
                        tiled=True,
                        switch_DiT_boundary=0.9,
                        **teacache_kwargs,
                    )
                elif mode == "ti2v":
                    video_frames = call_pipe(
                        pipe,
                        task_id=task_id,
                        stage=f"mkv.{mode} denoise",
                        is_task_cancelled=self.is_task_cancelled,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        input_image=input_image,
                        seed=effective_seed,
                        height=req.height,
                        width=req.width,
                        num_frames=num_frames,
                        num_inference_steps=req.num_inference_steps,
                        cfg_scale=req.guidance_scale,
                        tiled=True,
                        **teacache_kwargs,
                    )
                elif mode == "vicv":
                    video_frames = call_pipe(
                        pipe,
                        task_id=task_id,
                        stage=f"mkv.{mode} denoise",
                        is_task_cancelled=self.is_task_cancelled,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        control_video=control_video_frames,
                        reference_image=reference_image,
                        seed=effective_seed,
                        height=req.height,
                        width=req.width,
                        num_frames=num_frames,
                        num_inference_steps=req.num_inference_steps,
                        # cfg_scale=req.guidance_scale,
                        tiled=True,
                        **teacache_kwargs,
                    )
                elif mode == "ivv2v":
                    video_frames = call_pipe(
                        pipe,
                        task_id=task_id,
                        stage=f"mkv.{mode} denoise",
                        is_task_cancelled=self.is_task_cancelled,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        input_image=input_image,
                        animate_pose_video=pose_video_frames,
                        animate_face_video=face_video_frames,
                        seed=effective_seed,
                        height=req.height,
                        width=req.width,
                        num_frames=num_frames,
                        num_inference_steps=min(20, req.num_inference_steps),
                        cfg_scale=1.0,  # animate 示例里 cfg_scale=1
                        tiled=True,
                        **teacache_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported MKV mode: {mode}")
            finally:
                # best-effort clear hotloaded loras
                try:
                    clear_wan2_loras(pipe=pipe, logger=self.logger)
                except Exception:
                    pass

            self.logger.info(f"[Wan2][MKV] pipe_ready+infer elapsed={time.time() - t_pipe_start:.2f}s mode={mode}")
            self._raise_if_cancelled(task_id, f"before mkv.{mode} save video")
            save_video(video_frames, str(tmp_mp4), fps=fps, quality=VIDEO_SAVE_QUALITY)
            self._raise_if_cancelled(task_id, f"after mkv.{mode} save video")

        t0 = time.time()
        tmp_mp4 = Path(tempfile.gettempdir()) / f"vitoom_{task_id}_mkv_{mode}.mp4"
        try:
            pipe_name = {
                "t2v": "t2v",
                "i2v": "i2v",
                "ti2v": "ti2v",
                "vicv": "control",
                "ivv2v": "animate",
            }[mode]

            teacache_kwargs = build_wan2_teacache_kwargs(
                req,
                pipe_name=pipe_name,
                height=req.height,
                width=req.width,
                logger=self.logger,
            )

            for attempt in range(2):
                use_low_vram = bool(force_offload) or (attempt == 1)
                pipe: Any = None
                cache_key = ""
                cache_enabled = False
                try:
                    if self.check_cancelled and await self.check_cancelled(f"before mkv.{mode} acquire pipe attempt {attempt + 1}"):
                        return
                    pipe, cache_key, cache_enabled = await acquire_wan2_pipe(
                        name=pipe_name,
                        models_base_dir=self.inference_config.models_dir,
                        weights_base_dir=self.inference_config.weights_dir,
                        model_ref_override=model_override,
                        device=None,
                        torch_dtype="bf16",
                        vram_limit=None,
                        low_vram=use_low_vram,
                        pipeline_cache=self.pipeline_cache,
                        run_blocking=self.run_blocking,
                        log=self.logger,
                    )
                    if self.check_cancelled and await self.check_cancelled(f"after mkv.{mode} acquire pipe attempt {attempt + 1}"):
                        return
                    if self.run_blocking:
                        await self.run_blocking(_infer_and_save, pipe, tmp_mp4, teacache_kwargs)
                    else:
                        await asyncio.to_thread(_infer_and_save, pipe, tmp_mp4, teacache_kwargs)
                    if self.check_cancelled and await self.check_cancelled(f"after mkv.{mode} inference attempt {attempt + 1}"):
                        return
                    break
                except Exception as e:
                    if (attempt == 0) and (not force_offload) and is_oom(e):
                        self.logger.warning(
                            f"[Wan2][MKV] OOM, fallback to low_vram retry once. mode={mode} err={e}"
                        )
                        cleanup_after_oom()
                        continue
                    raise
                finally:
                    try:
                        if pipe is not None:
                            await finish_wan2_pipe_use(
                                pipe=pipe,
                                cache_key=cache_key,
                                cache_enabled=cache_enabled,
                                pipeline_cache=self.pipeline_cache,
                                run_blocking=self.run_blocking,
                                log=self.logger,
                            )
                    except Exception:
                        pass
            if self.check_cancelled and await self.check_cancelled(f"before mkv.{mode} read result"):
                return
            mp4_bytes = tmp_mp4.read_bytes()
        finally:
            try:
                tmp_mp4.unlink(missing_ok=True)
            except Exception:
                pass

        if self.check_cancelled and await self.check_cancelled(f"before mkv.{mode} result upload"):
            return
        await self.result_handler.process_single_result(
            file_data=mp4_bytes,
            request_params=req,
            generate_time=time.time() - t0,
            service_id=self.service_id,
            index=0,
            total=1,
        )


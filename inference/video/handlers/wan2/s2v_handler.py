"""
Wan2 专用：S2V（音频生视频）Handler

约定（用户确认）：
- ref_video 在 S2V 下复用为 pose_video（可选）
- prompt 对 S2V 可非空（规划为必需），url 必需，prompt_wav_path 必需

已实现具体推理；长视频支持分段生成并增量回传（覆盖同一 mp4，progress 递增）。
"""

from __future__ import annotations

import asyncio
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from common.io_utils import download_url_to_tempfile
from common.config_loader import InferenceConfig
from common.result_handler import ResultHandler
from common.logger import get_logger
from common.task_cancel import TaskCancelledError
from diffsynth.utils.data import VideoData, save_video, save_video_with_audio
from diffsynth.pipelines.wan_video import WanVideoUnit_S2V
from schemas import InferenceRequestParams
from common.Constant import VIDEO_SAVE_QUALITY, VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB
from video.runtime.io_utils import load_image_from_url_or_path
from common.pipeline_cache import PipelineCache
from video.runtime.wan2_pipeline_factory import is_oom, cleanup_after_oom
from video.runtime.wan2_pipeline_manager import acquire_wan2_pipe, finish_wan2_pipe_use
from video.runtime.wan2_call_utils import call_pipe, build_wan2_teacache_kwargs
from video.runtime.wan2_lora_manager import build_lora_list, append_trigger_words_to_prompt, apply_wan2_loras, clear_wan2_loras
from video.runtime.gpu_vram import decide_force_offload

logger = get_logger(__name__)


class Wan2S2vHandler:
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

    async def run(self, req: InferenceRequestParams, *, task_id: str) -> None:
        if req.type != "video":
            raise ValueError(f"Wan2S2vHandler expected req.type='video', got '{req.type}'")

        if not (req.url and req.prompt_wav_path):
            raise ValueError("S2V requires req.url (reference image) and req.prompt_wav_path (audio url)")

        if self.check_cancelled and await self.check_cancelled("before s2v"):
            return

        # Wan2 输出固定为 mp4；S2V 默认 fps=24，可通过 model_config.fps 覆盖
        req.file_type = "mp4"
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
        base_num_frames = int(req.duration * fps) + 1
        # S2V 推荐 4n+1
        num_frames = ((max(1, base_num_frames) - 1 + 3) // 4) * 4 + 1

        pose_video_url = req.ref_video  # 复用为 pose_video（可选）
        self.logger.info(
            f"[Wan2][S2V] task_id={task_id} pose_video={'set' if pose_video_url else 'none'} "
            f"duration={req.duration}s fps={fps} num_frames={num_frames}"
        )

        # 下载输入
        input_image = await load_image_from_url_or_path(req.url, size=(req.width, req.height))
        if self.check_cancelled and await self.check_cancelled("after s2v input image"):
            return
        audio_path = await download_url_to_tempfile(
            req.prompt_wav_path,
            default_suffix=".mp3",
            timeout_seconds=300.0,
            max_bytes=200 * 1024 * 1024,
        )
        if self.check_cancelled and await self.check_cancelled("after s2v audio download"):
            return
        pose_frames = None
        if pose_video_url:
            pose_path = await download_url_to_tempfile(
                pose_video_url,
                default_suffix=".mp4",
                timeout_seconds=300.0,
                max_bytes=500 * 1024 * 1024,
            )
            vd = VideoData(str(pose_path), height=req.height, width=req.width)
            frames = vd.raw_data()
            if len(frames) >= num_frames:
                pose_frames = frames[:num_frames]
            elif frames:
                pose_frames = frames + [frames[-1]] * (num_frames - len(frames))
            if self.check_cancelled and await self.check_cancelled("after s2v pose video"):
                return

        # librosa：在主协程中导入/加载（失败就明确报错）
        try:
            import librosa  # type: ignore
        except Exception as e:
            raise RuntimeError("S2V requires librosa to load audio. Please install librosa.") from e

        t0 = time.time()
        model_override = (req.load_name or "").strip() or None
        lora_list = build_lora_list(req.prompt or "", getattr(req, "loras", None))
        prompt = append_trigger_words_to_prompt((req.prompt or "").strip(), lora_list)
        negative_prompt = req.negative_prompt or ""
        input_audio, sample_rate = librosa.load(str(audio_path), sr=16000)
        if self.check_cancelled and await self.check_cancelled("after s2v audio load"):
            return
        force_offload = bool(model_cfg.get("force_offload")) if isinstance(model_cfg, dict) else False
        # 低显存策略：推理前探测一次可用 VRAM；若不足则强制进入 low_vram(cpu offload)
        force_offload, _free_vram = decide_force_offload(
            requested=force_offload,
            threshold_gib=VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB,
            device=None,
            logger=self.logger,
            log_prefix=f"[Wan2][S2V] task_id={task_id} ",
        )
        use_low_vram = bool(force_offload)
        # 为了避免 LoRA fuse 污染缓存权重：只允许在 low_vram(hotload) 模式启用 LoRA
        if lora_list and not use_low_vram:
            self.logger.info("[Wan2][S2V] loras specified -> enabling low_vram (force_offload) for reversible hotload")
            use_low_vram = True

        tmp_mp4 = Path(tempfile.gettempdir()) / f"vitoom_{task_id}_s2v.mp4"
        sent_any = False
        teacache_kwargs = build_wan2_teacache_kwargs(
            req, pipe_name="s2v", height=req.height, width=req.width, logger=self.logger
        )
        try:
            for attempt in range(2):
                pipe: Any = None
                cache_key = ""
                cache_enabled = False
                try:
                    if self.check_cancelled and await self.check_cancelled(f"before s2v acquire pipe attempt {attempt + 1}"):
                        return
                    pipe, cache_key, cache_enabled = await acquire_wan2_pipe(
                        name="s2v",
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
                    if self.check_cancelled and await self.check_cancelled(f"after s2v acquire pipe attempt {attempt + 1}"):
                        return
                    # 在推理线程内 hotload LoRA（避免阻塞 event loop；同时便于与释放路径保持一致）
                    def _load_loras() -> None:
                        self._raise_if_cancelled(task_id, "before s2v lora load")
                        apply_wan2_loras(
                            pipe=pipe, lora_list=lora_list, loras_dir=self.inference_config.loras_dir, logger=self.logger
                        )
                        self._raise_if_cancelled(task_id, "after s2v lora load")

                    if self.run_blocking:
                        await self.run_blocking(_load_loras)
                    else:
                        _load_loras()

                    # 短视频：单次生成，回传一次即可
                    if num_frames <= 81:
                        def _infer_short():
                            self._raise_if_cancelled(task_id, "before s2v short inference")
                            return call_pipe(
                                pipe,
                                task_id=task_id,
                                stage="s2v short denoise",
                                is_task_cancelled=self.is_task_cancelled,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                input_image=input_image,
                                input_audio=input_audio,
                                audio_sample_rate=sample_rate,
                                s2v_pose_video=pose_frames,
                                seed=effective_seed,
                                height=req.height,
                                width=req.width,
                                num_frames=num_frames,
                                num_inference_steps=max(1, min(60, req.num_inference_steps)),
                                # cfg_scale=req.guidance_scale,
                                tiled=True,
                                **teacache_kwargs,
                            )

                        video_frames = await (
                            self.run_blocking(_infer_short) if self.run_blocking else asyncio.to_thread(_infer_short)
                        )
                        if self.check_cancelled and await self.check_cancelled("after s2v short inference"):
                            return
                        frames_to_save = video_frames[1:] if isinstance(video_frames, list) and len(video_frames) > 1 else video_frames
                        self._raise_if_cancelled(task_id, "before s2v short save video")
                        try:
                            await asyncio.to_thread(save_video_with_audio, frames_to_save, str(tmp_mp4), str(audio_path), fps, VIDEO_SAVE_QUALITY)
                        except Exception:
                            await asyncio.to_thread(save_video, frames_to_save, str(tmp_mp4), fps, VIDEO_SAVE_QUALITY)
                        if self.check_cancelled and await self.check_cancelled("after s2v short save video"):
                            return

                        if self.check_cancelled and await self.check_cancelled("before s2v short read result"):
                            return
                        mp4_bytes = tmp_mp4.read_bytes()
                        if self.check_cancelled and await self.check_cancelled("before s2v short result upload"):
                            return
                        await self.result_handler.process_single_result(
                            file_data=mp4_bytes,
                            request_params=req,
                            generate_time=time.time() - t0,
                            service_id=self.service_id,
                            index=0,
                            total=1,
                        )
                        sent_any = True
                        return

                    # 长视频：多段拼接 + 每段回传一次（覆盖同一 mp4）
                    infer_frames = 80  # 4n
                    motion_frames = 73  # 超参（与示例一致）

                    def _precalc():
                        self._raise_if_cancelled(task_id, "before s2v precalc")
                        with torch.no_grad():
                            result = WanVideoUnit_S2V.pre_calculate_audio_pose(
                                pipe=pipe,
                                input_audio=input_audio,
                                audio_sample_rate=sample_rate,
                                s2v_pose_video=pose_frames,
                                num_frames=infer_frames + 1,
                                height=req.height,
                                width=req.width,
                                fps=fps,
                            )
                        self._raise_if_cancelled(task_id, "after s2v precalc")
                        return result

                    audio_embeds, pose_latents, num_repeat = await (
                        self.run_blocking(_precalc) if self.run_blocking else asyncio.to_thread(_precalc)
                    )
                    if self.check_cancelled and await self.check_cancelled("after s2v precalc"):
                        return

                    motion_video = None
                    all_frames = []

                    for r in range(num_repeat):
                        if self.check_cancelled and await self.check_cancelled(f"s2v clip {r+1}/{num_repeat}"):
                            return

                        def _infer_clip(mv, pe, pl):
                            self._raise_if_cancelled(task_id, f"before s2v clip {r+1}/{num_repeat} inference")
                            with torch.no_grad():
                                clip_tensor = call_pipe(
                                    pipe,
                                    task_id=task_id,
                                    stage=f"s2v clip {r+1}/{num_repeat} denoise",
                                    is_task_cancelled=self.is_task_cancelled,
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    input_image=input_image,
                                    audio_embeds=pe,
                                    s2v_pose_latents=pl,
                                    motion_video=mv,
                                    seed=effective_seed,
                                    height=req.height,
                                    width=req.width,
                                    num_frames=infer_frames + 1,
                                    num_inference_steps=max(1, min(60, req.num_inference_steps)),
                                    # cfg_scale=req.guidance_scale,
                                    tiled=True,
                                    output_type="floatpoint",
                                    **teacache_kwargs,
                                )

                                clip_tensor = clip_tensor[:, :, -infer_frames:, :, :]
                                if r == 0:
                                    clip_tensor = clip_tensor[:, :, 3:, :, :]
                                    overlap = min(motion_frames, clip_tensor.shape[2])
                                    mv2 = clip_tensor[:, :, -overlap:, :, :].clone()
                                else:
                                    overlap = min(motion_frames, clip_tensor.shape[2])
                                    mv2 = (
                                        torch.cat((mv[:, :, overlap:, :, :], clip_tensor[:, :, -overlap:, :, :]), dim=2)
                                        if mv is not None
                                        else None
                                    )

                                new_frames = pipe.vae_output_to_video(clip_tensor)
                                return mv2, new_frames

                        pl = pose_latents[r] if pose_latents is not None else None
                        motion_video, new_frames = await (
                            self.run_blocking(_infer_clip, motion_video, audio_embeds[r], pl)
                            if self.run_blocking
                            else asyncio.to_thread(_infer_clip, motion_video, audio_embeds[r], pl)
                        )
                        if self.check_cancelled and await self.check_cancelled(f"after s2v clip {r+1}/{num_repeat} inference"):
                            return
                        all_frames.extend(new_frames)

                        self._raise_if_cancelled(task_id, f"before s2v clip {r+1}/{num_repeat} save video")
                        try:
                            await asyncio.to_thread(save_video_with_audio, all_frames, str(tmp_mp4), str(audio_path), fps, VIDEO_SAVE_QUALITY)
                        except Exception:
                            await asyncio.to_thread(save_video, all_frames, str(tmp_mp4), fps, VIDEO_SAVE_QUALITY)
                        if self.check_cancelled and await self.check_cancelled(f"after s2v clip {r+1}/{num_repeat} save video"):
                            return

                        if self.check_cancelled and await self.check_cancelled(f"before s2v clip {r+1}/{num_repeat} read result"):
                            return
                        mp4_bytes = tmp_mp4.read_bytes()
                        if self.check_cancelled and await self.check_cancelled(f"before s2v clip {r+1}/{num_repeat} result upload"):
                            return
                        await self.result_handler.process_single_result(
                            file_data=mp4_bytes,
                            request_params=req,
                            generate_time=time.time() - t0,
                            service_id=self.service_id,
                            index=r,
                            total=num_repeat,
                            file_name_override=f"{req.task_id}.mp4",
                        )
                        sent_any = True
                    return
                except Exception as e:
                    # 只在“尚未产出任何结果”时允许自动重试，避免重复回传/覆盖造成混乱
                    if (attempt == 0) and (not force_offload) and (not sent_any) and is_oom(e):
                        self.logger.warning(f"[Wan2][S2V] OOM, fallback to low_vram retry once. err={e}")
                        cleanup_after_oom()
                        use_low_vram = True
                        continue
                    raise
                finally:
                    try:
                        if pipe is not None:
                            if self.run_blocking:
                                await self.run_blocking(lambda: clear_wan2_loras(pipe=pipe, logger=self.logger))
                            else:
                                clear_wan2_loras(pipe=pipe, logger=self.logger)
                    except Exception:
                        pass
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
        finally:
            try:
                tmp_mp4.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                # audio 临时文件仅在 URL 下载场景产生；本地路径不删
                if str(req.prompt_wav_path).startswith(("http://", "https://")):
                    audio_path.unlink(missing_ok=True)
            except Exception:
                pass


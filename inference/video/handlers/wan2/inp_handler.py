"""
Wan2 专用：INP（首尾帧补全）Handler

规划：
- 首帧：req.url（必需）
- 尾帧：req.image_file2（可选）
- req.prompt 允许为空

对应示例：Wan2.2-Fun-A14B-InP.py
"""

from __future__ import annotations

import asyncio
import random
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from common.config_loader import InferenceConfig
from common.result_handler import ResultHandler
from common.logger import get_logger
from common.Constant import VIDEO_SAVE_QUALITY, VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB
from common.task_cancel import TaskCancelledError
from diffsynth.utils.data import save_video
from schemas import InferenceRequestParams

from video.runtime.io_utils import load_image_from_url_or_path
from common.pipeline_cache import PipelineCache
from video.runtime.wan2_pipeline_factory import is_oom, cleanup_after_oom
from video.runtime.wan2_pipeline_manager import acquire_wan2_pipe, finish_wan2_pipe_use
from video.runtime.wan2_call_utils import call_pipe, build_wan2_teacache_kwargs
from video.runtime.wan2_lora_manager import build_lora_list, append_trigger_words_to_prompt, apply_wan2_loras, clear_wan2_loras
from video.runtime.gpu_vram import decide_force_offload

logger = get_logger(__name__)


class Wan2InpHandler:
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
            raise ValueError(f"Wan2InpHandler expected req.type='video', got '{req.type}'")
        if not req.url:
            raise ValueError("INP requires req.url (start image)")

        # Wan2 输出固定为 mp4
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
        num_frames = int(req.duration * fps) + 1

        end_image_url = req.image_file2 or None
        self.logger.info(f"[Wan2][INP] task_id={task_id} end_image={'set' if end_image_url else 'none'} num_frames={num_frames}")

        if self.check_cancelled and await self.check_cancelled("before inp"):
            return

        # 加载输入图片
        input_image = await load_image_from_url_or_path(req.url, size=(req.width, req.height))
        end_image = await load_image_from_url_or_path(end_image_url, size=(req.width, req.height)) if end_image_url else None
        if self.check_cancelled and await self.check_cancelled("after inp input load"):
            return
        force_offload = bool(model_cfg.get("force_offload")) if isinstance(model_cfg, dict) else False
        # 低显存策略：推理前探测一次可用 VRAM；若不足则强制进入 low_vram(cpu offload)
        force_offload, _free_vram = decide_force_offload(
            requested=force_offload,
            threshold_gib=VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB,
            device=None,
            logger=self.logger,
            log_prefix=f"[Wan2][INP] task_id={task_id} ",
        )
        lora_list = build_lora_list(req.prompt or "", getattr(req, "loras", None))
        prompt = append_trigger_words_to_prompt((req.prompt or "").strip(), lora_list)
        # 为了避免 LoRA fuse 污染缓存权重：只允许在 low_vram(hotload) 模式启用 LoRA
        if lora_list and not force_offload:
            self.logger.info("[Wan2][INP] loras specified -> enabling low_vram (force_offload) for reversible hotload")
            force_offload = True

        teacache_kwargs = build_wan2_teacache_kwargs(req, pipe_name="inp", height=req.height, width=req.width, logger=self.logger)

        def _infer_and_save(pipe: Any, tmp_mp4: Path) -> None:
            try:
                self._raise_if_cancelled(task_id, "before inp lora load")
                apply_wan2_loras(
                    pipe=pipe, lora_list=lora_list, loras_dir=self.inference_config.loras_dir, logger=self.logger
                )
                self._raise_if_cancelled(task_id, "after inp lora load")
                video_frames = call_pipe(
                    pipe,
                    task_id=task_id,
                    stage="inp denoise",
                    is_task_cancelled=self.is_task_cancelled,
                    prompt=prompt,
                    negative_prompt=(req.negative_prompt or ""),
                    input_image=input_image,
                    end_image=end_image,
                    seed=effective_seed,
                    height=req.height,
                    width=req.width,
                    num_frames=num_frames,
                    num_inference_steps=req.num_inference_steps,
                    # cfg_scale=req.guidance_scale,
                    tiled=True,
                    **teacache_kwargs,
                )
            finally:
                try:
                    clear_wan2_loras(pipe=pipe, logger=self.logger)
                except Exception:
                    pass
            self._raise_if_cancelled(task_id, "before inp save video")
            save_video(video_frames, str(tmp_mp4), fps=fps, quality=VIDEO_SAVE_QUALITY)
            self._raise_if_cancelled(task_id, "after inp save video")

        t0 = time.time()
        tmp_mp4 = Path(tempfile.gettempdir()) / f"vitoom_{task_id}_inp.mp4"
        try:
            model_override = (req.load_name or "").strip() or None
            for attempt in range(2):
                use_low_vram = bool(force_offload) or (attempt == 1)
                pipe: Any = None
                cache_key = ""
                cache_enabled = False
                try:
                    if self.check_cancelled and await self.check_cancelled(f"before inp acquire pipe attempt {attempt + 1}"):
                        return
                    pipe, cache_key, cache_enabled = await acquire_wan2_pipe(
                        name="inp",
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
                    if self.check_cancelled and await self.check_cancelled(f"after inp acquire pipe attempt {attempt + 1}"):
                        return
                    if self.run_blocking:
                        await self.run_blocking(_infer_and_save, pipe, tmp_mp4)
                    else:
                        await asyncio.to_thread(_infer_and_save, pipe, tmp_mp4)
                    if self.check_cancelled and await self.check_cancelled(f"after inp inference attempt {attempt + 1}"):
                        return
                    break
                except Exception as e:
                    if (attempt == 0) and (not force_offload) and is_oom(e):
                        self.logger.warning(f"[Wan2][INP] OOM, fallback to low_vram retry once. err={e}")
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
            if self.check_cancelled and await self.check_cancelled("before inp read result"):
                return
            mp4_bytes = tmp_mp4.read_bytes()
        finally:
            try:
                tmp_mp4.unlink(missing_ok=True)
            except Exception:
                pass

        if self.check_cancelled and await self.check_cancelled("before inp result upload"):
            return
        await self.result_handler.process_single_result(
            file_data=mp4_bytes,
            request_params=req,
            generate_time=time.time() - t0,
            service_id=self.service_id,
            index=0,
            total=1,
        )


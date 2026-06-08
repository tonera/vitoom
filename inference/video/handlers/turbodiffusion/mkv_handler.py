"""
TurboDiffusion MKV handler (T2V / I2V).

Routing:
- T2V: prompt 非空 且 url 为空
- I2V: url 非空（并要求 prompt 非空，TurboDiffusion 当前建议使用长英文 prompt）

Notes:
- 本项目接入只依赖 pip 安装的 TurboDiffusion（`pip install turbodiffusion --no-build-isolation`），不需要把官方源码复制到仓库。
- 推理逻辑不依赖 `serve/`（TUI/界面），只调用核心模块（rcm/ops/SLA + 模型工厂）。
- tokenizer 默认优先走模型目录内的 `google/umt5-xxl/`（完全离线）；width/height 优先使用前端传入值。
- 模型目录规则：所有组件自包含在同一个目录（符合本项目现有约束）
  - VAE: *VAE*.pth
  - Text encoder: *umt5*enc*.pth（或 *umt5*.pth）
  - T2V DiT: 目录内最大的 .pth（排除 VAE/umt5）
  - I2V DiT: 分别匹配包含 "high" / "low" 的 .pth（同样排除 VAE/umt5）
"""

import asyncio
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import hashlib

from common.config_loader import InferenceConfig
from common.result_handler import ResultHandler
from common.logger import get_logger
from common.Constant import VIDEO_SAVE_QUALITY, VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB
from common.task_cancel import TaskCancelledError
from schemas import InferenceRequestParams

from video.runtime.io_utils import download_and_preprocess_image_to_tempfile
from video.runtime.gpu_vram import decide_force_offload
from diffsynth.utils.data import save_video

logger = get_logger(__name__)

from common.pipeline_cache import PipelineCache
from .checkpoints import (
    ensure_pth_from_safetensors,
    infer_quant_linear,
    find_one_in_roots,
    resolve_tokenizer_dir_in_roots,
    select_umt5_ckpt_in_roots,
    select_i2v_high_low,
    select_t2v_dit,
)
from .engine import TurboArgs, TurboModelPaths, generate_i2v, generate_t2v, load_models, build_models_cache_key
from .imports import import_turbodiffusion_core
from .release import release_turbo_models_twice_async, cleanup_runtime_only


def _non_empty(s: Optional[str]) -> bool:
    return bool(str(s or "").strip())


def _align_num_frames_for_wan_vae_encode(num_frames: int, temporal_window: int = 4) -> int:
    """
    Wan2.1 VAE ``encode`` 按 ``temporal_window`` 在时间维分块编码；尾段长度为 2 时会在
    causal 时间卷积上触发 kernel>T。将像素级总帧数向上对齐到满足
    ``(num_frames - 1) % temporal_window == 0``（与 ``WanTokenizer.get_latent_num_frames`` 的 //4 一致）。
    """
    n = max(1, int(num_frames))
    w = max(1, int(temporal_window))
    r = (n - 1) % w
    if r == 0:
        return n
    return n + (w - r)


def _resolve_model_root(models_base_dir: str, model_ref: str) -> Path:
    p = Path(str(model_ref))
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        return p
    if "/" in model_ref or "\\" in model_ref:
        raise ValueError(f"load_name must be a directory name under models_dir (no slashes): {model_ref}")
    base = Path(models_base_dir).resolve()
    root = (base / model_ref).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model dir not found: {root}")
    return root


class TurboDiffusionMkvHandler:
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

    def _resolve_mode(self, req: InferenceRequestParams) -> str:
        if bool(req.url):
            return "i2v"
        return "t2v"

    def _build_job(self, req: InferenceRequestParams, *, mode: str) -> Tuple[TurboArgs, TurboModelPaths, int, int, int]:
        model_cfg = getattr(req, "model_cfg", None) if hasattr(req, "model_cfg") else None
        model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
        force_offload = bool(model_cfg.get("force_offload"))
        # 低显存策略：推理前探测一次可用 VRAM；若不足则强制进入 offload 模式
        force_offload, _free_vram = decide_force_offload(
            requested=force_offload,
            threshold_gib=VIDEO_FORCE_OFFLOAD_FREE_VRAM_GIB,
            device=None,
            logger=self.logger,
            log_prefix=f"[TurboDiffusion][MKV] task_id={getattr(req, 'task_id', '')} mode={mode} ",
        )
        # Offload level controls how aggressive we are when force_offload is enabled.
        # - balanced (default when force_offload=true): use GPU for compute stages, offload between stages.
        # - max: keep umt5 on CPU + VAE decode on CPU.
        offload_level = str(model_cfg.get("turbo_offload_level") or ("balanced" if force_offload else "none")).strip().lower()
        if offload_level not in ("none", "balanced", "max"):
            offload_level = "balanced" if force_offload else "none"

        if not req.load_name:
            raise ValueError("TurboDiffusion requires load_name (local dir name under models_dir, or an absolute path).")

        model_root = _resolve_model_root(self.inference_config.models_dir, str(req.load_name).strip())

        # Shared components fallback order:
        # 1) {models_dir}/{load_name}/...
        # 2) {models_dir}/WanVideo/...
        # 3) {weights_dir}/WanVideo/...
        shared_roots = [
            model_root,
            (Path(self.inference_config.models_dir).expanduser().resolve() / "WanVideo"),
            (Path(self.inference_config.weights_dir).expanduser().resolve() / "WanVideo"),
        ]

        tokenizer_dir = resolve_tokenizer_dir_in_roots(shared_roots)

        # VAE: upstream expects .pth/.pt; packed models may provide .safetensors
        vae_raw = find_one_in_roots(
            shared_roots,
            [
                "Wan2.1_VAE.pth",
                "*VAE*.pth",
                "*vae*.pth",
                "*VAE*.pt",
                "*vae*.pt",
                "vae_fp8.safetensors",
                "*VAE*.safetensors",
                "*vae*.safetensors",
            ],
        )
        vae_path = ensure_pth_from_safetensors(str(vae_raw), prefix="vae")

        # umT5: upstream asserts ".pth" for checkpoint; convert safetensors if needed
        text_encoder_raw = select_umt5_ckpt_in_roots(shared_roots)
        text_encoder_path = ensure_pth_from_safetensors(str(text_encoder_raw), prefix="umt5")

        # Size: prefer frontend-provided width/height. Fallback to (resolution, aspect_ratio) only if missing.
        w = int(req.width or 0)
        h = int(req.height or 0)
        if w <= 0 or h <= 0:
            resolution = str(getattr(req, "resolution", None) or (model_cfg.get("resolution") if model_cfg else "") or "").strip()
            if not resolution:
                resolution = "480p" if mode == "t2v" else "720p"
            aspect_ratio = str(getattr(req, "aspect_ratio", None) or (model_cfg.get("aspect_ratio") if model_cfg else "") or "").strip()
            if not aspect_ratio:
                aspect_ratio = "16:9"
            td = import_turbodiffusion_core()
            w, h = td.VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        if w <= 0 or h <= 0:
            raise ValueError("TurboDiffusion requires positive width/height.")

        # fps/时长逻辑与 Wan2 MKV 对齐：
        # - fps 默认 24，可通过 model_cfg.fps 覆盖（1~60）
        # - raw num_frames = int(duration * fps) + 1；再按 Wan VAE encode 分块对齐（见 _align_num_frames_for_wan_vae_encode）
        fps = 24
        try:
            if isinstance(model_cfg, dict) and model_cfg.get("fps") is not None:
                fps = int(model_cfg.get("fps"))
        except Exception:
            fps = 24
        fps = max(1, min(60, int(fps)))
        raw_num_frames = int(req.duration * fps) + 1
        num_frames = _align_num_frames_for_wan_vae_encode(raw_num_frames)
        if num_frames != raw_num_frames:
            self.logger.info(
                "[TurboDiffusion][MKV] num_frames aligned for Wan VAE temporal chunking: "
                "%s -> %s (duration=%s fps=%s)",
                raw_num_frames,
                num_frames,
                getattr(req, "duration", None),
                fps,
            )

        # TurboDiffusion distilled steps: 1~4
        raw_steps = model_cfg.get("turbo_num_steps", model_cfg.get("num_steps", req.num_inference_steps or 4))
        try:
            num_steps = int(raw_steps)
        except Exception:
            num_steps = 4
        num_steps = max(1, min(4, num_steps))

        # Sigma defaults depend on mode
        sigma_max = model_cfg.get("sigma_max", 80 if mode == "t2v" else 200)
        try:
            sigma_max = float(sigma_max)
        except Exception:
            sigma_max = 80.0 if mode == "t2v" else 200.0

        attention_type = str(model_cfg.get("attention_type", "sagesla")).strip().lower()
        if attention_type not in ("sagesla", "sla", "original"):
            attention_type = "sagesla"
        sla_topk = model_cfg.get("sla_topk", 0.1)
        try:
            sla_topk = float(sla_topk)
        except Exception:
            sla_topk = 0.1

        default_norm = bool(model_cfg.get("default_norm", False))
        # 统一 seed 语义：seed=None -> 生成随机 seed，并写回 req.seed（便于日志/结果追溯）
        raw_seed = req.seed
        seed = int(raw_seed) if (raw_seed is not None and int(raw_seed) >= 0) else random.randint(0, 2**32 - 1)
        try:
            req.seed = seed
        except Exception:
            pass

        if mode == "t2v":
            dit_path = select_t2v_dit(model_root)
            quant_linear = infer_quant_linear(model_cfg=model_cfg, ckpt_paths=[dit_path])
            model_arch = "Wan2.1-14B" if "14b" in str(req.load_name).lower() else "Wan2.1-1.3B"
            args = TurboArgs(
                num_steps=num_steps,
                num_frames=num_frames,
                num_samples=1,
                sigma_max=float(sigma_max),
                seed=seed,
                force_offload=bool(force_offload),
                offload_level=str(offload_level),
                attention_type=attention_type,
                sla_topk=float(sla_topk),
                quant_linear=bool(quant_linear),
                default_norm=bool(default_norm),
            )
            paths = TurboModelPaths(
                mode="t2v",
                model_root=model_root,
                vae_path=str(vae_path),
                text_encoder_path=str(text_encoder_path),
                tokenizer_dir=tokenizer_dir,
                dit_path=dit_path,
                model_arch=model_arch,
            )
            return args, paths, fps, w, h

        # i2v
        # Defensive validation: I2V requires two DiT checkpoints (high/low). If user accidentally sends `url`
        # with a T2V-only model, fail fast with a clear message.
        try:
            mn = str(req.load_name or "").strip().lower()
        except Exception:
            mn = ""
        try:
            ckpts = sorted([p.name for p in Path(model_root).glob("*.pth")])
        except Exception:
            ckpts = []
        has_high = any("high" in n.lower() for n in ckpts)
        has_low = any("low" in n.lower() for n in ckpts)
        if ("t2v" in mn) and not (has_high and has_low):
            raise ValueError(
                "请改用 I2V 模型"
            )
        high_path, low_path = select_i2v_high_low(model_root)
        quant_linear = infer_quant_linear(model_cfg=model_cfg, ckpt_paths=[high_path, low_path])
        boundary = model_cfg.get("boundary", 0.9)
        try:
            boundary = float(boundary)
        except Exception:
            boundary = 0.9
        adaptive_resolution = bool(model_cfg.get("adaptive_resolution", False))
        ode = bool(model_cfg.get("ode", False))
        args = TurboArgs(
            num_steps=num_steps,
            num_frames=num_frames,
            num_samples=1,
            sigma_max=float(sigma_max),
            seed=seed,
            force_offload=bool(force_offload),
            offload_level=str(offload_level),
            attention_type=attention_type,
            sla_topk=float(sla_topk),
            quant_linear=bool(quant_linear),
            default_norm=bool(default_norm),
            boundary=float(boundary),
            adaptive_resolution=bool(adaptive_resolution),
            ode=bool(ode),
        )
        paths = TurboModelPaths(
            mode="i2v",
            model_root=model_root,
            vae_path=str(vae_path),
            text_encoder_path=str(text_encoder_path),
            tokenizer_dir=tokenizer_dir,
            high_noise_model_path=high_path,
            low_noise_model_path=low_path,
        )
        return args, paths, fps, w, h

    async def run(self, req: InferenceRequestParams, *, task_id: str) -> None:
        if req.type != "video":
            raise ValueError(f"TurboDiffusionMkvHandler expected req.type='video', got '{req.type}'")

        mode = self._resolve_mode(req)
        self.logger.info(f"[TurboDiffusion][MKV] task_id={task_id} resolved mode={mode}")

        if mode == "t2v" and not _non_empty(req.prompt):
            raise ValueError("TurboDiffusion T2V requires non-empty prompt.")
        if mode == "i2v" and not req.url:
            raise ValueError("TurboDiffusion I2V requires url.")
        if mode == "i2v" and not _non_empty(req.prompt):
            raise ValueError("TurboDiffusion I2V requires non-empty prompt.")

        if self.check_cancelled and await self.check_cancelled(f"before mkv.{mode}"):
            return

        # TurboDiffusion 输出固定为 mp4
        req.file_type = "mp4"

        args, paths, fps, width, height = self._build_job(req, mode=mode)

        tmp_mp4 = Path(tempfile.gettempdir()) / f"vitoom_{task_id}_turbo_{mode}.mp4"

        # prepare image path if needed
        tmp_img: Optional[Path] = None
        if mode == "i2v":
            # 统一参考图预处理：EXIF 纠正 + center-crop 到目标比例 + resize 到模型尺寸（避免拉伸变形）
            tmp_img = await download_and_preprocess_image_to_tempfile(
                req.url,
                size=(int(width), int(height)),
                timeout_seconds=120.0,
            )
            if self.check_cancelled and await self.check_cancelled(f"after turbo mkv.{mode} input preprocess"):
                return

        def _infer_and_save() -> None:
            self._raise_if_cancelled(task_id, f"before turbo mkv.{mode} inference")
            models = models_holder
            def _cancel_checker(stage: str) -> None:
                self._raise_if_cancelled(task_id, stage)
            if mode == "t2v":
                frames = generate_t2v(
                    models=models,
                    args=args,
                    prompt=str(req.prompt),
                    width=int(width),
                    height=int(height),
                    logger=self.logger,
                    cancel_checker=_cancel_checker,
                )
            else:
                assert tmp_img is not None
                frames = generate_i2v(
                    models=models,
                    args=args,
                    prompt=str(req.prompt),
                    init_image_path=str(tmp_img),
                    width=int(width),
                    height=int(height),
                    logger=self.logger,
                    cancel_checker=_cancel_checker,
                )

            self._raise_if_cancelled(task_id, f"after turbo mkv.{mode} inference")
            frames_np = frames.numpy()
            self._raise_if_cancelled(task_id, f"before turbo mkv.{mode} save video")
            save_video(list(frames_np), str(tmp_mp4), fps=int(fps), quality=VIDEO_SAVE_QUALITY)
            self._raise_if_cancelled(task_id, f"after turbo mkv.{mode} save video")

            # TurboDiffusion pipeline currently saves with fps=16; keep for now but log for traceability
            self.logger.info(f"[TurboDiffusion][{mode}] saved mp4={tmp_mp4} fps={fps}")

        t0 = time.time()
        models_holder: Any = None
        cache_key = ""
        cache_enabled = False
        try:
            # ===== acquire models via PipelineCache(LRU=1+TTL) =====
            # build a stable key and shorten it (sha256) for cache/log readability
            raw_key = build_models_cache_key(paths=paths, args=args)
            cache_key = "turbo:" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            cache_enabled = bool(self.pipeline_cache is not None and self.pipeline_cache.enabled())

            def _create_models():
                return load_models(paths=paths, args=args, logger=self.logger)

            async def _create_models_async():
                if self.run_blocking:
                    return await self.run_blocking(_create_models)
                return await asyncio.to_thread(_create_models)

            if cache_enabled and self.pipeline_cache is not None:
                models_holder, hit = await self.pipeline_cache.acquire(key=cache_key, create_fn=_create_models_async)
                try:
                    self.logger.info(f"[turbodiffusion][pipeline-cache] {'HIT' if hit else 'MISS'} key={cache_key[:12]}")
                except Exception:
                    pass
            else:
                models_holder = await _create_models_async()
            if self.check_cancelled and await self.check_cancelled(f"after turbo mkv.{mode} model load"):
                return

            # ===== run inference =====
            if self.run_blocking:
                await self.run_blocking(_infer_and_save)
            else:
                await asyncio.to_thread(_infer_and_save)
            if self.check_cancelled and await self.check_cancelled(f"after turbo mkv.{mode} inference"):
                return
            if self.check_cancelled and await self.check_cancelled(f"before turbo mkv.{mode} read result"):
                return
            mp4_bytes = tmp_mp4.read_bytes()
        finally:
            try:
                tmp_mp4.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                if tmp_img is not None:
                    tmp_img.unlink(missing_ok=True)
            except Exception:
                pass
            # ===== release/return cache use =====
            try:
                if cache_enabled and self.pipeline_cache is not None and cache_key:
                    try:
                        # cached mode: light cleanup to drop peak reserved
                        if self.run_blocking:
                            await self.run_blocking(lambda: cleanup_runtime_only(log=self.logger))
                        else:
                            cleanup_runtime_only(log=self.logger)
                    except Exception:
                        pass
                    await self.pipeline_cache.release_use(key=cache_key)
                else:
                    # non-cached mode: full release
                    await release_turbo_models_twice_async(
                        models_holder,
                        log=self.logger,
                        run_blocking=self.run_blocking,
                        aggressive_cpu=False,
                    )
            except Exception:
                pass

        if self.check_cancelled and await self.check_cancelled(f"before turbo mkv.{mode} result upload"):
            return
        await self.result_handler.process_single_result(
            file_data=mp4_bytes,
            request_params=req,
            generate_time=time.time() - t0,
            service_id=self.service_id,
            index=0,
            total=1,
        )


"""
job_type = ID 的 handler（PuLID identity）

参考用户给的示例：
- NunchakuFluxTransformer2dModel + PuLIDFluxPipeline
- transformer.forward 绑定 pulid_forward
- pipeline(prompt, id_image=..., id_weight=..., num_inference_steps=..., guidance_scale=...)

说明：
- 为避免每次任务都重新加载超大权重，这里复用推理器提供的全局 PipelineCache（LRU=1 + TTL）。
"""

from __future__ import annotations

import time
from types import MethodType
from typing import Any, Callable, Optional
from common.logger import print_info

import numpy as np
import torch

from common.Constant import MODEL_SDXL
from common.image_utils import load_image
from image.runtime.postprocess_pipeline import apply_postprocess
from image.runtime.pipeline_release import finish_pipeline_use, resolve_release_targets
from image.runtime.lora_manager import (
    load_loras_into_pipe,
)
from image.runtime.pulid_flux_pipeline_with_paths import PuLIDFluxPipelineWithPaths
from common.pipeline_cache import PipelineCache
from image.runtime.pipeline_lifecycle import PipelineLifecycle
from schemas import InferenceRequestParams


class IdHandler:
    def __init__(
        self,
        *,
        inference_config: Any,
        device_planner: Any,
        seed_manager: Any,
        result_handler: Any,
        service_id: str,
        logger: Any,
        run_blocking: Callable[[Callable[[], Any]], Any],
        check_cancelled: Optional[Callable[[str], Any]] = None,
        pipeline_cache: Optional[PipelineCache] = None,
        lifecycle: Optional[Any] = None,
    ):
        self.inference_config = inference_config
        self.device_planner = device_planner
        self.seed_manager = seed_manager
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled
        self.pipeline_cache = pipeline_cache
        self.lifecycle = lifecycle

    def _resolve_pulid_assets(self, *, family: str, load_name: Optional[str]) -> dict[str, str]:
        """
        统一收敛 PuLID 相关路径变量，避免散落在多处 hardcode。
        """
        weights_dir = str(getattr(self.inference_config, "weights_dir", "resources/weights"))
        models_dir = str(getattr(self.inference_config, "models_dir", "resources/models"))
        roop_dir = f"{models_dir}/roop"
        base_model = f"{models_dir}/{load_name}" if load_name else ""
        return {
            "weights_dir": weights_dir,
            "models_dir": models_dir,
            "roop_dir": roop_dir,
            "base_model": base_model,
            "pulid_flux_path": f"{weights_dir}/PuLID/pulid_flux_v0.9.1.safetensors",
            "pulid_sdxl_path": f"{weights_dir}/PuLID/pulid_v1.1.safetensors",
            "eva_clip_path": f"{weights_dir}/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt",
        }

    def _build_cache_key(
        self,
        *,
        device: str,
        torch_dtype: torch.dtype,
        family: str,
        load_name: Optional[str],
        low_vram: bool,
    ) -> str:
        assets = self._resolve_pulid_assets(family=family, load_name=load_name)
        weights_dir = assets["weights_dir"]
        models_dir = assets["models_dir"]

        mv = (family or "").lower()
        if mv in {m.lower() for m in MODEL_SDXL}:
            if not load_name:
                # run() 里会强制校验，这里只是兜底
                load_name = ""
            base_model = assets["base_model"]
            pulid_path = assets["pulid_sdxl_path"]
            eva_clip_path = assets["eva_clip_path"]
            antelope_root = assets["roop_dir"]
            return (
                f"sdxl:{device}:{str(torch_dtype)}:low_vram={int(bool(low_vram))}:{weights_dir}:{models_dir}:"
                f"base={base_model}:pulid={pulid_path}:eva={eva_clip_path}:ante_root={antelope_root}"
            )

        # 默认：FLUX.1-dev + nunchaku-flux.1-dev 量化 transformer
        from nunchaku.utils import get_precision  # type: ignore

        facexlib_dirpath = assets["roop_dir"]
        insightface_dirpath = assets["roop_dir"]
        pulid_path = assets["pulid_flux_path"]
        eva_clip_path = assets["eva_clip_path"]

        precision = get_precision()
        return (
            f"flux:{device}:{str(torch_dtype)}:low_vram={int(bool(low_vram))}:{weights_dir}:{models_dir}:{precision}:"
            f"facexlib={facexlib_dirpath}:insightface={insightface_dirpath}:"
            f"pulid={pulid_path}:eva={eva_clip_path}"
        )

    def _move_to_device(self, pipe: Any, *, device: str, params: InferenceRequestParams) -> Any:
        if device == "cpu":
            return pipe
        if bool(getattr(params, "low_vram", False)) and self.lifecycle is not None:
            if self.lifecycle.enable_cpu_offload(pipe, params):
                return pipe
            inner = getattr(pipe, "pipe", None)
            if inner is not None and self.lifecycle.enable_cpu_offload(inner, params):
                return pipe
        if hasattr(pipe, "to"):
            return pipe.to(device)
        return pipe

    async def _acquire_pipeline(
        self,
        *,
        device: str,
        torch_dtype: torch.dtype,
        params: InferenceRequestParams,
        family: str,
        load_name: Optional[str],
    ) -> tuple[Any, str, bool]:
        """
        通过全局 PipelineCache 获取 pipeline（LRU=1 + TTL）。
        返回：(pipe, cache_key, cache_enabled)
        """
        cache_enabled = bool(self.pipeline_cache is not None and self.pipeline_cache.enabled())
        key = self._build_cache_key(
            device=device,
            torch_dtype=torch_dtype,
            family=family,
            load_name=load_name,
            low_vram=bool(getattr(params, "low_vram", False)),
        )

        def _create_and_place():
            assets = self._resolve_pulid_assets(family=family, load_name=load_name)
            weights_dir = assets["weights_dir"]
            models_dir = assets["models_dir"]
            mv = (family or "").lower()

            if mv in {m.lower() for m in MODEL_SDXL}:
                # SDXL: use project-local PuLID pipeline v1.1
                from image.controlnet.pulid.pipeline_v1_1 import PuLIDPipeline as PuLIDPipelineSDXL

                if not load_name:
                    raise ValueError("ID(SDXL) requires load_name")
                base_model = assets["base_model"]
                sampler = "dpmpp_sde" if "lightning" in load_name.lower() else "dpmpp_2m"

                pulid_path = assets["pulid_sdxl_path"]
                eva_clip_path = assets["eva_clip_path"]
                antelope_root = assets["roop_dir"]

                self.logger.info(
                    f"ID(SDXL): loading PuLIDPipeline v1.1 base_model={base_model} sampler={sampler}"
                )
                # v-prediction：在“有 params 的地方”判定，然后透传给 PuLIDPipeline，避免 pipeline 内部读取权重文件。
                scheduler_args = None
                try:
                    from image.runtime.scheduler_loader import _is_v_prediction_model  # type: ignore

                    if _is_v_prediction_model(params):
                        scheduler_args = {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}
                except Exception:
                    scheduler_args = None
                pipe = PuLIDPipelineSDXL(
                    sdxl_repo=base_model,
                    sampler=sampler,
                    models_dir=models_dir,
                    weights_dir=weights_dir,
                    pulid_path=pulid_path,
                    eva_clip_path=eva_clip_path,
                    antelope_root=antelope_root,
                    allow_download=False,
                    facexlib_dirpath=assets["roop_dir"],
                    scheduler_args=scheduler_args,
                    # insightface/onnxruntime：默认走 CPU，避免 CUDA provider 常驻显存导致“释放不掉”的观感
                    onnx_provider=(
                        str(getattr(params, "model_cfg", {}).get("onnx_provider", "cpu")).strip().lower()
                        if isinstance(getattr(params, "model_cfg", None), dict)
                        else "cpu"
                    ),
                    device=device,
                    torch_dtype=torch_dtype,
                )
                pipe = self._move_to_device(pipe, device=device, params=params)
                return pipe

            # FLUX: nunchaku PuLIDFluxPipeline
            from nunchaku.utils import get_precision  # type: ignore
            from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel  # type: ignore
            from nunchaku.models.pulid.pulid_forward import pulid_forward  # type: ignore

            facexlib_dirpath = assets["roop_dir"]
            insightface_dirpath = assets["roop_dir"]
            pulid_path = assets["pulid_flux_path"]
            eva_clip_path = assets["eva_clip_path"]

            precision = get_precision()
            transformer_path = f"{weights_dir}/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
            model_path = f"{models_dir}/FLUX.1-dev"

            self.logger.info(f"ID(FLUX): loading PuLIDFluxPipeline model_path={model_path} transformer={transformer_path}")
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_path)

            extra_kwargs: dict[str, Any] = {}
            extra_kwargs["facexlib_dirpath"] = facexlib_dirpath
            extra_kwargs["insightface_dirpath"] = insightface_dirpath
            extra_kwargs["pulid_path"] = pulid_path
            extra_kwargs["eva_clip_path"] = eva_clip_path
            # 默认强制走 CPU onnx provider，避免 ORT CUDA provider 常驻显存导致“释放不掉”的观感
            #（如需 GPU provider，可通过 params.model_cfg.onnx_provider 显式指定）
            extra_kwargs["onnx_provider"] = (
                str(getattr(params, "model_cfg", {}).get("onnx_provider", "cpu")).strip().lower()
                if isinstance(getattr(params, "model_cfg", None), dict)
                else "cpu"
            )

            print_info({"model_path": model_path, "transformer": transformer, "torch_dtype": torch_dtype, **extra_kwargs}, "管道参数:")
            pipe = PuLIDFluxPipelineWithPaths.from_pretrained(
                model_path, transformer=transformer, torch_dtype=torch_dtype, **extra_kwargs
            )
            pipe.transformer.forward = MethodType(pulid_forward, pipe.transformer)

            pipe = self._move_to_device(pipe, device=device, params=params)
            return pipe

        async def _create_new_pipe_async():
            return await self.run_blocking(_create_and_place)

        if cache_enabled and self.pipeline_cache is not None:
            pipe, hit = await self.pipeline_cache.acquire(key=key, create_fn=_create_new_pipe_async)
            self.logger.info(f"[pipeline-cache] ID {'HIT' if hit else 'MISS'} key={key[:12]}")
            return pipe, key, True

        pipe = await _create_new_pipe_async()
        return pipe, key, False

    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        if self.lifecycle is None:
            raise RuntimeError("IdHandler requires lifecycle")
        if not params.url:
            raise ValueError("ID requires url as id_image")
        if self.check_cancelled is not None:
            if await self.check_cancelled("before ID inference"):
                return

        mv = (getattr(params, "family", "") or "").lower()
        load_name = str(getattr(params, "load_name", "") or "").strip()
        if mv in {m.lower() for m in MODEL_SDXL}:
            import os

            if not load_name:
                raise ValueError("ID(SDXL) requires load_name")
            # 强制约定：base 模型来自 models_dir/{load_name}
            base_model_dir = f"{self.inference_config.models_dir}/{load_name}"
            if not os.path.exists(base_model_dir):
                raise ValueError(f"ID(SDXL) base model path not found: {base_model_dir}")

        plan = self.device_planner.plan(params)
        device = plan.device
        torch_dtype = plan.torch_dtype

        # 预处理（prompt/loras/trigger/sanitize）已在推理入口统一完成：
        # - params.prompt 为最终推理 prompt（已拼 trigger 且 sanitize）
        # - params.parsed_loras 为 sanitize 前解析出的 LoRA 列表
        prompt = params.prompt or ""
        parsed_loras = getattr(params, "parsed_loras", None)
        lora_list = list(parsed_loras) if isinstance(parsed_loras, list) else []
        id_image = load_image(params.url)
        if id_image is None:
            raise ValueError("ID id_image (url) is not loadable")

        total = int(getattr(params, "generate_num", 1) or 1)
        total = max(1, total)
        width = getattr(params, "width", 1024) or 1024
        height = getattr(params, "height", 1024) or 1024

        # FLUX: id_weight；SDXL: id_scale（约定从 params.strength 读取；默认 0.8）
        id_weight = 1.0
        id_scale = float(getattr(params, "strength", 0.8) or 0.8)

        pipe: Any = None
        failed = False
        oom_error = False
        try:
            pipe, cache_key, cache_enabled = await self._acquire_pipeline(
                device=device,
                torch_dtype=torch_dtype,
                params=params,
                family=mv,
                load_name=load_name if load_name else None,
            )
            if self.lifecycle is not None:
                self.lifecycle.set_cache_state(cache_key=cache_key, cache_enabled=cache_enabled)
            if lora_list:
                self.logger.info(
                    "ID LoRA load params: loras_dir=%s loras=%s",
                    getattr(self.inference_config, "loras_dir", None),
                    lora_list,
                )

                def _load_loras():
                    lora_target = getattr(pipe, "pipe", pipe)
                    load_loras_into_pipe(
                        lora_target,
                        getattr(params, "family", ""),
                        self.inference_config.loras_dir,
                        lora_list,
                        logger=self.logger,
                    )

                await self.run_blocking(_load_loras)
                try:
                    if hasattr(pipe, "get_list_adapters"):
                        active = pipe.get_list_adapters()
                        self.logger.info("ID LoRA active adapters: %s", active)
                    else:
                        self.logger.info("ID LoRA load done; no adapter list API on pipeline")
                except Exception:
                    self.logger.debug("ID LoRA active adapter query failed (ignored)", exc_info=True)

            for index in range(total):
                if self.check_cancelled is not None:
                    if await self.check_cancelled(f"before ID iteration {index + 1}"):
                        return
                seed = (
                    self.seed_manager.iteration_seed()
                    if total > 1
                    else self.seed_manager.initial_seed(params.seed)
                )
                generator = self.seed_manager.create_generator(device, seed)

                def _run():
                    with torch.inference_mode():
                        if mv in {m.lower() for m in MODEL_SDXL}:
                            id_np = np.array(id_image.convert("RGB"))
                            print_info(
                                {
                                    "prompt": prompt,
                                    "id_image": params.url,
                                    "id_scale": id_scale,
                                    "num_inference_steps": int(getattr(params, "num_inference_steps", 30) or 30),
                                    # guidance_scale 允许为 0；仅当缺失/为 None 时才回落默认值
                                    "guidance_scale": float(
                                        7.5
                                        if getattr(params, "guidance_scale", None) is None
                                        else getattr(params, "guidance_scale")
                                    ),
                                    "seed": seed,
                                    "width": width,
                                    "height": height,
                                    "load_name": load_name,
                                },
                                "推理参数(SDXL):",
                            )
                            uncond_id_embedding, id_embedding = pipe.get_id_embedding([id_np])
                            images = pipe.inference(
                                prompt,
                                (1, int(height), int(width)),
                                str(getattr(params, "negative_prompt", "") or ""),
                                id_embedding,
                                uncond_id_embedding,
                                float(id_scale),
                                float(
                                    7.5
                                    if getattr(params, "guidance_scale", None) is None
                                    else getattr(params, "guidance_scale")
                                ),
                                int(getattr(params, "num_inference_steps", 30) or 30),
                                int(seed),
                            )
                            return images

                        print_info(
                            {
                                "prompt": prompt,
                                "id_image": id_image,
                                "id_weight": id_weight,
                                "num_inference_steps": int(getattr(params, "num_inference_steps", 30) or 30),
                                # guidance_scale 允许为 0；仅当缺失/为 None 时才回落默认值
                                "guidance_scale": float(
                                    3.5
                                    if getattr(params, "guidance_scale", None) is None
                                    else getattr(params, "guidance_scale")
                                ),
                                "seed": seed,
                                "width": width,
                                "height": height,
                            },
                            "推理参数(FLUX):",
                        )
                        return pipe(
                            prompt,
                            id_image=id_image,
                            id_weight=float(id_weight),
                            num_inference_steps=int(getattr(params, "num_inference_steps", 30) or 30),
                            guidance_scale=float(
                                3.5
                                if getattr(params, "guidance_scale", None) is None
                                else getattr(params, "guidance_scale")
                            ),
                            generator=generator,
                            width=width,
                            height=height,
                        )

                start = time.time()
                result = await self.run_blocking(_run)
                used = time.time() - start

                if mv in {m.lower() for m in MODEL_SDXL}:
                    if not isinstance(result, list) or not result:
                        raise ValueError("ID(SDXL) inference returned no images")
                    generated_image = result[0]
                else:
                    if not hasattr(result, "images") or not result.images:
                        raise ValueError("ID(FLUX) inference returned no images")
                    generated_image = result.images[0]
                generated_image = apply_postprocess(generated_image, params)
                await self.result_handler.process_single_result(
                    file_data=generated_image,
                    request_params=params,
                    generate_time=used,
                    service_id=self.service_id,
                    file_seed=seed,
                    index=index,
                    total=total,
                )
                # 及时断开对结果对象的引用（某些 pipeline output 可能携带 tensor）
                result = None
                generated_image = None
        except Exception as e:
            failed = True
            oom_error = PipelineLifecycle.is_oom_error(e)
            if oom_error:
                self.logger.warning("OOM detected during ID inference; forcing cached pipeline eviction")
                try:
                    e.__traceback__ = None
                except Exception:
                    pass
            raise
        finally:
            cache_key, cache_enabled = self.lifecycle.get_cache_state()
            unload_pipe, release_pipe, extra_release_targets = resolve_release_targets(
                pipe,
                family=getattr(params, "family", ""),
            )
            try:
                await finish_pipeline_use(
                    pipe=release_pipe,
                    params=params,
                    inference_params=None,
                    logger=self.logger,
                    failed=failed,
                    oom=oom_error,
                    pipeline_cache=self.lifecycle.pipeline_cache if cache_enabled else None,
                    cache_key=cache_key,
                    run_blocking=self.run_blocking,
                    unload_pipe=unload_pipe,
                    extra_release_targets=extra_release_targets,
                )
            finally:
                pipe = release_pipe = unload_pipe = None
                self.lifecycle.clear_cache_state()



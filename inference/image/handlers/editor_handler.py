"""
图片编辑任务 handler（JT_ED / JT_POSE）。

设计约束：
- 编辑输入统一来自 tpl_list
- flux_kontext / FLUX.1-Depth-dev / FLUX.1-Canny-dev 只使用第一张图
- flux2_klein / qwen.edit 支持多图编辑（上游预处理已限制 <= 9）
- 运行期清理/后处理/结果回传与 DiffusionHandler 保持一致
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from common.image_utils import calc_fit_size_with_multiple, constrain_size
from common.pipeline_component_injector import build_component_overrides
from common.task_cancel import TaskCancelledError
from image.runtime.controlnet_image_builder import ControlnetImageBuildError, build_controlnet_image
from image.runtime.lora_manager import build_lora_list, load_loras_into_pipe
from image.runtime.postprocess_pipeline import apply_postprocess
from image.runtime.pipeline_release import finish_pipeline_use
from image.runtime.scheduler_loader import load_scheduler_from_pipe
from image.runtime.vae_dtype_fixer import ensure_vae_dtype
from schemas import InferenceRequestParams


SPECIAL_FLUX_CONTROL_MODELS = {
    "flux.1-depth-dev": "depth",
    "flux.1-canny-dev": "canny",
}


@dataclass
class IterationSpec:
    index: int
    seed: int
    generator: Any
    fail_on_empty: bool = False


class EditorHandler:
    def __init__(
        self,
        *,
        inference_config: Any,
        lifecycle: Any,
        seed_manager: Any,
        result_handler: Any,
        service_id: str,
        logger: Any,
        run_blocking: Callable[[Callable[[], Any]], Any],
        check_cancelled: Callable[[str], Any],
        is_task_cancelled: Callable[[], bool],
    ):
        self.inference_config = inference_config
        self.lifecycle = lifecycle
        self.seed_manager = seed_manager
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled
        self.is_task_cancelled = is_task_cancelled

    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        pipe = None
        inference_params: Optional[dict] = None
        failed = False
        oom_error = False
        try:
            pipe, inference_params, device = await self._prepare_runtime(params)
            specs = self._build_iteration_specs(params, device)
            await self._drive_iterations(
                pipe=pipe,
                base_params=inference_params or {},
                params=params,
                specs=specs,
            )
        except TaskCancelledError:
            raise
        except Exception as e:
            failed = True
            oom_error = bool(self.lifecycle.is_oom_error(e))
            if oom_error:
                self.logger.warning("OOM detected during editor inference; forcing cached pipeline eviction")
                try:
                    e.__traceback__ = None
                except Exception:
                    pass
            raise
        finally:
            cache_key, cache_enabled = self.lifecycle.get_cache_state()
            try:
                await finish_pipeline_use(
                    pipe=pipe,
                    params=params,
                    inference_params=inference_params,
                    logger=self.logger,
                    failed=failed,
                    oom=oom_error,
                    pipeline_cache=self.lifecycle.pipeline_cache if cache_enabled else None,
                    cache_key=cache_key,
                    run_blocking=self.run_blocking,
                )
            finally:
                pipe = None
                inference_params = None
                self.lifecycle.clear_cache_state()

    async def _prepare_runtime(self, params: InferenceRequestParams) -> tuple[Any, dict, str]:
        special_mode = self._special_flux_control_mode(params)
        if special_mode:
            return await self._prepare_flux_control_runtime(params, special_mode=special_mode)

        pipe, inference_params, device_plan = await self.lifecycle.create_pipeline(params)
        pipe = load_scheduler_from_pipe(pipe, params)
        self.lifecycle.apply_fast_mode_cache(pipe, params)
        self.lifecycle.pretouch_cpu_tensors(
            pipe,
            device_plan,
            getattr(params, "family", ""),
            low_vram=bool(getattr(params, "low_vram", False)),
        )
        pipe, device = self.lifecycle.move_to_device(pipe, device_plan, params)
        self.logger.info(f"Using device: {device} for editor inference")
        return pipe, inference_params or {}, device

    async def _prepare_flux_control_runtime(self, params: InferenceRequestParams, *, special_mode: str) -> tuple[Any, dict, str]:
        device_plan = self.lifecycle.device_planner.plan(params)
        device = device_plan.device
        torch_dtype = device_plan.torch_dtype

        def _create_pipe():
            from diffusers import FluxControlPipeline  # type: ignore

            model_path = self._resolve_model_path(str(getattr(params, "load_name", "") or ""))
            model_info = self._build_model_info(model_path, load_name=str(getattr(params, "load_name", "") or model_path))
            pipe_kwargs = build_component_overrides(
                family="flux",
                model_config=getattr(params, "model_cfg", None),
                inference_config=self.inference_config,
                device_plan=device_plan,
                model_info=model_info,
                logger=self.logger,
                num_inference_steps=int(getattr(params, "num_inference_steps", 0) or 0),
                fast_mode=bool(getattr(params, "fast_mode", False)),
                low_vram=bool(getattr(params, "low_vram", False)),
            )
            pipe_kwargs.setdefault("torch_dtype", torch_dtype)
            pipe_kwargs.setdefault("local_files_only", True)
            return self.lifecycle.pipeline_service.instantiate_pipeline(
                pipeline_cls=FluxControlPipeline,
                model_info=model_info,
                pipeline_params=pipe_kwargs,
            )

        pipe = await self.run_blocking(_create_pipe)
        pipe = load_scheduler_from_pipe(pipe, params)
        await self._load_loras_into_pipe(pipe, params)
        self.lifecycle.apply_fast_mode_cache(pipe, params)
        self.lifecycle.pretouch_cpu_tensors(
            pipe,
            device_plan,
            getattr(params, "family", ""),
            low_vram=bool(getattr(params, "low_vram", False)),
        )

        if bool(getattr(params, "low_vram", False)):
            self.lifecycle.enable_cpu_offload(pipe, params)
        else:
            pipe = pipe.to(device)

        tpl_list = list(getattr(params, "tpl_list", None) or [])
        source_url = str(tpl_list[0] or "")
        try:
            control_image = await self.run_blocking(
                lambda: build_controlnet_image(
                    special_mode,
                    source_url,
                    weights_dir=self.inference_config.weights_dir,
                    logger=self.logger,
                )
            )
        except ControlnetImageBuildError as e:
            raise ValueError(str(e)) from e

        width, height = await self._resolve_flux_control_size(source_url)
        inference_params = {
            "prompt": params.prompt,
            "num_inference_steps": int(getattr(params, "num_inference_steps", 30) or 30),
            "guidance_scale": float(getattr(params, "guidance_scale", 7.5)),
            "width": int(width),
            "height": int(height),
            "control_image": control_image,
        }

        negative_prompt = str(getattr(params, "negative_prompt", "") or "")
        sig = self._get_call_signature(pipe)
        if negative_prompt and self._supports_kwarg(sig, "negative_prompt"):
            inference_params["negative_prompt"] = negative_prompt

        cscale = getattr(params, "controlnet_conditioning_scale", None)
        if cscale is None:
            cscale = getattr(params, "strength", 1.0)
        if self._supports_kwarg(sig, "controlnet_conditioning_scale"):
            inference_params["controlnet_conditioning_scale"] = float(cscale)

        if sig is not None and "control_image" not in sig.parameters and self._supports_kwarg(sig, "image"):
            inference_params["image"] = inference_params.pop("control_image")

        self.logger.info(f"Using device: {device} for editor inference")
        return pipe, inference_params, device

    def _build_iteration_specs(self, params: InferenceRequestParams, device: str) -> list[IterationSpec]:
        if int(getattr(params, "generate_num", 1) or 1) > 1:
            specs: list[IterationSpec] = []
            for i in range(int(getattr(params, "generate_num", 1) or 1)):
                seed = self.seed_manager.iteration_seed()
                generator = self.seed_manager.create_generator(device, seed)
                specs.append(IterationSpec(index=i, seed=seed, generator=generator))
            return specs

        seed = self.seed_manager.initial_seed(getattr(params, "seed", None))
        generator = self.seed_manager.create_generator(device, seed)
        return [IterationSpec(index=0, seed=seed, generator=generator, fail_on_empty=True)]

    async def _drive_iterations(
        self,
        *,
        pipe: Any,
        base_params: dict,
        params: InferenceRequestParams,
        specs: list[IterationSpec],
    ) -> None:
        sig = self._get_call_signature(pipe)
        total = len(specs)

        for spec in specs:
            if await self.check_cancelled("during editor"):
                return

            iter_params = base_params.copy()
            if self._supports_kwarg(sig, "generator"):
                iter_params["generator"] = spec.generator
            iter_params = self._filter_kwargs_for_signature(sig, iter_params)

            await self._run_inference_iteration(
                pipe=pipe,
                iter_params=iter_params,
                request_params=params,
                file_seed=spec.seed,
                index=spec.index,
                total=total,
                fail_on_empty=spec.fail_on_empty,
                task_id=getattr(params, "task_id", None),
            )

    async def _run_inference_iteration(
        self,
        *,
        pipe: Any,
        iter_params: dict,
        request_params: InferenceRequestParams,
        file_seed: int,
        index: int,
        total: int,
        fail_on_empty: bool,
        task_id: str,
    ) -> None:
        from common.logger import print_info

        print_info(iter_params, "编辑推理参数:")
        iter_start_time = time.time()
        call_kwargs = self._build_callable_kwargs(
            pipe=pipe,
            iter_params=iter_params,
            task_id=task_id,
            stage=f"editor iteration {index + 1}",
        )

        def _run_pipe():
            if self.is_task_cancelled():
                raise TaskCancelledError(task_id, f"before editor iteration {index + 1}")
            with torch.inference_mode():
                ensure_vae_dtype(pipe, logger=self.logger)
                return pipe(**call_kwargs)

        result = await self.run_blocking(_run_pipe)
        if self.is_task_cancelled():
            raise TaskCancelledError(task_id, f"after editor iteration {index + 1}")
        if not hasattr(result, "images") or not result.images:
            msg = f"Editor inference returned no images for iteration {index + 1}"
            if fail_on_empty:
                raise ValueError(msg)
            self.logger.error(msg)
            return

        generated_image = result.images[0]
        generated_image = apply_postprocess(generated_image, request_params)
        iter_time = time.time() - iter_start_time

        await self.result_handler.process_single_result(
            file_data=generated_image,
            request_params=request_params,
            generate_time=iter_time,
            service_id=self.service_id,
            file_seed=file_seed,
            index=index,
            total=total,
        )

        try:
            del result
        except Exception:
            pass
        del generated_image

    def _build_callable_kwargs(
        self,
        *,
        pipe: Any,
        iter_params: dict,
        task_id: str,
        stage: str,
    ) -> dict:
        sig = self._get_call_signature(pipe)
        call_kwargs = dict(iter_params)

        def _raise_cancelled(step_index: Any = None) -> None:
            if self.is_task_cancelled():
                step_suffix = ""
                if isinstance(step_index, int):
                    step_suffix = f" step={step_index}"
                raise TaskCancelledError(task_id, f"{stage}{step_suffix}")

        if sig is not None and self._supports_kwarg(sig, "callback_on_step_end"):
            def _callback_on_step_end(_pipe, step_index, _timestep, callback_kwargs):
                _raise_cancelled(step_index)
                return callback_kwargs

            call_kwargs["callback_on_step_end"] = _callback_on_step_end
            if self._supports_kwarg(sig, "callback_on_step_end_tensor_inputs"):
                call_kwargs.setdefault("callback_on_step_end_tensor_inputs", [])
        elif sig is not None and self._supports_kwarg(sig, "callback"):
            def _callback(step_index, _timestep, _latents=None):
                _raise_cancelled(step_index)

            call_kwargs["callback"] = _callback
            if self._supports_kwarg(sig, "callback_steps"):
                call_kwargs["callback_steps"] = 1

        return self._filter_kwargs_for_signature(sig, call_kwargs)

    async def _load_loras_into_pipe(self, pipe: Any, params: InferenceRequestParams) -> None:
        lora_list = []
        try:
            lora_list = list(getattr(params, "parsed_loras", None) or [])
        except Exception:
            lora_list = []
        if not lora_list:
            lora_list = build_lora_list(params.prompt or "", getattr(params, "loras", None))
        if not lora_list:
            return

        def _load():
            load_loras_into_pipe(
                pipe,
                getattr(params, "family", ""),
                self.inference_config.loras_dir,
                lora_list,
                logger=self.logger,
            )

        await self.run_blocking(_load)

    @staticmethod
    def _supports_kwarg(sig: Optional[inspect.Signature], name: str) -> bool:
        if sig is None:
            return True
        if name in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    @staticmethod
    def _filter_kwargs_for_signature(sig: Optional[inspect.Signature], kwargs: dict) -> dict:
        if sig is None:
            return kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = {name for name in sig.parameters.keys() if name not in {"self", "cls"}}
        return {k: v for k, v in kwargs.items() if k in allowed}

    @staticmethod
    def _get_call_signature(pipe: Any) -> Optional[inspect.Signature]:
        try:
            return inspect.signature(pipe.__call__)
        except Exception:
            return None

    @staticmethod
    def _build_model_info(model_path: str, *, load_name: str) -> Any:
        from image.runtime.model_locator import ModelInfo

        p = Path(model_path)
        return ModelInfo(
            repo_id=str(p),
            method="from_pretrained" if p.is_dir() else "from_single_file",
            load_name=str(load_name or model_path),
        )

    def _resolve_model_path(self, raw_name: str) -> str:
        p = Path(str(raw_name or "").strip())
        if p.is_absolute() or p.exists():
            model_path = p
        else:
            model_path = Path(str(self.inference_config.models_dir)) / p
        return str(model_path)

    @staticmethod
    def _special_flux_control_mode(params: InferenceRequestParams) -> Optional[str]:
        load_name = str(getattr(params, "load_name", "") or "").strip().lower()
        return SPECIAL_FLUX_CONTROL_MODELS.get(load_name)

    async def _resolve_flux_control_size(self, source_url: str) -> tuple[int, int]:
        ow, oh = await self.run_blocking(lambda: self._load_source_image_size_sync(source_url))
        w, h = constrain_size(int(ow), int(oh))
        return int(w), int(h)

    def _load_source_image_size_sync(self, source_url: str) -> tuple[int, int]:
        from common.image_utils import load_image

        image = load_image(source_url, resize=False)
        if image is None:
            raise ValueError(f"Editor source image is not loadable: {source_url}")
        try:
            ow, oh = getattr(image, "size", (0, 0))
            return int(ow), int(oh)
        finally:
            try:
                del image
            except Exception:
                pass

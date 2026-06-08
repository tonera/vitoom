"""
扩散类任务 handler（MK/ED/SED 等需要 diffusers pipeline 的 job_type）
把 inferrer.py 中的 diffusion 主流程拆出来
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Callable

import torch

from common.image_utils import load_image, calc_fit_size_with_multiple
from common.task_cancel import TaskCancelledError
from image.runtime.postprocess_pipeline import apply_postprocess
from image.runtime.pipeline_release import finish_pipeline_use
from image.runtime.vae_dtype_fixer import ensure_vae_dtype
from image.runtime.scheduler_loader import load_scheduler_from_pipe
from schemas import InferenceRequestParams


@dataclass
class IterationSpec:
    index: int
    seed: int
    generator: Any
    image_path: Optional[str] = None
    fail_on_empty: bool = False


class DiffusionHandler:
    def __init__(
        self,
        *,
        lifecycle: Any,
        seed_manager: Any,
        result_handler: Any,
        service_id: str,
        logger: Any,
        run_blocking: Callable[[Callable[[], Any]], Any],
        check_cancelled: Callable[[str], Any],
        is_task_cancelled: Callable[[], bool],
    ):
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
            self.logger.info(f"Using device: {device} for inference")

            mode = self._resolve_mode(params)
            specs = self._build_iteration_specs(mode, params, device)

            await self._drive_iterations(
                pipe=pipe,
                base_params=inference_params or {},
                params=params,
                device=device,
                task_id=task_id,
                specs=specs,
                mode=mode,
            )
        except TaskCancelledError:
            raise
        except Exception as e:
            failed = True
            oom_error = bool(self.lifecycle.is_oom_error(e))
            if oom_error:
                self.logger.warning("OOM detected during diffusion inference; forcing cached pipeline eviction")
                # OOM traceback 持有 VAE decode 等中间巨型 tensor 的栈帧引用，
                # 必须在 cleanup 前清掉，否则 gc 无法回收那些 tensor。
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

    def _resolve_mode(self, params: InferenceRequestParams) -> str:
        if params.tpl_list and params.job_type in ["SED", "RBG", "SR", "FS"]:
            return "batch"
        if params.generate_num > 1:
            return "multi"
        return "single"

    def _build_iteration_specs(self, mode: str, params: InferenceRequestParams, device: str) -> list[IterationSpec]:
        if mode == "batch":
            if not params.tpl_list:
                raise ValueError(f"tpl_list is required for batch job_type {params.job_type}")
            batch_seed = self.seed_manager.batch_seed(params.seed)
            generator = self.seed_manager.create_generator(device, batch_seed)
            return [IterationSpec(index=i, seed=batch_seed, generator=generator, image_path=tpl) for i, tpl in enumerate(params.tpl_list)]

        if mode == "multi":
            specs: list[IterationSpec] = []
            for i in range(params.generate_num):
                seed = self.seed_manager.iteration_seed()
                generator = self.seed_manager.create_generator(device, seed)
                specs.append(IterationSpec(index=i, seed=seed, generator=generator))
            return specs

        seed = self.seed_manager.initial_seed(params.seed)
        generator = self.seed_manager.create_generator(device, seed)
        return [IterationSpec(index=0, seed=seed, generator=generator, fail_on_empty=True)]

    def _compose_iter_params(self, base_params: dict, spec: IterationSpec, params: InferenceRequestParams) -> tuple[Optional[dict], Optional[Any]]:
        iter_params = base_params.copy()
        iter_params["generator"] = spec.generator

        input_image = None
        if spec.image_path:
            input_image = load_image(spec.image_path)
            if input_image is None:
                self.logger.warning(f"Failed to load image from tpl_list[{spec.index}]: {spec.image_path}, skipping")
                return None, None
            if params.job_type == "SED":
                try:
                    ow, oh = getattr(input_image, "size", (0, 0))
                    w, h = calc_fit_size_with_multiple(int(ow), int(oh), 1024, 1024, multiple=16)
                    if w and h and ("width" in iter_params) and ("height" in iter_params):
                        iter_params["width"] = int(w)
                        iter_params["height"] = int(h)
                except Exception as e:
                    self.logger.warning(f"JT_SED calc_fit_size_with_multiple failed, using original params: {e}")
            iter_params["image"] = input_image
        return iter_params, input_image

    async def _drive_iterations(
        self,
        *,
        pipe: Any,
        base_params: dict,
        params: InferenceRequestParams,
        device: str,
        task_id: str,
        specs: Iterable[IterationSpec],
        mode: str,
    ) -> None:
        specs_list = list(specs)
        initial_total = len(specs_list)
        actual_total = initial_total
        self.logger.info(f"{mode.capitalize()} mode: planning {initial_total} iterations")

        for spec in specs_list:
            if await self.check_cancelled(f"during {mode}"):
                return

            iter_params, input_image = self._compose_iter_params(base_params, spec, params)
            if not iter_params:
                actual_total -= 1
                continue

            success = await self._run_inference_iteration(
                pipe=pipe,
                iter_params=iter_params,
                request_params=params,
                file_seed=spec.seed,
                index=spec.index,
                total=initial_total if params.job_type == "SED" else actual_total,
                fail_on_empty=spec.fail_on_empty,
                task_id=task_id,
            )

            if not success:
                actual_total -= 1

            if input_image is not None:
                del input_image

    async def _run_inference_iteration(
        self,
        *,
        pipe: Any,
        iter_params: dict,
        request_params: InferenceRequestParams,
        file_seed: int,
        index: int,
        total: Optional[int],
        fail_on_empty: bool,
        task_id: str,
    ) -> bool:
        from common.logger import print_info

        print_info(iter_params, "推理参数:")
        # 额外补一条“确定性”日志：直接输出 generator 的 device（CPU/CUDA）
        try:
            gen = iter_params.get("generator")
            if isinstance(gen, torch.Generator):
                self.logger.debug(f"iter_params.generator device={gen.device}, seed={gen.initial_seed()}")
            elif gen is not None:
                self.logger.debug(f"iter_params.generator type={type(gen)}")
        except Exception:
            self.logger.debug("Failed to introspect iter_params.generator (ignored)", exc_info=True)
        iter_start_time = time.time()
        call_kwargs = self._build_callable_kwargs(
            pipe=pipe,
            iter_params=iter_params,
            task_id=task_id,
            stage=f"diffusion iteration {index + 1}",
        )

        def _run_pipe():
            if self.is_task_cancelled():
                raise TaskCancelledError(task_id, f"before diffusion iteration {index + 1}")
            with torch.inference_mode():
                # 某些 SDXL/VAE 组合会出现 post_quant_conv bias=float32 而输入 latents=float16 的情况；
                # 这里在真正调用 pipeline 前做一次轻量修复，避免 decode 阶段 dtype mismatch。
                ensure_vae_dtype(pipe, logger=self.logger)
                return pipe(**call_kwargs)

        result = await self.run_blocking(_run_pipe)
        if self.is_task_cancelled():
            raise TaskCancelledError(task_id, f"after diffusion iteration {index + 1}")

        if not hasattr(result, "images") or not result.images:
            msg = f"Inference returned no images for iteration {index + 1}"
            if fail_on_empty:
                raise ValueError(msg)
            self.logger.error(msg)
            return False

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
        return True

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



"""
Pipeline 生命周期管理（从 inferrer.py 抽离）

目标：让 inferrer.py 只做流程编排，一眼能看清。
"""

from __future__ import annotations

from typing import Any, Optional
import json
import hashlib
import os
import time

import torch

from common.runtime_cleanup import is_oom_error as common_is_oom_error
from common.torch_transfer_utils import pretouch_pipeline_cpu_tensors, should_pretouch
from common.pipeline_cache import PipelineCache
from schemas import InferenceRequestParams
from image.runtime.pipeline_service import PipelineService


class PipelineLifecycle:
    def __init__(
        self,
        *,
        detector: Any,
        device_planner: Any,
        model_locator: Any,
        inference_config: Any,
        logger: Any,
        pipeline_cache: Optional[PipelineCache] = None,
        run_blocking: Optional[Any] = None,
    ):
        self.detector = detector
        self.device_planner = device_planner
        self.model_locator = model_locator
        self.inference_config = inference_config
        self.logger = logger
        self.pipeline_cache = pipeline_cache
        # 可选：将“模型加载/参数构建”等重活放到线程执行，避免阻塞事件循环（保持 WS 心跳/收包稳定）
        self.run_blocking = run_blocking
        self._cache_key: Optional[str] = None
        self._cache_enabled: bool = False
        self.pipeline_service = PipelineService(
            detector=self.detector,
            device_planner=self.device_planner,
            model_locator=self.model_locator,
            inference_config=self.inference_config,
            logger=self.logger,
            pipeline_cache=self.pipeline_cache,
            run_blocking=self.run_blocking,
            build_cache_key_fn=self._build_cache_key,
        )

    @staticmethod
    def _stable_hash(sig: dict) -> str:
        payload = json.dumps(sig, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_cache_key(
        self,
        *,
        params: InferenceRequestParams,
        pipeline_cls: Any,
        model_info: Any,
        device_plan: Any,
        pipeline_params: dict,
    ) -> str:
        """
        构建“base pipeline 兼容性签名”。

        只包含会影响 base pipeline 结构/权重注入的字段；scheduler / LoRA 等按 request 动态处理，不进入签名。
        """
        sig = {
            "load_name": getattr(params, "load_name", None),
            "family": getattr(params, "family", None),
            "is_img2img": bool(getattr(params, "url", None)),
            "pipeline_cls": getattr(pipeline_cls, "__name__", str(pipeline_cls)),
            "repo_id": getattr(model_info, "repo_id", None),
            "method": getattr(model_info, "method", None),
            "device": getattr(device_plan, "device", None),
            "torch_dtype": str(getattr(device_plan, "torch_dtype", None)),
            "low_vram": bool(getattr(params, "low_vram", False)),
            # from_single_file 的 original_config / config / cache_dir 等可能影响 pipeline 结构
            "pipeline_params_sig": {
                k: pipeline_params.get(k)
                for k in (
                    "original_config",
                    "config",
                    "use_safetensors",
                    "variant",
                    "cache_dir",
                    "__component_sig",
                    "__dropped_kwargs",
                )
                if k in pipeline_params
            },
        }
        return self._stable_hash(sig)

    def cache_enabled(self) -> bool:
        return bool(self._cache_enabled and self.pipeline_cache is not None and self._cache_key)

    def set_cache_state(self, *, cache_key: Optional[str], cache_enabled: bool) -> None:
        self._cache_key = cache_key
        self._cache_enabled = bool(cache_enabled and self.pipeline_cache is not None and cache_key)

    def get_cache_state(self) -> tuple[Optional[str], bool]:
        return self._cache_key, self.cache_enabled()

    def clear_cache_state(self) -> None:
        self._cache_key = None
        self._cache_enabled = False

    @staticmethod
    def is_oom_error(exc: BaseException) -> bool:
        return common_is_oom_error(exc)

    def _resolve_fast_mode(self, params: InferenceRequestParams) -> bool:
        fast = bool(getattr(params, "fast_mode", False))
        raw = str(os.getenv("VITOOM_FAST_MODE", "") or "").strip().lower()
        if not raw:
            return fast

        truthy = {"1", "true", "on", "yes", "y"}
        falsy = {"0", "false", "off", "no", "n"}
        if raw in truthy:
            forced = True
        elif raw in falsy:
            forced = False
        else:
            try:
                self.logger.warning(f"ignore invalid env VITOOM_FAST_MODE={raw!r}; fallback to params.fast_mode={fast}")
            except Exception:
                pass
            return fast

        if forced != fast:
            try:
                self.logger.info(f"fast_mode overridden by env VITOOM_FAST_MODE={raw!r}: {fast} -> {forced}")
            except Exception:
                pass
        return forced

    async def create_pipeline(self, params: InferenceRequestParams) -> tuple[Any, dict, Any]:
        """创建 pipeline + 基础推理参数（不包含 generator）。"""
        # 注意：params 的预处理应在推理入口统一完成（ImageInferrer），这里不再做二次预处理，避免链路变长/语义分裂。
        result = await self.pipeline_service.acquire(params)
        self.set_cache_state(cache_key=result.cache_key, cache_enabled=result.cache_enabled)
        return result.pipe, result.inference_params, result.device_plan

    def apply_fast_mode_cache(self, pipe: Any, params: InferenceRequestParams) -> None:
        """
        开关职责约定：
        - fast_mode：控制是否启用运行期 fast cache
        - low_vram：不参与 cache 开关决策，只影响 move_to_device 中的 offload / 设备迁移策略

        fast_mode 加速开关（运行期）：
        - TeaCache：fast_mode=True 时按 family 尝试启用
        - FBCache：fast_mode=True 且 SDXL 时尝试启用
        - MeanCache：fast_mode=True 时尝试启用

        注意：pipeline cache 可能复用同一个 pipe；因此 fast_mode=False 时也要 best-effort 关闭/回滚相关 patch，
        避免“上一次 fast_mode 打开后，下一次没开也仍在加速”的串开关问题。
        """
        from common.family_utils import to_model_family

        mv = to_model_family(getattr(params, "family", None))
        fast = self._resolve_fast_mode(params)

        # ---------- helpers ----------
        def _disable_teacache_on_transformer(tr: Any) -> None:
            try:
                if tr is None:
                    return
                # instance attrs
                for name in (
                    "previous_modulated_input",
                    "previous_residual",
                    "accumulated_rel_l1_distance",
                    "cnt",
                    "_nunchaku_teacache_ctx",
                    "_nunchaku_teacache_ctx_qwenimage",
                ):
                    try:
                        if hasattr(tr, name):
                            setattr(tr, name, None)
                    except Exception:
                        pass
                try:
                    if hasattr(tr, "enable_teacache"):
                        setattr(tr, "enable_teacache", False)
                except Exception:
                    pass
            except Exception:
                return

        def _enable_teacache_on_pipe() -> None:
            tr = getattr(pipe, "transformer", None)
            if tr is None:
                return
            try:
                from common.teacache import set_tea_cache  # type: ignore

                # family -> load_name (polynomial selection)
                load_name = "flux-kontext" if mv in {"flux_kontext", "flux-kontext"} else "flux"
                if mv == "qwen":
                    load_name = "qwenimage"
                if mv == "chroma":
                    load_name = "chroma"

                # thresholds: keep existing defaults (tunable later)
                rel_l1 = 0.3
                set_tea_cache(tr, int(getattr(params, "num_inference_steps", 0) or 0), rel_l1_thresh=rel_l1, skip_steps=2, load_name=load_name)
                self.logger.info(f"fast_mode: enabled teacache for family={mv} load_name={load_name} rel_l1={rel_l1}")
            except Exception as e:
                self.logger.warning(f"fast_mode: failed to enable teacache for family={mv}: {e}")

        def _disable_sdxl_fbcache() -> None:
            # Try to restore UNet forward if we can (best-effort).
            try:
                unet = getattr(pipe, "unet", None)
                if unet is not None and hasattr(unet, "_original_forward"):
                    try:
                        unet.forward = unet._original_forward  # type: ignore[assignment]
                        setattr(unet, "_is_cached", False)
                    except Exception:
                        pass
            except Exception:
                pass

            # 回滚 pipeline __call__ 的 instance subclass。
            # FBCache 可能叠加在 MeanCache 之上，因此这里只回到 FBCache 记录的“上一层类”。
            try:
                base_cls = getattr(pipe, "_fbcache_pipeline_base_class", None) or getattr(pipe, "_pipeline_base_class", None)
                if bool(getattr(pipe, "_fbcache_call_isolated", False)) and base_cls is not None:
                    pipe.__class__ = base_cls  # type: ignore[assignment]
                    setattr(pipe, "_fbcache_call_isolated", False)
            except Exception:
                pass

        def _enable_sdxl_fbcache() -> None:
            try:
                from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

                if not isinstance(pipe, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)):
                    return
                from common.fbcache_sdxl import apply_cache_on_pipe as apply_sdxl_cache

                already = bool(getattr(pipe, "_fbcache_call_isolated", False)) or bool(getattr(pipe, "_is_cached", False))
                apply_sdxl_cache(pipe, residual_diff_threshold=0.12, verbose=False)
                if already:
                    self.logger.debug(f"fast_mode: local SDXL fbcache already enabled for {type(pipe).__name__}")
                else:
                    self.logger.info(f"fast_mode: enabled local SDXL fbcache for {type(pipe).__name__}")
            except Exception as e:
                self.logger.warning(f"fast_mode: failed SDXL fbcache for {type(pipe).__name__}: {e}")

        def _disable_meancache_on_module(mod: Any) -> None:
            try:
                if mod is None:
                    return
                # 快速禁用（即使 forward 已回滚也无害）
                try:
                    if hasattr(mod, "enable_meancache"):
                        setattr(mod, "enable_meancache", False)
                except Exception:
                    pass

                # 恢复原 forward（best-effort）
                try:
                    if hasattr(mod, "_meancache_original_forward"):
                        mod.forward = mod._meancache_original_forward  # type: ignore[assignment]
                except Exception:
                    pass

                # 清理可能持有大 tensor 的字段，避免跨请求占用内存
                for name in (
                    "_meancache_engine",
                    "_meancache_last_template_by_pred",
                    "_meancache_last_packer_by_pred",
                    "_meancache_x_param_name",
                    "_meancache_x_param_pos",
                    "_meancache_timestep_param_name",
                    "_meancache_timestep_param_pos",
                    "_meancache_forward_calls",
                    "_meancache_is_patched",
                ):
                    try:
                        if hasattr(mod, name):
                            delattr(mod, name)
                    except Exception:
                        pass

                # _meancache_original_forward 放最后删，防止上面恢复 forward 依赖它
                try:
                    if hasattr(mod, "_meancache_original_forward"):
                        delattr(mod, "_meancache_original_forward")
                except Exception:
                    pass
            except Exception:
                return

        def _disable_meancache_on_pipe() -> None:
            # 1) 先尽量回滚 transformer/unet 的 forward
            _disable_meancache_on_module(getattr(pipe, "transformer", None))
            _disable_meancache_on_module(getattr(pipe, "unet", None))
            _disable_meancache_on_module(getattr(pipe, "_meancache_target_module", None))

            # 2) reset engine state（释放缓存 tensor）
            try:
                eng = getattr(pipe, "_meancache_engine", None)
                if eng is not None and hasattr(eng, "reset"):
                    eng.reset()
            except Exception:
                pass
            try:
                if hasattr(pipe, "_meancache_engine"):
                    setattr(pipe, "_meancache_engine", None)
            except Exception:
                pass

            # 3) 回滚 __call__ 的 instance subclass（只在 meancache 打过 patch 时处理）
            try:
                base_cls = getattr(pipe, "_meancache_pipeline_base_class", None) or getattr(pipe, "_pipeline_base_class", None)
                if bool(getattr(pipe, "_meancache_call_isolated", False)) and base_cls is not None:
                    pipe.__class__ = base_cls  # type: ignore[assignment]
                    setattr(pipe, "_meancache_call_isolated", False)
            except Exception:
                pass

        def _enable_meancache_on_pipe() -> None:
            """
            MeanCache：对 pipeline 做实例级 patch（优先 transformer，否则 unet）。
            当前 fast_mode 策略会与其他 cache 一起按 best-effort 方式尝试启用。
            """
            try:
                from cache import apply_meancache_on_pipe  # type: ignore

                already = bool(getattr(pipe, "_meancache_call_isolated", False)) or getattr(pipe, "_meancache_engine", None) is not None
                apply_meancache_on_pipe(
                    pipe,
                    # 保持 meancache 内部默认参数（仓库内已有默认调参）
                    cache_device="cpu",
                    preset_name="fast_mode",
                    debug=False,
                )
                if already:
                    self.logger.debug(f"fast_mode: local meancache already enabled for family={mv} {type(pipe).__name__}")
                else:
                    self.logger.info(f"fast_mode: enabled local meancache for family={mv} {type(pipe).__name__}")
            except Exception as e:
                self.logger.warning(f"fast_mode: failed meancache for family={mv} {type(pipe).__name__}: {e}")

        # ---------- main ----------
        if not fast:
            # Explicitly disable/cleanup to avoid cross-request leakage under pipeline cache.
            _disable_teacache_on_transformer(getattr(pipe, "transformer", None))
            _disable_sdxl_fbcache()
            _disable_meancache_on_pipe()
            return

        try:
            _enable_meancache_on_pipe()
        except Exception:
            self.logger.debug(f"fast_mode: 启用 meancache 失败 for family={mv} {type(pipe).__name__}")
            pass

        if mv in {"flux", "flux_kontext", "flux-kontext", "qwen", "chroma"}:
            _enable_teacache_on_pipe()

        if mv == "sdxl":
            _enable_sdxl_fbcache()

    def pretouch_cpu_tensors(
        self,
        pipe: Any,
        plan: Any,
        family: str,
        *,
        low_vram: bool = False,
    ) -> None:
        """
        仅在“明确会整管道 .to(cuda)”且不走 offload 的场景才做 CPU 侧 pretouch。
        对 Qwen/QwenEdit 场景默认跳过，由上层显式策略自行控制。
        """
        from common.Constant import MODEL_QWEN, MODEL_QWEN_EDIT

        if low_vram:
            self.logger.info("pretouch cpu tensors: skipped (low_vram=true, using CPU offload)")
            return

        mv = (family or "").lower()
        if mv in {m.lower() for m in MODEL_QWEN} or mv in {m.lower() for m in MODEL_QWEN_EDIT}:
            self.logger.info(f"pretouch cpu tensors: skipped (family={family!r}, qwen handled separately)")
            return

        resolved_device = getattr(plan, "device", None)
        if not should_pretouch(resolved_device):
            self.logger.info(
                f"pretouch cpu tensors: skipped (should_pretouch=false device={resolved_device!r})"
            )
            return

        touched: list[str] = []
        t0 = time.perf_counter()

        def _on_component(name: str) -> None:
            touched.append(name)
            self.logger.info(f"pretouch cpu tensors: {name} done")

        pretouch_pipeline_cpu_tensors(pipe, on_component=_on_component)
        elapsed = time.perf_counter() - t0
        if touched:
            self.logger.info(
                f"pretouch cpu tensors: completed components={touched} elapsed={elapsed:.2f}s"
            )
        else:
            self.logger.info(f"pretouch cpu tensors: no CPU tensors to touch elapsed={elapsed:.2f}s")

    def enable_cpu_offload(self, pipe: Any, params: InferenceRequestParams) -> bool:
        """
        启用 CPU offload（统一策略入口）。
        - Qwen/QwenEdit：优先 transformer.set_offload + sequential offload，并排除 transformer 避免重复 hook
        - 其他：优先 enable_model_cpu_offload，否则 enable_sequential_cpu_offload
        """
        from common.Constant import MODEL_QWEN, MODEL_QWEN_EDIT

        mv = (getattr(params, "family", "") or "").lower()
        model_qwen = {m.lower() for m in MODEL_QWEN}
        model_qwen_edit = {m.lower() for m in MODEL_QWEN_EDIT}

        try:
            if mv in model_qwen or mv in model_qwen_edit:
                tr = getattr(pipe, "transformer", None)
                # if tr is not None and hasattr(tr, "set_offload"):
                #     tr.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                # if hasattr(pipe, "_exclude_from_cpu_offload"):
                #     try:
                #         ex = getattr(pipe, "_exclude_from_cpu_offload")
                #         if isinstance(ex, list) and "transformer" not in ex:
                #             ex.append("transformer")
                #     except Exception:
                #         pass
                # if hasattr(pipe, "enable_sequential_cpu_offload"):
                #     pipe.enable_sequential_cpu_offload()
                #     return True
                try:
                    if tr.__class__.__name__ == "NunchakuQwenImageTransformer2DModel":
                        tr.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                        pipe._exclude_from_cpu_offload.append("transformer")
                        pipe.enable_sequential_cpu_offload()
                        return True
                except Exception:
                    pass

            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
                return True
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                return True
            return False
        except Exception as e:
            self.logger.warning(f"启用 CPU offload 失败: {e}")
        return False

    def enable_low_vram_slicing(self, pipe: Any) -> None:
        """在低显存模式下尽量启用切片优化，进一步降低推理峰值显存。"""
        try:
            if hasattr(pipe, "enable_vae_slicing"):
                # pipe.enable_vae_slicing()
                pipe.vae.enable_slicing()
        except Exception as e:
            self.logger.warning(f"启用 VAE slicing 失败: {e}")

        try:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        except Exception as e:
            self.logger.warning(f"启用 attention slicing 失败: {e}")

    def move_to_device(self, pipe: Any, plan: Any, params: InferenceRequestParams) -> tuple[Any, str]:
        """
        从原 inferrer.py 迁移：根据设备规划迁移管道到目标设备，并在低显存模式应用优化策略。
        开关职责约定：
        - low_vram=True：走 CPU offload + slicing，不直接整管道 .to(cuda)
        - low_vram=False：走常规设备迁移；若当前环境可用，则优先使用 cuda
        """
        from common.Constant import MODEL_SDXL, MODEL_15

        mv = (getattr(params, "family", "") or "").lower()
        model_sdxl = {m.lower() for m in MODEL_SDXL}
        model_15 = {m.lower() for m in MODEL_15}
        target = plan.device
        low_vram = bool(getattr(params, "low_vram", False))
        # 精简日志：仅在关键分支记录

        if mv in model_sdxl or mv in model_15:
            try:
                pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
            except Exception:
                pass
        if low_vram:
            self.logger.info("low_vram=true，启用 slicing 优化并跳过 direct .to，直接启用 CPU offload")
            self.enable_low_vram_slicing(pipe)
            self.enable_cpu_offload(pipe, params)
            return pipe, target

        if torch.cuda.is_available():
            target = "cuda"

        if target == "cpu":
            return pipe, target

        pipe = pipe.to(target)
        return pipe, target





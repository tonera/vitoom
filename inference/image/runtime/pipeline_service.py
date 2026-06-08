from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Callable
import inspect

from common.logger import print_info
from common.family_utils import to_model_family
from image.inference_params_builder import build_inference_params
from image.runtime.lora_manager import build_lora_list, load_loras_into_pipe
from schemas import InferenceRequestParams


@dataclass(frozen=True)
class PipelineAcquireResult:
    pipe: Any
    inference_params: dict
    device_plan: Any
    cache_key: Optional[str]
    cache_enabled: bool


class PipelineService:
    """
    Pipeline 装配门面（运行期）：
    Resolve -> Acquire(base pipe) -> Configure (LoRA) -> Build kwargs
    （Place/oom/offload 仍由 PipelineLifecycle 现有方法负责，后续可继续拆）
    """

    def __init__(
        self,
        *,
        detector: Any,
        device_planner: Any,
        model_locator: Any,
        inference_config: Any,
        logger: Any,
        pipeline_cache: Optional[Any],
        run_blocking: Optional[Any],
        build_cache_key_fn: Callable[..., str],
    ):
        self.detector = detector
        self.device_planner = device_planner
        self.model_locator = model_locator
        self.inference_config = inference_config
        self.logger = logger
        self.pipeline_cache = pipeline_cache
        self.run_blocking = run_blocking
        self._build_cache_key = build_cache_key_fn

    @staticmethod
    def _filter_kwargs_for_callable(fn: Any, kwargs: dict) -> tuple[dict, list[str]]:
        """
        仅为“diffusers 体系内 pipeline from_pretrained/from_single_file”的 kwargs 兼容：
        - 若 fn 支持 **kwargs：不做过滤
        - 否则按 signature 过滤出可接受的参数名
        返回：(filtered_kwargs, dropped_keys)
        """
        if not kwargs:
            return {}, []
        try:
            sig = inspect.signature(fn)
            for p in sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return kwargs, []
            allowed = set(sig.parameters.keys())
            # classmethod: 跳过 cls/self 参数（不会出现在 kwargs 里，但保守）
            allowed.discard("self")
            allowed.discard("cls")
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            dropped = [k for k in kwargs.keys() if k not in allowed]
            return filtered, sorted(dropped)
        except Exception:
            # 无法获取 signature 时不过滤（尽量不改变现有行为）
            return kwargs, []

    def instantiate_pipeline(self, *, pipeline_cls: Any, model_info: Any, pipeline_params: dict) -> Any:
        """
        Materialize lazy component overrides and instantiate a pipeline factory.

        This keeps POSE and the regular pipeline path on the same injection/materialization logic.
        """
        if not hasattr(pipeline_cls, model_info.method):
            raise ValueError(f"Pipeline {pipeline_cls.__name__} does not have method {model_info.method}")
        pipe_factory = getattr(pipeline_cls, model_info.method)
        raw_kwargs = dict(pipeline_params)
        lazy_loaders = raw_kwargs.pop("__lazy_component_loaders", None)
        if isinstance(lazy_loaders, dict):
            try:
                self.logger.info(
                    f"[inject-load] materializing lazy components count={len(lazy_loaders)} "
                    f"pipeline={pipeline_cls.__name__} method={model_info.method}"
                )
            except Exception:
                pass
            for name in sorted(lazy_loaders.keys()):
                loader = lazy_loaders.get(name)
                if not callable(loader):
                    continue
                try:
                    self.logger.info(f"[inject-load] materialize component name={name}")
                except Exception:
                    pass
                raw_kwargs[name] = loader()
        # 内部 metadata 只用于 cache key/日志，不参与 signature 过滤，也不透传给 pipeline 工厂
        for k in ("__component_sig", "__component_sources", "__dropped_kwargs", "__lazy_component_loaders"):
            raw_kwargs.pop(k, None)
        call_kwargs, dropped = self._filter_kwargs_for_callable(pipe_factory, raw_kwargs)
        if dropped:
            try:
                self.logger.info(
                    f"[pipeline-kwargs] {pipeline_cls.__name__}.{model_info.method} dropped={dropped}"
                )
            except Exception:
                pass
        return pipe_factory(model_info.repo_id, **call_kwargs)

    async def acquire(
        self,
        params: InferenceRequestParams,
    ) -> PipelineAcquireResult:
        self.logger.info(f"Creating pipeline for model: {params.load_name}")

        pipeline_cls = self.detector.get_pipeline(params)

        # ===== family 统一：入口已归一化为 canonical；这里仅兜底 + 不一致告警 =====
        cur = to_model_family(getattr(params, "family", None))
        det = to_model_family(getattr(self.detector, "family", None))
        if not cur:
            if det:
                params.family = det
            else:
                raise ValueError("family missing and detector failed to detect model type")
        else:
            # 若入口 canonical 与 detector 不一致，仅告警并继续沿用入口 canonical（按需求：用户为主）
            if det and det != cur:
                try:
                    self.logger.warning(
                        f"[family] entry(canonical)={cur} 与 detector={det} 不一致；仍以 entry 为准"
                    )
                except Exception:
                    pass

        model_info = self.model_locator.locate(params)
        device_plan = self.device_planner.plan(params)
        self.logger.info(
            f"载入推理管道: {pipeline_cls.__name__}, family: {params.family} "
            f"torch_dtype: {device_plan.torch_dtype}, device: {device_plan.device}"
        )

        pipeline_params = self.detector.build_pipeline_params(
            params,
            device_plan=device_plan,
            model_info=model_info,
        )
        display_pipeline_params = {k: v for k, v in pipeline_params.items() if k != "__lazy_component_loaders"}
        print_info(display_pipeline_params, "管道参数:")

        cache_ttl = int(getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0) or 0)
        cache_enabled = bool(self.pipeline_cache is not None and cache_ttl > 0 and self.pipeline_cache.enabled())

        def _create_new_pipe():
            return self.instantiate_pipeline(
                pipeline_cls=pipeline_cls,
                model_info=model_info,
                pipeline_params=pipeline_params,
            )

        cache_key: Optional[str] = None
        if cache_enabled:
            # cache key 必须基于“最终生效的 kwargs”（包含 dropped 结果的稳定记录）
            try:
                pipe_factory = getattr(pipeline_cls, model_info.method)
                raw_kwargs = dict(pipeline_params)
                for k in ("__component_sig", "__component_sources", "__dropped_kwargs", "__lazy_component_loaders"):
                    raw_kwargs.pop(k, None)
                call_kwargs, dropped = self._filter_kwargs_for_callable(pipe_factory, raw_kwargs)
                # 记录 dropped（用于 key 语义，不传给 pipe_factory）
                if dropped:
                    pipeline_params["__dropped_kwargs"] = dropped
            except Exception:
                pass
            key = self._build_cache_key(
                params=params,
                pipeline_cls=pipeline_cls,
                model_info=model_info,
                device_plan=device_plan,
                pipeline_params=pipeline_params,
            )
            cache_key = key

            async def _create_new_pipe_async():
                if self.run_blocking:
                    return await self.run_blocking(_create_new_pipe)
                return _create_new_pipe()

            pipe, hit = await self.pipeline_cache.acquire(key=key, create_fn=_create_new_pipe_async)
            try:
                if hit:
                    self.logger.info(f"[pipeline-cache] HIT 复用缓存pipeline key={key[:12]} model={params.load_name}")
                else:
                    self.logger.info(f"[pipeline-cache] MISS 新建并缓存pipeline key={key[:12]} model={params.load_name}")
            except Exception:
                pass
        else:
            if self.run_blocking:
                pipe = await self.run_blocking(_create_new_pipe)
            else:
                pipe = _create_new_pipe()

        # LoRA load（必须在 build_inference_params 前）
        lora_list = []
        try:
            lora_list = list(getattr(params, "parsed_loras", None) or [])
        except Exception:
            lora_list = []
        # 兜底：若未经过预处理，则尝试从原始字段解析（注意：sanitize_prompt 会移除 <lora...>）
        if not lora_list:
            lora_list = build_lora_list(params.prompt or "", getattr(params, "loras", None))

        if lora_list:
            fam = to_model_family(getattr(params, "family", None))
            if fam == "anima":
                # 仅在 runtime backend（无 diffusers adapters API）时跳过。
                # 未来官方 diffusers 版若支持 LoRA，这里不会阻断。
                supports_adapters = bool(getattr(pipe, "load_lora_weights", None)) and bool(getattr(pipe, "set_adapters", None))
                if not supports_adapters:
                    try:
                        self.logger.warning("anima(runtime) ignores LoRA params (not supported)")
                    except Exception:
                        pass
                    lora_list = []

        if lora_list:
            def _load_loras():
                load_loras_into_pipe(
                    pipe,
                    getattr(params, "family", ""),
                    self.inference_config.loras_dir,
                    lora_list,
                    logger=self.logger,
                )

            if self.run_blocking:
                await self.run_blocking(_load_loras)
            else:
                _load_loras()

        if self.run_blocking:
            inference_params = await self.run_blocking(lambda: build_inference_params(pipe, params))
        else:
            inference_params = build_inference_params(pipe, params)

        inference_params.pop("generator", None)
        return PipelineAcquireResult(
            pipe=pipe,
            inference_params=inference_params,
            device_plan=device_plan,
            cache_key=cache_key,
            cache_enabled=bool(cache_enabled and cache_key),
        )


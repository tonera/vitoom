from __future__ import annotations

import asyncio
import gc
import inspect
import json
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

from common.base_inferrer import BaseInferrer
from common.config_loader import load_inference_config
from common.logger import get_logger, print_info
from common.pipeline_cache import PipelineCache
from common.result_handler import ResultHandler
from common.tts_speakers import get_default_speaker, voxcpm_speaker_presets
from schemas import InferenceRequestParams

from audio.engines.qwen_tts_engine import QwenTtsEngine
from audio.engines.tts_engine import TtsEngine
from audio.engines.voxcpm_tts_engine import VoxCPMTtsEngine
from audio.handlers.qwen_asr_handler import QwenAsrHandler
from audio.handlers.qwen_tts_handler import QwenTtsHandler
from audio.handlers.voxcpm_tts_handler import VoxCPMTtsHandler
from audio.runtime.qwen_asr_bridge import (
    load_asr_bundle as load_qwen_asr_bundle,
    resolve_qwen_asr_bundle_options,
)
from audio.runtime.qwen_asr_vllm_streaming import (
    create_vllm_streaming_session,
    load_vllm_asr_bundle,
    resolve_qwen_asr_vllm_options,
)
from audio.runtime.qwen_tts_bridge import (
    load_tts_bundle as load_qwen_tts_bundle,
    prewarm_qwen_tts_bundle,
)
from audio.runtime.runtime_resolver import (
    merge_qwen_tts_loader_runtime_cfg,
    merge_voxcpm_loader_runtime_cfg,
    resolve_audio_backend,
    resolve_audio_model_ref,
    resolve_audio_runtime_policy,
    resolve_audio_runtime,
)
from audio.runtime.voxcpm_bridge import (
    load_realtime_bundle as load_voxcpm_realtime_bundle,
    load_tts_bundle as load_voxcpm_tts_bundle,
)
from audio.session_runtime import AudioSessionRuntime, AudioSessionState

logger = get_logger(__name__)


class _ReleasingStreamingSession:
    """Wrap realtime ASR sessions and release the cache use after finish()."""

    def __init__(self, session: Any, release_fn: Any):
        self._session = session
        self._release_fn = release_fn
        self._loop = asyncio.get_running_loop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def finish(self) -> Any:
        try:
            return self._session.finish()
        finally:
            future = asyncio.run_coroutine_threadsafe(self._release_fn(), self._loop)
            future.result()


class _ReleasingTtsEngine:
    """Wrap a session TTS engine and release all acquired bundles per request."""

    def __init__(self, engine: TtsEngine, used_bundles: Dict[str, tuple[Dict[str, Any], bool]], release_fn: Any):
        self._engine = engine
        self._used_bundles = used_bundles
        self._release_fn = release_fn

    async def synthesize_stream(self, **kwargs: Any):
        self._used_bundles.clear()
        try:
            async for chunk in self._engine.synthesize_stream(**kwargs):  # type: ignore[call-arg]
                yield chunk
        finally:
            for key, (bundle, cache_enabled) in list(self._used_bundles.items()):
                await self._release_fn(key, bundle, cache_enabled)
            self._used_bundles.clear()


def _stable_json_fingerprint(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload or {}, sort_keys=True, default=str)
    except Exception:
        return str(payload)


def _params_with_load_name_override(params: Any, new_load_name: str) -> SimpleNamespace:
    """构造一份与 ``params`` 同字段、仅 ``load_name`` 被覆盖的 ``SimpleNamespace``。

    用途：handler 在同一推理任务里需要先后加载多个权重（如 qwen-tts drama 的
    VoiceDesign → Base 切换）时，借此触发 ``LRU=1`` 的 bundle_cache 驱逐前一个
    bundle、释放 VRAM。``family`` 等字段保持不变，确保仍走相同 runtime。
    """
    if isinstance(params, SimpleNamespace):
        base_dict = dict(params.__dict__)
    elif hasattr(params, "model_dump") and callable(getattr(params, "model_dump")):
        try:
            base_dict = dict(params.model_dump())
        except Exception:
            base_dict = dict(getattr(params, "__dict__", {}) or {})
    else:
        base_dict = dict(getattr(params, "__dict__", {}) or {})
    base_dict["load_name"] = new_load_name
    return SimpleNamespace(**base_dict)


class AudioInferrer(BaseInferrer):
    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.inference_config = load_inference_config(service_id=service_id)
        self.result_handler: Optional[ResultHandler] = None
        # 与 image/video/mini 一致：LRU=1 + TTL。无论是否 fixed_model，音频进程
        # 同一时刻只保留一个 bundle；key 变化会先驱逐旧 bundle。
        self.bundle_cache: PipelineCache = PipelineCache(
            ttl_seconds=getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0),
            logger=logger,
            release_fn=self._release_bundle,
        )
        self._session_runtime: Optional[AudioSessionRuntime] = None
        self._audio_backend: str = "transformers"
        self._supported_models: list[str] = []
        self._capabilities: list[str] = []
        self._fixed_model: Optional[str] = None
        self._fixed_family: Optional[str] = None
        # TTS 模型实例通常不是并发安全的。task 通道和 session.tts 通道共享
        # bundle/model 时必须串行生成，否则可能出现语音流和落盘文件串音。
        self._audio_generation_lock = asyncio.Lock()

    async def initialize(self):
        await super().initialize()
        self.result_handler = ResultHandler(
            ws_client=self.ws_client,
            storage_base_path=self.inference_config.outputs_dir,
            inference_config=self.inference_config,
        )
        try:
            self.bundle_cache.start()
        except Exception:
            logger.warning("audio bundle_cache.start failed", exc_info=True)
        service_runtime_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
        self._audio_backend = resolve_audio_backend(service_runtime_cfg, default="transformers")
        self._fixed_model = str(service_runtime_cfg.get("fixed_model") or "").strip() or None
        self._fixed_family = (
            str(service_runtime_cfg.get("fixed_family") or "").strip() or None
        )
        self._supported_models = self._resolve_supported_models(service_runtime_cfg)
        self._capabilities = self._resolve_capabilities(service_runtime_cfg)
        self._enforce_fixed_model_consistency()
        if self._fixed_model is not None:
            logger.info(
                "Audio inferrer fixed_model=%s fixed_family=%s "
                "(task load_name/family will be coerced to these values)",
                self._fixed_model,
                self._fixed_family,
            )
        if self._ws_transport is not None:
            self._session_runtime = AudioSessionRuntime(
                sender=self._ws_transport.send_message,
                backend=self._audio_backend,
                fixed_family=self._fixed_family,
                streaming_session_factory=(
                    self._create_realtime_streaming_session if self._audio_backend == "vllm" else None
                ),
                tts_engine_factory=self._create_session_tts_engine,
                stream_lock=self._audio_generation_lock,
            )
        logger.info(
            "Audio inferrer initialized backend=%s supported_models=%s capabilities=%s",
            self._audio_backend,
            self._supported_models,
            self._capabilities,
        )

    def _resolve_supported_models(self, service_runtime_cfg: Dict[str, Any]) -> list[str]:
        raw_models = service_runtime_cfg.get("supported_models")
        if not isinstance(raw_models, list):
            raise RuntimeError(
                f"Audio service '{self.service_id}' requires config.supported_models as a non-empty list"
            )

        supported_models = [str(item).strip() for item in raw_models if str(item).strip()]
        supported_models = list(dict.fromkeys(supported_models))
        if not supported_models:
            raise RuntimeError(
                f"Audio service '{self.service_id}' requires at least one configured supported model"
            )
        return supported_models

    def _resolve_capabilities(self, service_runtime_cfg: Dict[str, Any]) -> list[str]:
        """读取并规范 ``config.capabilities``。

        语义：本实例在 audio 大类下对外提供的子能力（``tts`` / ``asr`` 等）。
        audio 服务**必填**；去重后保持小写；空列表或非 list 直接报错——由上层
        dispatch 在 pin 路径下按 capability 过滤候选，保证 TTS 请求不会被随机
        打到 ASR 服务（反之亦然）。能力值与 ``fixed_family`` / runtime 的
        一致性由 ``_create_session_tts_engine`` 的 runtime 白名单兜底，这里不做
        交叉映射，避免在代码里再落一份"能力→runtime"表。
        """
        raw = service_runtime_cfg.get("capabilities")
        if not isinstance(raw, list):
            raise RuntimeError(
                f"Audio service '{self.service_id}' requires config.capabilities as a non-empty list "
                "(e.g. ['tts'] or ['asr'])"
            )

        capabilities = [str(item).strip().lower() for item in raw if str(item).strip()]
        capabilities = list(dict.fromkeys(capabilities))
        if not capabilities:
            raise RuntimeError(
                f"Audio service '{self.service_id}' requires at least one non-empty capability "
                "(e.g. ['tts'] or ['asr'])"
            )
        return capabilities

    def _enforce_fixed_model_consistency(self) -> None:
        """Validate `fixed_model` / `fixed_family` / `supported_models` relationships.

        三种合法状态：

          1) 都为空——完全无 pin、无服务级默认；调用方必须显式传 ``family``，
             否则 ``resolve_audio_runtime`` 会报错。
          2) 仅 ``fixed_family``——声明本服务的"默认 family"，但不 pin
             具体权重；request/session 没传 ``family`` 时由 inferrer 兜底填上，
             ``load_name`` 仍由调用方决定（或被 handler 默认规则补）。这种模式下：
                - dispatch 仍按非 pin 的 ``supported_models`` 收窄候选；
                - ``_coerce_params_for_pin`` 只填空，不覆盖调用方显式传的 ``family``。
          3) 都设置——传统 pin 模式：dispatch 把 ``fixed_model`` 当作"当前服务的唯一
             可用模型"，``_coerce_params_for_pin`` 硬覆盖请求里的 ``load_name`` 与
             ``family``。pin 模式下 ``fixed_model`` 必须出现在 ``supported_models`` 中。

        非法：``fixed_model`` 设置但 ``fixed_family`` 为空——pin 模式必须知道
        runtime 选哪个，缺 class 没法决定。

        注意：pin 模式下不会改写 ``supported_models``——静态能力清单保持原样，
        "当前实际只服务 fixed_model" 由 dispatch 层按 ``fixed_model`` 再次收窄。
        """
        has_model = bool(self._fixed_model)
        has_class = bool(self._fixed_family)
        if has_model and not has_class:
            raise RuntimeError(
                f"Audio service '{self.service_id}' inconsistent pin config: "
                f"fixed_model={self._fixed_model!r} requires fixed_family to also be set "
                "(pin mode must know which runtime to use)."
            )
        if not has_model:
            # 都为空 或 仅 fixed_family——都合法，跳过对 supported_models 的 pin 检查。
            return
        if self._fixed_model not in self._supported_models:
            raise RuntimeError(
                f"Audio service '{self.service_id}' fixed_model={self._fixed_model!r} "
                f"is not listed in supported_models={self._supported_models}. "
                "supported_models must declare fixed_model as part of the class capability."
            )

    def _coerce_params_for_pin(self, params: Any) -> Any:
        """按服务 pin / 默认 class 配置改写请求的 load_name / family。

        三种行为分支（与 ``_enforce_fixed_model_consistency`` 三种合法状态对齐）：

          - 都未配置：原样返回，调用方爱传啥传啥；
          - 仅 ``fixed_family``：当请求的 ``family`` 为空时回填它，
            ``load_name`` 不动（保留请求空值，让 handler 默认规则在更下游决定权重）；
          - pin 模式（两者都设置）：硬覆盖 ``load_name`` 和 ``family``，
            请求传的值失效（这是 pin 的承诺，调用方不能绕过）。

        所有情况都不修改原始对象——必要时返回带覆盖字段的新 ``SimpleNamespace``。
        """
        if not self._fixed_family:
            return params

        incoming_name = str(getattr(params, "load_name", None) or "").strip()
        incoming_class = str(getattr(params, "family", None) or "").strip()

        overrides: Dict[str, Any] = {}
        if self._fixed_model:
            # pin 模式：硬覆盖
            if (
                incoming_name == self._fixed_model
                and incoming_class.casefold() == self._fixed_family.casefold()
            ):
                return params
            if incoming_name or incoming_class:
                logger.info(
                    "Audio pin mode: coercing task load_name=%r family=%r to fixed_model=%r fixed_family=%r",
                    incoming_name,
                    incoming_class,
                    self._fixed_model,
                    self._fixed_family,
                )
            overrides["load_name"] = self._fixed_model
            overrides["family"] = self._fixed_family
        else:
            # 仅 fixed_family：缺 class 才补，不动 load_name
            if incoming_class:
                return params
            overrides["family"] = self._fixed_family
            logger.debug(
                "Audio service '%s': filling missing family with service default %r "
                "(no fixed_model pinned)",
                self.service_id,
                self._fixed_family,
            )

        if isinstance(params, SimpleNamespace):
            base_dict = dict(params.__dict__)
        elif hasattr(params, "model_dump") and callable(getattr(params, "model_dump")):
            try:
                base_dict = dict(params.model_dump())
            except Exception:
                base_dict = dict(getattr(params, "__dict__", {}) or {})
        else:
            base_dict = dict(getattr(params, "__dict__", {}) or {})
        base_dict.update(overrides)
        return SimpleNamespace(**base_dict)

    def _merge_service_runtime_cfg(self, params: Any) -> Any:
        """Merge service-level config.runtime into request model_cfg.runtime.

        任务通道不像 session.open 那样提前合并服务 runtime；这里统一补齐，
        让 qwen_asr.yaml 中的 forced_aligner / allow_remote_assets 等配置对普通
        task 请求也生效。请求里的 model_cfg.runtime 仍保留最高优先级。
        """
        service_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
        service_runtime_cfg = (
            service_cfg.get("runtime") if isinstance(service_cfg.get("runtime"), dict) else {}
        )
        if not service_runtime_cfg:
            return params

        if isinstance(params, SimpleNamespace):
            base_dict = dict(params.__dict__)
        elif hasattr(params, "model_dump") and callable(getattr(params, "model_dump")):
            try:
                base_dict = dict(params.model_dump())
            except Exception:
                base_dict = dict(getattr(params, "__dict__", {}) or {})
        else:
            base_dict = dict(getattr(params, "__dict__", {}) or {})

        model_cfg = base_dict.get("model_cfg")
        model_cfg = dict(model_cfg) if isinstance(model_cfg, dict) else {}
        request_runtime_cfg = model_cfg.get("runtime")
        request_runtime_cfg = dict(request_runtime_cfg) if isinstance(request_runtime_cfg, dict) else {}
        model_cfg["runtime"] = {
            **dict(service_runtime_cfg),
            **request_runtime_cfg,
        }
        base_dict["model_cfg"] = model_cfg
        return SimpleNamespace(**base_dict)

    async def cleanup(self):
        try:
            await self.bundle_cache.stop()
        except Exception:
            logger.warning("audio bundle_cache.stop failed", exc_info=True)
        await self._empty_device_cache()
        await super().cleanup()

    async def _release_bundle(self, bundle: Any) -> None:
        """Release one audio bundle when evicted from the LRU=1 cache."""
        if bundle is None:
            return
        model = bundle.get("model") if isinstance(bundle, dict) else None

        if model is not None:
            stop_fn = getattr(model, "stop_zmq_tasks", None)
            if callable(stop_fn):
                try:
                    await self._call_release_hook(stop_fn)
                except Exception:
                    logger.warning("Failed stopping audio model ZMQ tasks", exc_info=True)

            shutdown_fn = getattr(model, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    await self._call_release_hook(shutdown_fn)
                except Exception:
                    logger.warning("Failed shutting down audio model", exc_info=True)

        bundle_shutdown = getattr(bundle, "shutdown", None)
        if callable(bundle_shutdown):
            try:
                await self._call_release_hook(bundle_shutdown)
            except Exception:
                logger.warning("Failed shutting down audio bundle", exc_info=True)

        if isinstance(bundle, dict):
            try:
                bundle.clear()
            except Exception:
                pass
        await self._empty_device_cache()

    async def _call_release_hook(self, fn: Any) -> None:
        if inspect.iscoroutinefunction(fn):
            await fn()
            return
        result = await self.run_blocking(fn)
        if inspect.isawaitable(result):
            await result

    async def _empty_device_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    async def _release_bundle_use(
        self,
        cache_key: str,
        bundle: Dict[str, Any],
        cache_enabled: bool,
    ) -> None:
        if cache_enabled:
            await self.bundle_cache.release_use(key=cache_key)
            return
        await self._release_bundle(bundle)

    async def _on_ws_disconnect(self, reason: str):
        await super()._on_ws_disconnect(reason)

    async def _after_ws_connected(self):
        await super()._after_ws_connected()
        if self._session_runtime is not None:
            await self._session_runtime.register_service(
                service_type="audio",
                supported_models=self._supported_models,
                capabilities=self._capabilities,
                fixed_model=self._fixed_model,
                fixed_family=self._fixed_family,
            )

    async def _on_session_message(self, message: Dict[str, Any]) -> bool:
        if self._session_runtime is None:
            return False
        return await self._session_runtime.handle_message(message)

    async def _check_cancelled(self, task_id: str, stage: str) -> bool:
        if self.task_processor and self.task_processor.is_task_cancelled(task_id):
            logger.info(f"Audio task {task_id} was cancelled {stage}")
            if self.ws_client and self.ws_client.is_connected():
                await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return True
        return False

    async def _send_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
        *,
        progress: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        if not self.ws_client:
            return
        payload = {"task_id": task_id, "status": status}
        if progress is not None:
            payload["progress"] = max(0, min(100, int(progress)))
        if message:
            payload["message"] = message
        if status == "completed":
            payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        if error:
            payload["error"] = error
        await self.ws_client.send_task_status(**payload)

    async def _send_stream_event(self, message: Dict[str, Any]) -> bool:
        if not self.ws_client or not hasattr(self.ws_client, "send_stream_event"):
            raise RuntimeError("stream egress is not available for audio service")
        return await self.ws_client.send_stream_event(message)

    async def _get_realtime_qwen_asr_bundle(self, state: AudioSessionState) -> Dict[str, Any]:
        bundle, _, _ = await self._get_realtime_qwen_asr_bundle_with_key(state)
        return bundle

    async def _get_realtime_qwen_asr_bundle_with_key(
        self, state: AudioSessionState
    ) -> tuple[Dict[str, Any], str, bool]:
        effective_load_name = state.load_name or self._fixed_model or ""
        effective_family = state.family or self._fixed_family or ""
        if not effective_load_name:
            raise RuntimeError("realtime ASR session_open is missing load_name")
        if not effective_family:
            raise RuntimeError("realtime ASR session_open is missing family")
        if resolve_audio_runtime(SimpleNamespace(family=effective_family)) != "qwen_asr":
            raise RuntimeError(
                f"realtime ASR currently supports only Qwen-asr, got family={effective_family}"
            )

        service_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
        runtime_cfg = service_cfg.get("runtime") if isinstance(service_cfg.get("runtime"), dict) else {}
        merged_model_cfg = dict(state.model_cfg or {})
        if not isinstance(merged_model_cfg.get("runtime"), dict):
            merged_model_cfg["runtime"] = {}
        merged_model_cfg["runtime"] = {
            **runtime_cfg,
            **dict(merged_model_cfg.get("runtime") or {}),
        }
        dummy_params = self._coerce_params_for_pin(
            SimpleNamespace(
                load_name=effective_load_name,
                family=effective_family,
                timestamps=False,
                model_cfg=merged_model_cfg,
            )
        )
        model_ref = resolve_audio_model_ref(
            dummy_params,
            models_dir=getattr(self.inference_config, "models_dir", None),
            weights_dir=getattr(self.inference_config, "weights_dir", None),
            fixed_model=self._fixed_model,
        )
        policy = resolve_audio_runtime_policy(dummy_params, audio_mode="asr")
        bundle_options = resolve_qwen_asr_bundle_options(
            dummy_params,
            models_dir=getattr(self.inference_config, "models_dir", None),
            weights_dir=getattr(self.inference_config, "weights_dir", None),
        )
        vllm_options = resolve_qwen_asr_vllm_options(service_cfg)
        cache_key = _stable_json_fingerprint(
            {
                "runtime": "qwen_asr",
                "mode": "realtime_asr",
                "model_ref": model_ref,
                "policy": f"backend=vllm|{policy.cache_key}|{bundle_options.cache_key}",
            }
        )

        async def _create_bundle() -> Dict[str, Any]:
            return await self.run_blocking(
                load_vllm_asr_bundle,
                model_ref,
                policy,
                bundle_options,
                vllm_options,
            )

        bundle, hit = await self.bundle_cache.acquire(key=cache_key, create_fn=_create_bundle)
        logger.info("[audio-cache] realtime ASR %s key=%s", "HIT" if hit else "MISS", cache_key[:120])
        return bundle, cache_key, self.bundle_cache.enabled()

    async def _create_realtime_streaming_session(self, state: AudioSessionState) -> Any:
        bundle, cache_key, cache_enabled = await self._get_realtime_qwen_asr_bundle_with_key(state)
        session = create_vllm_streaming_session(bundle)

        async def _release() -> None:
            await self._release_bundle_use(cache_key, bundle, cache_enabled)

        return _ReleasingStreamingSession(session, _release)

    def _voxcpm_speaker_preset_overrides(self) -> Dict[str, Any]:
        """从共享 `config/tts_speakers.json` 读取 VoxCPM 参考音色映射。

        返回的 dict 形态直接适配 `VoxCPMTtsEngine` / `VoxCPMTtsHandler` 的 kwargs；
        若未配置则为空 dict。
        """
        out: Dict[str, Any] = {}
        presets = voxcpm_speaker_presets()
        default_speaker = get_default_speaker("voxcpm")
        if presets:
            out["speaker_presets"] = presets
        if default_speaker:
            out["default_speaker"] = default_speaker.strip()
        return out

    async def _create_session_tts_engine(self, state: AudioSessionState) -> TtsEngine:
        """为 session.tts.* 构造一个 TtsEngine 实例（同一 session 内跨 request 复用）。

        ``load_name`` 在 qwen_tts runtime 下允许 session.open 留空——engine 每次
        synthesize 时按 ``voice_cfg`` 自己挑权重并通过 bundle_loader 的 ``load_name``
        kwarg override 触发加载；voxcpm 没有这条默认规则，仍要求 session.open 带上。
        """
        effective_load_name = state.load_name or self._fixed_model or ""
        effective_family = state.family or self._fixed_family or ""
        if not effective_family:
            raise RuntimeError("session.open for TTS requires family")

        service_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
        service_runtime_cfg = (
            service_cfg.get("runtime") if isinstance(service_cfg.get("runtime"), dict) else {}
        )

        runtime_name = resolve_audio_runtime(SimpleNamespace(family=effective_family))
        if runtime_name not in ("qwen_tts", "voxcpm"):
            raise RuntimeError(
                f"session.tts unsupported runtime={runtime_name} (family={effective_family})"
            )
        if not effective_load_name and runtime_name != "qwen_tts":
            raise RuntimeError(f"session.open for TTS requires load_name (runtime={runtime_name})")

        base_model_cfg = dict(state.model_cfg or {})
        if not isinstance(base_model_cfg.get("runtime"), dict):
            base_model_cfg["runtime"] = {}
        base_model_cfg["runtime"] = {
            **service_runtime_cfg,
            **dict(base_model_cfg.get("runtime") or {}),
        }

        used_bundles: Dict[str, tuple[Dict[str, Any], bool]] = {}

        def _bundle_loader_factory(runtime_name: str):
            async def _loader(mode: str, *, load_name: Optional[str] = None) -> Dict[str, Any]:
                name_to_use = (load_name or "").strip() or effective_load_name
                if not name_to_use:
                    raise RuntimeError(
                        f"qwen-tts session.tts: voice_cfg did not resolve a weight (mode={mode})"
                    )
                dummy_params = self._coerce_params_for_pin(
                    SimpleNamespace(
                        load_name=name_to_use,
                        family=effective_family,
                        model_cfg=dict(base_model_cfg),
                        audio_mode=mode,
                    )
                )
                bundle, cache_key, cache_enabled = await self._get_bundle_with_key(dummy_params, mode)
                used_bundles[cache_key] = (bundle, cache_enabled)
                return bundle

            return _loader

        if runtime_name == "qwen_tts":
            engine = QwenTtsEngine(
                bundle_loader=_bundle_loader_factory("qwen_tts"),
                logger=logger,
            )
        else:
            # voxcpm：session 通道默认用 realtime_tts 模式（兼容非流式退化）
            engine = VoxCPMTtsEngine(
                audio_mode="realtime_tts",
                bundle_loader=_bundle_loader_factory("voxcpm"),
                logger=logger,
                **self._voxcpm_speaker_preset_overrides(),
            )
        return _ReleasingTtsEngine(engine, used_bundles, self._release_bundle_use)

    async def _get_bundle(self, params: InferenceRequestParams, audio_mode: str) -> Dict[str, Any]:
        bundle, _, _ = await self._get_bundle_with_key(params, audio_mode)
        return bundle

    async def _get_bundle_with_key(
        self, params: InferenceRequestParams, audio_mode: str
    ) -> tuple[Dict[str, Any], str, bool]:
        params = self._coerce_params_for_pin(params)
        params = self._merge_service_runtime_cfg(params)
        runtime = resolve_audio_runtime(params)
        model_ref = resolve_audio_model_ref(
            params,
            models_dir=getattr(self.inference_config, "models_dir", None),
            weights_dir=getattr(self.inference_config, "weights_dir", None),
            fixed_model=self._fixed_model,
        )
        # VoxCPM：`tts` 与 `realtime_tts` 共用同一 loader/权重；若 cache_key 区分 audio_mode，
        # 会先加载 tts 再为 realtime_tts 再加载一份，显存约翻倍易 OOM。
        if runtime == "voxcpm" and audio_mode in ("tts", "realtime_tts"):
            policy = resolve_audio_runtime_policy(params, audio_mode="tts")
            bundle_cache_mode = "tts"
        else:
            policy = resolve_audio_runtime_policy(params, audio_mode=audio_mode)
            bundle_cache_mode = audio_mode
        bundle_variant_key = ""
        if runtime == "qwen_asr":
            bundle_options = resolve_qwen_asr_bundle_options(
                params,
                models_dir=getattr(self.inference_config, "models_dir", None),
                weights_dir=getattr(self.inference_config, "weights_dir", None),
            )
            bundle_variant_key = f"backend={self._audio_backend}|{bundle_options.cache_key}"
        elif runtime == "voxcpm":
            service_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
            runtime_cfg = service_cfg.get("runtime") if isinstance(service_cfg.get("runtime"), dict) else {}
            voxcpm_effective = merge_voxcpm_loader_runtime_cfg(runtime_cfg, backend=self._audio_backend)
            bundle_variant_key = f"backend={self._audio_backend}|voxrtc={_stable_json_fingerprint(voxcpm_effective)}"
        else:
            bundle_options = None

        cache_policy_key = f"{policy.cache_key}|{bundle_variant_key}" if bundle_variant_key else policy.cache_key
        cache_key = _stable_json_fingerprint(
            {
                "runtime": runtime,
                "mode": bundle_cache_mode,
                "model_ref": model_ref,
                "policy": cache_policy_key,
            }
        )

        loader = {
            "voxcpm": {
                "tts": load_voxcpm_tts_bundle,
                "realtime_tts": load_voxcpm_realtime_bundle,
            },
            "qwen_tts": {
                "tts": load_qwen_tts_bundle,
            },
            "qwen_asr": {
                "asr": load_qwen_asr_bundle,
            },
        }.get(runtime, {}).get(audio_mode)
        if loader is None:
            raise ValueError(f"Unsupported audio runtime/mode combination: runtime={runtime} audio_mode={audio_mode}")

        async def _create_bundle() -> Dict[str, Any]:
            if runtime == "qwen_asr" and self._audio_backend == "vllm":
                vllm_options = resolve_qwen_asr_vllm_options(getattr(self.config, "config", {}) if self.config else {})
                return await self.run_blocking(load_vllm_asr_bundle, model_ref, policy, bundle_options, vllm_options)
            elif runtime == "qwen_asr":
                return await self.run_blocking(loader, model_ref, policy, bundle_options)
            elif runtime == "qwen_tts":
                service_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
                runtime_cfg = service_cfg.get("runtime") if isinstance(service_cfg.get("runtime"), dict) else {}
                qwen_tts_effective = merge_qwen_tts_loader_runtime_cfg(
                    runtime_cfg, backend=self._audio_backend
                )
                return await self.run_blocking(
                    loader,
                    model_ref,
                    policy,
                    self._audio_backend,
                    dict(qwen_tts_effective),
                )
            elif runtime == "voxcpm":
                service_cfg = getattr(self.config, "config", {}) if self.config is not None else {}
                runtime_cfg = service_cfg.get("runtime") if isinstance(service_cfg.get("runtime"), dict) else {}
                voxcpm_effective = merge_voxcpm_loader_runtime_cfg(runtime_cfg, backend=self._audio_backend)
                return await self.run_blocking(
                    loader,
                    model_ref,
                    policy,
                    self._audio_backend,
                    dict(voxcpm_effective),
                )
            else:
                raise RuntimeError(f"Unhandled audio bundle loader for runtime={runtime!r}")

        bundle, hit = await self.bundle_cache.acquire(key=cache_key, create_fn=_create_bundle)
        logger.info("[audio-cache] %s key=%s", "HIT" if hit else "MISS", cache_key[:120])
        # 冷加载（cache MISS）后对 nano_vllm 流式 bundle 跑一次最小 generate，
        # 把 start_zmq_tasks + 首批 codec frame 的 ~5s warmup 摊到本次 bundle 创建期，
        # 后续真实请求 TTFB 直接降到 100ms 量级。失败仅 warn，不阻塞调用方。
        if not hit and runtime == "qwen_tts":
            await prewarm_qwen_tts_bundle(bundle, logger_=logger)
        return bundle, cache_key, self.bundle_cache.enabled()

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        task_id = params.task_id
        logger.info(f"Starting audio inference for task: {task_id}")
        try:
            data = params.model_dump() if hasattr(params, "model_dump") else params
        except Exception:
            data = params
        print_info(data, prefix=f"[audio][task_id={task_id}] ")

        if await self._check_cancelled(task_id, "before inference"):
            return None
        if not self.result_handler:
            raise RuntimeError("ResultHandler not initialized")

        audio_mode = str(getattr(params, "audio_mode", "tts") or "tts").strip().lower()
        effective_params = self._coerce_params_for_pin(params)
        runtime = resolve_audio_runtime(effective_params)
        used_bundles: Dict[str, tuple[Dict[str, Any], bool]] = {}

        async def _load_tracked_bundle(
            mode: str,
            *,
            load_name: Optional[str] = None,
        ) -> Dict[str, Any]:
            """加载本任务用到的 bundle，并登记到 ``used_bundles``。

            ``load_name``（可选）：当 handler 需要在同一个推理任务里串行加载多个权重
            （如 qwen-tts drama 的 VoiceDesign → Base 切换）时，可传入覆盖名。底层会
            构造一份临时 params（仅替换 load_name），通过 ``_get_bundle_with_key``
            生成新的 cache_key；``LRU=1`` 的 ``bundle_cache`` 会立即驱逐旧 bundle，
            从而释放上一段权重的 VRAM。pin 模式下此覆盖会被 ``_coerce_params_for_pin``
            合规化（通常 pin 模式只跑单个 fixed_model，不应走多权重切换路径）。
            """
            params_to_use = effective_params
            override = (load_name or "").strip() if load_name is not None else ""
            if override and override != str(getattr(effective_params, "load_name", "") or "").strip():
                params_to_use = _params_with_load_name_override(effective_params, override)
            bundle, cache_key, cache_enabled = await self._get_bundle_with_key(params_to_use, mode)
            used_bundles[cache_key] = (bundle, cache_enabled)
            return bundle

        common_kwargs = dict(
            result_handler=self.result_handler,
            service_id=self.service_id,
            logger=logger,
            bundle_loader=_load_tracked_bundle,
            stream_sender=self._send_stream_event,
            check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
            status_sender=self._send_task_status,
        )

        try:
            if runtime == "voxcpm":
                if audio_mode == "asr":
                    raise ValueError("VoxCPM runtime does not support audio_mode=asr")
                handler = VoxCPMTtsHandler(
                    audio_mode=audio_mode,
                    **common_kwargs,
                    **self._voxcpm_speaker_preset_overrides(),
                )
            elif runtime == "qwen_tts":
                if audio_mode != "tts":
                    raise ValueError("Qwen-tts runtime currently supports only audio_mode=tts")
                handler = QwenTtsHandler(audio_mode=audio_mode, **common_kwargs)
            elif runtime == "qwen_asr":
                if audio_mode != "asr":
                    raise ValueError("Qwen-asr runtime currently supports only audio_mode=asr")
                handler = QwenAsrHandler(**common_kwargs)
            else:
                raise ValueError(f"Unsupported audio runtime={runtime}")

            # 与 session.tts 共用同一把锁，避免同一个 TTS bundle/model 并发生成时
            # 把聊天语音流和 task 音频文件产物混在一起。ASR task 保持原路径。
            if runtime in {"voxcpm", "qwen_tts"} and audio_mode != "asr":
                async with self._audio_generation_lock:
                    await handler.run(effective_params, task_id=task_id)
            else:
                await handler.run(effective_params, task_id=task_id)

            if not (self.task_processor and self.task_processor.is_task_cancelled(task_id)):
                await self._send_task_status(task_id, "completed")
        except Exception as e:
            logger.error(f"Error in audio inference for task {task_id}: {e}", exc_info=True)
            if not (self.task_processor and self.task_processor.is_task_cancelled(task_id)):
                await self._send_task_status(task_id, "failed", error=str(e))
            raise
        finally:
            for key, (bundle, cache_enabled) in list(used_bundles.items()):
                await self._release_bundle_use(key, bundle, cache_enabled)

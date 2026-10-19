"""Prompt 扩写推理器：注册、排队、取消，以及按 TTL 决定权重是否常驻。"""

from __future__ import annotations

import time
from threading import Event
from typing import Any

from common.base_inferrer import BaseInferrer
from common.config_loader import load_inference_config
from common.logger import get_logger
from common.pipeline_cache import PipelineCache
from common.result_handler import ResultHandler
from common.service_register import build_service_register_message
from common.task_cancel import TaskCancelledError
from schemas import InferenceRequestParams

from prompt.runtime.request import (
    DEFAULT_MAX_NEW_TOKENS,
    MAX_IMAGES,
    PromptRequestError,
    resolve_max_new_tokens,
    resolve_upsample_input,
)
from prompt.runtime.upsampler import (
    MAX_IMAGE_BYTES,
    MODEL_NAME,
    load_image,
    load_upsampler_bundle,
    release_upsampler_bundle,
    resolve_model_paths,
    upsample,
)

logger = get_logger(__name__)


class PromptInferrer(BaseInferrer):
    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.inference_config = load_inference_config(service_id=service_id)
        self.result_handler: ResultHandler | None = None
        self.bundle_cache = PipelineCache(
            ttl_seconds=getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0),
            logger=logger,
            release_fn=self._release_bundle,
        )
        self._fixed_model = MODEL_NAME
        self._device = "cuda"
        self._dtype_name = "bfloat16"
        self._max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self._max_images = MAX_IMAGES
        self._max_image_bytes = MAX_IMAGE_BYTES

    async def initialize(self) -> None:
        await super().initialize()
        runtime = self._runtime_cfg()
        configured = str(self._service_model_cfg().get("fixed_model") or "").strip()
        self._fixed_model = configured or MODEL_NAME
        self._device = str(runtime.get("device") or "cuda").strip() or "cuda"
        self._dtype_name = str(runtime.get("dtype") or "bfloat16").strip() or "bfloat16"
        self._max_new_tokens = resolve_max_new_tokens(
            runtime.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            default=DEFAULT_MAX_NEW_TOKENS,
            cap=DEFAULT_MAX_NEW_TOKENS,
        )
        try:
            self._max_images = max(1, int(runtime.get("max_images") or MAX_IMAGES))
        except (TypeError, ValueError):
            self._max_images = MAX_IMAGES
        try:
            self._max_image_bytes = max(1, int(runtime.get("max_image_bytes") or MAX_IMAGE_BYTES))
        except (TypeError, ValueError):
            self._max_image_bytes = MAX_IMAGE_BYTES

        self.result_handler = ResultHandler(
            ws_client=self.ws_client,
            storage_base_path=self.inference_config.outputs_dir,
            inference_config=self.inference_config,
        )
        try:
            self.bundle_cache.start()
        except Exception:
            logger.warning("prompt bundle_cache.start failed", exc_info=True)
        logger.info(
            "Prompt inferrer initialized fixed_model=%s pipeline_cache_ttl_seconds=%s resident=%s",
            self._fixed_model,
            self.bundle_cache.ttl_seconds,
            self.bundle_cache.enabled(),
        )

    async def _before_backend_registration(self) -> None:
        await super()._before_backend_registration()
        if not self.bundle_cache.enabled():
            logger.info(
                "Prompt weights will be released after each task (pipeline_cache_ttl_seconds=%s)",
                self.bundle_cache.ttl_seconds,
            )
            return
        logger.info(
            "Prompt weights will stay resident until idle for %ss",
            self.bundle_cache.ttl_seconds,
        )
        await self._warmup_load()

    def _build_service_register_message(self) -> dict[str, Any]:
        service_cfg = self._service_model_cfg()
        capabilities = service_cfg.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            capabilities = ["prompt.upsample"]
        return build_service_register_message(
            service_type=self._resolve_service_type(),
            service_config=service_cfg,
            supports_task=True,
            supported_models=[self._fixed_model],
            capabilities=capabilities,
            fixed_model=self._fixed_model,
            queue_length=self.get_queue_length(),
        )

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        task_id = str(params.task_id or "").strip()
        requested = str(getattr(params, "load_name", "") or "").strip()
        load_name = self._fixed_model
        if requested and requested != load_name:
            logger.info(
                "Prompt task load_name=%s ignored; fixed_model=%s task_id=%s",
                requested,
                load_name,
                task_id,
            )

        if self._check_cancelled(task_id):
            raise TaskCancelledError(task_id, "before prompt upsample")

        extract = getattr(params, "extract", None)
        extract = dict(extract) if isinstance(extract, dict) else {}
        messages = extract.get("messages")
        if not isinstance(messages, list):
            messages = None
        text, sources = resolve_upsample_input(
            prompt=str(getattr(params, "prompt", "") or ""),
            image_sources=list(getattr(params, "tpl_list", None) or []),
            messages=messages,
            max_images=self._max_images,
        )
        max_new_tokens = resolve_max_new_tokens(
            _first_present(extract, "max_new_tokens", "max_completion_tokens", "max_tokens"),
            default=self._max_new_tokens,
            cap=self._max_new_tokens,
        )
        cache_key = self._cache_key(load_name)
        bundle = None
        failed = False
        try:
            bundle, cache_hit = await self.bundle_cache.acquire(
                key=cache_key,
                create_fn=lambda: self._load_bundle(load_name),
            )
            logger.info(
                "[prompt] bundle %s key=%s task_id=%s images=%s max_new_tokens=%s",
                "hit" if cache_hit else "load",
                cache_key,
                task_id,
                len(sources),
                max_new_tokens,
            )
            if self._check_cancelled(task_id):
                raise TaskCancelledError(task_id, "after prompt bundle")

            images = []
            for index, source in enumerate(sources):
                if self._check_cancelled(task_id):
                    raise TaskCancelledError(task_id, f"before image {index}")
                images.append(
                    await self.run_blocking(load_image, source, max_bytes=self._max_image_bytes)
                )

            if self._check_cancelled(task_id):
                raise TaskCancelledError(task_id, "before upsample")

            started = time.perf_counter()
            caption, prompt_tokens, completion_tokens, finish = await self.run_blocking(
                upsample,
                bundle,
                prompt=text,
                images=images,
                max_new_tokens=max_new_tokens,
                cancel_event=self._cancel_event(task_id),
                task_id=task_id,
            )
            elapsed = time.perf_counter() - started
            if self._check_cancelled(task_id):
                raise TaskCancelledError(task_id, "after upsample")

            logger.info(
                "[prompt] task=%s images=%s in=%s out=%s finish=%s %.2fs",
                task_id,
                len(images),
                prompt_tokens,
                completion_tokens,
                finish,
                elapsed,
            )
            if self.result_handler is None:
                raise RuntimeError("prompt result handler is not initialized")
            params_for_send = params.model_copy(update={"file_type": "txt", "load_name": load_name})
            await self.result_handler.process_single_result(
                file_data=caption.encode("utf-8"),
                request_params=params_for_send,
                generate_time=elapsed,
                service_id=self.service_id,
                index=0,
                total=1,
                extra_message_fields={
                    "content": caption,
                    "finish_reason": finish,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            )
            if self.ws_client is not None:
                try:
                    await self.ws_client.send_task_status(task_id=task_id, status="completed")
                except Exception:
                    logger.warning("send_task_status completed failed for task=%s", task_id, exc_info=True)
        except TaskCancelledError:
            raise
        except PromptRequestError:
            raise
        except Exception:
            failed = True
            raise
        finally:
            if bundle is not None:
                await self._finish_bundle(bundle, cache_key=cache_key, failed=failed)
        return caption

    async def stop(self) -> None:
        try:
            await self.bundle_cache.stop()
        except Exception:
            logger.warning("prompt bundle_cache.stop failed", exc_info=True)
        await super().stop()

    async def _warmup_load(self) -> None:
        started = time.perf_counter()
        logger.info("Prompt startup warmup begin load_name=%s", self._fixed_model)
        cache_key = self._cache_key(self._fixed_model)
        bundle = None
        try:
            bundle, _ = await self.bundle_cache.acquire(
                key=cache_key,
                create_fn=lambda: self._load_bundle(self._fixed_model),
            )
        except Exception:
            logger.warning("Prompt startup warmup failed load_name=%s", self._fixed_model, exc_info=True)
            return
        finally:
            if bundle is not None:
                await self.bundle_cache.release_use(key=cache_key)
        logger.info(
            "Prompt startup warmup completed load_name=%s elapsed=%.2fs",
            self._fixed_model,
            time.perf_counter() - started,
        )

    async def _load_bundle(self, load_name: str):
        return await self.run_blocking(
            load_upsampler_bundle,
            models_dir=self.inference_config.models_dir,
            weights_dir=self.inference_config.weights_dir,
            load_name=load_name,
            device=self._device,
            dtype_name=self._dtype_name,
        )

    async def _release_bundle(self, bundle: Any) -> None:
        if bundle is None:
            return
        try:
            await self.run_blocking(release_upsampler_bundle, bundle)
        except Exception as exc:
            logger.warning("prompt bundle release raised: %s", exc, exc_info=True)

    async def _finish_bundle(self, bundle: Any, *, cache_key: str, failed: bool) -> None:
        if self.bundle_cache.enabled():
            if failed:
                await self.bundle_cache.evict(force=True)
                return
            await self.bundle_cache.release_use(key=cache_key)
            return
        await self._release_bundle(bundle)

    def _cache_key(self, load_name: str) -> str:
        try:
            pup, quant = resolve_model_paths(
                self.inference_config.models_dir,
                self.inference_config.weights_dir,
                load_name,
            )
        except Exception:
            pup, quant = load_name, ""
        return f"prompt|{pup}|{quant}|{self._device}|{self._dtype_name}"

    def _check_cancelled(self, task_id: str) -> bool:
        if self.task_processor is None:
            return False
        try:
            return bool(self.task_processor.is_task_cancelled(task_id))
        except Exception:
            return False

    def _cancel_event(self, task_id: str) -> Event:
        if self.task_processor is None:
            return Event()
        return self.task_processor.get_task_cancel_event(task_id)

    def _service_model_cfg(self) -> dict[str, Any]:
        if not self.config or not isinstance(self.config.config, dict):
            return {}
        return dict(self.config.config)

    def _runtime_cfg(self) -> dict[str, Any]:
        runtime = self._service_model_cfg().get("runtime")
        return dict(runtime) if isinstance(runtime, dict) else {}


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data.get(key) not in (None, ""):
            return data.get(key)
    return None

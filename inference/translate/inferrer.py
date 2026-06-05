from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from common.base_inferrer import BaseInferrer
from common.config_loader import load_inference_config
from common.logger import get_logger
from common.message_cache import MessageCache
from common.pipeline_cache import PipelineCache
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams
from text.runtime.common import deep_merge_dict

from translate.handlers.translategemma_handler import TranslateGemmaHandler

logger = get_logger(__name__)

_HANDLER_REGISTRY: Dict[str, str] = {
    "TranslateGemma": "translategemma",
}


def _get_nested_config_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    return default if current is None else current


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        yaml_mod = importlib.import_module("yaml")
        with path.open("r", encoding="utf-8") as f:
            data = yaml_mod.safe_load(f)
    except Exception as e:
        logger.warning("Failed to read config file %s: %s", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def _merged_app_config() -> Dict[str, Any]:
    config_dir = Path(__file__).resolve().parents[2] / "config"
    merged: Dict[str, Any] = {}
    for name in ("default.yaml", "app.yaml"):
        merged = deep_merge_dict(merged, _load_yaml_config(config_dir / name))
    return merged


def _resolve_default_translate_load_name() -> str:
    return str(
        _get_nested_config_value(
            _merged_app_config(),
            "agents.tools.translate.default_model_name",
            "",
        )
        or ""
    ).strip()


def _resolve_default_translate_family() -> str:
    return str(
        _get_nested_config_value(
            _merged_app_config(),
            "agents.tools.translate.default_family",
            "TranslateGemma",
        )
        or "TranslateGemma"
    ).strip()


class TranslateInferrer(BaseInferrer):
    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.result_handler: Optional[ResultHandler] = None
        self.message_cache: Optional[MessageCache] = None
        self.inference_config = load_inference_config(service_id=service_id)
        self.bundle_cache: PipelineCache = PipelineCache(
            ttl_seconds=getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0),
            logger=logger,
            release_fn=self._release_bundle,
        )
        self._handlers: Dict[str, Any] = {}
        self._fixed_model: Optional[str] = None
        self._fixed_family: Optional[str] = None

    async def _release_bundle(self, bundle: Any) -> None:
        if bundle is None:
            return
        shutdown = getattr(bundle, "shutdown", None)
        if callable(shutdown):
            try:
                await self.run_blocking(shutdown)
            except Exception as e:
                logger.warning("translate bundle shutdown raised: %s", e, exc_info=True)

    async def initialize(self):
        await super().initialize()
        service_cfg = self._service_model_cfg()
        self._fixed_model = str(service_cfg.get("fixed_model") or "").strip() or None
        self._fixed_family = str(service_cfg.get("fixed_family") or "").strip() or None
        if self._fixed_model:
            logger.info(
                "Translate inferrer fixed_model=%s fixed_family=%s",
                self._fixed_model,
                self._fixed_family or _resolve_default_translate_family(),
            )

        self.result_handler = ResultHandler(
            ws_client=self.ws_client,
            storage_base_path=self.inference_config.outputs_dir,
            inference_config=self.inference_config,
        )
        try:
            self.bundle_cache.start()
        except Exception:
            logger.warning("translate bundle_cache.start failed", exc_info=True)

        self._handlers = {
            "translategemma": TranslateGemmaHandler(
                inference_config=self.inference_config,
                bundle_cache=self.bundle_cache,
                result_handler=self.result_handler,
                service_id=self.service_id,
                logger=logger,
                run_blocking=self.run_blocking,
                check_cancelled=self._check_cancelled,
                service_model_cfg=service_cfg,
            ),
        }
        logger.info(
            "TranslateInferrer initialized registry=%s handlers=%s default_model=%s",
            _HANDLER_REGISTRY,
            list(self._handlers.keys()),
            self._effective_load_name(""),
        )

    def _service_model_cfg(self) -> Dict[str, Any]:
        if not self.config or not isinstance(self.config.config, dict):
            return {}
        return dict(self.config.config)

    def _effective_load_name(self, requested: str) -> str:
        if self._fixed_model:
            return self._fixed_model
        resolved = str(requested or "").strip()
        if resolved:
            return resolved
        return _resolve_default_translate_load_name()

    def _effective_family(self, requested: str) -> str:
        if self._fixed_family:
            return self._fixed_family
        resolved = str(requested or "").strip()
        if resolved:
            return resolved
        return _resolve_default_translate_family()

    def _check_cancelled(self, task_id: str) -> bool:
        if self.task_processor is None:
            return False
        try:
            return bool(self.task_processor.is_task_cancelled(task_id))
        except Exception:
            return False

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        task_id = params.task_id
        requested_load_name = str(getattr(params, "load_name", "") or "").strip()
        requested_family = str(getattr(params, "family", "") or "").strip()
        load_name = self._effective_load_name(requested_load_name)
        family = self._effective_family(requested_family)

        logger.info(
            "[translate] task received task_id=%s family=%s load_name=%s%s",
            task_id,
            family,
            load_name,
            f" (default/fixed override; request had load_name={requested_load_name!r})"
            if load_name != requested_load_name
            else "",
        )

        if self._check_cancelled(task_id):
            if self.ws_client:
                await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return None

        if not load_name:
            raise ValueError(
                "translate task requires load_name (or config.fixed_model / "
                "agents.tools.translate.default_model_name)"
            )
        if not family:
            raise ValueError(
                "translate task requires family (or config.fixed_family / "
                "agents.tools.translate.default_family)"
            )

        handler_id = _HANDLER_REGISTRY.get(family)
        if handler_id is None:
            raise ValueError(
                f"translate service does not support family={family!r}; "
                f"registered classes: {sorted(_HANDLER_REGISTRY.keys())}"
            )
        handler = self._handlers.get(handler_id)
        if handler is None:
            raise RuntimeError(f"handler_id={handler_id!r} is not instantiated")

        params = params.model_copy(update={"load_name": load_name, "family": family})
        await handler.handle(params)

        if self._check_cancelled(task_id):
            if self.ws_client:
                await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return None

        if self.ws_client:
            try:
                await self.ws_client.send_task_status(task_id=task_id, status="completed")
            except Exception:
                logger.warning("send_task_status completed failed for task=%s", task_id, exc_info=True)
        return None

    async def stop(self):
        try:
            await self.bundle_cache.stop()
        except Exception:
            logger.warning("translate bundle_cache.stop failed", exc_info=True)
        await super().stop()

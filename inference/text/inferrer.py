from __future__ import annotations

import asyncio
import copy
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from typing import Any, AsyncIterator, Dict, Optional

from common.base_inferrer import BaseInferrer
from common.config_loader import load_inference_config
from common.logger import get_logger
from schemas import InferenceRequestParams

from text.runtime.common import build_messages_from_prompt, deep_merge_dict, normalize_chat_messages
from text.runtime.runtime_resolver import (
    resolve_text_model_ref,
    resolve_text_runtime,
    resolve_text_runtime_policy,
)
from text.runtime.ollama_bridge import (
    abort_chat_request as abort_ollama_chat_request,
    generate_chat_text as generate_ollama_chat_text,
    load_ollama_text_bundle,
    shutdown_ollama_text_bundle,
    stream_chat_text as stream_ollama_chat_text,
)
from text.runtime.transformers_bridge import (
    abort_chat_request as abort_transformers_chat_request,
    generate_chat_text as generate_transformers_chat_text,
    load_transformers_text_bundle,
    shutdown_transformers_text_bundle,
    stream_chat_text as stream_transformers_chat_text,
)
from text.runtime.vllm_bridge import (
    abort_chat_request,
    generate_chat_text,
    load_vllm_text_bundle,
    shutdown_vllm_text_bundle,
    stream_chat_text,
)
from text.session_runtime import TextSessionRuntime

logger = get_logger(__name__)

_STREAM_STATS_KEYS = (
    "finish_reason",
    "prompt_tokens",
    "output_tokens",
    "ttft_seconds",
    "total_seconds",
    "decode_seconds",
    "tok_s_total",
    "tok_s_decode",
)


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


def _resolve_default_text_load_name() -> str:
    config_dir = Path(__file__).resolve().parents[2] / "config"
    merged: Dict[str, Any] = {}
    for name in ("default.yaml", "app.yaml"):
        merged = deep_merge_dict(merged, _load_yaml_config(config_dir / name))
    return str(_get_nested_config_value(merged, "agents.default_model", "") or "").strip()


class TextInferrer(BaseInferrer):
    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.inference_config = load_inference_config(service_id=service_id)
        self._session_runtime: Optional[TextSessionRuntime] = None
        self._model_cache: Dict[tuple[str, str, str], Any] = {}
        self._model_load_locks: Dict[tuple[str, str, str], asyncio.Lock] = {}
        self._fixed_model: Optional[str] = None
        self._startup_warmup_done = False

    async def initialize(self):
        await super().initialize()
        self._fixed_model = str(self._service_model_cfg().get("fixed_model") or "").strip() or None
        if self._fixed_model:
            logger.info(
                "Text inferrer fixed_model=%s (weights / ollama tag resolution ignores request load_name)",
                self._fixed_model,
            )
        if self._ws_transport is not None:
            self._session_runtime = TextSessionRuntime(
                sender=self._ws_transport.send_message,
                stream_text=self._stream_session_text,
                abort_request=self._abort_session_request,
            )
        logger.info("Text inferrer initialized")

    async def _after_ws_connected(self):
        await super()._after_ws_connected()
        if self._session_runtime is not None:
            await self._session_runtime.register_service(service_type="text")

    async def _before_backend_registration(self):
        await super()._before_backend_registration()
        await self._maybe_run_startup_warmup()

    async def _on_session_message(self, message: Dict[str, Any]) -> bool:
        if self._session_runtime is None:
            return False
        return await self._session_runtime.handle_message(message)

    async def _send_stream_event(self, message: Dict[str, Any]) -> bool:
        if not self.ws_client or not hasattr(self.ws_client, "send_stream_event"):
            raise RuntimeError("stream egress is not available for text service")
        return await self.ws_client.send_stream_event(message)

    def _service_model_cfg(self) -> Dict[str, Any]:
        if not self.config or not isinstance(self.config.config, dict):
            return {}
        return self.config.config

    def _effective_load_name(self, requested: str) -> str:
        if self._fixed_model:
            return self._fixed_model
        return str(requested or "").strip()

    async def _maybe_run_startup_warmup(self) -> None:
        if self._startup_warmup_done:
            return
        self._startup_warmup_done = True

        load_name = self._effective_load_name(_resolve_default_text_load_name())
        if not load_name:
            logger.warning("Text startup warmup skipped: configure agents.default_model or config.fixed_model")
            return

        t0 = time.perf_counter()
        logger.info("Text startup warmup begin load_name=%s", load_name)
        try:
            await self._generate_text(
                load_name=load_name,
                family="warmup",
                messages=build_messages_from_prompt("你好"),
                temperature=0.0,
                max_tokens=1,
            )
        except Exception as e:
            logger.warning("Text startup warmup failed load_name=%s: %s", load_name, e, exc_info=True)
        else:
            logger.info("Text startup warmup completed load_name=%s elapsed=%.2fs", load_name, time.perf_counter() - t0)

    def _build_request_spec(
        self,
        *,
        load_name: str,
        family: str,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> SimpleNamespace:
        """合并服务 config 与请求 ``runtime_config``；``config.runtime`` 仅来自服务 YAML。

        请求里的 ``runtime_config[\"runtime\"]`` 会被忽略（不参与 merge、不覆盖后端），
        与 audio/mini 推理服务一致；解析侧只读 ``spec.service_runtime``。
        """
        service_cfg = self._service_model_cfg()
        svc_rt = service_cfg.get("runtime")
        svc_rt = dict(svc_rt) if isinstance(svc_rt, dict) else {}
        request_cfg = dict(runtime_config or {})
        request_cfg.pop("runtime", None)
        merged_runtime_config = deep_merge_dict(service_cfg, request_cfg)
        merged_runtime_config["runtime"] = copy.deepcopy(svc_rt)
        return SimpleNamespace(
            load_name=load_name,
            family=family,
            runtime_config=merged_runtime_config,
            service_runtime=copy.deepcopy(svc_rt),
        )

    @staticmethod
    def _estimate_message_package_bytes(messages: list[dict[str, Any]]) -> int:
        try:
            payload = json.dumps(
                messages,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        except Exception:
            payload = str(messages)
        return len(payload.encode("utf-8"))

    @staticmethod
    def _extract_tool_names(tools: Optional[list[dict[str, Any]]]) -> list[str]:
        names: list[str] = []
        if not isinstance(tools, list):
            return names
        for item in tools:
            if not isinstance(item, dict):
                continue
            function = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = str(function.get("name") or item.get("name") or "").strip()
            if name:
                names.append(name)
        return names

    def _log_inference_request(
        self,
        *,
        request_id: str,
        load_name: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        logger.info(
            "Text inference request request_id=%s load_name=%s message_bytes=%s tool_names=%s",
            request_id or "-",
            load_name or "<empty>",
            self._estimate_message_package_bytes(messages),
            self._extract_tool_names(tools),
        )

    async def _get_bundle(self, spec: Any) -> Any:
        runtime = resolve_text_runtime(spec)
        models_dir = getattr(self.inference_config, "models_dir", None)
        weights_dir = getattr(self.inference_config, "weights_dir", None)
        model_ref = resolve_text_model_ref(
            spec,
            models_dir=models_dir,
            weights_dir=weights_dir,
        )
        policy = resolve_text_runtime_policy(spec)
        cache_key = (runtime, model_ref, policy.cache_key)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached

        lock = self._model_load_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                return cached

            if runtime == "vllm":
                # 首次冷加载 vLLM bundle 可能触发较长时间的 autotune / warmup。
                # 放到线程池里执行，避免阻塞事件循环导致 WS ping/pong 超时断开。
                bundle = await self.run_blocking(
                    load_vllm_text_bundle,
                    model_ref,
                    policy,
                    models_dir=models_dir,
                    weights_dir=weights_dir,
                )
            elif runtime == "transformers":
                bundle = await self.run_blocking(
                    load_transformers_text_bundle,
                    model_ref,
                    policy,
                    models_dir=models_dir,
                    weights_dir=weights_dir,
                )
            elif runtime == "ollama":
                # 首次触发 `ollama create` 可能要把大 gguf 拷进 blob 仓库，同样耗时；
                # 放线程池避免阻塞 WS 事件循环。
                bundle = await self.run_blocking(load_ollama_text_bundle, model_ref, policy)
            else:
                raise ValueError(f"Unsupported text runtime={runtime}")
            self._model_cache[cache_key] = bundle
            return bundle

    async def _stream_text(
        self,
        *,
        load_name: str,
        family: str,
        request_id: str,
        messages: list[dict[str, Any]],
        runtime_config: Optional[Dict[str, Any]] = None,
        temperature: Any = None,
        max_tokens: Any = None,
        enable_thinking: Optional[bool] = None,
        top_p: Any = None,
        top_k: Any = None,
        presence_penalty: Any = None,
        frequency_penalty: Any = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        spec = self._build_request_spec(
            load_name=load_name,
            family=family,
            runtime_config=runtime_config,
        )
        self._log_inference_request(
            request_id=request_id,
            load_name=load_name,
            messages=messages,
            tools=tools,
        )
        runtime = resolve_text_runtime(spec)
        bundle = await self._get_bundle(spec)
        policy = resolve_text_runtime_policy(spec)
        effective_max_tokens = (
            policy.service_max_tokens if policy.service_max_tokens is not None else max_tokens
        )
        if runtime == "vllm":
            async for item in stream_chat_text(
                bundle,
                messages=messages,
                request_id=request_id,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                enable_thinking=enable_thinking,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mm_processor_kwargs=mm_processor_kwargs,
                tools=tools,
            ):
                yield item
            return
        if runtime == "transformers":
            async for item in stream_transformers_chat_text(
                bundle,
                messages=messages,
                request_id=request_id,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                enable_thinking=enable_thinking,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mm_processor_kwargs=mm_processor_kwargs,
                tools=tools,
            ):
                yield item
            return
        if runtime == "ollama":
            async for item in stream_ollama_chat_text(
                bundle,
                messages=messages,
                request_id=request_id,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                enable_thinking=enable_thinking,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mm_processor_kwargs=mm_processor_kwargs,
                tools=tools,
            ):
                yield item
            return
        raise ValueError(f"Unsupported text runtime={runtime}")

    async def _abort_text_request(
        self,
        *,
        load_name: str,
        family: str,
        request_id: str,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not request_id:
            return
        spec = self._build_request_spec(
            load_name=load_name,
            family=family,
            runtime_config=runtime_config,
        )
        runtime = resolve_text_runtime(spec)
        bundle = await self._get_bundle(spec)
        if runtime == "vllm":
            await abort_chat_request(bundle, request_id)
            return
        if runtime == "transformers":
            await abort_transformers_chat_request(bundle, request_id)
            return
        if runtime == "ollama":
            await abort_ollama_chat_request(bundle, request_id)
            return
        raise ValueError(f"Unsupported text runtime={runtime}")

    async def _generate_text(
        self,
        *,
        load_name: str,
        family: str,
        messages: list[dict[str, Any]],
        runtime_config: Optional[Dict[str, Any]] = None,
        temperature: Any = None,
        max_tokens: Any = None,
        enable_thinking: Optional[bool] = None,
        top_p: Any = None,
        top_k: Any = None,
        presence_penalty: Any = None,
        frequency_penalty: Any = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        spec = self._build_request_spec(
            load_name=load_name,
            family=family,
            runtime_config=runtime_config,
        )
        request_id = f"oneshot:{uuid4()}"
        self._log_inference_request(
            request_id=request_id,
            load_name=load_name,
            messages=messages,
            tools=tools,
        )
        runtime = resolve_text_runtime(spec)
        bundle = await self._get_bundle(spec)
        policy = resolve_text_runtime_policy(spec)
        effective_max_tokens = (
            policy.service_max_tokens if policy.service_max_tokens is not None else max_tokens
        )
        if runtime == "vllm":
            return await generate_chat_text(
                bundle,
                messages=messages,
                request_id=request_id,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                enable_thinking=enable_thinking,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mm_processor_kwargs=mm_processor_kwargs,
                tools=tools,
            )
        if runtime == "transformers":
            return await generate_transformers_chat_text(
                bundle,
                messages=messages,
                request_id=request_id,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                enable_thinking=enable_thinking,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mm_processor_kwargs=mm_processor_kwargs,
                tools=tools,
            )
        if runtime == "ollama":
            return await generate_ollama_chat_text(
                bundle,
                messages=messages,
                request_id=request_id,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                enable_thinking=enable_thinking,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                mm_processor_kwargs=mm_processor_kwargs,
                tools=tools,
            )
        raise ValueError(f"Unsupported text runtime={runtime}")

    async def _stream_session_text(self, request: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        raw_tools = request.get("tools")
        tools_list = [dict(item) for item in raw_tools if isinstance(item, dict)] if isinstance(raw_tools, list) else None
        async for item in self._stream_text(
            load_name=self._effective_load_name(str(request.get("load_name") or "")),
            family=str(request.get("family") or ""),
            request_id=str(request.get("request_id") or ""),
            runtime_config=request.get("runtime_config") if isinstance(request.get("runtime_config"), dict) else {},
            messages=normalize_chat_messages(request.get("messages") or []),
            temperature=request.get("temperature"),
            max_tokens=request.get("max_tokens"),
            enable_thinking=request.get("enable_thinking"),
            top_p=request.get("top_p"),
            top_k=request.get("top_k"),
            presence_penalty=request.get("presence_penalty"),
            frequency_penalty=request.get("frequency_penalty"),
            mm_processor_kwargs=request.get("mm_processor_kwargs")
            if isinstance(request.get("mm_processor_kwargs"), dict)
            else None,
            tools=tools_list,
        ):
            yield item

    async def _abort_session_request(self, request: Dict[str, Any]) -> None:
        await self._abort_text_request(
            load_name=self._effective_load_name(str(request.get("load_name") or "")),
            family=str(request.get("family") or ""),
            request_id=str(request.get("request_id") or ""),
            runtime_config=request.get("runtime_config") if isinstance(request.get("runtime_config"), dict) else {},
        )

    async def stop(self):
        for cache_key, bundle in list(self._model_cache.items()):
            runtime = str(cache_key[0] or "").strip().lower() if isinstance(cache_key, tuple) and cache_key else ""
            try:
                if runtime == "transformers":
                    shutdown_transformers_text_bundle(bundle)
                elif runtime == "ollama":
                    shutdown_ollama_text_bundle(bundle)
                else:
                    shutdown_vllm_text_bundle(bundle)
            except Exception:
                pass
        self._model_cache.clear()
        self._model_load_locks.clear()
        await super().stop()

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        task_id = params.task_id
        prompt = str(getattr(params, "prompt", "") or "").strip()
        requested_model = str(getattr(params, "load_name", "") or "").strip()
        load_name = self._effective_load_name(requested_model)
        family = str(getattr(params, "family", "") or "").strip()

        logger.info(
            "Starting text inference for task: %s load_name=%s%s",
            task_id,
            load_name,
            f" (fixed_model override; request had {requested_model!r})" if self._fixed_model else "",
        )

        if self.task_processor and self.task_processor.is_task_cancelled(task_id):
            await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return None

        if not load_name:
            raise ValueError("text task requires load_name (or config.fixed_model)")
        if not family:
            raise ValueError("text task requires family")

        request_id = f"task:{task_id}"
        reply_parts: list[str] = []
        if bool(getattr(params, "stream", False)):
            sequence = 0
            started = await self._send_stream_event(
                {
                    "type": "text_stream_delta",
                    "task_id": task_id,
                    "service_id": self.service_id,
                    "sequence": sequence,
                    "delta": "",
                    "is_final": False,
                }
            )
            if not started:
                raise RuntimeError("stream transport disconnected, text task aborted")
            sequence += 1

            async for item in self._stream_text(
                load_name=load_name,
                family=family,
                request_id=request_id,
                runtime_config=params.runtime_config if isinstance(params.runtime_config, dict) else {},
                messages=build_messages_from_prompt(prompt),
                temperature=getattr(params, "temperature", None),
                max_tokens=getattr(params, "max_tokens", None),
            ):
                if self.task_processor and self.task_processor.is_task_cancelled(task_id):
                    await self._abort_text_request(
                        load_name=load_name,
                        family=family,
                        request_id=request_id,
                        runtime_config=params.runtime_config if isinstance(params.runtime_config, dict) else {},
                    )
                    await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
                    return None
                chunk = str(item.get("delta") or "")
                if chunk:
                    reply_parts.append(chunk)
                ok = await self._send_stream_event(
                    {
                        "type": "text_stream_delta",
                        "task_id": task_id,
                        "service_id": self.service_id,
                        "sequence": sequence,
                        "delta": chunk,
                        "is_final": bool(item.get("finished")),
                        **{key: item.get(key) for key in _STREAM_STATS_KEYS if item.get(key) is not None},
                    }
                )
                if not ok:
                    raise RuntimeError("stream transport disconnected, text task aborted")
                sequence += 1
                await asyncio.sleep(0)
        else:
            reply = await self._generate_text(
                load_name=load_name,
                family=family,
                runtime_config=params.runtime_config if isinstance(params.runtime_config, dict) else {},
                messages=build_messages_from_prompt(prompt),
                temperature=getattr(params, "temperature", None),
                max_tokens=getattr(params, "max_tokens", None),
            )
            await self._send_stream_event(
                {
                    "type": "text_stream_delta",
                    "task_id": task_id,
                    "service_id": self.service_id,
                    "sequence": 0,
                    "delta": reply,
                    "is_final": True,
                }
            )
            reply_parts.append(reply)

        if self.task_processor and self.task_processor.is_task_cancelled(task_id):
            await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return None

        await self.ws_client.send_task_status(task_id=task_id, status="completed")
        return "".join(reply_parts)

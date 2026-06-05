# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import gc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from common.logger import get_logger

from text.runtime.common import count_multimodal_parts
from text.runtime.runtime_resolver import TextRuntimePolicy, resolve_speculative_config
from text.runtime.vllm_bridge import render_chat_prompt

logger = get_logger(__name__)


@dataclass
class _RequestControl:
    stop_event: Any


@dataclass
class TransformersTextBundle:
    model_ref: str
    tokenizer: Any
    model: Any
    policy: TextRuntimePolicy
    chat_adapter: Any = None
    assistant_model: Any = None
    tool_arguments_shape: Optional[str] = None
    active_requests: Dict[str, _RequestControl] = field(default_factory=dict, repr=False)
    request_lock: Any = field(default_factory=threading.Lock, repr=False)
    generation_lock: Any = field(default_factory=threading.Lock, repr=False)


def _is_local_model_ref(model_ref: str) -> bool:
    try:
        return Path(model_ref).expanduser().exists()
    except Exception:
        return False


def _resolve_torch_dtype(dtype: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    normalized = str(dtype or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(normalized, "auto")


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    return number if number > 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_count_tokens(tokenizer: Any, text: str) -> Optional[int]:
    try:
        token_ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
    except Exception:
        return None
    try:
        return len(token_ids)
    except Exception:
        return None


def _get_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if device is None:
        return inputs
    try:
        return inputs.to(device)
    except Exception:
        pass

    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            try:
                moved[key] = value.to(device)
                continue
            except Exception:
                pass
        moved[key] = value
    return moved


def _normalize_device_placement(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value or "").strip().lower()
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _summarize_device_map(model: Any) -> Dict[str, int]:
    raw_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_map, dict):
        return {}
    summary: Dict[str, int] = {}
    for value in raw_map.values():
        placement = _normalize_device_placement(value) or "<unknown>"
        summary[placement] = summary.get(placement, 0) + 1
    return summary


def _has_cpu_offload(model: Any) -> bool:
    summary = _summarize_device_map(model)
    return any(place in {"cpu", "disk", "meta"} for place in summary)


def _cleanup_loaded_model(model: Any) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _register_request(bundle: TransformersTextBundle, request_id: str, stop_event: Any) -> None:
    if not request_id:
        return
    with bundle.request_lock:
        bundle.active_requests[request_id] = _RequestControl(stop_event=stop_event)


def _unregister_request(bundle: TransformersTextBundle, request_id: str) -> None:
    if not request_id:
        return
    with bundle.request_lock:
        bundle.active_requests.pop(request_id, None)


_LOADER_TUNING_KEYS = frozenset({"disable_mmap", "pin_memory", "patch_accelerate_pin_memory"})


def _resolve_pin_memory_setting(pin_memory: Any, device: Any) -> bool:
    try:
        from common.detect_pin_memory import resolve_pin_memory as _resolve
    except Exception:
        if pin_memory == "auto":
            return str(device).startswith("cuda")
        return bool(pin_memory)
    return _resolve(pin_memory, device)


def _split_loader_tuning_kwargs(model_kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    remaining = dict(model_kwargs)
    tuning: Dict[str, Any] = {}
    for key in _LOADER_TUNING_KEYS:
        if key in remaining:
            tuning[key] = remaining.pop(key)
    return remaining, tuning


def _patch_accelerate_pin_memory(*, enabled: bool) -> bool:
    """Monkey-patch accelerate 权重搬运：CPU→CUDA 前先 pin，规避 GB10 many-small H2D 极慢路径。"""
    if not enabled:
        return False
    try:
        import accelerate.utils.modeling as am  # type: ignore[import-not-found]
    except Exception:
        return False

    if getattr(am.set_module_tensor_to_device, "_vitoom_patched", False):
        return True

    orig = am.set_module_tensor_to_device

    def wrapped_set_module_tensor_to_device(  # type: ignore[no-untyped-def]
        module,
        tensor_name,
        device,
        value=None,
        dtype=None,
        fp16_statistics=None,
        **kwargs,
    ):
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            return orig(
                module,
                tensor_name,
                device,
                value=value,
                dtype=dtype,
                fp16_statistics=fp16_statistics,
                **kwargs,
            )
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            dev_str = str(device)
            if dev_str.startswith("cuda"):
                try:
                    if not value.is_pinned():
                        value = value.pin_memory()
                except Exception:
                    pass
        return orig(
            module,
            tensor_name,
            device,
            value=value,
            dtype=dtype,
            fp16_statistics=fp16_statistics,
            **kwargs,
        )

    wrapped_set_module_tensor_to_device._vitoom_patched = True  # type: ignore[attr-defined]
    am.set_module_tensor_to_device = wrapped_set_module_tensor_to_device  # type: ignore[assignment]
    return True


def _build_generation_kwargs(
    bundle: TransformersTextBundle,
    *,
    inputs: Any,
    max_tokens: Any = None,
    temperature: Any = None,
    top_p: Any = None,
    top_k: Any = None,
) -> Dict[str, Any]:
    tokenizer = bundle.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    max_new_tokens = _coerce_positive_int(max_tokens, 1024)
    sample_temperature = _coerce_float(temperature, 0.7)
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample_temperature > 0.0,
        "temperature": sample_temperature if sample_temperature > 0.0 else None,
        "pad_token_id": pad_token_id,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if sample_temperature > 0.0:
        kwargs["top_p"] = _coerce_float(top_p, 1.0)
        if top_k is not None:
            try:
                parsed_top_k = int(top_k)
            except Exception:
                parsed_top_k = None
            if parsed_top_k is not None and parsed_top_k > 0:
                kwargs["top_k"] = parsed_top_k

    if isinstance(inputs, dict):
        kwargs.update(inputs)
    else:
        kwargs.update(dict(inputs))
    return {key: value for key, value in kwargs.items() if value is not None}


def _load_transformers_assistant_model(
    assistant_ref: str,
    policy: TextRuntimePolicy,
    *,
    local_files_only: bool,
) -> Any:
    from transformers import AutoModelForCausalLM  # type: ignore

    assistant_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": assistant_ref,
        "torch_dtype": _resolve_torch_dtype(policy.dtype),
        "device_map": policy.device_map,
        "trust_remote_code": policy.trust_remote_code,
        "local_files_only": local_files_only,
    }
    model = AutoModelForCausalLM.from_pretrained(**assistant_kwargs)
    try:
        model.eval()
    except Exception:
        pass
    return model


def load_transformers_text_bundle(
    model_ref: str,
    policy: TextRuntimePolicy,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> TransformersTextBundle:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Failed to import transformers for text runtime. "
            "Please install a recent transformers version in inference/text/requirements.txt."
        ) from e

    local_files_only = _is_local_model_ref(model_ref)
    logger.info(
        "Loading transformers text bundle model_ref=%s dtype=%s device_map=%s enable_thinking=%s",
        model_ref,
        policy.dtype,
        policy.device_map,
        policy.enable_thinking,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=policy.trust_remote_code,
        local_files_only=local_files_only,
    )
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_ref,
        "torch_dtype": _resolve_torch_dtype(policy.dtype),
        "device_map": policy.device_map,
        "trust_remote_code": policy.trust_remote_code,
        "local_files_only": local_files_only,
    }
    if policy.model_kwargs:
        model_kwargs.update(policy.model_kwargs)

    hf_kwargs, tuning = _split_loader_tuning_kwargs(model_kwargs)
    device_map = hf_kwargs.get("device_map", policy.device_map)
    pin_raw = tuning.get("pin_memory", "auto")
    pin_enabled = _resolve_pin_memory_setting(
        pin_raw,
        device_map if isinstance(device_map, str) else "cuda:0",
    )
    patch_accel = tuning.get("patch_accelerate_pin_memory", pin_enabled)
    if patch_accel and pin_enabled:
        patched = _patch_accelerate_pin_memory(enabled=True)
        if patched:
            logger.info(
                "Enabled accelerate pin_memory patch for transformers load model_ref=%s",
                model_ref,
            )
        else:
            logger.warning(
                "pin_memory=%s but accelerate patch unavailable for model_ref=%s; "
                "GB10/DGX Spark loads may stay slow",
                pin_raw,
                model_ref,
            )
    disable_mmap = tuning.get("disable_mmap")
    if disable_mmap is not None:
        hf_kwargs["disable_mmap"] = bool(disable_mmap)
    # pin_memory 不传给 from_pretrained（Gemma4 等模型类不支持该 kwarg）；
    # GB10/DGX Spark 的加速依赖上方的 accelerate set_module_tensor_to_device patch。

    logger.info(
        "Transformers load tuning model_ref=%s disable_mmap=%s pin_memory=%s accelerate_patch=%s",
        model_ref,
        hf_kwargs.get("disable_mmap"),
        pin_raw if pin_raw != "auto" else f"auto->{pin_enabled}",
        bool(patch_accel and pin_enabled),
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(**hf_kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "disable_mmap" in msg and "disable_mmap" in hf_kwargs:
            hf_kwargs.pop("disable_mmap", None)
            logger.warning(
                "Transformers rejected disable_mmap for model_ref=%s; retrying without it: %s",
                model_ref,
                exc,
            )
            model = AutoModelForCausalLM.from_pretrained(**hf_kwargs)
        else:
            raise
    logger.info(
        "Transformers from_pretrained completed model_ref=%s class=%s",
        model_ref,
        type(model).__name__,
    )
    try:
        model.eval()
    except Exception:
        pass

    device_map_summary = _summarize_device_map(model)
    if device_map_summary:
        logger.info("Transformers model device map model_ref=%s placements=%s", model_ref, device_map_summary)
    if _has_cpu_offload(model) and not policy.allow_cpu_offload:
        _cleanup_loaded_model(model)
        raise RuntimeError(
            "transformers text runtime detected CPU offload in hf_device_map, which usually causes very slow "
            "TTFT and stuck-looking interactive requests. Please set runtime.device_map to 'cuda:0' (or a pure "
            "multi-GPU CUDA map), use a smaller model, or switch backend to 'vllm'. If you really want this "
            "behavior, set runtime.allow_cpu_offload=true."
        )

    assistant_model = None
    resolved_spec = resolve_speculative_config(
        policy.speculative_config,
        models_dir=models_dir,
        weights_dir=weights_dir,
    )
    if policy.speculative_config and resolved_spec is None:
        assistant_name = str(policy.speculative_config.get("model") or "").strip()
        logger.warning(
            "transformers speculative_config is set but assistant model was not found under "
            "models_dir/weights_dir (model=%r); MTP disabled",
            assistant_name or policy.speculative_config,
        )
    elif resolved_spec and resolved_spec.get("model"):
        assistant_ref = str(resolved_spec["model"])
        logger.info("Loading transformers MTP assistant model_ref=%s", assistant_ref)
        assistant_model = _load_transformers_assistant_model(
            assistant_ref,
            policy,
            local_files_only=_is_local_model_ref(assistant_ref),
        )
        logger.info(
            "Transformers MTP enabled assistant=%s method=%s",
            assistant_ref,
            resolved_spec.get("method"),
        )

    return TransformersTextBundle(
        model_ref=model_ref,
        tokenizer=tokenizer,
        model=model,
        policy=policy,
        chat_adapter=tokenizer,
        assistant_model=assistant_model,
    )


def _prepare_generation_inputs(
    bundle: TransformersTextBundle,
    *,
    messages: List[Dict[str, Any]],
    enable_thinking: bool | None = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Any, int]:
    image_count, video_count = count_multimodal_parts(messages)
    if image_count > 0 or video_count > 0:
        raise RuntimeError(
            "transformers text runtime currently only supports text-only chat messages. "
            f"Got images={image_count}, videos={video_count}. Please switch backend to vllm "
            "for multimodal text requests."
        )

    prompt = render_chat_prompt(
        bundle,
        messages,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    tokenizer = bundle.tokenizer
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize text prompt for transformers runtime: {e}") from e
    try:
        inputs.pop("token_type_ids", None)
    except Exception:
        pass
    inputs = _move_inputs_to_device(inputs, _get_model_device(bundle.model))

    prompt_tokens = None
    try:
        prompt_tokens = int(inputs["input_ids"].shape[-1])
    except Exception:
        prompt_tokens = _safe_count_tokens(tokenizer, prompt)
    return prompt, inputs, int(prompt_tokens or 0)


async def _join_worker(worker: Any, *, timeout: float = 5.0) -> None:
    await asyncio.to_thread(worker.join, timeout)
    if worker.is_alive():
        logger.warning("transformers generation worker still alive after timeout=%s", timeout)


async def _iterate_streamer(streamer: Any) -> AsyncIterator[str]:
    iterator = iter(streamer)
    sentinel = object()
    while True:
        chunk = await asyncio.to_thread(next, iterator, sentinel)
        if chunk is sentinel:
            break
        yield str(chunk or "")


def _build_stream_stats(
    *,
    tokenizer: Any,
    prompt: str,
    output_text: str,
    prompt_tokens: int,
    started_at: float,
    first_delta_at: Optional[float],
    finished_at: float,
) -> Dict[str, Any]:
    output_tokens = _safe_count_tokens(tokenizer, output_text)
    ttft_seconds = None if first_delta_at is None else max(0.0, first_delta_at - started_at)
    total_seconds = max(0.0, finished_at - started_at)
    decode_seconds = None
    if ttft_seconds is not None:
        decode_seconds = max(0.0, total_seconds - ttft_seconds)
    elif first_delta_at is not None:
        decode_seconds = max(0.0, finished_at - first_delta_at)

    stats: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens or _safe_count_tokens(tokenizer, prompt),
        "total_seconds": total_seconds,
    }
    if output_tokens is not None:
        stats["output_tokens"] = int(output_tokens)
    if ttft_seconds is not None:
        stats["ttft_seconds"] = ttft_seconds
    if decode_seconds is not None:
        stats["decode_seconds"] = decode_seconds
    if isinstance(output_tokens, int) and total_seconds > 0:
        stats["tok_s_total"] = float(output_tokens) / total_seconds
        if decode_seconds is not None and decode_seconds > 0:
            stats["tok_s_decode"] = float(output_tokens) / decode_seconds
    return stats


async def stream_chat_text(
    bundle: TransformersTextBundle,
    *,
    messages: List[Dict[str, Any]],
    request_id: str,
    max_tokens: Any = None,
    temperature: Any = None,
    enable_thinking: bool | None = None,
    top_p: Any = None,
    top_k: Any = None,
    presence_penalty: Any = None,
    frequency_penalty: Any = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    _ = presence_penalty
    _ = frequency_penalty
    _ = mm_processor_kwargs

    try:
        import torch  # type: ignore
        from transformers.generation import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    except Exception as e:
        raise RuntimeError("transformers streaming runtime requires torch and TextIteratorStreamer") from e

    waited_for_slot = bundle.generation_lock.locked()
    if waited_for_slot:
        logger.info("transformers runtime waiting for active generation model_ref=%s request_id=%s", bundle.model_ref, request_id)
    await asyncio.to_thread(bundle.generation_lock.acquire)
    try:
        prompt, inputs, prompt_tokens = _prepare_generation_inputs(
            bundle,
            messages=messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )

        stop_event = threading.Event()
        result_box: Dict[str, Any] = {"error": None}
        streamer = TextIteratorStreamer(
            bundle.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        class _FlagStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                return stop_event.is_set()

        generation_kwargs = _build_generation_kwargs(
            bundle,
            inputs=inputs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generation_kwargs["streamer"] = streamer
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([_FlagStoppingCriteria()])
        if bundle.assistant_model is not None:
            generation_kwargs["assistant_model"] = bundle.assistant_model

        def _generate() -> None:
            try:
                logger.info(
                    "Transformers generate begin model_ref=%s request_id=%s max_new_tokens=%s",
                    bundle.model_ref,
                    request_id or "-",
                    generation_kwargs.get("max_new_tokens"),
                )
                with torch.inference_mode():
                    bundle.model.generate(**generation_kwargs)
                logger.info(
                    "Transformers generate finished model_ref=%s request_id=%s",
                    bundle.model_ref,
                    request_id or "-",
                )
            except Exception as exc:
                result_box["error"] = exc
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        worker = threading.Thread(target=_generate, name=f"text-hf-{request_id or 'unknown'}", daemon=True)
        _register_request(bundle, request_id, stop_event)
        worker.start()

        started_at = time.perf_counter()
        first_delta_at: Optional[float] = None
        output_parts: List[str] = []
        try:
            async for piece in _iterate_streamer(streamer):
                if piece:
                    output_parts.append(piece)
                    if first_delta_at is None:
                        first_delta_at = time.perf_counter()
                    yield {"delta": piece, "finished": False}
                await asyncio.sleep(0)

            await _join_worker(worker)
            if result_box.get("error") is not None:
                raise RuntimeError(
                    f"transformers text generation failed: {type(result_box['error']).__name__}: {result_box['error']}"
                ) from result_box["error"]

            finished_at = time.perf_counter()
            payload: Dict[str, Any] = {
                "delta": "",
                "finished": True,
                "finish_reason": "cancelled" if stop_event.is_set() else "stop",
            }
            payload.update(
                _build_stream_stats(
                    tokenizer=bundle.tokenizer,
                    prompt=prompt,
                    output_text="".join(output_parts),
                    prompt_tokens=prompt_tokens,
                    started_at=started_at,
                    first_delta_at=first_delta_at,
                    finished_at=finished_at,
                )
            )
            yield payload
        except asyncio.CancelledError:
            stop_event.set()
            raise
        finally:
            stop_event.set()
            _unregister_request(bundle, request_id)
            if worker.is_alive():
                await _join_worker(worker, timeout=1.0)
    finally:
        try:
            bundle.generation_lock.release()
        except Exception:
            pass


async def generate_chat_text(
    bundle: TransformersTextBundle,
    *,
    messages: List[Dict[str, Any]],
    request_id: str,
    max_tokens: Any = None,
    temperature: Any = None,
    enable_thinking: bool | None = None,
    top_p: Any = None,
    top_k: Any = None,
    presence_penalty: Any = None,
    frequency_penalty: Any = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts: List[str] = []
    async for item in stream_chat_text(
        bundle,
        messages=messages,
        request_id=request_id,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        mm_processor_kwargs=mm_processor_kwargs,
        tools=tools,
    ):
        delta = str(item.get("delta") or "")
        if delta:
            parts.append(delta)
    return "".join(parts)


async def abort_chat_request(bundle: TransformersTextBundle, request_id: Optional[str]) -> None:
    if not request_id:
        return
    with bundle.request_lock:
        control = bundle.active_requests.get(request_id)
    if control is not None:
        control.stop_event.set()


def shutdown_transformers_text_bundle(bundle: TransformersTextBundle) -> None:
    try:
        with bundle.request_lock:
            controls = list(bundle.active_requests.values())
        for control in controls:
            try:
                control.stop_event.set()
            except Exception:
                pass
    except Exception:
        pass

    model = getattr(bundle, "model", None)
    assistant_model = getattr(bundle, "assistant_model", None)
    tokenizer = getattr(bundle, "tokenizer", None)
    try:
        bundle.model = None  # type: ignore[assignment]
    except Exception:
        pass
    try:
        bundle.tokenizer = None  # type: ignore[assignment]
    except Exception:
        pass
    try:
        bundle.chat_adapter = None  # type: ignore[assignment]
    except Exception:
        pass
    try:
        bundle.assistant_model = None  # type: ignore[assignment]
    except Exception:
        pass

    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
    if assistant_model is not None:
        try:
            assistant_model.to("cpu")
        except Exception:
            pass
        try:
            del assistant_model
        except Exception:
            pass
    if tokenizer is not None:
        try:
            del tokenizer
        except Exception:
            pass

    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

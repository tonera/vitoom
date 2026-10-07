# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from common.logger import get_logger

from text.runtime.common import count_multimodal_parts
from text.runtime.runtime_resolver import TextRuntimePolicy, resolve_speculative_config

logger = get_logger(__name__)


@dataclass
class VllmTextBundle:
    model_ref: str
    tokenizer: Any
    engine: Any
    policy: TextRuntimePolicy
    chat_adapter: Any = None
    # chat_template 对 assistant.tool_calls.function.arguments 的期望形态：
    #   None   —— 未探测；首次渲染时按 OpenAI 原生 string 试，失败再回退到 dict。
    #   "str"  —— 期望 JSON 字符串（OpenAI 官方形态，Llama3/Qwen2.5-text 等）。
    #   "dict" —— 期望已反序列化的 mapping（Qwen3-VL 等模板，会对 arguments 调 items()）。
    tool_arguments_shape: Optional[str] = None


def _import_async_vllm_symbols() -> tuple[Any, Any, Any]:
    try:
        from vllm import AsyncLLMEngine  # type: ignore
    except Exception:
        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import AsyncLLMEngine from vllm. "
                "Please install a recent vllm version in inference/text/requirements.txt."
            ) from e

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    except Exception as e:
        raise RuntimeError("Failed to import AsyncEngineArgs from vllm.engine.arg_utils") from e

    try:
        from vllm.sampling_params import RequestOutputKind  # type: ignore
    except Exception as e:
        raise RuntimeError("Failed to import RequestOutputKind from vllm.sampling_params") from e

    return AsyncLLMEngine, AsyncEngineArgs, RequestOutputKind


def _normalize_legacy_rope_scaling(value: Any) -> tuple[Optional[Dict[str, Any]], bool]:
    if not isinstance(value, dict):
        return None, False

    normalized = dict(value)
    changed = False
    if "type" in normalized:
        legacy_type = normalized.pop("type")
        if normalized.get("rope_type") in (None, "") and legacy_type not in (None, ""):
            normalized["rope_type"] = legacy_type
        changed = True
    return normalized, changed


def _load_local_hf_config(model_ref: str) -> Dict[str, Any]:
    config_path = Path(model_ref) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to read raw HF config from %s: %s", config_path, e)
        return {}
    return data if isinstance(data, dict) else {}


def _build_hf_compat_overrides(model_ref: str, existing_hf_overrides: Any) -> Optional[Dict[str, Any]]:
    if isinstance(existing_hf_overrides, dict):
        existing_rope_scaling, changed = _normalize_legacy_rope_scaling(existing_hf_overrides.get("rope_scaling"))
        if changed and existing_rope_scaling is not None:
            return {"rope_scaling": existing_rope_scaling}
        if isinstance(existing_hf_overrides.get("rope_scaling"), dict):
            return None

    raw_cfg = _load_local_hf_config(model_ref)
    rope_scaling, changed = _normalize_legacy_rope_scaling(raw_cfg.get("rope_scaling"))
    if changed and rope_scaling is not None:
        return {"rope_scaling": rope_scaling}
    return None


def load_vllm_text_bundle(
    model_ref: str,
    policy: TextRuntimePolicy,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> VllmTextBundle:
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Failed to import transformers for text runtime. "
            "Gemma/Qwen text runtime requires a transformers version new enough to load these tokenizers."
        ) from e

    AsyncLLMEngine, AsyncEngineArgs, _RequestOutputKind = _import_async_vllm_symbols()

    speculative_config = resolve_speculative_config(
        policy.speculative_config,
        models_dir=models_dir,
        weights_dir=weights_dir,
    )
    if policy.speculative_config and speculative_config is None:
        assistant_name = str(policy.speculative_config.get("model") or "").strip()
        logger.warning(
            "vLLM speculative_config is set but assistant model was not found under "
            "models_dir/weights_dir (model=%r); MTP disabled, using target model only",
            assistant_name or policy.speculative_config,
        )
    elif speculative_config:
        logger.info(
            "vLLM MTP enabled method=%s assistant=%s num_speculative_tokens=%s",
            speculative_config.get("method"),
            speculative_config.get("model"),
            speculative_config.get("num_speculative_tokens"),
        )

    logger.info(
        "Loading async vLLM text bundle model_ref=%s tp=%s max_model_len=%s enable_thinking=%s",
        model_ref,
        policy.tensor_parallel_size,
        policy.max_model_len,
        policy.enable_thinking,
    )

    chat_adapter = None
    tokenizer = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_ref,
            trust_remote_code=policy.trust_remote_code,
            local_files_only=True,
        )
        if hasattr(processor, "apply_chat_template"):
            chat_adapter = processor
            tokenizer = getattr(processor, "tokenizer", None)
    except Exception as e:
        logger.info("Falling back to AutoTokenizer for model_ref=%s: %s", model_ref, e)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=policy.trust_remote_code,
            local_files_only=True,
        )
    if chat_adapter is None:
        chat_adapter = tokenizer

    engine_kwargs: Dict[str, Any] = {
        "model": model_ref,
        "tensor_parallel_size": policy.tensor_parallel_size,
        "gpu_memory_utilization": policy.gpu_memory_utilization,
        "trust_remote_code": policy.trust_remote_code,
        # 开启 vLLM 自动前缀缓存：同一会话内"system prompt + tool schemas
        # + 历史"前缀不变时，prefill 阶段可跳过，大幅降低第 2 轮起的 TTFT
        # 与 GPU 负载。vLLM >= 0.6.0 默认已为 True；这里统一显式设置，
        # 兼容老版本。若某个模型不兼容（极少数），可通过 model 配置
        # runtime_cfg.engine_kwargs.enable_prefix_caching=false 覆盖。
        "enable_prefix_caching": True,
    }
    if policy.max_model_len:
        engine_kwargs["max_model_len"] = policy.max_model_len
    engine_kwargs.update(policy.engine_kwargs)
    if speculative_config:
        engine_kwargs["speculative_config"] = speculative_config

    direct_rope_scaling, direct_changed = _normalize_legacy_rope_scaling(engine_kwargs.get("rope_scaling"))
    if direct_changed and direct_rope_scaling is not None:
        engine_kwargs["rope_scaling"] = direct_rope_scaling
        logger.info("Normalized legacy engine rope_scaling for model_ref=%s", model_ref)

    compat_hf_overrides = _build_hf_compat_overrides(model_ref, engine_kwargs.get("hf_overrides"))
    if compat_hf_overrides:
        merged_hf_overrides = (
            dict(engine_kwargs.get("hf_overrides"))
            if isinstance(engine_kwargs.get("hf_overrides"), dict)
            else {}
        )
        for key, value in compat_hf_overrides.items():
            merged_hf_overrides[key] = value
        engine_kwargs["hf_overrides"] = merged_hf_overrides
        logger.info(
            "Applied HF config compatibility overrides for model_ref=%s keys=%s",
            model_ref,
            sorted(compat_hf_overrides.keys()),
        )

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return VllmTextBundle(
        model_ref=model_ref,
        tokenizer=tokenizer,
        engine=engine,
        policy=policy,
        chat_adapter=chat_adapter,
    )


def _get_chat_adapter(bundle: VllmTextBundle) -> Any:
    return bundle.chat_adapter or bundle.tokenizer


def _inflate_tool_call_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 assistant.tool_calls[*].function.arguments 从 JSON 字符串反序列化回 dict。

    OpenAI 协议规定 ``arguments`` 是 JSON 字符串，但 Qwen3-VL 等模型的 chat_template
    直接对 ``tool_call.arguments`` 调 ``| items``，要求是 mapping，否则抛
    ``TypeError: Can only get item pairs from a mapping.``。这里在渲染前做一次反序列化，
    让两边都舒服。无法解析的值保留原样并包一层 ``{"_raw": <str>}``，至少不至于炸模板。
    """
    inflated: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            inflated.append(message)
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            inflated.append(message)
            continue

        new_tool_calls: List[Dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                new_tool_calls.append(call)
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                new_tool_calls.append(call)
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                text = arguments.strip()
                parsed: Any
                if not text:
                    parsed = {}
                else:
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        parsed = {"_raw": arguments}
                if not isinstance(parsed, dict):
                    parsed = {"_raw": parsed}
                new_function = dict(function)
                new_function["arguments"] = parsed
                new_call = dict(call)
                new_call["function"] = new_function
                new_tool_calls.append(new_call)
            else:
                new_tool_calls.append(call)

        new_message = dict(message)
        new_message["tool_calls"] = new_tool_calls
        inflated.append(new_message)
    return inflated


def render_chat_prompt(
    bundle: VllmTextBundle,
    messages: List[Dict[str, Any]],
    *,
    enable_thinking: bool | None = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """把 messages 渲染成底层模型需要的 prompt。

    如果传入 ``tools``，会尝试走 tokenizer/chat_template 的 ``tools=`` 入口（Qwen2.5+
    及新版 Llama chat templates 支持）。若模板不接受 ``tools`` 参数，抛 TypeError 时
    回落到普通 chat 模式——这是故意的保守行为：若模型本身不支持 tool use，强塞 tools
    只会导致奇怪的幻觉输出，让调用方在响应里看到空的 ``tool_calls`` 即可明确降级。
    """
    thinking = bundle.policy.enable_thinking if enable_thinking is None else bool(enable_thinking)
    adapter = _get_chat_adapter(bundle)
    normalized_tools = [dict(tool) for tool in (tools or []) if isinstance(tool, dict)]

    # assistant.tool_calls.arguments 形态按需切换：默认尊重 OpenAI 协议的 string，
    # 仅当已探测到当前模型期望 dict 时才提前 inflate。首次命中 mapping-items 错误则
    # 自动 inflate 重试并把结果缓存到 bundle 上，以后该模型不再重复试错。
    string_messages = messages
    dict_messages: Optional[List[Dict[str, Any]]] = None

    def _get_dict_messages() -> List[Dict[str, Any]]:
        nonlocal dict_messages
        if dict_messages is None:
            dict_messages = _inflate_tool_call_arguments(string_messages)
        return dict_messages

    if bundle.tool_arguments_shape == "dict":
        messages = _get_dict_messages()
    else:
        messages = string_messages

    def _is_mapping_items_error(exc: BaseException) -> bool:
        return "Can only get item pairs from a mapping" in str(exc)

    def _is_unsupported_kwarg_error(exc: TypeError, kwarg: str) -> bool:
        msg = str(exc)
        return "unexpected keyword argument" in msg and f"'{kwarg}'" in msg

    def _auto_inflate_and_retry(call: Any) -> Optional[str]:
        """若本次失败因 tool_call.arguments 形态不匹配，inflate 后重试一次并记住偏好。"""
        if bundle.tool_arguments_shape == "dict":
            return None
        inflated = _get_dict_messages()
        if inflated == string_messages:
            return None
        try:
            rendered = call(inflated)
        except Exception as retry_exc:
            logger.warning(
                "retry with inflated tool_call.arguments still failed for model_ref=%s: %s",
                bundle.model_ref,
                retry_exc,
            )
            return None
        bundle.tool_arguments_shape = "dict"
        logger.info(
            "chat_template for model_ref=%s requires dict-shaped tool_call.arguments; "
            "future renders will auto-inflate.",
            bundle.model_ref,
        )
        return rendered

    if normalized_tools:
        # 先用当前 adapter（优先 AutoProcessor，便于保留多模态占位符）尝试渲染 tools。
        # 若 processor 的 apply_chat_template 不认 `tools`（Qwen-VL 的 processor 模板
        # 通常只负责 vision placeholder，不带 tool-use jinja），回落到底层 tokenizer
        # 的 apply_chat_template：tool-use 模板都挂在 tokenizer 层，Qwen2.5+/Qwen3
        # 的 tokenizer 会正确展开工具 schema。
        #
        # 关键细节：Jinja 模板内部的错误（例如对非 mapping 调 `.items()` 抛
        # "Can only get item pairs from a mapping."）会以 TypeError 形式透传出来，
        # 与 "kwarg 不支持" 长得一模一样。因此我们只在**首次**调用且异常确实是
        # "unexpected keyword argument" 时才判定为 kwarg 不支持并回退，其它 TypeError
        # 直接让它抛，免得把真正的 schema 问题伪装成 "模板不支持 tools" 静默丢掉。
        render_candidates: list[Any] = [adapter]
        if bundle.tokenizer is not None and bundle.tokenizer is not adapter:
            render_candidates.append(bundle.tokenizer)

        last_tools_error: Optional[Exception] = None

        def _invoke(candidate: Any, msgs: List[Dict[str, Any]]) -> str:
            try:
                return candidate.apply_chat_template(
                    msgs,
                    tools=normalized_tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except TypeError as exc:
                if _is_unsupported_kwarg_error(exc, "enable_thinking"):
                    return candidate.apply_chat_template(
                        msgs,
                        tools=normalized_tools,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                raise

        def _try_render(candidate: Any) -> Optional[str]:
            nonlocal last_tools_error
            try:
                return _invoke(candidate, messages)
            except TypeError as exc:
                last_tools_error = exc
                if _is_unsupported_kwarg_error(exc, "tools"):
                    return None
                if _is_mapping_items_error(exc):
                    rendered = _auto_inflate_and_retry(lambda m: _invoke(candidate, m))
                    if rendered is not None:
                        return rendered
                logger.warning(
                    "apply_chat_template raised TypeError while rendering tools on %s "
                    "(will try next candidate): %s",
                    type(candidate).__name__,
                    exc,
                )
                return None

        for candidate in render_candidates:
            rendered = _try_render(candidate)
            if rendered is not None:
                return rendered

        logger.error(
            "All chat-template candidates failed to render tools; dropping tool schemas. "
            "model_ref=%s candidates=%s last_error=%s tools=%s",
            bundle.model_ref,
            [type(c).__name__ for c in render_candidates],
            last_tools_error,
            normalized_tools,
        )

    def _invoke_no_tools(msgs: List[Dict[str, Any]]) -> str:
        try:
            return adapter.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError as exc:
            if _is_unsupported_kwarg_error(exc, "enable_thinking"):
                return adapter.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            raise

    try:
        return _invoke_no_tools(messages)
    except TypeError as exc:
        if _is_mapping_items_error(exc):
            rendered = _auto_inflate_and_retry(_invoke_no_tools)
            if rendered is not None:
                return rendered
        raise


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    return number if number > 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number


def _safe_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_count_tokens(tokenizer: Any, text: str) -> Optional[int]:
    try:
        token_ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
    except Exception:
        return None
    try:
        return len(token_ids)
    except Exception:
        return None


def _module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _torchvision_has_read_video() -> bool:
    try:
        from torchvision import io as torchvision_io
    except Exception:
        return False
    return callable(getattr(torchvision_io, "read_video", None))


def _apply_qwen_video_reader_backend(backend: str) -> None:
    normalized_backend = str(backend or "").strip().lower()
    if normalized_backend not in {"torchcodec", "decord", "torchvision"}:
        return

    os.environ["FORCE_QWENVL_VIDEO_READER"] = normalized_backend
    vision_process = sys.modules.get("qwen_vl_utils.vision_process")
    if vision_process is None:
        return

    try:
        setattr(vision_process, "FORCE_QWENVL_VIDEO_READER", normalized_backend)
        cache_clear = getattr(getattr(vision_process, "get_video_reader_backend", None), "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
    except Exception:
        logger.warning("Failed to refresh qwen-vl-utils video backend cache backend=%s", normalized_backend)


def _ensure_qwen_video_reader_backend(video_count: int) -> None:
    if video_count <= 0:
        return

    forced_backend = str(os.environ.get("FORCE_QWENVL_VIDEO_READER") or "").strip().lower()
    if forced_backend:
        if forced_backend == "torchvision" and not _torchvision_has_read_video():
            raise RuntimeError(
                "FORCE_QWENVL_VIDEO_READER=torchvision, but current torchvision does not expose "
                "`torchvision.io.read_video`. Install `torchcodec` (recommended) or `decord`, "
                "or pin torchvision to a version that still provides video decoding support."
            )
        _apply_qwen_video_reader_backend(forced_backend)
        logger.info("Using forced Qwen video reader backend=%s", forced_backend)
        return

    if _module_available("torchcodec"):
        _apply_qwen_video_reader_backend("torchcodec")
        logger.info("Selected Qwen video reader backend=torchcodec")
        return

    if _module_available("decord"):
        _apply_qwen_video_reader_backend("decord")
        logger.info("Selected Qwen video reader backend=decord")
        return

    if _torchvision_has_read_video():
        _apply_qwen_video_reader_backend("torchvision")
        logger.info("Selected Qwen video reader backend=torchvision")
        return

    raise RuntimeError(
        "Video understanding requires a supported decoder backend for `qwen-vl-utils`, but none "
        "is usable in the current environment. Please install `torchcodec` (recommended) or "
        "`decord`, then restart the text inference service. Current torchvision does not expose "
        "`torchvision.io.read_video`."
    )


def _build_multimodal_payload(messages: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    image_count, video_count = count_multimodal_parts(messages)
    if image_count <= 0 and video_count <= 0:
        return {}, {}

    _ensure_qwen_video_reader_backend(video_count)

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Multi-modal Qwen requests require `qwen-vl-utils`. "
            "Please install it in inference/text/requirements.txt."
        ) from e

    image_inputs = None
    video_inputs = None
    video_processor_kwargs: Dict[str, Any] = {}
    vision_result = None
    try:
        vision_result = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        try:
            vision_result = process_vision_info(messages, return_video_metadata=True)
        except TypeError:
            vision_result = process_vision_info(messages)

    if isinstance(vision_result, tuple):
        if len(vision_result) >= 1:
            image_inputs = vision_result[0]
        if len(vision_result) >= 2:
            video_inputs = vision_result[1]
        if len(vision_result) >= 3 and isinstance(vision_result[2], dict):
            video_processor_kwargs = dict(vision_result[2])

    payload: Dict[str, Any] = {}
    if image_inputs is not None:
        payload["image"] = image_inputs
    if video_inputs is not None:
        payload["video"] = video_inputs
    return payload, video_processor_kwargs


def _convert_openai_part_to_qwen_local(part: Dict[str, Any]) -> Dict[str, Any]:
    raw_type = str(part.get("type") or "").strip().lower()
    if raw_type == "image_url":
        image_url = part.get("image_url")
        url = ""
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "").strip()
        elif isinstance(image_url, str):
            url = image_url.strip()
        converted: Dict[str, Any] = {"type": "image", "image": url}
        for key in ("detail", "min_pixels", "max_pixels", "resized_height", "resized_width"):
            value = part.get(key)
            if value not in (None, ""):
                converted[key] = value
        return converted

    if raw_type == "video_url":
        video_url = part.get("video_url")
        url = ""
        if isinstance(video_url, dict):
            url = str(video_url.get("url") or "").strip()
        elif isinstance(video_url, str):
            url = video_url.strip()
        converted = {"type": "video", "video": url}
        for key in (
            "fps",
            "nframes",
            "min_frames",
            "max_frames",
            "min_pixels",
            "max_pixels",
            "total_pixels",
            "resized_height",
            "resized_width",
            "video_start",
            "video_end",
        ):
            value = part.get(key)
            if value not in (None, ""):
                converted[key] = value
        return converted

    return dict(part)


def _adapt_messages_for_qwen_local(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    adapted: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            adapted.append(dict(message))
            continue

        adapted_content: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict):
                adapted_content.append(_convert_openai_part_to_qwen_local(part))
            else:
                adapted_content.append({"type": "text", "text": str(part or "")})
        adapted.append(
            {
                **dict(message),
                "content": adapted_content,
            }
        )
    return adapted


def _extract_vllm_stats(
    *,
    tokenizer: Any,
    prompt: str,
    output: Any,
    completion: Any,
    full_text: str,
    started_at: float,
    first_delta_at: Optional[float],
    finished_at: float,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        try:
            stats["prompt_tokens"] = len(prompt_token_ids)
        except Exception:
            pass
    if "prompt_tokens" not in stats:
        prompt_tokens = _safe_count_tokens(tokenizer, prompt)
        if prompt_tokens is not None:
            stats["prompt_tokens"] = prompt_tokens

    metrics = getattr(output, "metrics", None)
    output_tokens = None
    if metrics is not None:
        output_tokens = getattr(metrics, "num_generation_tokens", None)
    if output_tokens is None:
        completion_token_ids = getattr(completion, "token_ids", None)
        if completion_token_ids is not None:
            try:
                output_tokens = len(completion_token_ids)
            except Exception:
                output_tokens = None
    if output_tokens is None:
        output_tokens = _safe_count_tokens(tokenizer, full_text)
    if output_tokens is not None:
        try:
            stats["output_tokens"] = int(output_tokens)
        except Exception:
            pass

    ttft_seconds = None
    if metrics is not None:
        ttft_seconds = _safe_optional_float(getattr(metrics, "first_token_latency", None))
        if ttft_seconds is None:
            arrival_time = _safe_optional_float(getattr(metrics, "arrival_time", None))
            first_token_ts = _safe_optional_float(getattr(metrics, "first_token_ts", None))
            if arrival_time is not None and first_token_ts is not None and first_token_ts >= arrival_time:
                ttft_seconds = first_token_ts - arrival_time
    if ttft_seconds is None and first_delta_at is not None:
        ttft_seconds = max(0.0, first_delta_at - started_at)

    total_seconds = max(0.0, finished_at - started_at)
    decode_seconds = None
    if ttft_seconds is not None:
        decode_seconds = max(0.0, total_seconds - ttft_seconds)
    elif first_delta_at is not None:
        decode_seconds = max(0.0, finished_at - first_delta_at)

    stats["total_seconds"] = total_seconds
    if ttft_seconds is not None:
        stats["ttft_seconds"] = ttft_seconds
    if decode_seconds is not None:
        stats["decode_seconds"] = decode_seconds

    tokens_value = stats.get("output_tokens")
    if isinstance(tokens_value, int) and total_seconds > 0:
        stats["tok_s_total"] = float(tokens_value) / total_seconds
        if decode_seconds is not None and decode_seconds > 0:
            stats["tok_s_decode"] = float(tokens_value) / decode_seconds

    return stats


def _build_sampling_params(
    *,
    max_tokens: Any = None,
    temperature: Any = None,
    top_p: Any = None,
    top_k: Any = None,
    presence_penalty: Any = None,
    frequency_penalty: Any = None,
    output_kind: Any = None,
) -> Any:
    try:
        from vllm import SamplingParams
    except Exception as e:
        raise RuntimeError("Failed to import vllm SamplingParams") from e

    kwargs: Dict[str, Any] = {
        "temperature": _coerce_float(temperature, 0.7),
        "max_tokens": _coerce_positive_int(max_tokens, 1024),
    }
    if top_p is not None:
        kwargs["top_p"] = _coerce_float(top_p, 1.0)
    if top_k is not None:
        kwargs["top_k"] = _coerce_positive_int(top_k, 0)
    if presence_penalty is not None:
        kwargs["presence_penalty"] = _coerce_float(presence_penalty, 0.0)
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = _coerce_float(frequency_penalty, 0.0)
    if output_kind is not None:
        kwargs["output_kind"] = output_kind
    return SamplingParams(**kwargs)


async def stream_chat_text(
    bundle: VllmTextBundle,
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
    _AsyncLLMEngine, _AsyncEngineArgs, RequestOutputKind = _import_async_vllm_symbols()

    model_messages = _adapt_messages_for_qwen_local(messages)
    prompt = render_chat_prompt(
        bundle,
        model_messages,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    multi_modal_data, inferred_mm_processor_kwargs = _build_multimodal_payload(model_messages)
    merged_mm_processor_kwargs: Dict[str, Any] = dict(inferred_mm_processor_kwargs or {})
    if mm_processor_kwargs:
        merged_mm_processor_kwargs.update(dict(mm_processor_kwargs))
    image_count, video_count = count_multimodal_parts(model_messages)
    if image_count > 0 or video_count > 0:
        logger.info(
            "Preparing multimodal vLLM request request_id=%s images=%s videos=%s mm_processor_kwargs=%s",
            request_id,
            image_count,
            video_count,
            merged_mm_processor_kwargs,
        )
    started_at = time.perf_counter()
    first_delta_at: Optional[float] = None
    output_text_parts: List[str] = []
    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        output_kind=RequestOutputKind.DELTA,
    )

    prompt_input: Any = prompt
    if multi_modal_data or merged_mm_processor_kwargs:
        prompt_input = {"prompt": prompt}
        if multi_modal_data:
            prompt_input["multi_modal_data"] = multi_modal_data
        if merged_mm_processor_kwargs:
            prompt_input["mm_processor_kwargs"] = merged_mm_processor_kwargs

    generate_kwargs: Dict[str, Any] = {
        "request_id": request_id,
        "prompt": prompt_input,
        "sampling_params": sampling_params,
    }

    try:
        async for output in bundle.engine.generate(**generate_kwargs):
            completions = getattr(output, "outputs", None) or []
            if not completions and getattr(output, "finished", False):
                finished_at = time.perf_counter()
                payload = {"delta": "", "finished": True}
                payload.update(
                    _extract_vllm_stats(
                        tokenizer=bundle.tokenizer,
                        prompt=prompt,
                        output=output,
                        completion=None,
                        full_text="".join(output_text_parts),
                        started_at=started_at,
                        first_delta_at=first_delta_at,
                        finished_at=finished_at,
                    )
                )
                yield payload
                break

            for completion in completions:
                delta = str(getattr(completion, "text", "") or "")
                if delta:
                    output_text_parts.append(delta)
                    if first_delta_at is None:
                        first_delta_at = time.perf_counter()
                payload = {
                    "delta": delta,
                    "finished": bool(getattr(output, "finished", False)),
                    "finish_reason": getattr(completion, "finish_reason", None),
                }
                if payload["finished"]:
                    finished_at = time.perf_counter()
                    payload.update(
                        _extract_vllm_stats(
                            tokenizer=bundle.tokenizer,
                            prompt=prompt,
                            output=output,
                            completion=completion,
                            full_text="".join(output_text_parts),
                            started_at=started_at,
                            first_delta_at=first_delta_at,
                            finished_at=finished_at,
                        )
                    )
                if delta or payload["finished"]:
                    yield payload
            if getattr(output, "finished", False):
                break
    except asyncio.CancelledError:
        await abort_chat_request(bundle, request_id)
        raise


async def generate_chat_text(
    bundle: VllmTextBundle,
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


async def abort_chat_request(bundle: VllmTextBundle, request_id: Optional[str]) -> None:
    if not request_id:
        return
    try:
        await bundle.engine.abort(request_id)
    except Exception as e:
        logger.warning("Failed to abort vLLM request_id=%s: %s", request_id, e)


def shutdown_vllm_text_bundle(bundle: VllmTextBundle) -> None:
    try:
        bundle.engine.shutdown()
    except Exception as e:
        logger.warning("Failed to shutdown vLLM engine for model_ref=%s: %s", bundle.model_ref, e)

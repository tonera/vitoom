from __future__ import annotations

import importlib
import logging
import random
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Sequence

from common.logger import get_logger

from audio.runtime.runtime_resolver import AudioRuntimePolicy

logger = get_logger(__name__)

DEFAULT_QWEN_TTS_SAMPLE_RATE = 24000
DEFAULT_QWEN_CUSTOM_SPEAKER = "Vivian"
DEFAULT_QWEN_TTS_BACKEND = "transformers"
NANO_QWEN_TTS_BACKEND = "nano_vllm"

_QWEN_SPEAKER_CANONICAL = {
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle-fu": "Uncle_Fu",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "ono-anna": "Ono_Anna",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
}

_QWEN_LANGUAGE_ALIASES = {
    "": "Auto",
    "auto": "Auto",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "ko-kr": "Korean",
    "de": "German",
    "de-de": "German",
    "fr": "French",
    "fr-fr": "French",
    "ru": "Russian",
    "ru-ru": "Russian",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "es": "Spanish",
    "es-es": "Spanish",
    "it": "Italian",
    "it-it": "Italian",
    "chinese": "Chinese",
    "english": "English",
    "japanese": "Japanese",
    "korean": "Korean",
    "german": "German",
    "french": "French",
    "russian": "Russian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "italian": "Italian",
}


def _get_qwen_tts_version() -> str:
    try:
        return str(version("qwen-tts"))
    except (PackageNotFoundError, Exception):
        return "unknown"


def _normalize_choice(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def normalize_qwen_language(value: Any, *, supported_languages: Sequence[str] | None = None) -> str:
    raw = str(value or "").strip()
    key = _normalize_choice(raw)
    canonical = _QWEN_LANGUAGE_ALIASES.get(key)
    if canonical is None:
        canonical = raw if raw else "Auto"

    supported_map = {
        _normalize_choice(item): str(item).strip()
        for item in (supported_languages or [])
        if str(item).strip()
    }
    if supported_map:
        if _normalize_choice(canonical) in supported_map:
            return supported_map[_normalize_choice(canonical)]
        if not raw:
            auto_key = _normalize_choice("Auto")
            if auto_key in supported_map:
                return supported_map[auto_key]

    return canonical


def resolve_qwen_custom_speaker(
    value: Any,
    *,
    supported_speakers: Sequence[str] | None = None,
) -> str:
    raw = str(value or "").strip()
    supported_map = {
        _normalize_choice(item): str(item).strip()
        for item in (supported_speakers or [])
        if str(item).strip()
    }

    if not raw:
        fallback_key = _normalize_choice(DEFAULT_QWEN_CUSTOM_SPEAKER)
        if fallback_key in supported_map:
            return supported_map[fallback_key]
        if supported_map:
            return next(iter(supported_map.values()))
        return DEFAULT_QWEN_CUSTOM_SPEAKER

    key = _normalize_choice(raw)
    canonical = _QWEN_SPEAKER_CANONICAL.get(key)
    if canonical is None and key in supported_map:
        canonical = supported_map[key]
    if canonical is None:
        supported_text = sorted(set(supported_map.values())) or sorted(set(_QWEN_SPEAKER_CANONICAL.values()))
        fallback = random.choice(supported_text)
        logger.warning(
            "Unsupported qwen-tts speaker=%r; fallback to random speaker=%r. Supported speakers: %s",
            raw,
            fallback,
            supported_text,
        )
        return fallback

    if supported_map and _normalize_choice(canonical) not in supported_map:
        supported_text = sorted(set(supported_map.values()))
        fallback = random.choice(supported_text)
        logger.warning(
            "Qwen-tts speaker=%r resolved to %r but current model does not support it; "
            "fallback to random speaker=%r. Supported speakers: %s",
            raw,
            canonical,
            fallback,
            supported_text,
        )
        return fallback
    return canonical


def _resolve_attn_implementation(attn_implementation: str) -> str:
    if attn_implementation != "flash_attention_2":
        return attn_implementation or "sdpa"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        logger.info("flash_attn is not installed; using sdpa for qwen-tts")
        return "sdpa"
    return "flash_attention_2"


def _resolve_target_device(policy: AudioRuntimePolicy) -> str:
    if policy.device == "cuda":
        try:
            import torch

            return f"cuda:{int(torch.cuda.current_device())}"
        except Exception:
            return "cuda:0"
    return policy.device


def _resolve_initial_device_map(policy: AudioRuntimePolicy) -> str | None:
    if policy.device == "cuda":
        return _resolve_target_device(policy)
    if policy.device == "cpu":
        return "cpu"
    return None


def _describe_load_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    described = dict(kwargs)
    if "torch_dtype" in described:
        described["torch_dtype"] = str(described["torch_dtype"]).replace("torch.", "")
    if "dtype" in described:
        described["dtype"] = str(described["dtype"]).replace("torch.", "")
    return described


def _finalize_loaded_model(model: Any, policy: AudioRuntimePolicy, *, used_device_map: str | None) -> Any:
    target_device = _resolve_target_device(policy)
    should_move = (used_device_map is None and target_device not in {"", "cpu"})

    target = getattr(model, "model", None)
    if should_move and target is not None and hasattr(target, "to"):
        try:
            target.to(target_device)
        except Exception:
            logger.warning("Failed to move qwen-tts model to device=%s", target_device, exc_info=True)
    elif should_move and hasattr(model, "to"):
        try:
            model.to(target_device)
        except Exception:
            logger.warning("Failed to move qwen-tts wrapper to device=%s", target_device, exc_info=True)

    if target is not None and hasattr(target, "eval"):
        target.eval()
    elif hasattr(model, "eval"):
        model.eval()
    return model


def _resolve_sample_rate(model: Any) -> int:
    candidates = [
        getattr(model, "sample_rate", None),
        getattr(getattr(model, "processor", None), "sample_rate", None),
        getattr(getattr(getattr(model, "processor", None), "audio_tokenizer", None), "sample_rate", None),
        getattr(getattr(getattr(model, "processor", None), "feature_extractor", None), "sampling_rate", None),
        getattr(getattr(getattr(model, "model", None), "config", None), "sample_rate", None),
        getattr(getattr(getattr(model, "model", None), "generation_config", None), "sample_rate", None),
    ]
    for candidate in candidates:
        try:
            sample_rate = int(candidate)
        except Exception:
            continue
        if sample_rate > 0:
            return sample_rate
    return DEFAULT_QWEN_TTS_SAMPLE_RATE


def _infer_qwen_tts_capabilities(model_ref: str, model: Any) -> Dict[str, bool]:
    name = Path(str(model_ref or "")).name.lower()
    if "customvoice" in name:
        return {
            "custom_voice": True,
            "voice_design": False,
            "voice_clone": False,
            "create_voice_clone_prompt": False,
        }
    if "voicedesign" in name:
        return {
            "custom_voice": False,
            "voice_design": True,
            "voice_clone": False,
            "create_voice_clone_prompt": False,
        }
    if name.endswith("-base") or "base" in name:
        return {
            "custom_voice": False,
            "voice_design": False,
            "voice_clone": True,
            "create_voice_clone_prompt": True,
        }
    return {
        "custom_voice": hasattr(model, "generate_custom_voice"),
        "voice_design": hasattr(model, "generate_voice_design"),
        "voice_clone": hasattr(model, "generate_voice_clone"),
        "create_voice_clone_prompt": hasattr(model, "create_voice_clone_prompt"),
    }


def _resolve_qwen_tts_runtime_options(runtime_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(runtime_cfg or {})
    options: Dict[str, Any] = {
        "tensor_parallel_size": max(int(cfg.get("tensor_parallel_size", 1) or 1), 1),
        "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.9) or 0.9),
        "enforce_eager": bool(cfg.get("enforce_eager", False)),
    }
    for key in ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "kvcache_block_size"):
        raw_value = cfg.get(key)
        if raw_value in (None, ""):
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"qwen-tts runtime.{key} must be > 0, got {raw_value!r}")
        options[key] = value
    return options


def _ensure_nano_qwen3tts_importable() -> None:
    """Append ``inference/third_party`` to ``sys.path`` so ``nano_qwen3tts_vllm`` resolves."""
    package_name = "nano_qwen3tts_vllm"
    if package_name in sys.modules:
        return

    inference_root = Path(__file__).resolve().parents[2]
    third_party_dir = inference_root / "third_party"
    init_file = third_party_dir / package_name / "__init__.py"
    if not init_file.is_file():
        raise RuntimeError(
            f"Vendored nano-qwen3tts-vllm is missing: expected {init_file}"
        )

    third_party_str = str(third_party_dir)
    if third_party_str not in sys.path:
        sys.path.append(third_party_str)


def _load_nano_qwen_tts_model(
    model_ref: str,
    policy: AudioRuntimePolicy,
    runtime_cfg: Dict[str, Any] | None = None,
) -> Any:
    _ensure_nano_qwen3tts_importable()
    try:
        Qwen3TTSInterface = importlib.import_module("nano_qwen3tts_vllm.interface").Qwen3TTSInterface
    except Exception as exc:
        raise RuntimeError(
            "nano-qwen3tts-vllm runtime is not importable. "
            "Please ensure vendored sources are present and runtime dependencies are installed."
        ) from exc

    runtime_options = _resolve_qwen_tts_runtime_options(runtime_cfg)
    load_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_ref,
        "enforce_eager": runtime_options["enforce_eager"],
        "tensor_parallel_size": runtime_options["tensor_parallel_size"],
        "gpu_memory_utilization": runtime_options["gpu_memory_utilization"],
        "local_files_only": (not policy.allow_remote_assets),
    }
    for opt_key in ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "kvcache_block_size"):
        if opt_key in runtime_options:
            load_kwargs[opt_key] = runtime_options[opt_key]
    logger.info(
        "Loading nano-qwen3tts-vllm model_ref=%s kwargs=%s",
        model_ref,
        load_kwargs,
    )
    return Qwen3TTSInterface.from_pretrained(**load_kwargs)


def load_qwen_tts_weight(
    model_ref: str,
    policy: AudioRuntimePolicy,
    backend: str = DEFAULT_QWEN_TTS_BACKEND,
    runtime_cfg: Dict[str, Any] | None = None,
) -> Any:
    """Public: 按 policy 加载一个 Qwen3-TTS 权重包（CustomVoice / VoiceDesign / Base 任一）。

    这里不区分权重的"玩法"—— `Qwen3TTSModel.from_pretrained` 返回的是同一种类型，
    区别仅在于之后该对象暴露 `generate_custom_voice` / `generate_voice_design` /
    `generate_voice_clone` 中的哪几个方法。handler 侧按 `tts_mode` 调用对应方法即可。
    """
    if str(backend or DEFAULT_QWEN_TTS_BACKEND).strip().lower() == NANO_QWEN_TTS_BACKEND:
        return _load_nano_qwen_tts_model(model_ref, policy, runtime_cfg)
    return _load_qwen_tts_model(model_ref, policy)


def _load_qwen_tts_model(model_ref: str, policy: AudioRuntimePolicy) -> Any:
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "qwen-tts runtime is not importable. "
            "Either the 'qwen-tts' package is not installed, or one of its runtime "
            "dependencies is missing (common culprits when installed via --no-deps: "
            "einops, soundfile, onnxruntime, gradio, sox). "
            f"Underlying ImportError: {exc!r}"
        ) from exc

    attn_impl = _resolve_attn_implementation(policy.attn_implementation)
    initial_device_map = _resolve_initial_device_map(policy)
    base_kwargs: Dict[str, Any] = {}
    if initial_device_map is not None:
        base_kwargs["device_map"] = initial_device_map
    if not policy.allow_remote_assets:
        base_kwargs["local_files_only"] = True

    attempts: list[Dict[str, Any]] = []
    for dtype_key in ("torch_dtype", "dtype"):
        candidate = dict(base_kwargs)
        candidate[dtype_key] = policy.torch_dtype
        candidate["attn_implementation"] = attn_impl
        attempts.append(candidate)
        if attn_impl != "sdpa":
            fallback_candidate = dict(candidate)
            fallback_candidate["attn_implementation"] = "sdpa"
            attempts.append(fallback_candidate)

    if initial_device_map is not None:
        cpu_fallback = {k: v for k, v in base_kwargs.items() if k != "device_map"}
        for dtype_key in ("torch_dtype", "dtype"):
            candidate = dict(cpu_fallback)
            candidate[dtype_key] = policy.torch_dtype
            candidate["attn_implementation"] = "sdpa"
            attempts.append(candidate)

    last_exc: Exception | None = None
    for attempt_idx, kwargs in enumerate(attempts, start=1):
        try:
            logger.info(
                "Loading qwen-tts model_ref=%s attempt=%s kwargs=%s",
                model_ref,
                attempt_idx,
                _describe_load_kwargs(kwargs),
            )
            model = Qwen3TTSModel.from_pretrained(model_ref, **kwargs)
            return _finalize_loaded_model(model, policy, used_device_map=kwargs.get("device_map"))
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "qwen-tts load attempt=%s failed model_ref=%s kwargs=%s error=%s",
                attempt_idx,
                model_ref,
                _describe_load_kwargs(kwargs),
                exc,
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to load qwen-tts model from {model_ref}")


def load_tts_bundle(
    model_ref: str,
    policy: AudioRuntimePolicy,
    backend: str = DEFAULT_QWEN_TTS_BACKEND,
    runtime_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    backend_name = str(backend or DEFAULT_QWEN_TTS_BACKEND).strip().lower() or DEFAULT_QWEN_TTS_BACKEND
    logger.info(
        "Loading qwen-tts bundle model_ref=%s backend=%s policy=%s source=%s qwen_tts=%s",
        model_ref,
        backend_name,
        policy.cache_key,
        policy.policy_source,
        _get_qwen_tts_version(),
    )
    model = load_qwen_tts_weight(model_ref, policy, backend_name, runtime_cfg)
    sample_rate = _resolve_sample_rate(model)
    supported_speakers = None
    supported_languages = None
    if backend_name != NANO_QWEN_TTS_BACKEND:
        try:
            supported_speakers = model.get_supported_speakers()
        except Exception:
            logger.info("qwen-tts model did not expose supported speakers", exc_info=True)
        try:
            supported_languages = model.get_supported_languages()
        except Exception:
            logger.info("qwen-tts model did not expose supported languages", exc_info=True)
    else:
        sample_rate = int(
            getattr(getattr(model, "speech_tokenizer", None), "sample_rate", None)
            or sample_rate
            or DEFAULT_QWEN_TTS_SAMPLE_RATE
        )

    capabilities = _infer_qwen_tts_capabilities(model_ref, model)
    logger.info(
        "qwen-tts bundle loaded successfully from %s backend=%s sample_rate=%s speakers=%s languages=%s caps=%s",
        model_ref,
        backend_name,
        sample_rate,
        len(supported_speakers or []),
        len(supported_languages or []),
        capabilities,
    )
    return {
        "device": policy.device,
        "torch_dtype": policy.torch_dtype,
        "model": model,
        "model_ref": model_ref,
        "sample_rate": sample_rate,
        "streaming_variant": (backend_name == NANO_QWEN_TTS_BACKEND),
        "runtime_backend": backend_name,
        "runtime_config": dict(runtime_cfg or {}),
        "runtime_policy": policy,
        "qwen_tts_version": _get_qwen_tts_version(),
        "supported_speakers": supported_speakers,
        "supported_languages": supported_languages,
        "capabilities": capabilities,
    }


async def prewarm_qwen_tts_bundle(
    bundle: Dict[str, Any],
    *,
    logger_: logging.Logger | None = None,
    frames: int = 4,
) -> None:
    """对 nano_vllm 流式 bundle 跑一次最小 generate，把首请求 5s+ TTFB 摊掉。

    具体摊掉的开销包含：``start_zmq_tasks`` 启动 talker / predictor 子进程、
    首批 codec frame 的 vLLM scheduler 暖身、（如开启）CUDA Graph capture。
    transformers 后端无此开销，直接跳过。

    设计点：
    - 只跑 ``frames`` 帧（默认 4，约 0.33s 音频，足够触发上述所有 warmup）；
    - 出错时仅 warn，不抛——产线首请求继续按原路径走，至多 TTFB 没省下；
    - 选择最廉价的 prewarm 入口（custom_voice → voice_design → 跳过），
      voice_clone Base 权重需要 ref_audio，留给真实请求触发首帧。
    """
    log = logger_ or logger
    if not bundle.get("streaming_variant"):
        return
    if str(bundle.get("runtime_backend") or "").strip().lower() != NANO_QWEN_TTS_BACKEND:
        return
    model = bundle.get("model")
    start_fn = getattr(model, "start_zmq_tasks", None) if model is not None else None
    if not callable(start_fn):
        log.warning("[qwen-tts][prewarm] bundle missing start_zmq_tasks; skip")
        return

    caps = bundle.get("capabilities") or {}
    speakers = list(bundle.get("supported_speakers") or [])

    if caps.get("custom_voice"):
        speaker = next(iter(speakers), None) or DEFAULT_QWEN_CUSTOM_SPEAKER
        gen_factory = lambda: model.generate_custom_voice_async(  # noqa: E731
            text="嗯。",
            language="Chinese",
            speaker=speaker,
            instruct=None,
            non_streaming_mode=False,
        )
        path = f"custom_voice(speaker={speaker})"
    elif caps.get("voice_design"):
        gen_factory = lambda: model.generate_voice_design_async(  # noqa: E731
            text="嗯。",
            language="Chinese",
            instruct="自然平和的女声",
            non_streaming_mode=False,
        )
        path = "voice_design"
    else:
        log.info(
            "[qwen-tts][prewarm] no cheap prewarm path for model_ref=%s caps=%s; skip",
            bundle.get("model_ref"),
            caps,
        )
        return

    t0 = time.perf_counter()
    consumed = 0
    try:
        await start_fn()
        gen = gen_factory()
        try:
            async for _chunk in gen:
                consumed += 1
                if consumed >= max(int(frames or 1), 1):
                    break
        finally:
            close = getattr(gen, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:  # pragma: no cover
                    pass
        elapsed = time.perf_counter() - t0
        log.info(
            "[qwen-tts][prewarm] done model_ref=%s path=%s frames=%s elapsed=%.2fs",
            bundle.get("model_ref"),
            path,
            consumed,
            elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.warning(
            "[qwen-tts][prewarm] failed model_ref=%s path=%s frames=%s elapsed=%.2fs err=%r",
            bundle.get("model_ref"),
            path,
            consumed,
            elapsed,
            exc,
            exc_info=True,
        )

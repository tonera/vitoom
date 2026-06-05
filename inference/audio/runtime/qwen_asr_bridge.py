from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Sequence

from common.logger import get_logger

from audio.runtime.runtime_resolver import AudioRuntimePolicy

logger = get_logger(__name__)

DEFAULT_QWEN_ASR_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_QWEN_ASR_FORCED_ALIGNER_DIR_NAME = "Qwen3-ForcedAligner-0.6B"
DEFAULT_QWEN_ASR_MAX_NEW_TOKENS = 1024
DEFAULT_QWEN_ASR_MAX_BATCH_SIZE = 1

_QWEN_ASR_LANGUAGE_ALIASES = {
    "": None,
    "auto": None,
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-tw": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "yue": "Cantonese",
    "zh-yue": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "portuguese": "Portuguese",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "ko": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "russian": "Russian",
    "th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ms": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "filipino": "Filipino",
    "fa": "Persian",
    "persian": "Persian",
    "el": "Greek",
    "greek": "Greek",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "macedonian": "Macedonian",
    "ro": "Romanian",
    "romanian": "Romanian",
    "chinese": "Chinese",
    "english": "English",
}


@dataclass(frozen=True)
class QwenAsrBundleOptions:
    forced_aligner_ref: str | None
    max_new_tokens: int
    max_inference_batch_size: int
    timestamps_requested: bool
    forced_aligner_source: str
    searched_roots: tuple[str, ...]

    @property
    def cache_key(self) -> str:
        return (
            f"forced_aligner={self.forced_aligner_ref or 'none'}|"
            f"max_new_tokens={self.max_new_tokens}|"
            f"max_inference_batch_size={self.max_inference_batch_size}|"
            f"timestamps_requested={int(self.timestamps_requested)}|"
            f"forced_aligner_source={self.forced_aligner_source}"
        )


def _get_qwen_asr_version() -> str:
    try:
        return str(version("qwen-asr"))
    except (PackageNotFoundError, Exception):
        return "unknown"


def _normalize_choice(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def normalize_qwen_asr_language(value: Any) -> str | None:
    raw = str(value or "").strip()
    key = _normalize_choice(raw)
    if key in _QWEN_ASR_LANGUAGE_ALIASES:
        return _QWEN_ASR_LANGUAGE_ALIASES[key]
    return raw or None


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except Exception:
        return default
    return coerced if coerced > 0 else default


def _resolve_optional_model_ref(
    raw: str | None,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise ValueError(f"Optional audio model path not found: {candidate}")
        return str(candidate.resolve())

    for root in (models_dir, weights_dir):
        root_text = str(root or "").strip()
        if not root_text:
            continue
        rooted = (Path(root_text).expanduser().resolve() / text).resolve()
        if rooted.exists():
            return str(rooted)

    return text


def _resolve_local_optional_model_ref(
    raw: str | None,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve()) if candidate.exists() else None

    for root in (models_dir, weights_dir):
        root_text = str(root or "").strip()
        if not root_text:
            continue
        rooted = (Path(root_text).expanduser().resolve() / text).resolve()
        if rooted.exists():
            return str(rooted)
    return None


def resolve_qwen_asr_bundle_options(
    params: Any,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
) -> QwenAsrBundleOptions:
    model_cfg = getattr(params, "model_cfg", None)
    runtime_cfg = model_cfg.get("runtime") if isinstance(model_cfg, dict) and isinstance(model_cfg.get("runtime"), dict) else {}
    searched_roots = tuple(str(x).strip() for x in (models_dir, weights_dir) if str(x or "").strip())

    timestamps_enabled = bool(getattr(params, "timestamps", False))
    raw_aligner = None
    forced_aligner_source = "none"
    if timestamps_enabled:
        raw_aligner = (
            runtime_cfg.get("forced_aligner")
            or (model_cfg.get("forced_aligner") if isinstance(model_cfg, dict) else None)
        )
    if raw_aligner:
        forced_aligner_ref = _resolve_optional_model_ref(
            raw_aligner,
            models_dir=models_dir,
            weights_dir=weights_dir,
        )
        forced_aligner_source = "explicit"
    elif timestamps_enabled:
        forced_aligner_ref = _resolve_local_optional_model_ref(
            DEFAULT_QWEN_ASR_FORCED_ALIGNER_DIR_NAME,
            models_dir=models_dir,
            weights_dir=weights_dir,
        )
        forced_aligner_source = "auto_local" if forced_aligner_ref else "missing_local_default"
    else:
        forced_aligner_ref = None

    max_new_tokens = _coerce_positive_int(
        runtime_cfg.get("max_new_tokens")
        if "max_new_tokens" in runtime_cfg else (
            model_cfg.get("max_new_tokens") if isinstance(model_cfg, dict) else None
        ),
        DEFAULT_QWEN_ASR_MAX_NEW_TOKENS,
    )
    max_inference_batch_size = _coerce_positive_int(
        runtime_cfg.get("max_inference_batch_size")
        if "max_inference_batch_size" in runtime_cfg else (
            model_cfg.get("max_inference_batch_size") if isinstance(model_cfg, dict) else None
        ),
        DEFAULT_QWEN_ASR_MAX_BATCH_SIZE,
    )
    return QwenAsrBundleOptions(
        forced_aligner_ref=forced_aligner_ref,
        max_new_tokens=max_new_tokens,
        max_inference_batch_size=max_inference_batch_size,
        timestamps_requested=timestamps_enabled,
        forced_aligner_source=forced_aligner_source,
        searched_roots=searched_roots,
    )


def _resolve_attn_implementation(attn_implementation: str) -> str:
    if attn_implementation != "flash_attention_2":
        return attn_implementation or "sdpa"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        logger.info("flash_attn is not installed; using sdpa for qwen-asr")
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
            logger.warning("Failed to move qwen-asr model to device=%s", target_device, exc_info=True)
    elif should_move and hasattr(model, "to"):
        try:
            model.to(target_device)
        except Exception:
            logger.warning("Failed to move qwen-asr wrapper to device=%s", target_device, exc_info=True)

    if target is not None and hasattr(target, "eval"):
        target.eval()
    elif hasattr(model, "eval"):
        model.eval()
    return model


def load_asr_bundle(
    model_ref: str,
    policy: AudioRuntimePolicy,
    options: QwenAsrBundleOptions,
) -> Dict[str, Any]:
    if options.timestamps_requested and not options.forced_aligner_ref:
        raise RuntimeError(
            "Qwen-asr timestamps=true requires a local forced aligner. "
            f"Expected '{DEFAULT_QWEN_ASR_FORCED_ALIGNER_DIR_NAME}' under "
            f"{list(options.searched_roots) or ['<models_dir>', '<weights_dir>']}, "
            "or set model_cfg.runtime.forced_aligner to an explicit model/path "
            "and keep allow_remote_assets=true if you want remote download."
        )
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "qwen-asr runtime is not installed. "
            "Please install the 'qwen-asr' package before using Qwen-asr audio models."
        ) from exc

    attn_impl = _resolve_attn_implementation(policy.attn_implementation)
    initial_device_map = _resolve_initial_device_map(policy)
    base_kwargs: Dict[str, Any] = {
        "max_inference_batch_size": options.max_inference_batch_size,
        "max_new_tokens": options.max_new_tokens,
    }
    if initial_device_map is not None:
        base_kwargs["device_map"] = initial_device_map
    if not policy.allow_remote_assets:
        base_kwargs["local_files_only"] = True
    if options.forced_aligner_ref:
        base_kwargs["forced_aligner"] = options.forced_aligner_ref
        forced_aligner_kwargs: Dict[str, Any] = {}
        if initial_device_map is not None:
            forced_aligner_kwargs["device_map"] = initial_device_map
        if not policy.allow_remote_assets:
            forced_aligner_kwargs["local_files_only"] = True
        base_kwargs["forced_aligner_kwargs"] = forced_aligner_kwargs

    attempts: list[Dict[str, Any]] = []
    for dtype_key in ("dtype", "torch_dtype"):
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
        if isinstance(cpu_fallback.get("forced_aligner_kwargs"), dict):
            cpu_fallback["forced_aligner_kwargs"] = {
                k: v for k, v in cpu_fallback["forced_aligner_kwargs"].items() if k != "device_map"
            }
        for dtype_key in ("dtype", "torch_dtype"):
            candidate = dict(cpu_fallback)
            candidate[dtype_key] = policy.torch_dtype
            candidate["attn_implementation"] = "sdpa"
            attempts.append(candidate)

    last_exc: Exception | None = None
    for attempt_idx, kwargs in enumerate(attempts, start=1):
        try:
            logger.info(
                "Loading qwen-asr model_ref=%s attempt=%s kwargs=%s",
                model_ref,
                attempt_idx,
                _describe_load_kwargs(kwargs),
            )
            model = Qwen3ASRModel.from_pretrained(model_ref, **kwargs)
            model = _finalize_loaded_model(model, policy, used_device_map=kwargs.get("device_map"))
            logger.info(
                "qwen-asr bundle loaded successfully from %s forced_aligner=%s qwen_asr=%s",
                model_ref,
                options.forced_aligner_ref,
                _get_qwen_asr_version(),
            )
            return {
                "device": policy.device,
                "torch_dtype": policy.torch_dtype,
                "model": model,
                "sample_rate": 16000,
                "streaming_variant": False,
                "runtime_policy": policy,
                "qwen_asr_version": _get_qwen_asr_version(),
                "forced_aligner_ref": options.forced_aligner_ref,
                "max_new_tokens": options.max_new_tokens,
                "max_inference_batch_size": options.max_inference_batch_size,
            }
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "qwen-asr load attempt=%s failed model_ref=%s kwargs=%s error=%s",
                attempt_idx,
                model_ref,
                _describe_load_kwargs(kwargs),
                exc,
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to load qwen-asr model from {model_ref}")

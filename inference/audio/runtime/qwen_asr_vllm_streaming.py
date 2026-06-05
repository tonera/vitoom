from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import sys
import traceback
from typing import Any, Dict, Optional

from common.logger import get_logger

from audio.runtime.qwen_asr_bridge import (
    DEFAULT_QWEN_ASR_FORCED_ALIGNER_DIR_NAME,
    QwenAsrBundleOptions,
)
from audio.runtime.runtime_resolver import AudioRuntimePolicy

logger = get_logger(__name__)

DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.80
DEFAULT_STREAMING_UNFIXED_CHUNK_NUM = 2
DEFAULT_STREAMING_UNFIXED_TOKEN_NUM = 5
DEFAULT_STREAMING_CHUNK_SIZE_SEC = 2.0


@dataclass(frozen=True)
class QwenAsrVllmOptions:
    gpu_memory_utilization: float
    max_model_len: int | None
    enforce_eager: bool
    streaming_unfixed_chunk_num: int
    streaming_unfixed_token_num: int
    streaming_chunk_size_sec: float


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(raw or "").replace("-", ".").split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed_version(dist_name: str) -> str | None:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None


def _validate_qwen_asr_vllm_compatibility() -> None:
    vllm_version = _installed_version("vllm")
    qwen_asr_version = _installed_version("qwen-asr")
    if not vllm_version or not qwen_asr_version:
        return

    if _parse_version_tuple(vllm_version) >= (0, 16) and _parse_version_tuple(qwen_asr_version) <= (0, 0, 6):
        raise RuntimeError(
            "qwen-asr vLLM runtime compatibility check failed: "
            f"qwen-asr=={qwen_asr_version} is not compatible with vllm=={vllm_version}. "
            "For realtime Qwen-ASR, please pin vllm to 0.14 "
        )


def _validate_vllm_importable() -> None:
    """Fail early with the real vLLM import error before qwen-asr wraps it."""
    try:
        import torch  # noqa: F401  # preload libtorch_cuda.so before vLLM C extension loads
        from vllm import LLM as _VllmLLM  # noqa: F401
        from vllm import SamplingParams as _SamplingParams  # noqa: F401
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        logger.error(
            "qwen-asr vLLM backend import failed: python=%s vllm=%s qwen-asr=%s error=%s",
            sys.executable,
            _installed_version("vllm") or "<not installed>",
            _installed_version("qwen-asr") or "<not installed>",
            detail,
            exc_info=True,
        )
        raise RuntimeError(
            "qwen-asr vLLM backend is not importable in this Python process. "
            f"python={sys.executable}; "
            f"vllm={_installed_version('vllm') or '<not installed>'}; "
            f"qwen-asr={_installed_version('qwen-asr') or '<not installed>'}; "
            f"cause={detail}"
        ) from exc


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except Exception:
        return default
    return coerced if coerced > 0 else default


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        coerced = float(value)
    except Exception:
        return default
    return coerced if coerced > 0 else default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _runtime_cfg(source: Any) -> Dict[str, Any]:
    if isinstance(source, dict):
        runtime_cfg = source.get("runtime")
        if isinstance(runtime_cfg, dict):
            return dict(runtime_cfg)
        return dict(source)
    config_obj = getattr(source, "config", None)
    if isinstance(config_obj, dict):
        runtime_cfg = config_obj.get("runtime")
        if isinstance(runtime_cfg, dict):
            return dict(runtime_cfg)
    model_cfg = getattr(source, "model_cfg", None)
    if isinstance(model_cfg, dict):
        runtime_cfg = model_cfg.get("runtime")
        if isinstance(runtime_cfg, dict):
            return dict(runtime_cfg)
    return {}


def resolve_qwen_asr_vllm_options(source: Any) -> QwenAsrVllmOptions:
    runtime_cfg = _runtime_cfg(source)
    # 与 `service_text_qwen.yaml` 一致：vLLM 项与 `backend` 同级，放在 `runtime.vllm`。
    # 兼容旧写法 `runtime.qwen_asr.vllm`（本服务已是 Qwen-ASR，无需再套一层 qwen_asr）。
    vllm_cfg = runtime_cfg.get("vllm") if isinstance(runtime_cfg.get("vllm"), dict) else {}
    if not vllm_cfg:
        qwen_cfg = runtime_cfg.get("qwen_asr") if isinstance(runtime_cfg.get("qwen_asr"), dict) else {}
        vllm_cfg = qwen_cfg.get("vllm") if isinstance(qwen_cfg.get("vllm"), dict) else {}
    streaming_cfg = vllm_cfg.get("streaming") if isinstance(vllm_cfg.get("streaming"), dict) else {}

    max_model_len_raw = vllm_cfg.get("max_model_len")
    try:
        max_model_len = int(max_model_len_raw) if max_model_len_raw not in (None, "") else None
    except Exception:
        max_model_len = None

    return QwenAsrVllmOptions(
        gpu_memory_utilization=_coerce_positive_float(
            vllm_cfg.get("gpu_memory_utilization"),
            DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        ),
        max_model_len=max_model_len,
        enforce_eager=_coerce_bool(vllm_cfg.get("enforce_eager"), default=False),
        streaming_unfixed_chunk_num=_coerce_positive_int(
            streaming_cfg.get("unfixed_chunk_num"),
            DEFAULT_STREAMING_UNFIXED_CHUNK_NUM,
        ),
        streaming_unfixed_token_num=_coerce_positive_int(
            streaming_cfg.get("unfixed_token_num"),
            DEFAULT_STREAMING_UNFIXED_TOKEN_NUM,
        ),
        streaming_chunk_size_sec=_coerce_positive_float(
            streaming_cfg.get("chunk_size_sec"),
            DEFAULT_STREAMING_CHUNK_SIZE_SEC,
        ),
    )


def load_vllm_asr_bundle(
    model_ref: str,
    policy: AudioRuntimePolicy,
    options: QwenAsrBundleOptions,
    vllm_options: QwenAsrVllmOptions,
) -> Dict[str, Any]:
    _validate_qwen_asr_vllm_compatibility()
    _validate_vllm_importable()
    if options.timestamps_requested and not options.forced_aligner_ref:
        raise RuntimeError(
            "Qwen-asr timestamps=true requires a forced aligner before using vLLM. "
            f"Expected '{DEFAULT_QWEN_ASR_FORCED_ALIGNER_DIR_NAME}' under "
            f"{list(options.searched_roots) or ['<models_dir>', '<weights_dir>']}, "
            "or set config.runtime.forced_aligner / model_cfg.runtime.forced_aligner "
            "to an explicit model/path."
        )
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "qwen-asr[vllm] runtime is not installed. "
            "Please install the 'qwen-asr[vllm]' package on the runtime machine."
        ) from exc

    kwargs: Dict[str, Any] = {
        "model": model_ref,
        "gpu_memory_utilization": vllm_options.gpu_memory_utilization,
        "max_inference_batch_size": options.max_inference_batch_size,
        "max_new_tokens": options.max_new_tokens,
    }
    if vllm_options.max_model_len is not None:
        kwargs["max_model_len"] = vllm_options.max_model_len
    if vllm_options.enforce_eager:
        kwargs["enforce_eager"] = True
    if options.forced_aligner_ref:
        kwargs["forced_aligner"] = options.forced_aligner_ref
        kwargs["forced_aligner_kwargs"] = {}

    logger.info("Loading qwen-asr vLLM bundle model_ref=%s kwargs=%s", model_ref, kwargs)
    model = Qwen3ASRModel.LLM(**kwargs)
    return {
        "device": "vllm",
        "torch_dtype": None,
        "model": model,
        "sample_rate": 16000,
        "streaming_variant": True,
        "runtime_policy": policy,
        "forced_aligner_ref": options.forced_aligner_ref,
        "max_new_tokens": options.max_new_tokens,
        "max_inference_batch_size": options.max_inference_batch_size,
        "vllm_options": vllm_options,
    }


class QwenAsrVllmStreamingSession:
    def __init__(self, bundle: Dict[str, Any]):
        self._model = bundle["model"]
        self._options: QwenAsrVllmOptions = bundle["vllm_options"]
        self._state = self._model.init_streaming_state(
            unfixed_chunk_num=self._options.streaming_unfixed_chunk_num,
            unfixed_token_num=self._options.streaming_unfixed_token_num,
            chunk_size_sec=self._options.streaming_chunk_size_sec,
        )
        self._last_text = ""

    def push_chunk(self, audio_chunk: Any) -> Dict[str, Any]:
        self._model.streaming_transcribe(audio_chunk, self._state)
        return self._snapshot(is_final=False)

    def finish(self) -> Dict[str, Any]:
        self._model.finish_streaming_transcribe(self._state)
        return self._snapshot(is_final=True)

    @staticmethod
    def _longest_common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _snapshot(self, *, is_final: bool) -> Dict[str, Any]:
        text = str(getattr(self._state, "text", "") or "").strip()
        language = str(getattr(self._state, "language", "") or "").strip()
        # qwen-asr 流式解码的末尾 unfixed_token_num 个 token 可能被下一次改写，
        # 不能简单用 startswith 判前缀；用最长公共前缀计算真正的增量，
        # 并把被改写的旧尾部通过 replaced 字段暴露出来，方便消费者做"回退+覆盖"。
        prefix_len = self._longest_common_prefix_len(self._last_text, text)
        replaced = self._last_text[prefix_len:]
        delta = text[prefix_len:]
        self._last_text = text
        return {
            "text": text,
            "delta": delta,
            "replaced": replaced,
            "language": language,
            "is_final": is_final,
        }


def create_vllm_streaming_session(bundle: Dict[str, Any]) -> QwenAsrVllmStreamingSession:
    return QwenAsrVllmStreamingSession(bundle)

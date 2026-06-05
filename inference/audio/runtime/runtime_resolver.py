from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.logger import get_logger

logger = get_logger(__name__)

# VoxCPM 服务 YAML 中 `config.runtime` 下与 `service_text_qwen.yaml` 相同风格的
# backend 分块键；合并后再传给 loader（公共项与 backend 同级，专属项在子 dict）。
_VOXCPM_BACKEND_RUNTIME_BLOCKS = frozenset({"transformers", "nano_vllm_voxcpm"})
_QWEN_TTS_BACKEND_RUNTIME_BLOCKS = frozenset({"transformers", "nano_vllm"})


@dataclass(frozen=True)
class AudioRuntimePolicy:
    audio_mode: str
    low_vram: bool
    fast_mode: bool
    policy_source: str
    device: str
    torch_dtype: Any
    torch_dtype_name: str
    attn_implementation: str
    device_map: str | None
    low_cpu_mem_usage: bool
    allow_remote_assets: bool

    @property
    def cache_key(self) -> str:
        return (
            f"mode={self.audio_mode}|low_vram={int(self.low_vram)}|"
            f"fast_mode={int(self.fast_mode)}|"
            f"device={self.device}|dtype={self.torch_dtype_name}|"
            f"attn={self.attn_implementation}|device_map={self.device_map or 'none'}|"
            f"low_cpu_mem_usage={int(self.low_cpu_mem_usage)}|"
            f"allow_remote_assets={int(self.allow_remote_assets)}"
        )


def _runtime_cfg(source: Any) -> dict[str, Any]:
    if isinstance(source, dict):
        runtime_cfg = source.get("runtime")
        return dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}

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


def _extract_low_vram_setting(params: Any) -> tuple[bool, str]:
    model_cfg = getattr(params, "model_cfg", None)
    if isinstance(model_cfg, dict):
        runtime_cfg = model_cfg.get("runtime")
        if isinstance(runtime_cfg, dict):
            for key in ("low_vram", "force_offload"):
                if key in runtime_cfg:
                    return _coerce_bool(runtime_cfg.get(key)), f"model_cfg.runtime.{key}"
        for key in ("low_vram", "force_offload"):
            if key in model_cfg:
                return _coerce_bool(model_cfg.get(key)), f"model_cfg.{key}"

    if hasattr(params, "low_vram"):
        return _coerce_bool(getattr(params, "low_vram", False)), "params.low_vram"
    return False, "default"


def _extract_allow_remote_assets(params: Any) -> tuple[bool, str]:
    model_cfg = getattr(params, "model_cfg", None)
    if isinstance(model_cfg, dict):
        runtime_cfg = model_cfg.get("runtime")
        if isinstance(runtime_cfg, dict):
            for key in ("allow_remote_assets", "allow_download", "offline"):
                if key in runtime_cfg:
                    value = runtime_cfg.get(key)
                    if key == "offline":
                        return (not _coerce_bool(value, default=False)), f"model_cfg.runtime.{key}"
                    return _coerce_bool(value, default=True), f"model_cfg.runtime.{key}"
        for key in ("allow_remote_assets", "allow_download", "offline"):
            if key in model_cfg:
                value = model_cfg.get(key)
                if key == "offline":
                    return (not _coerce_bool(value, default=False)), f"model_cfg.{key}"
                return _coerce_bool(value, default=True), f"model_cfg.{key}"
    return True, "default"


def _extract_fast_mode_setting(params: Any) -> tuple[bool, str]:
    model_cfg = getattr(params, "model_cfg", None)
    if isinstance(model_cfg, dict):
        runtime_cfg = model_cfg.get("runtime")
        if isinstance(runtime_cfg, dict) and "fast_mode" in runtime_cfg:
            return _coerce_bool(runtime_cfg.get("fast_mode"), default=True), "model_cfg.runtime.fast_mode"
        if "fast_mode" in model_cfg:
            return _coerce_bool(model_cfg.get("fast_mode"), default=True), "model_cfg.fast_mode"

    if hasattr(params, "fast_mode"):
        return _coerce_bool(getattr(params, "fast_mode", True), default=True), "params.fast_mode"
    return True, "default"


def merge_voxcpm_loader_runtime_cfg(
    runtime: dict[str, Any] | None,
    *,
    backend: str,
) -> dict[str, Any]:
    """将 ``config.runtime`` 合并为传给 VoxCPM bundle loader 的扁平 dict。

    - 与 ``backend`` 同级的键视为各后端可共享的公共项（以及向后兼容的顶层 legacy 项）。
    - ``transformers`` / ``nano_vllm_voxcpm`` 子 dict 为对应后端专属，会覆盖同名公共键。
    - 返回结果不含 ``backend`` 键，避免进入引擎 kwargs。
    """
    if not isinstance(runtime, dict):
        return {}
    merged: dict[str, Any] = {}
    for key, value in runtime.items():
        if key in _VOXCPM_BACKEND_RUNTIME_BLOCKS:
            continue
        merged[key] = value
    section = runtime.get(backend)
    if isinstance(section, dict):
        merged = {**merged, **section}
    merged.pop("backend", None)
    return merged


def merge_qwen_tts_loader_runtime_cfg(
    runtime: dict[str, Any] | None,
    *,
    backend: str,
) -> dict[str, Any]:
    """将 ``config.runtime`` 合并为传给 Qwen-TTS bundle loader 的扁平 dict。

    布局与 ``service_text_qwen.yaml`` / ``voxcpm.yaml`` 一致：公共项与 ``backend`` 同级；
    ``transformers`` / ``nano_vllm`` 为各推理后端专属子 dict（覆盖同名公共键）。
    返回结果不含 ``backend``。
    """
    if not isinstance(runtime, dict):
        return {}
    merged: dict[str, Any] = {}
    for key, value in runtime.items():
        if key in _QWEN_TTS_BACKEND_RUNTIME_BLOCKS:
            continue
        merged[key] = value
    section = runtime.get(backend)
    if isinstance(section, dict):
        merged = {**merged, **section}
    merged.pop("backend", None)
    return merged


def resolve_audio_backend(source: Any, *, default: str = "transformers") -> str:
    runtime_cfg = _runtime_cfg(source)
    backend_raw = str(runtime_cfg.get("backend") or "").strip().lower()
    if not backend_raw:
        return default
    if backend_raw in {"hf", "huggingface", "transformers"}:
        return "transformers"
    if backend_raw == "vllm":
        return "vllm"
    if backend_raw in {"nano_vllm", "nano-vllm", "nano_qwen3tts_vllm", "nano-qwen3tts-vllm"}:
        return "nano_vllm"
    if backend_raw in {
        "nano_vllm_voxcpm",
        "nano-vllm-voxcpm",
        "nanovllm_voxcpm",
        "nanovllm-voxcpm",
    }:
        return "nano_vllm_voxcpm"
    raise ValueError(
        f"Unsupported audio runtime.backend={backend_raw!r}; expected one of "
        "transformers/vllm/nano_vllm/nano_vllm_voxcpm"
    )


def resolve_audio_runtime(params: Any) -> str:
    raw_family = str(getattr(params, "family", None) or "").strip().lower()
    if not raw_family:
        raise ValueError("audio task requires family to select runtime")

    alias_map = {
        "voxcpm": "voxcpm",
        "qwen-tts": "qwen_tts",
        "qwen_tts": "qwen_tts",
        "qwentts": "qwen_tts",
        "qwen-asr": "qwen_asr",
        "qwen_asr": "qwen_asr",
        "qwenasr": "qwen_asr",
    }
    runtime = alias_map.get(raw_family)
    if runtime:
        return runtime

    raise ValueError(
        f"Unsupported audio family='{getattr(params, 'family', None)}'. "
        "Current audio runtime only supports family in {Voxcpm, Qwen-tts, Qwen-asr}."
    )


def resolve_audio_runtime_policy(params: Any, *, audio_mode: str) -> AudioRuntimePolicy:
    import torch

    low_vram, low_vram_source = _extract_low_vram_setting(params)
    allow_remote_assets, allow_remote_source = _extract_allow_remote_assets(params)
    requested_fast_mode, fast_mode_source = _extract_fast_mode_setting(params)
    fast_mode = False
    if requested_fast_mode:
        logger.info(
            "Ignoring fast_mode for audio runtime audio_mode=%s source=%s",
            audio_mode,
            fast_mode_source,
        )
    policy_source = f"{low_vram_source};fast={fast_mode_source};remote={allow_remote_source}"

    if low_vram:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        torch_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    else:
        torch_dtype = torch.float32
        attn_impl = "sdpa"

    return AudioRuntimePolicy(
        audio_mode=audio_mode,
        low_vram=low_vram,
        fast_mode=fast_mode,
        policy_source=policy_source,
        device=device,
        torch_dtype=torch_dtype,
        torch_dtype_name=str(torch_dtype).replace("torch.", ""),
        attn_implementation=attn_impl,
        device_map=None,
        low_cpu_mem_usage=False,
        allow_remote_assets=allow_remote_assets,
    )


def resolve_audio_model_ref(
    params: Any,
    *,
    models_dir: str | None = None,
    weights_dir: str | None = None,
    fixed_model: str | None = None,
) -> str:
    pinned = str(fixed_model or "").strip()
    load_name = pinned or str(getattr(params, "load_name", None) or "").strip()
    if not load_name:
        raise ValueError("audio task is missing load_name")

    candidate = Path(load_name).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise ValueError(f"Audio model path not found: {candidate}")
        return str(candidate.resolve())

    for root in (models_dir, weights_dir):
        root_text = str(root or "").strip()
        if not root_text:
            continue
        rooted = (Path(root_text).expanduser().resolve() / load_name).resolve()
        if rooted.exists():
            if str(rooted) != load_name:
                logger.info("Resolved audio model reference %s -> %s", load_name, rooted)
            return str(rooted)

    searched_roots = [str(x).strip() for x in (models_dir, weights_dir) if str(x or "").strip()]
    raise ValueError(
        f"Audio model path not found for load_name='{load_name}'. "
        f"Searched under: {searched_roots or ['<none>']}"
    )

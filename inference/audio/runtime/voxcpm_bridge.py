from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict

from common.logger import get_logger

from audio.runtime.runtime_resolver import AudioRuntimePolicy
from audio.runtime.voxcpm_nano_adapter import VoxCPMNanoVllmFacade

logger = get_logger(__name__)

VOXCPM_BACKEND_TRANSFORMERS = "transformers"
VOXCPM_BACKEND_NANO_VLLM = "nano_vllm_voxcpm"


def _get_voxcpm_version() -> str:
    try:
        return str(version("voxcpm"))
    except (PackageNotFoundError, Exception):
        return "unknown"


def _finalize_loaded_model(model: Any, policy: AudioRuntimePolicy) -> Any:
    if policy.device == "cpu":
        return model
    if not hasattr(model, "to"):
        return model
    try:
        return model.to(policy.device)
    except TypeError:
        try:
            return model.to(device=policy.device)
        except Exception:
            logger.warning("Failed to move VoxCPM model to device=%s", policy.device, exc_info=True)
            return model
    except Exception:
        logger.warning("Failed to move VoxCPM model to device=%s", policy.device, exc_info=True)
        return model


def _normalize_voxcpm_backend(backend: str | None) -> str:
    raw = str(backend or VOXCPM_BACKEND_TRANSFORMERS).strip().lower()
    if raw in ("nano_vllm", "nano-vllm"):
        logger.warning(
            "VoxCPM service uses backend=%r; treating as Nano-vLLM-VoxCPM. "
            "Prefer explicit %r in config to avoid confusion with Qwen nano_vllm.",
            raw,
            VOXCPM_BACKEND_NANO_VLLM,
        )
        return VOXCPM_BACKEND_NANO_VLLM
    if raw in (
        VOXCPM_BACKEND_NANO_VLLM,
        "nanovllm_voxcpm",
        "nano-vllm-voxcpm",
        "nanovllm-voxcpm",
    ):
        return VOXCPM_BACKEND_NANO_VLLM
    return VOXCPM_BACKEND_TRANSFORMERS


def _load_transformers_bundle(model_ref: str, policy: AudioRuntimePolicy) -> Dict[str, Any]:
    from voxcpm import VoxCPM

    logger.info(
        "Loading VoxCPM (transformers) bundle model_ref=%s policy=%s source=%s voxcpm=%s optimize=%s",
        model_ref,
        policy.cache_key,
        policy.policy_source,
        _get_voxcpm_version(),
        False,
    )
    # VoxCPM FAQ 明确说明：torch.compile（optimize=True 默认开启）在后台线程/
    # 多线程场景下会与 CUDA Graphs 冲突。当前音频服务通过 worker thread 执行推理，
    # 因此这里固定关闭 optimize，优先保证稳定性。
    model = VoxCPM.from_pretrained(
        model_ref,
        load_denoiser=False,
        optimize=False,
    )
    model = _finalize_loaded_model(model, policy)
    sample_rate = int(getattr(getattr(model, "tts_model", None), "sample_rate", 48000) or 48000)
    logger.info("VoxCPM bundle loaded successfully from %s sample_rate=%s", model_ref, sample_rate)
    return {
        "device": policy.device,
        "torch_dtype": policy.torch_dtype,
        "model": model,
        "sample_rate": sample_rate,
        "streaming_variant": True,
        "runtime_policy": policy,
        "voxcpm_version": _get_voxcpm_version(),
        "runtime_backend": VOXCPM_BACKEND_TRANSFORMERS,
    }


def _load_nano_vllm_bundle(
    model_ref: str,
    policy: AudioRuntimePolicy,
    runtime_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if policy.device != "cuda":
        raise RuntimeError(
            "Nano-vLLM-VoxCPM requires CUDA (Linux + NVIDIA). "
            f"Current audio runtime policy selected device={policy.device!r}."
        )
    try:
        from nanovllm_voxcpm import VoxCPM as NanoVoxCPM
    except ImportError as exc:
        raise RuntimeError(
            "Nano-vLLM-VoxCPM backend is selected but `nanovllm_voxcpm` is not importable. "
            "Install with: pip install nano-vllm-voxcpm "
            "(see https://github.com/a710128/nanovllm-voxcpm )."
        ) from exc

    cfg = dict(runtime_cfg or {})
    try:
        import torch

        default_device_idx = int(torch.cuda.current_device())
    except Exception:
        default_device_idx = 0

    devices_raw = cfg.get("devices")
    if isinstance(devices_raw, list) and devices_raw:
        devices = [int(x) for x in devices_raw]
    else:
        devices = [default_device_idx]

    raw_gpu_mu = cfg.get("gpu_memory_utilization", 0.9)
    gpu_mu = float(raw_gpu_mu if raw_gpu_mu is not None else 0.9)
    if gpu_mu <= 0.0 or gpu_mu > 1.0:
        raise ValueError(
            "Nano-vLLM-VoxCPM runtime.gpu_memory_utilization must be in (0, 1], "
            f"got {gpu_mu!r}"
        )

    load_kwargs: Dict[str, Any] = {
        "model": model_ref,
        "devices": devices,
        "inference_timesteps": max(1, int(cfg.get("inference_timesteps", 10))),
        "max_num_batched_tokens": max(256, int(cfg.get("max_num_batched_tokens", 16384))),
        "max_num_seqs": max(1, int(cfg.get("max_num_seqs", 16))),
        "max_model_len": max(256, int(cfg.get("max_model_len", 4096))),
        "gpu_memory_utilization": gpu_mu,
        "enforce_eager": bool(cfg.get("enforce_eager", False)),
    }

    logger.info(
        "Loading VoxCPM (Nano-vLLM-VoxCPM) bundle model_ref=%s devices=%s kwargs=%s policy=%s",
        model_ref,
        devices,
        {k: v for k, v in load_kwargs.items() if k != "model"},
        policy.cache_key,
    )

    pool = NanoVoxCPM.from_pretrained(**load_kwargs)
    info = pool.get_model_info()
    sample_rate = int(info.get("sample_rate") or info.get("output_sample_rate") or 48000)
    facade = VoxCPMNanoVllmFacade(pool, sample_rate=sample_rate)
    logger.info(
        "VoxCPM Nano-vLLM-VoxCPM bundle loaded model_ref=%s sample_rate=%s",
        model_ref,
        sample_rate,
    )
    return {
        "device": policy.device,
        "torch_dtype": policy.torch_dtype,
        "model": facade,
        "sample_rate": sample_rate,
        "streaming_variant": True,
        "runtime_policy": policy,
        "voxcpm_version": _get_voxcpm_version(),
        "runtime_backend": VOXCPM_BACKEND_NANO_VLLM,
        "runtime_config": dict(runtime_cfg or {}),
    }


def load_tts_bundle(
    model_ref: str,
    policy: AudioRuntimePolicy,
    backend: str = VOXCPM_BACKEND_TRANSFORMERS,
    runtime_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved = _normalize_voxcpm_backend(backend)
    if resolved == VOXCPM_BACKEND_NANO_VLLM:
        return _load_nano_vllm_bundle(model_ref, policy, runtime_cfg)
    return _load_transformers_bundle(model_ref, policy)


def load_realtime_bundle(
    model_ref: str,
    policy: AudioRuntimePolicy,
    backend: str = VOXCPM_BACKEND_TRANSFORMERS,
    runtime_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return load_tts_bundle(model_ref, policy, backend, runtime_cfg)

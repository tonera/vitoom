"""
TurboDiffusion 模型释放/清理工具。

目标：
- 对齐 PipelineCache 的 TTL 驱逐：切模型时能稳定回收显存与内存，避免 OOM。
"""

from __future__ import annotations

from typing import Any, Optional

from common.logger import get_logger
from common.runtime_cleanup import cleanup_after_release, cleanup_runtime_only as common_cleanup_runtime_only

logger = get_logger(__name__)


def _move_vae_interface(tokenizer: object, *, device: str) -> None:
    # Mirror logic from turbodiffusion.engine._move_vae_interface (best-effort)
    vae = getattr(tokenizer, "model", None)
    if vae is None:
        return
    inner = getattr(vae, "model", None)
    if inner is not None and hasattr(inner, "to"):
        inner.to(device)  # type: ignore[call-arg]
    for name in ("mean", "std"):
        t = getattr(vae, name, None)
        if t is not None and hasattr(t, "to"):
            setattr(vae, name, t.to(device))
    if hasattr(vae, "device"):
        setattr(vae, "device", device)


def _move_t5_encoder(t5_encoder: object, *, device: str) -> None:
    m = getattr(t5_encoder, "model", None)
    if m is not None and hasattr(m, "to"):
        m.to(device)  # type: ignore[call-arg]
    if hasattr(t5_encoder, "device"):
        setattr(t5_encoder, "device", device)


def cleanup_runtime_only(*, log: Any = logger) -> None:
    common_cleanup_runtime_only()


def release_turbo_models(models: Any, *, aggressive_cpu: bool = False, log: Any = logger) -> None:
    """
    释放 TurboModels 占用（best-effort）。
    aggressive_cpu=True：驱逐/切模型场景尽量把模块搬回 CPU 并断引用。
    """
    try:
        if aggressive_cpu and models is not None:
            try:
                tok = getattr(models, "tokenizer", None)
                if tok is not None:
                    _move_vae_interface(tok, device="cpu")
            except Exception:
                pass
            try:
                te = getattr(models, "t5_encoder", None)
                if te is not None:
                    _move_t5_encoder(te, device="cpu")
            except Exception:
                pass
            # 断开 tokenizer / t5_encoder 内部对大模型的引用（让 GC 更稳定）
            try:
                tok = getattr(models, "tokenizer", None)
                if tok is not None:
                    for attr in ("model", "vae", "vae_model"):
                        if hasattr(tok, attr):
                            try:
                                setattr(tok, attr, None)
                            except Exception:
                                pass
            except Exception:
                pass
            try:
                te = getattr(models, "t5_encoder", None)
                if te is not None and hasattr(te, "model"):
                    try:
                        setattr(te, "model", None)
                    except Exception:
                        pass
            except Exception:
                pass
            # DiT / high-low models: try move to cpu then detach
            for name in ("net", "high_noise_model", "low_noise_model"):
                try:
                    m = getattr(models, name, None)
                    if m is not None and hasattr(m, "to"):
                        m.to("cpu")  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    if hasattr(models, name):
                        setattr(models, name, None)
                except Exception:
                    pass
            # 最后再断开 models 对 tokenizer/t5 的引用
            try:
                if hasattr(models, "tokenizer"):
                    setattr(models, "tokenizer", None)
            except Exception:
                pass
            try:
                if hasattr(models, "t5_encoder"):
                    setattr(models, "t5_encoder", None)
            except Exception:
                pass

        try:
            if models is not None:
                del models
        except Exception:
            pass

        cleanup_after_release(trim_malloc=True)
    except Exception:
        try:
            log.debug("release_turbo_models failed (ignored)", exc_info=True)
        except Exception:
            pass


async def release_turbo_models_once_async(
    models: Any,
    *,
    log: Any = logger,
    run_blocking: Optional[Any] = None,
    aggressive_cpu: bool = False,
) -> None:
    if run_blocking is not None:
        await run_blocking(lambda: release_turbo_models(models, aggressive_cpu=aggressive_cpu, log=log))
        return
    release_turbo_models(models, aggressive_cpu=aggressive_cpu, log=log)


async def release_turbo_models_twice_async(
    models: Any,
    *,
    log: Any = logger,
    run_blocking: Optional[Any] = None,
    aggressive_cpu: bool = False,
) -> None:
    await release_turbo_models_once_async(models, log=log, run_blocking=run_blocking, aggressive_cpu=aggressive_cpu)
    await release_turbo_models_once_async(None, log=log, run_blocking=run_blocking, aggressive_cpu=aggressive_cpu)


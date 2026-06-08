"""
图片推理统一释放入口。

约定：
- 成功且缓存命中：这里只负责轻清理 + release_use，不强释放权重
- 推理失败 / TTL 驱逐 / 非缓存模式结束：统一走强释放
- 对象级特殊释放保留在各 pipeline 自身的 `release()` 中；这里只负责统一编排
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from common.Constant import MODEL_SDXL
from common.runtime_cleanup import (
    cleanup_after_release as cleanup_runtime_after_release,
    cleanup_after_oom as cleanup_runtime_after_oom,
    cleanup_runtime_only as common_cleanup_runtime_only,
)
from image.runtime.lora_manager import unload_loras_from_pipe


def clear_inference_params(
    inference_params: Optional[dict],
    *,
    logger: Any,
    extra_keys: Iterable[str] = (),
) -> None:
    keys = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "image",
        "images",
        "mask_image",
        "control_image",
        *tuple(extra_keys),
    )
    try:
        if isinstance(inference_params, dict):
            for key in dict.fromkeys(keys).keys():
                inference_params.pop(key, None)
            inference_params.clear()
    except Exception:
        logger.debug("cleanup: clear inference_params failed (ignored)", exc_info=True)


def _try_clear_fbcache_buffers() -> None:
    debug_log = None
    try:
        from common.logger import get_logger

        debug_log = get_logger(__name__)
    except Exception:
        debug_log = None

    try:
        from common import fbcache_sdxl as _fbc  # type: ignore

        fn = getattr(_fbc, "clear_fbcache_buffers", None)
        if callable(fn):
            fn()
            return
    except Exception:
        if debug_log is not None:
            debug_log.debug("_try_clear_fbcache_buffers: project clear_fbcache_buffers failed (ignored)", exc_info=True)

    try:
        from nunchaku.caching.fbcache import get_current_cache_context, set_buffer  # type: ignore

        ctx = get_current_cache_context()
        if ctx is not None:
            for attr in ("buffers", "buffer", "_buffers", "_buffer", "cache", "_cache"):
                data = getattr(ctx, attr, None)
                if isinstance(data, dict):
                    data.clear()

            for key in ("final_output", "first_single_hidden_states_residual"):
                try:
                    set_buffer(key, None)  # type: ignore[arg-type]
                except Exception:
                    if debug_log is not None:
                        debug_log.debug("_try_clear_fbcache_buffers: set_buffer(%s) failed (ignored)", key, exc_info=True)
    except Exception:
        if debug_log is not None:
            debug_log.debug("_try_clear_fbcache_buffers: direct nunchaku fbcache cleanup failed (ignored)", exc_info=True)

    try:
        import nunchaku.caching.fbcache as _nfbc  # type: ignore

        for attr in ("BUFFERS", "_BUFFERS", "buffers", "_buffers", "CACHE", "_CACHE", "cache", "_cache"):
            data = getattr(_nfbc, attr, None)
            if isinstance(data, dict):
                data.clear()
    except Exception:
        if debug_log is not None:
            debug_log.debug("_try_clear_fbcache_buffers: module-level fbcache cleanup failed (ignored)", exc_info=True)


def cleanup_runtime_only(*, logger: Any) -> None:
    common_cleanup_runtime_only(pre_cleanup=_try_clear_fbcache_buffers)


def resolve_release_targets(pipe: Any, *, family: str) -> tuple[Any, Any, tuple[Any, ...]]:
    """
    统一解析需要卸载 LoRA 的对象和额外释放目标。

    返回值：
    - unload_pipe: 卸载 adapter 的目标
    - release_pipe: 主释放目标
    - extra_release_targets: 需要先于主目标释放的对象
    """
    unload_pipe = pipe
    release_pipe = pipe
    extra_release_targets: list[Any] = []

    mv = (family or "").lower()
    if release_pipe is not None and mv in {m.lower() for m in MODEL_SDXL} and hasattr(release_pipe, "pipe"):
        inner_pipe = getattr(release_pipe, "pipe", None)
        if inner_pipe is not None:
            unload_pipe = inner_pipe
            extra_release_targets.append(inner_pipe)

    return unload_pipe, release_pipe, tuple(extra_release_targets)


def _gpu_mem_summary() -> str:
    try:
        import torch as _t
        if _t.cuda.is_available():
            alloc = _t.cuda.memory_allocated() / (1024 ** 3)
            reserved = _t.cuda.memory_reserved() / (1024 ** 3)
            return f"gpu_alloc={alloc:.2f}GiB gpu_reserved={reserved:.2f}GiB"
    except Exception:
        pass
    return "gpu_mem=<unavailable>"


def _nuke_module_gpu_tensors(module: Any) -> None:
    """Replace all CUDA parameter/buffer data with empty CPU tensors in-place.

    Unlike ``module.to("cpu")`` which copies data from GPU to CPU (slow and
    requires allocating equivalent CPU memory), this simply swaps out the
    underlying storage so the GPU tensors become reclaimable by
    ``gc.collect() + empty_cache()`` immediately.  The model is destroyed
    and cannot be used afterwards.
    """
    import torch as _t

    try:
        for param in module.parameters():
            try:
                if param.data.is_cuda:
                    param.data = _t.empty(0, device="cpu", dtype=param.dtype)
            except Exception:
                pass
        for buf in module.buffers():
            try:
                if buf.is_cuda:
                    buf.data = _t.empty(0, device="cpu", dtype=buf.dtype)
            except Exception:
                pass
    except Exception:
        pass


def _clear_nunchaku_lora_snapshots(module: Any, *, logger: Any) -> int:
    """Clear LoRA weight snapshots (_quantized_part_sd etc.) that hold cloned
    CUDA tensors outside the normal ``nn.Module`` parameter tree.

    These are created by ``nunchaku SVDQLoRAMixin._init_lora_state()`` at model
    load time and survive ``module.to("cpu")`` because they are plain dict values,
    not registered parameters.

    Returns the number of tensors cleared.
    """
    count = 0
    for attr in (
        "_quantized_part_sd",
        "_unquantized_part_sd",
        "_cached_quantized_loras",
        "_cached_unquantized_loras",
    ):
        d = getattr(module, attr, None)
        if isinstance(d, dict):
            count += len(d)
            d.clear()
            try:
                setattr(module, attr, {})
            except Exception:
                pass
    for attr in ("_quantized_part_ranks",):
        try:
            if hasattr(module, attr):
                setattr(module, attr, {})
        except Exception:
            pass
    if count:
        logger.info(f"[release_pipeline] cleared {count} nunchaku LoRA snapshot tensors from {type(module).__name__}")
    return count


def _clear_meancache_state(pipe: Any, *, logger: Any) -> None:
    """Reset and detach MeanCache engine state and module-level caches."""
    freed = False
    try:
        eng = getattr(pipe, "_meancache_engine", None)
        if eng is not None and hasattr(eng, "reset"):
            eng.reset()
            freed = True
        if eng is not None:
            setattr(pipe, "_meancache_engine", None)
    except Exception:
        logger.debug("_clear_meancache_state: engine reset failed (ignored)", exc_info=True)

    for mod_name in ("transformer", "unet"):
        mod = getattr(pipe, mod_name, None)
        if mod is None:
            continue
        for attr in (
            "_meancache_engine",
            "_meancache_last_template_by_pred",
            "_meancache_last_packer_by_pred",
            "_meancache_forward_calls",
            "_meancache_is_patched",
            "_meancache_original_forward",
            "_meancache_x_param_name",
            "_meancache_x_param_pos",
            "_meancache_timestep_param_name",
            "_meancache_timestep_param_pos",
        ):
            try:
                data = getattr(mod, attr, None)
                if isinstance(data, dict):
                    data.clear()
                if hasattr(mod, attr):
                    delattr(mod, attr)
                    freed = True
            except Exception:
                pass
    if freed:
        logger.info("[release_pipeline] cleared meancache engine state")


def release_pipeline(pipe: Any, *, logger: Any, aggressive_cpu: bool = False) -> None:
    try:
        logger.info(f"[release_pipeline] START aggressive_cpu={aggressive_cpu} pipe={type(pipe).__name__ if pipe else None} {_gpu_mem_summary()}")

        try:
            _try_clear_fbcache_buffers()
        except Exception:
            logger.debug("release_pipeline: clear fbcache buffers failed (ignored)", exc_info=True)

        try:
            if pipe is not None and hasattr(pipe, "release") and callable(getattr(pipe, "release")):
                pipe.release()  # type: ignore[call-arg]
        except Exception:
            logger.debug("release_pipeline: pipe.release() failed (ignored)", exc_info=True)

        try:
            if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
                pipe.maybe_free_model_hooks()
        except Exception:
            logger.debug("release_pipeline: maybe_free_model_hooks failed (ignored)", exc_info=True)
        try:
            if pipe is not None and hasattr(pipe, "_remove_all_hooks"):
                pipe._remove_all_hooks()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("release_pipeline: _remove_all_hooks failed (ignored)", exc_info=True)

        if aggressive_cpu and pipe is not None:
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None:
                for attr in (
                    "previous_modulated_input",
                    "previous_residual",
                    "accumulated_rel_l1_distance",
                    "_nunchaku_teacache_ctx",
                    "_nunchaku_teacache_ctx_qwenimage",
                ):
                    try:
                        if hasattr(transformer, attr):
                            setattr(transformer, attr, None)
                    except Exception:
                        pass
                cls = transformer.__class__
                for attr in (
                    "previous_modulated_input",
                    "previous_residual",
                    "accumulated_rel_l1_distance",
                    "_nunchaku_teacache_ctx",
                    "_nunchaku_teacache_ctx_qwenimage",
                ):
                    try:
                        if hasattr(cls, attr):
                            setattr(cls, attr, None)
                    except Exception:
                        pass
                try:
                    if hasattr(cls, "enable_teacache"):
                        setattr(cls, "enable_teacache", False)
                except Exception:
                    pass

            _clear_meancache_state(pipe, logger=logger)

            for name in ("text_encoder_2", "transformer", "unet", "vae", "text_encoder"):
                module = getattr(pipe, name, None)
                if module is None:
                    continue
                _clear_nunchaku_lora_snapshots(module, logger=logger)
                _nuke_module_gpu_tensors(module)
                try:
                    setattr(pipe, name, None)
                except Exception:
                    pass

        pipe = None

        import gc as _gc
        _gc.collect()

        cleanup_runtime_after_release()
        logger.info(f"[release_pipeline] END {_gpu_mem_summary()}")
    except Exception:
        logger.debug("release_pipeline failed (ignored)", exc_info=True)


async def _release_pipeline_once_async(
    pipe: Any,
    *,
    logger: Any,
    run_blocking: Optional[Any] = None,
    aggressive_cpu: bool = False,
) -> None:
    if run_blocking is not None:
        await run_blocking(lambda: release_pipeline(pipe, logger=logger, aggressive_cpu=aggressive_cpu))
        return
    release_pipeline(pipe, logger=logger, aggressive_cpu=aggressive_cpu)


async def force_release_pipeline(
    pipe: Any,
    *,
    logger: Any,
    run_blocking: Optional[Any] = None,
    aggressive_cpu: bool = False,
    extra_release_targets: Iterable[Any] = (),
) -> None:
    if pipe is not None and extra_release_targets and hasattr(pipe, "pipe"):
        inner_pipe = getattr(pipe, "pipe", None)
        if inner_pipe is not None and any(target is inner_pipe for target in extra_release_targets):
            try:
                setattr(pipe, "pipe", None)
            except Exception:
                pass
    release_targets = [target for target in (*tuple(extra_release_targets), pipe) if target is not None]
    for target in release_targets:
        await _release_pipeline_once_async(
            target,
            logger=logger,
            run_blocking=run_blocking,
            aggressive_cpu=aggressive_cpu,
        )
    await _release_pipeline_once_async(
        None,
        logger=logger,
        run_blocking=run_blocking,
        aggressive_cpu=aggressive_cpu,
    )


async def finish_pipeline_use(
    *,
    pipe: Any,
    params: Any,
    inference_params: Optional[dict],
    logger: Any,
    failed: bool,
    oom: bool = False,
    pipeline_cache: Any = None,
    cache_key: Optional[str] = None,
    run_blocking: Optional[Any] = None,
    unload_pipe: Any = None,
    extra_release_targets: Iterable[Any] = (),
    extra_inference_param_keys: Iterable[str] = (),
) -> None:
    clear_inference_params(
        inference_params,
        logger=logger,
        extra_keys=extra_inference_param_keys,
    )

    if unload_pipe is None:
        unload_pipe = pipe
    try:
        if unload_pipe is not None:
            unload_loras_from_pipe(unload_pipe, getattr(params, "family", ""), logger=logger)
    except Exception:
        logger.debug("cleanup: unload_adapters failed (ignored)", exc_info=True)

    cache_enabled = bool(pipeline_cache is not None and cache_key)
    evict_after_use = bool(
        getattr(pipe, "_vitoom_evict_after_use", False)
        or getattr(unload_pipe, "_vitoom_evict_after_use", False)
    )
    if cache_enabled and not failed and not evict_after_use:
        try:
            cleanup_runtime_only(logger=logger)
        except Exception:
            logger.debug("cleanup: cleanup_runtime_only failed (ignored)", exc_info=True)
        try:
            await pipeline_cache.release_use(key=cache_key)
            return
        except Exception:
            logger.debug("cleanup: release_use failed, fallback to full release", exc_info=True)

    evicted = False
    if cache_enabled and (failed or evict_after_use):
        try:
            if evict_after_use:
                reason = getattr(pipe, "_vitoom_evict_reason", None) or getattr(unload_pipe, "_vitoom_evict_reason", "marked")
                logger.info(f"[pipeline-cache] evict after use: reason={reason}")
            await pipeline_cache.evict(force=True)
            evicted = True
        except Exception:
            logger.debug("cleanup: evict failed, fallback to full release", exc_info=True)

    if not evicted:
        # 非缓存模式（TTL=0）或失败：必须强释放 GPU 权重；缓存命中成功路径已在上方 early return。
        await force_release_pipeline(
            pipe,
            logger=logger,
            run_blocking=run_blocking,
            aggressive_cpu=bool(not cache_enabled or failed),
            extra_release_targets=extra_release_targets,
        )

    if oom:
        try:
            _try_clear_fbcache_buffers()
        except Exception:
            logger.debug("cleanup: oom post-evict clear fbcache failed (ignored)", exc_info=True)
        cleanup_runtime_after_oom(trim_malloc=True)
        logger.info(f"[finish_pipeline_use] OOM cleanup done {_gpu_mem_summary()}")

"""
Wan2 pipeline cache 适配层（对齐 image 侧 PipelineCache 的使用方式）。

- 负责：构造稳定 cache key、通过 PipelineCache(LRU=1 + TTL) 获取/驱逐 pipeline
- 不负责：具体推理调用（由 handlers 负责）
"""

from __future__ import annotations

import asyncio
import json
import hashlib
from dataclasses import asdict
from typing import Any, Optional, Tuple

from common.pipeline_cache import PipelineCache

from video.runtime.wan2_pipeline_factory import compute_wan2_pipe_key, create_wan2_pipe, is_oom, cleanup_after_oom
from video.runtime.wan2_pipeline_release import (
    cleanup_runtime_only,
    release_wan2_pipeline_twice_async,
)


def _stable_hash(payload: Any) -> str:
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def build_wan2_cache_key_str(*, key_obj: Any) -> str:
    """
    将 Wan2PipeKey 转为稳定 key 字符串（用于 PipelineCache）。
    """
    try:
        return _stable_hash(asdict(key_obj))
    except Exception:
        return _stable_hash(str(key_obj))


async def acquire_wan2_pipe(
    *,
    name: str,
    models_base_dir: str,
    weights_base_dir: Optional[str],
    model_ref_override: Optional[str],
    device: Optional[str],
    torch_dtype: str,
    vram_limit: Optional[float],
    low_vram: bool,
    pipeline_cache: Optional[PipelineCache],
    run_blocking: Optional[Any],
    log: Any,
) -> Tuple[Any, str, bool]:
    """
    通过 PipelineCache 获取 Wan2 pipe。
    返回：(pipe, cache_key, cache_enabled)
    """
    cache_enabled = bool(pipeline_cache is not None and pipeline_cache.enabled())

    key_obj = compute_wan2_pipe_key(
        name=name,
        models_base_dir=models_base_dir,
        weights_base_dir=weights_base_dir,
        model_ref_override=model_ref_override,
        device=device,
        torch_dtype=torch_dtype,
        vram_limit=vram_limit,
        low_vram=low_vram,
    )
    key = build_wan2_cache_key_str(key_obj=key_obj)

    def _create_new():
        return create_wan2_pipe(
            name=name,
            models_base_dir=models_base_dir,
            weights_base_dir=weights_base_dir,
            model_ref_override=model_ref_override,
            device=device,
            torch_dtype=torch_dtype,
            vram_limit=vram_limit,
            low_vram=low_vram,
        )

    async def _create_new_async():
        # 与推理线程一致：在 run_blocking 的 worker 线程里创建 pipeline
        if run_blocking is not None:
            return await run_blocking(_create_new)
        return await asyncio.to_thread(_create_new)

    if cache_enabled and pipeline_cache is not None:
        pipe, hit = await pipeline_cache.acquire(key=key, create_fn=_create_new_async)
        try:
            log.info(f"[wan2][pipeline-cache] {'HIT' if hit else 'MISS'} key={key[:12]} name={name} low_vram={bool(low_vram)}")
        except Exception:
            pass
        return pipe, key, True

    pipe = await _create_new_async()
    return pipe, key, False


async def finish_wan2_pipe_use(
    *,
    pipe: Any,
    cache_key: str,
    cache_enabled: bool,
    pipeline_cache: Optional[PipelineCache],
    run_blocking: Optional[Any],
    log: Any,
) -> None:
    """
    任务结束后的统一收尾：
    - cache_enabled=True：轻量清理 + release_use（不释放权重）
    - cache_enabled=False：全释放（两次清理）
    """
    if cache_enabled:
        try:
            if run_blocking is not None:
                await run_blocking(lambda: cleanup_runtime_only(log=log))
            else:
                cleanup_runtime_only(log=log)
        except Exception:
            pass
        try:
            if pipeline_cache is not None and cache_key:
                await pipeline_cache.release_use(key=cache_key)
        except Exception:
            pass
        return

    # 非缓存：释放权重/显存
    try:
        await release_wan2_pipeline_twice_async(pipe, log=log, run_blocking=run_blocking, aggressive_cpu=False)
    except Exception:
        pass


async def run_wan2_with_oom_fallback(
    *,
    try_once_fn: Any,
    force_offload: bool,
    log: Any,
) -> Any:
    """
    统一 OOM 回退策略（与现有 handler 逻辑对齐）：
    - force_offload=True：不做自动重试
    - 否则：OOM 时 cleanup_after_oom() 后允许上层切换 low_vram 重试一次
    """
    try:
        return await try_once_fn()
    except Exception as e:
        if (not force_offload) and is_oom(e):
            try:
                log.warning(f"[wan2] OOM detected, cleanup_before_retry: {e}")
            except Exception:
                pass
            cleanup_after_oom()
        raise


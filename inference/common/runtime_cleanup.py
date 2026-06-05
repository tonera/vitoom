"""
PyTorch 运行期清理工具。

目标：
- 统一 OOM 判断逻辑
- 统一“任务结束轻量清理 / OOM 后强清理 / 释放后强清理”策略
- 让 image/video 两侧只保留各自的 pipeline 差异，避免重复维护 CUDA/MPS allocator 清理细节
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import gc

import torch

from common.memory_utils import try_malloc_trim


DEFAULT_CUDA_SLACK_BYTES = 1024**3  # 1GiB


def is_oom_error(exc: BaseException) -> bool:
    """统一判断是否为 OOM（CUDA/MPS/部分 RuntimeError 文本）。"""
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda out of memory" in msg) or ("mps out of memory" in msg)


def _run_pre_cleanup(pre_cleanup: Optional[Callable[[], Any]]) -> None:
    if pre_cleanup is None:
        return
    try:
        pre_cleanup()
    except Exception:
        pass


def _cleanup_cuda_allocator(*, force_empty_cache: bool, slack_threshold_bytes: Optional[int]) -> None:
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    should_empty = bool(force_empty_cache)
    if not should_empty and slack_threshold_bytes is not None:
        try:
            alloc = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            slack = max(0, reserved - alloc)
            should_empty = slack > int(slack_threshold_bytes)
        except Exception:
            should_empty = False

    if not should_empty:
        return

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _cleanup_mps_allocator() -> None:
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def cleanup_runtime_only(
    *,
    pre_cleanup: Optional[Callable[[], Any]] = None,
    slack_threshold_bytes: int = DEFAULT_CUDA_SLACK_BYTES,
) -> None:
    """
    任务结束的轻量清理：
    - 可选前置清理（例如 fbcache buffers）
    - gc.collect
    - 仅当 CUDA allocator 闲置 reserved 明显过大时才 empty_cache/ipc_collect
    """
    _run_pre_cleanup(pre_cleanup)
    gc.collect()
    _cleanup_cuda_allocator(force_empty_cache=False, slack_threshold_bytes=slack_threshold_bytes)
    _cleanup_mps_allocator()


def cleanup_after_oom(*, pre_cleanup: Optional[Callable[[], Any]] = None, trim_malloc: bool = False) -> None:
    """
    OOM 后强清理：
    - 可选前置清理（例如 fbcache buffers）
    - gc.collect
    - 无条件 empty_cache/ipc_collect
    """
    _run_pre_cleanup(pre_cleanup)
    gc.collect()
    if trim_malloc:
        try:
            try_malloc_trim(0)
        except Exception:
            pass
    _cleanup_cuda_allocator(force_empty_cache=True, slack_threshold_bytes=None)
    _cleanup_mps_allocator()


def cleanup_after_release(*, pre_cleanup: Optional[Callable[[], Any]] = None, trim_malloc: bool = False) -> None:
    """
    全量释放后的强清理：
    - 可选前置清理（例如 fbcache buffers）
    - gc.collect
    - 可选 malloc_trim
    - 无条件 empty_cache/ipc_collect
    """
    cleanup_after_oom(pre_cleanup=pre_cleanup, trim_malloc=trim_malloc)

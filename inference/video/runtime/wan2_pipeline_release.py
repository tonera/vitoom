"""
Wan2 pipeline 释放/清理工具。

目标：
- 对齐 image 侧 PipelineLifecycle 的体验：TTL 驱逐/模型切换时能稳定回收显存与内存。
- best-effort：尽量兼容 diffsynth 的 WanVideoPipeline（以及未来可能的变体 pipeline）。
"""

from __future__ import annotations

from typing import Any, Optional

from common.logger import get_logger
from common.runtime_cleanup import cleanup_after_release, cleanup_runtime_only as common_cleanup_runtime_only

logger = get_logger(__name__)


def cleanup_runtime_only(*, log: Any = logger) -> None:
    common_cleanup_runtime_only()


def release_wan2_pipeline(pipe: Any, *, aggressive_cpu: bool = False, log: Any = logger) -> None:
    """
    释放 Wan2 pipeline 的显存/内存占用（best-effort）。
    - aggressive_cpu=True：驱逐/切模型场景更激进：尽量 to(cpu)+断开大模块引用
    """
    try:
        # 0) 项目/三方 pipeline 若提供 release()，优先调用做 best-effort 清理。
        try:
            if pipe is not None and hasattr(pipe, "release") and callable(getattr(pipe, "release")):
                pipe.release()  # type: ignore[call-arg]
        except Exception:
            pass

        # 1) 驱逐场景：尽量把权重搬回 CPU，避免残留引用导致显存顽固占用。
        if aggressive_cpu:
            try:
                if pipe is not None and hasattr(pipe, "to") and callable(getattr(pipe, "to")):
                    pipe.to("cpu")  # type: ignore[call-arg]
            except Exception:
                pass

            # best-effort 断开常见大组件引用（具体字段依赖 diffsynth 实现；失败忽略）
            try:
                if pipe is not None:
                    for name in (
                        "dit",
                        "dit2",
                        "vae",
                        "text_encoder",
                        "text_encoder_2",
                        "tokenizer",
                        "tokenizer_2",
                        "audio_processor",
                    ):
                        if hasattr(pipe, name):
                            try:
                                setattr(pipe, name, None)
                            except Exception:
                                pass
            except Exception:
                pass

        # 2) 断开引用（真实释放依赖 refcount/GC）
        try:
            if pipe is not None:
                del pipe
        except Exception:
            pass

        # 3) GC + CUDA cache 清理
        cleanup_after_release(trim_malloc=True)
    except Exception:
        # 永远不让释放逻辑影响主流程
        try:
            log.debug("release_wan2_pipeline failed (ignored)", exc_info=True)
        except Exception:
            pass


async def release_wan2_pipeline_once_async(
    pipe: Any,
    *,
    log: Any = logger,
    run_blocking: Optional[Any] = None,
    aggressive_cpu: bool = False,
) -> None:
    """
    统一的“释放一次”入口（可选放到 run_blocking 的 worker 线程执行）。
    说明：一些库/缓存的回收对线程上下文敏感；把释放放到与推理相同的线程会更稳。
    """
    if run_blocking is not None:
        await run_blocking(lambda: release_wan2_pipeline(pipe, aggressive_cpu=aggressive_cpu, log=log))
        return
    release_wan2_pipeline(pipe, aggressive_cpu=aggressive_cpu, log=log)


async def release_wan2_pipeline_twice_async(
    pipe: Any,
    *,
    log: Any = logger,
    run_blocking: Optional[Any] = None,
    aggressive_cpu: bool = False,
) -> None:
    """
    统一的“释放 + 二次清理”入口（对齐 image 侧经验做法）。
    """
    await release_wan2_pipeline_once_async(pipe, log=log, run_blocking=run_blocking, aggressive_cpu=aggressive_cpu)
    await release_wan2_pipeline_once_async(None, log=log, run_blocking=run_blocking, aggressive_cpu=aggressive_cpu)


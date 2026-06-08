"""
PipelineCache

设计目标：
- 单进程/单 worker 推理器的“LRU=1 + TTL” pipeline 缓存。
- 解决场景：用户一段时间频繁使用同一个模型时，复用 pipeline 显著减少加载/上GPU耗时。
- 稳定优先：不做高风险的“热替换 unet/vae/transformer”；当兼容性签名变化，直接重建 pipeline。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union


@dataclass
class _CacheItem:
    key: str
    pipe: Any
    last_used_at: float
    in_use: bool = False


class PipelineCache:
    """
    LRU=1 pipeline cache with TTL eviction.

    注意：
    - 本缓存假设“推理主流程是串行的”（当前项目 BaseInferrer 的 executor=max_workers=1）。
      为安全起见仍使用 asyncio.Lock 防止并发 acquire/evict。
    """

    def __init__(
        self,
        *,
        ttl_seconds: int,
        logger: Any,
        release_fn: Callable[[Any], Union[None, Awaitable[None]]],
        tick_seconds: float = 1.0,
    ):
        self.ttl_seconds = int(ttl_seconds or 0)
        self.logger = logger
        self._release_fn = release_fn
        self._tick_seconds = float(tick_seconds)

        self._lock = asyncio.Lock()
        self._item: Optional[_CacheItem] = None

        self._evict_task: Optional[asyncio.Task] = None
        self._stopping = False

    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    def start(self) -> None:
        if not self.enabled():
            return
        if self._evict_task is not None:
            return
        self._stopping = False
        self._evict_task = asyncio.create_task(self._evict_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._evict_task is not None:
            self._evict_task.cancel()
            try:
                await self._evict_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._evict_task = None

        # 进程退出时：无论 TTL，尽力释放缓存 pipeline
        await self.evict(force=True)

    async def acquire(
        self,
        *,
        key: str,
        create_fn: Callable[[], Union[Any, Awaitable[Any]]],
    ) -> tuple[Any, bool]:
        """
        获取 pipeline。
        返回：(pipe, cache_hit)
        """
        if not self.enabled():
            created = create_fn()
            if inspect.isawaitable(created):
                return await created, False
            return created, False

        async with self._lock:
            now = time.time()
            if self._item is not None and self._item.key == key and self._item.pipe is not None:
                self._item.in_use = True
                self._item.last_used_at = now
                return self._item.pipe, True

            # miss：先驱逐旧的（若存在）
            if self._item is not None and self._item.pipe is not None:
                try:
                    self.logger.info(f"[pipeline-cache] key changed, evict old pipeline: {self._item.key} -> {key}")
                except Exception:
                    pass
                # 关键：先从缓存中断开强引用，再释放（否则释放时 gc/empty_cache 可能无法生效）
                old_pipe = self._item.pipe
                try:
                    self._item.pipe = None
                except Exception:
                    pass
                try:
                    released = self._release_fn(old_pipe)
                    if inspect.isawaitable(released):
                        await released
                except Exception:
                    pass
                self._item = None

            created = create_fn()
            pipe = await created if inspect.isawaitable(created) else created
            self._item = _CacheItem(key=key, pipe=pipe, last_used_at=now, in_use=True)
            return pipe, False

    async def release_use(self, *, key: str) -> None:
        """标记 pipeline 不再被本次任务使用（用于 TTL 驱逐）。"""
        if not self.enabled():
            return
        async with self._lock:
            if self._item is None:
                return
            if self._item.key != key:
                return
            self._item.in_use = False
            self._item.last_used_at = time.time()

    async def evict(self, *, force: bool = False) -> None:
        """按 TTL 或 force 驱逐缓存 pipeline。"""
        if not self.enabled() and not force:
            return
        async with self._lock:
            if self._item is None or self._item.pipe is None:
                return
            if self._item.in_use and not force:
                return

            now = time.time()
            expired = (now - self._item.last_used_at) > float(self.ttl_seconds or 0)
            if not force and not expired:
                return

            try:
                reason = "force" if force else f"ttl>{self.ttl_seconds}s"
                self.logger.info(f"[pipeline-cache] evict cached pipeline ({reason}): {self._item.key}")
            except Exception:
                pass
            # 关键：先断开缓存引用，再释放（否则释放时 gc/empty_cache 可能无法生效）
            pipe_to_release = self._item.pipe
            try:
                self._item.pipe = None
            except Exception:
                pass
            self._item = None

            try:
                released = self._release_fn(pipe_to_release)
                if inspect.isawaitable(released):
                    await released
            except Exception:
                pass
            pipe_to_release = None

    async def _evict_loop(self) -> None:
        try:
            while not self._stopping:
                await asyncio.sleep(self._tick_seconds)
                try:
                    await self.evict(force=False)
                except Exception:
                    # 不让驱逐线程影响主流程
                    pass
        except asyncio.CancelledError:
            pass


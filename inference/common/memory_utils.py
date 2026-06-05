"""
内存工具（best-effort）。

目标：
- 在 Linux(glibc) 上尽量把 Python/torch 已释放但仍保留在 allocator 内的内存归还给 OS，
  改善 RSS “看起来不下降”的问题。
"""

from __future__ import annotations

from typing import Optional


def try_malloc_trim(pad: int = 0) -> bool:
    """
    Best-effort 调用 glibc malloc_trim，把空闲堆内存归还给 OS。
    - 仅在 Linux/glibc 上有效；其它平台会直接返回 False。
    - 失败不抛异常。
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        fn = getattr(libc, "malloc_trim", None)
        if fn is None:
            return False
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_size_t]
        rc = int(fn(int(pad)))
        return rc == 1
    except Exception:
        return False


def read_rss_bytes() -> Optional[int]:
    """
    读取当前进程 RSS（bytes）。
    - Linux: /proc/self/statm
    - 失败返回 None
    """
    try:
        import os

        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            return None
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size
    except Exception:
        return None


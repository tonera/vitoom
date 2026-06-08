"""
将下载进度/错误写入 models.description 的保留区段工具。

约定区段格式：

--- download ---
status: downloading
progress: ...
error: ...
worker: download_01
updated_at: 2026-01-21T...
--- /download ---
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


START = "--- download ---"
END = "--- /download ---"


def build_download_block(
    *,
    status: str,
    progress: str = "",
    error: str = "",
    worker: str = "",
    updated_at: Optional[str] = None,
) -> str:
    ts = updated_at or datetime.utcnow().isoformat()
    # 保持字段顺序稳定，便于前端/排查
    lines = [
        START,
        f"status: {str(status or '').strip()}",
        f"progress: {str(progress or '').strip()}",
        f"error: {str(error or '').strip()}",
        f"worker: {str(worker or '').strip()}",
        f"updated_at: {ts}",
        END,
    ]
    return "\n".join(lines)


def upsert_download_block(
    description: Optional[str],
    *,
    status: str,
    progress: str = "",
    error: str = "",
    worker: str = "",
    updated_at: Optional[str] = None,
) -> str:
    """
    在原 description 中插入或替换下载区段，不破坏用户原描述。
    """
    base = str(description or "")
    block = build_download_block(
        status=status,
        progress=progress,
        error=error,
        worker=worker,
        updated_at=updated_at,
    )

    if START in base and END in base:
        pre, rest = base.split(START, 1)
        _, post = rest.split(END, 1)
        # 保持前后空行更自然
        pre = pre.rstrip()
        post = post.lstrip()
        out = pre + ("\n\n" if pre else "") + block + ("\n\n" if post else "") + post
        return out.strip("\n") + "\n"

    # 不存在区段：追加到末尾
    base = base.rstrip()
    out = base + ("\n\n" if base else "") + block
    return out.strip("\n") + "\n"


def clear_download_block(description: Optional[str]) -> str:
    """
    清空下载区段（完成后调用）。若不存在区段则原样返回。
    """
    base = str(description or "")
    if START not in base or END not in base:
        return base
    pre, rest = base.split(START, 1)
    _, post = rest.split(END, 1)
    out = (pre.rstrip() + "\n\n" + post.lstrip()).strip("\n")
    return (out + "\n") if out else ""


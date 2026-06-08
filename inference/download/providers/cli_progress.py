from __future__ import annotations

"""
从 CLI / tqdm 的输出行中提取下载进度信息。

说明：
- 解析发生在 provider 读取子进程 stdout 的“原始流”阶段，不依赖 worker 的 download_log（后者会裁剪/节流）。
- 返回的 bytes 可能是估算值（当 CLI 只输出百分比时）。
"""

from typing import Dict


def _unit_to_bytes(n: float, unit: str) -> int:
    u = str(unit or "").strip().lower()
    if u in {"b", "byte", "bytes"} or not u:
        return int(n)
    # 去掉可能的 "/s" 之类尾巴
    u = u.replace("/s", "").strip()
    # 兼容：GiB/MiB/KiB 以及 GB/MB/KB（展示用，不纠结 1000 vs 1024）
    if u in {"kb", "kib"}:
        return int(n * 1024)
    if u in {"mb", "mib"}:
        return int(n * 1024**2)
    if u in {"gb", "gib"}:
        return int(n * 1024**3)
    if u in {"tb", "tib"}:
        return int(n * 1024**4)
    return int(n)


def parse_progress_from_cli_line(line: str) -> Dict[str, int]:
    """
    返回字段（可能只有部分）：
    - progress: 0..100
    - bytes_downloaded: >=0
    - bytes_total: >=0
    """
    import re

    s = str(line or "").strip()
    if not s:
        return {}

    # 1) "123/456 bytes"
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*bytes\b", s, flags=re.IGNORECASE)
    if m:
        got = int(m.group(1))
        total = int(m.group(2))
        pct = int((got / total) * 100) if total > 0 else 0
        return {
            "progress": max(0, min(100, pct)),
            "bytes_downloaded": max(0, got),
            "bytes_total": max(0, total),
        }

    # 2) "1.23GiB/4.56GiB" / "1.23GB/4.56GB"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*([KMGT]?i?B|[KMGT]?B)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]?i?B|[KMGT]?B)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        got = _unit_to_bytes(float(m.group(1)), m.group(2))
        total = _unit_to_bytes(float(m.group(3)), m.group(4))
        pct = int((got / total) * 100) if total > 0 else 0
        return {
            "progress": max(0, min(100, pct)),
            "bytes_downloaded": max(0, got),
            "bytes_total": max(0, total),
        }

    # 3) "30%"（兜底）
    m = re.search(r"(?<!\d)(\d{1,3})\s*%\b", s)
    if m:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return {"progress": pct}

    return {}


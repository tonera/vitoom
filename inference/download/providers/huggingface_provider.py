from __future__ import annotations

import asyncio
import contextlib
import math
import time
import fnmatch
import shlex
import sys
from pathlib import Path
from typing import Any, List

from common.Constant import DOWNLOAD_SHARE_EXTS, DOWNLOAD_SHARE_MIN_SIZE_BYTES
from common.logger import get_logger
from .cli_progress import parse_progress_from_cli_line
from .subprocess_utils import (
    best_effort_set_child_die_with_parent,
    register_download_cli_proc,
    terminate_subprocess_tree,
    unregister_download_cli_proc,
    wrap_cmd_with_pdeathsig,
)

logger = get_logger(__name__)


def _excluded(path: str, patterns: List[str]) -> bool:
    p = str(path or "").strip()
    if not p:
        return True
    for pat in patterns:
        if not pat:
            continue
        if fnmatch.fnmatch(p, pat):
            return True
    return False


def _clean_cli_line(s: str) -> str:
    """
    清洗 tqdm/hf-cli 等输出中的控制字符：
    - 去掉回车覆盖（\\r）
    - 去掉 ANSI 控制码（例如 \\x1b[A 光标上移）
    """
    import re

    t = str(s or "")
    t = t.replace("\r", "")
    t = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", t)
    t = re.sub(r"\x1B\][^\x07]*(\x07|\x1B\\)", "", t)
    return t.strip("\n")


async def download_huggingface(
    *,
    repo_id: str,
    models_dir: Path,
    sha_index: Any,
    send_log,
    send_status,
    cancel_ev: asyncio.Event,
    default_excludes: List[str] | None = None,
) -> str:
    """
    HuggingFace：
    - 用 HfApi.model_info(repo_id, files_metadata=True) 获取 siblings（rfilename/size/lfs.sha256）
    - 对于 sha 命中本地索引：先在 local_dir 创建链接，再将该文件加入 `hf download` 的 `--exclude`
    - 最后执行：
      hf download "<repo_id>" --local-dir "<abs_dir>" --no-quiet --exclude "<pattern>" --exclude "<pattern>" ...
      依赖其断点续传补齐剩余文件
    """
    # 本地目录语义：{models_dir}/{model_name}（repo_id 最后一级）
    model_name = repo_id.split("/")[-1].strip() or repo_id
    rel_dir = Path(model_name)
    abs_dir = (models_dir / rel_dir).resolve()
    abs_dir.mkdir(parents=True, exist_ok=True)

    log_seq = 0
    await send_log([f"[huggingface] repo_id={repo_id}", f"local_dir={rel_dir.as_posix()}"], log_seq)
    log_seq += 1
    # 终端可观测：把关键信息打印到运行窗口，方便排障
    try:
        print(f"[huggingface] repo_id={repo_id}", flush=True)
        print(f"[huggingface] local_dir={rel_dir.as_posix()}", flush=True)
    except Exception:
        pass
    with contextlib.suppress(Exception):
        logger.info(f"[huggingface] repo_id={repo_id}")
        logger.info(f"[huggingface] local_dir={rel_dir.as_posix()}")

    IMPORT_TIMEOUT_S = 5
    API_INIT_TIMEOUT_S = 5
    MODEL_INFO_TIMEOUT_S = 15

    # 重要：某些环境下 import 可能非常慢甚至卡住；放到线程并加超时，避免黑盒
    try:
        def _import_hfapi():
            from huggingface_hub import HfApi  # type: ignore

            return HfApi

        HfApi = await asyncio.wait_for(asyncio.to_thread(_import_hfapi), timeout=IMPORT_TIMEOUT_S)
    except asyncio.TimeoutError:
        msg = f"huggingface_hub import timeout (>{IMPORT_TIMEOUT_S}s)"
        await send_log([f"[huggingface] ERROR: {msg}"], log_seq)
        raise RuntimeError(msg)
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not installed: {e}")

    try:
        api = await asyncio.wait_for(asyncio.to_thread(HfApi), timeout=API_INIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        msg = f"huggingface HfApi init timeout (>{API_INIT_TIMEOUT_S}s)"
        await send_log([f"[huggingface] ERROR: {msg}"], log_seq)
        raise RuntimeError(msg)

    await send_log([f"[huggingface] fetching model_info (files_metadata=True), timeout={MODEL_INFO_TIMEOUT_S}s ..."], log_seq)
    log_seq += 1
    with contextlib.suppress(Exception):
        print(f"[huggingface] fetching model_info (files_metadata=True), timeout={MODEL_INFO_TIMEOUT_S}s ...", flush=True)
    # 重要：HfApi 的网络调用可能卡死；放到线程里并加超时，避免任务永久 stuck 在 downloading
    try:
        info = await asyncio.wait_for(
            asyncio.to_thread(api.model_info, repo_id=repo_id, files_metadata=True),
            timeout=MODEL_INFO_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        base_cmd = f'hf download "{repo_id}" --local-dir "{str(abs_dir)}" --no-quiet'
        msg = f"huggingface model_info timeout (>{MODEL_INFO_TIMEOUT_S}s)"
        await send_log(
            [
                f"[huggingface] ERROR: {msg}",
                "[huggingface] NOTE: failed before building full `hf download ... --exclude ...` command.",
                "[huggingface] base_cmd: " + base_cmd,
            ],
            log_seq,
        )
        with contextlib.suppress(Exception):
            print(f"[huggingface] ERROR: {msg}", flush=True)
            print("[huggingface] base_cmd: " + base_cmd, flush=True)
        raise RuntimeError(msg)

    siblings = list(getattr(info, "siblings", []) or [])
    if not siblings:
        await send_log(["[huggingface] No siblings found"], log_seq)
        raise RuntimeError("No files found in huggingface model_info.siblings")

    exclude_args: list[str] = []
    if default_excludes:
        exclude_args.extend([x for x in default_excludes if x])

    # 预期文件清单（用于进度估算）
    expected: List[tuple[str, int, str]] = []  # (rfilename, size, sha256)
    for f in siblings:
        rfilename = str(getattr(f, "rfilename", "") or "").strip()
        size = int(getattr(f, "size", 0) or 0)
        lfs = getattr(f, "lfs", None)
        sha256 = str(getattr(lfs, "sha256", "") or "").strip().lower() if lfs else ""
        if not rfilename or not sha256 or size <= 0:
            continue
        expected.append((rfilename, size, sha256))

    linked_cnt = 0
    sha_index.load()
    for (rfilename, size, sha256) in expected:
        if not rfilename.lower().endswith(DOWNLOAD_SHARE_EXTS):
            continue
        if size and size < DOWNLOAD_SHARE_MIN_SIZE_BYTES:
            continue

        # 自清理：索引里记录的路径若已不存在，应删除该 sha 下的脏路径（避免索引长期积累无效记录）
        try:
            removed = sha_index.prune_missing(sha256)
            if removed:
                await sha_index.save()
        except Exception:
            pass

        existing = sha_index.find_one(sha256)
        if not existing:
            continue

        dst_abs = (abs_dir / rfilename).resolve()
        try:
            dst_abs.relative_to(abs_dir)
        except Exception:
            continue

        ok = sha_index.try_link(existing, dst_abs)
        if ok:
            linked_cnt += 1
            # hf 的 exclude 支持 glob，并可重复指定；这里精确排除该文件路径
            exclude_args.append(rfilename)
            sha_index.add(sha256, (rel_dir / rfilename).as_posix())

    if linked_cnt > 0:
        await sha_index.save()

    await send_log([f"[huggingface] linked={linked_cnt}, exclude_count={len(exclude_args)}"], log_seq)
    log_seq += 1

    # 用于进度估算的总大小（当 CLI 只有百分比、没有 bytes 信息时使用）
    total_bytes = sum(sz for (_, sz, _) in expected) or 0

    cmd = ["hf", "download", repo_id, "--local-dir", str(abs_dir), "--no-quiet"]
    for ex in exclude_args:
        cmd += ["--exclude", ex]

    shell_cmd = " ".join(shlex.quote(x) for x in cmd)
    await send_log(["[huggingface] exec: " + shell_cmd], log_seq)
    log_seq += 1
    # 终端可观测：把完整命令（含 --exclude）打印到运行窗口，便于复现/排障
    print("[huggingface] exec: " + shell_cmd, flush=True)

    cmd2 = wrap_cmd_with_pdeathsig(cmd)
    proc = await asyncio.create_subprocess_exec(
        *cmd2,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # 关键：让子进程成为独立进程组 leader，便于在父进程被 Ctrl+C/取消时 killpg 清理整个子进程树
        start_new_session=True,
    )
    best_effort_set_child_die_with_parent(proc)
    register_download_cli_proc(models_dir=models_dir, pid=int(proc.pid), kind="huggingface", cmd=cmd)

    rc = 0
    try:
        assert proc.stdout is not None
        # 终端：原始 stdout/stderr 原样写回运行窗口
        # WS：每秒最多发一条 download_log（取最新一行、做轻度清洗）
        buf = ""
        last_emit_t = 0.0
        last_log_line = ""
        # CLI 进度解析：取最大值推进，避免 per-file tqdm 重置导致回退
        cli_last_emit_t = 0.0
        cli_max_percent = -1
        cli_max_bytes = -1
        while True:
            if cancel_ev.is_set():
                await send_log(["[huggingface] canceled by user"], log_seq)
                raise asyncio.CancelledError()

            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except Exception:
                pass
            text = chunk.decode("utf-8", errors="replace")
            buf += text
            buf = buf.replace("\r\n", "\n").replace("\r", "\n")
            parts = buf.split("\n")
            buf = parts.pop()

            for raw in parts:
                s = _clean_cli_line(raw)
                if s:
                    last_log_line = s
                    # 从 CLI 输出提取进度（单一来源，避免 disk-scan 误导/卡 0）
                    try:
                        info = parse_progress_from_cli_line(s)
                        if info:
                            pct = int(info.get("progress") or 0)
                            got_b = info.get("bytes_downloaded")
                            total_b = info.get("bytes_total")

                            bt = int(total_b) if isinstance(total_b, int) and total_b > 0 else int(total_bytes or 0)

                            gb: int | None
                            if isinstance(got_b, int) and got_b >= 0:
                                gb = got_b
                            elif bt > 0 and 0 <= pct <= 100:
                                gb = int(math.floor((bt * pct) / 100.0))
                            else:
                                gb = None

                            improved = False
                            if 0 <= pct <= 100 and pct > cli_max_percent:
                                cli_max_percent = pct
                                improved = True
                            if isinstance(gb, int) and gb > cli_max_bytes:
                                cli_max_bytes = gb
                                improved = True

                            now2 = time.monotonic()
                            if improved and (now2 - cli_last_emit_t) >= 1.0:
                                cli_last_emit_t = now2
                                await send_status(
                                    "downloading",
                                    f"{(cli_max_bytes if cli_max_bytes >= 0 else 0)}/{bt} bytes ({max(cli_max_percent, 0)}%)",
                                    "",
                                    "",
                                    int(cli_max_bytes if cli_max_bytes >= 0 else 0),
                                    int(bt),
                                    int(max(cli_max_percent, 0)),
                                )
                    except Exception:
                        pass
            now = time.monotonic()
            if last_log_line and (now - last_emit_t) >= 1.0:
                last_emit_t = now
                try:
                    await send_log([last_log_line], log_seq)
                    log_seq += 1
                except Exception:
                    pass

        tail = _clean_cli_line(buf)
        if tail:
            try:
                await send_log([tail], log_seq)
                log_seq += 1
            except Exception:
                pass

        rc = await proc.wait()
    finally:
        with contextlib.suppress(Exception):
            await terminate_subprocess_tree(proc)
        unregister_download_cli_proc(models_dir=models_dir, pid=int(proc.pid))

    if rc != 0:
        raise RuntimeError(f"hf download failed: exit_code={rc}")

    # 完成后增量写入索引（只对大权重）
    try:
        sha_index.load()
        for f in siblings:
            rfilename = str(getattr(f, "rfilename", "") or "").strip()
            size = int(getattr(f, "size", 0) or 0)
            lfs = getattr(f, "lfs", None)
            sha256 = str(getattr(lfs, "sha256", "") or "").strip().lower() if lfs else ""
            if not rfilename or not sha256:
                continue
            if not rfilename.lower().endswith(DOWNLOAD_SHARE_EXTS):
                continue
            if size and size < DOWNLOAD_SHARE_MIN_SIZE_BYTES:
                continue
            p = (abs_dir / rfilename)
            if p.exists():
                sha_index.add(sha256, (rel_dir / rfilename).as_posix())
        await sha_index.save()
    except Exception as e:
        await send_log([f"[huggingface] sha index update failed (ignored): {e}"], log_seq)

    return rel_dir.as_posix()


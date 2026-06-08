from __future__ import annotations

import asyncio
import contextlib
import math
import time
import fnmatch
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    清洗 tqdm/modelscope-cli 等输出中的控制字符：
    - 去掉回车覆盖（\\r）
    - 去掉 ANSI 控制码（例如 \\x1b[A 光标上移）
    """
    import re

    t = str(s or "")
    t = t.replace("\r", "")
    t = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", t)
    t = re.sub(r"\x1B\][^\x07]*(\x07|\x1B\\)", "", t)
    return t.strip("\n")


async def download_modelscope(
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
    返回 local_path（相对 models_dir）。

    策略：
    - HubApi.get_model(repo_id) 拿到 files(name/sha256/size)
    - sha 命中本地索引：先在 local_dir 创建链接，再把该文件加入 --exclude
    - 执行：modelscope download --model repo_id --local_dir <abs_dir> --exclude ...
      依赖其断点续传补齐剩余文件
    """
    # 本地目录语义：{models_dir}/{model_name}（repo_id 最后一级）
    model_name = repo_id.split("/")[-1].strip() or repo_id
    rel_dir = Path(model_name)
    abs_dir = (models_dir / rel_dir).resolve()
    abs_dir.mkdir(parents=True, exist_ok=True)

    log_seq = 0
    await send_log([f"[modelscope] repo_id={repo_id}", f"local_dir={rel_dir.as_posix()}"], log_seq)
    log_seq += 1
    # 终端可观测（强约束）：在下载器进程终端直接打印关键信息
    try:
        print(f"[modelscope] repo_id={repo_id}", flush=True)
        print(f"[modelscope] local_dir={rel_dir.as_posix()}", flush=True)
    except Exception:
        pass
    try:
        logger.info(f"[modelscope] repo_id={repo_id}")
        logger.info(f"[modelscope] local_dir={rel_dir.as_posix()}")
    except Exception:
        pass

    try:
        # 注意：某些环境下 modelscope import 可能非常慢甚至卡死（例如依赖/环境异常）
        # 放到线程并加超时，避免阻塞事件循环导致 WS 心跳/日志都“黑盒”
        def _import_hubapi():
            from modelscope.hub.api import HubApi  # type: ignore

            return HubApi

        HubApi = await asyncio.wait_for(asyncio.to_thread(_import_hubapi), timeout=5)
    except Exception as e:
        raise RuntimeError(f"modelscope not installed: {e}")

    try:
        api = await asyncio.wait_for(asyncio.to_thread(HubApi), timeout=5)
    except asyncio.TimeoutError:
        raise RuntimeError("modelscope HubApi init timeout (>5s)")
    await send_log(["[modelscope] fetching model_info (HubApi.get_model)..."], log_seq)
    log_seq += 1
    # 重要：ModelScope API 网络调用可能卡死；放到线程里并加超时，避免任务永久 stuck 在 downloading
    try:
        model_info = await asyncio.wait_for(
            asyncio.to_thread(api.get_model, repo_id),
            timeout=15,
        )
    except asyncio.TimeoutError:
        raise RuntimeError("modelscope get_model timeout (>15s)")

    files: list[dict] = []
    mi = model_info.get("ModelInfos") if isinstance(model_info, dict) else None
    if isinstance(mi, dict):
        for _, v in mi.items():
            if isinstance(v, dict) and isinstance(v.get("files"), list):
                files.extend([x for x in v.get("files") if isinstance(x, dict)])

    if not files:
        await send_log(["[modelscope] No files found in ModelInfos"], log_seq)
        raise RuntimeError("No files found in modelscope model_info.ModelInfos")

    exclude_args: list[str] = []
    if default_excludes:
        exclude_args.extend([x for x in default_excludes if x])

    # 预期文件清单（用于进度估算）
    expected: List[tuple[str, int, str]] = []  # (name, size, sha256)
    for f in files:
        name = str(f.get("name") or "").strip()
        sha256 = str(f.get("sha256") or "").strip().lower()
        try:
            size = int(f.get("size") or 0)
        except Exception:
            size = 0
        if not name or not sha256 or size <= 0:
            continue
        expected.append((name, size, sha256))

    linked_cnt = 0
    sha_index.load()
    for (name, size, sha256) in expected:
        if not name.lower().endswith(DOWNLOAD_SHARE_EXTS):
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

        dst_abs = (abs_dir / name).resolve()
        try:
            dst_abs.relative_to(abs_dir)
        except Exception:
            continue

        ok = sha_index.try_link(existing, dst_abs)
        if ok:
            linked_cnt += 1
            exclude_args.append(name)
            sha_index.add(sha256, (rel_dir / name).as_posix())

    if linked_cnt > 0:
        await sha_index.save()

    await send_log([f"[modelscope] linked={linked_cnt}, exclude_count={len(exclude_args)}"], log_seq)
    log_seq += 1

    # 用于进度估算的总大小（当 CLI 只有百分比、没有 bytes 信息时使用）
    total_bytes = sum(sz for (_, sz, _) in expected) or 0

    cmd = ["modelscope", "download", "--model", repo_id, "--local_dir", str(abs_dir)]
    for ex in exclude_args:
        cmd += ["--exclude", ex]

    shell_cmd = " ".join(shlex.quote(x) for x in cmd)
    await send_log(["[modelscope] exec: " + shell_cmd], log_seq)
    log_seq += 1
    # 终端：原始命令直接输出在运行窗口
    print("[modelscope] exec: " + shell_cmd, flush=True)

    cmd2 = wrap_cmd_with_pdeathsig(cmd)
    proc = await asyncio.create_subprocess_exec(
        *cmd2,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # 关键：让子进程成为独立进程组 leader，便于在父进程被 Ctrl+C/取消时 killpg 清理整个子进程树
        # 否则 proc.terminate() 只能杀当前进程，若 modelscope 内部再派生子进程可能残留导致锁死
        start_new_session=True,
    )
    best_effort_set_child_die_with_parent(proc)
    register_download_cli_proc(models_dir=models_dir, pid=int(proc.pid), kind="modelscope", cmd=cmd)

    try:
        assert proc.stdout is not None
        # 终端：把 CLI 的原始 stdout/stderr 原样写回运行窗口（最简单、最直观）
        # WS：每秒最多发一条 download_log（取最新一行、做轻度清洗）
        buf = ""
        last_emit_t = 0.0
        last_log_line = ""
        # CLI 进度解析：取最大值推进，避免 tqdm per-file 进度重置导致回退
        cli_last_emit_t = 0.0
        cli_max_percent = -1
        cli_max_bytes = -1
        while True:
            if cancel_ev.is_set():
                await send_log(["[modelscope] canceled by user"], log_seq)
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
            # 规范化换行：把 \r\n / \r 都转成 \n
            buf = buf.replace("\r\n", "\n").replace("\r", "\n")
            parts = buf.split("\n")
            buf = parts.pop()  # 最后一个可能是不完整行，留到下个 chunk

            for raw in parts:
                s = _clean_cli_line(raw)
                if s:
                    last_log_line = s
                    # 从 CLI 输出提取进度（当 local_dir 文件大小不可用时，这是关键）
                    try:
                        info = parse_progress_from_cli_line(s)
                        if info:
                            pct = int(info.get("progress") or 0)
                            got_b = info.get("bytes_downloaded")
                            total_b = info.get("bytes_total")

                            # bytes_total：优先 CLI，否则用预期汇总
                            bt = int(total_b) if isinstance(total_b, int) and total_b > 0 else int(total_bytes or 0)

                            # bytes_downloaded：CLI 没给就按百分比估算
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
                            # provider 侧每秒最多推一次；worker 侧还会做 3s 节流
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

        # flush 尾巴（如果最后一段没有换行）
        tail = _clean_cli_line(buf)
        if tail:
            # WS 最后一条补发
            try:
                await send_log([tail], log_seq)
                log_seq += 1
            except Exception:
                pass

        rc = await proc.wait()
    finally:
        # 无论是 Ctrl+C、任务被 CancelledError 取消、还是其他异常，都必须清理子进程，避免遗留“modelscope download”孤儿
        # terminate_subprocess_tree 内部会检查 returncode；进程已正常退出时不会误杀
        with contextlib.suppress(Exception):
            await terminate_subprocess_tree(proc)
        unregister_download_cli_proc(models_dir=models_dir, pid=int(proc.pid))

    if rc != 0:
        raise RuntimeError(f"modelscope download failed: exit_code={rc}")

    # 完成后增量写入索引（只对大权重）
    try:
        sha_index.load()
        for f in files:
            name = str(f.get("name") or "").strip()
            sha256 = str(f.get("sha256") or "").strip().lower()
            if not name or not sha256:
                continue
            if not name.lower().endswith(DOWNLOAD_SHARE_EXTS):
                continue
            try:
                size = int(f.get("size") or 0)
            except Exception:
                size = 0
            if size and size < DOWNLOAD_SHARE_MIN_SIZE_BYTES:
                continue
            p = (abs_dir / name)
            if p.exists():
                sha_index.add(sha256, (rel_dir / name).as_posix())
        await sha_index.save()
    except Exception as e:
        await send_log([f"[modelscope] sha index update failed (ignored): {e}"], log_seq)

    return rel_dir.as_posix()


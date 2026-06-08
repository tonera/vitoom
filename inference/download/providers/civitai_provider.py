from __future__ import annotations

import asyncio
import contextlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import aiohttp  # type: ignore

from common.Constant import DOWNLOAD_SHARE_EXTS, DOWNLOAD_SHARE_MIN_SIZE_BYTES
from common.logger import get_logger
from download.sha_index import file_sha256

logger = get_logger(__name__)


async def download_civitai(
    *,
    version_id: str,
    models_dir: Path,
    sha_index: Any,
    send_log,
    send_status,
    cancel_ev: asyncio.Event,
    civitai_token: Optional[str] = None,
) -> str:
    """
    Civitai：resource_id = modelVersionId（数字）
    - GET https://civitai.com/api/v1/model-versions/{id} 获取文件信息与 sha
    - 若 sha 命中 models_sha256.json：硬链/软链复用并直接完成
    - 否则流式下载到 local_dir=civitai/{versionId}/...
    返回 local_path（相对 models_dir）。
    """
    vid = str(version_id or "").strip()
    if not re.fullmatch(r"\d+", vid):
        raise ValueError("civitai resource_id must be numeric modelVersionId")

    api_url = f"https://civitai.com/api/v1/model-versions/{vid}"
    # 一些环境/网关对缺少 UA 的请求会限流/阻断；补一个稳定的 UA，提升可观测性与兼容性
    headers = {
        "Accept": "application/json",
        "User-Agent": "vitoom-download/1.0",
    }
    if civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"

    await send_log([f"[civitai] GET {api_url}"], 0)
    # 终端可观测：打印关键阶段信息，避免“黑盒”
    with contextlib.suppress(Exception):
        print(f"[civitai] version_id={vid}", flush=True)
        print(f"[civitai] token={'present' if civitai_token else 'absent'}", flush=True)
        print(f"[civitai] api_url={api_url}", flush=True)
        print(f"[civitai] meta: GET {api_url}", flush=True)
    with contextlib.suppress(Exception):
        logger.info(f"[civitai] version_id={vid}")
        logger.info(f"[civitai] token={'present' if civitai_token else 'absent'}")
        logger.info(f"[civitai] api_url={api_url}")
        logger.info(f"[civitai] meta: GET {api_url}")

    # meta 请求不应该无限等待：加超时，避免卡死看不到任何进展
    META_TIMEOUT_S = 30
    try:
        meta_timeout = aiohttp.ClientTimeout(total=META_TIMEOUT_S, connect=10, sock_connect=10, sock_read=20)
        async with aiohttp.ClientSession(timeout=meta_timeout, trust_env=True) as session:
            async with session.get(api_url, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    with contextlib.suppress(Exception):
                        print(f"[civitai] meta request failed: HTTP {resp.status}: {text[:200]}", flush=True)
                    raise RuntimeError(f"Civitai meta request failed: HTTP {resp.status}: {text[:200]}")
                meta = await resp.json()
    except asyncio.TimeoutError:
        msg = f"Civitai meta request timeout (>{META_TIMEOUT_S}s)"
        await send_log([f"[civitai] ERROR: {msg}", f"[civitai] url={api_url}"], 0)
        with contextlib.suppress(Exception):
            print(f"[civitai] ERROR: {msg}", flush=True)
        raise RuntimeError(msg)
    except aiohttp.ClientError as e:
        msg = f"Civitai meta request client error: {type(e).__name__}: {e}"
        await send_log([f"[civitai] ERROR: {msg}", f"[civitai] url={api_url}"], 0)
        with contextlib.suppress(Exception):
            print(f"[civitai] ERROR: {msg}", flush=True)
        raise RuntimeError(msg)

    files = (meta or {}).get("files") or []
    if not files:
        raise RuntimeError("Civitai version has no files")
    with contextlib.suppress(Exception):
        print(f"[civitai] meta ok: files={len(files)}", flush=True)

    def _pick_key(f: Dict[str, Any]) -> Tuple[int, int]:
        name = str(f.get("name") or "")
        size_kb = f.get("sizeKB")
        try:
            size = int(float(size_kb) * 1024) if size_kb is not None else 0
        except Exception:
            size = 0
        ext_ok = 1 if name.lower().endswith(DOWNLOAD_SHARE_EXTS) else 0
        return (ext_ok, size)

    best = max(files, key=_pick_key)
    filename = str(best.get("name") or "model.bin")
    download_url = str(best.get("downloadUrl") or best.get("download_url") or "").strip()
    if not download_url:
        raise RuntimeError("Civitai file missing downloadUrl")

    sha256 = None
    try:
        hashes = best.get("hashes") or {}
        for k in ("SHA256", "sha256", "Sha256"):
            if hashes.get(k):
                sha256 = str(hashes.get(k)).strip().lower()
                break
    except Exception:
        sha256 = None
    with contextlib.suppress(Exception):
        logger.info(f"[civitai] picked file: {filename} sha256={'yes' if sha256 else 'no'}")
        print(f"[civitai] picked file: {filename} sha256={'yes' if sha256 else 'no'}", flush=True)
        print(f"[civitai] download_url={download_url}", flush=True)
        logger.info(f"[civitai] download_url={download_url}")

    # 落盘语义（按当前约定）：civitai 下载的模型文件直接放在 models_dir 根目录下
    rel_dir = Path(".")
    abs_dir = Path(models_dir).resolve()
    abs_dir.mkdir(parents=True, exist_ok=True)

    rel_path = Path(filename).as_posix()
    abs_path = (abs_dir / filename).resolve()
    with contextlib.suppress(Exception):
        print(f"[civitai] local_dir={rel_dir.as_posix()}", flush=True)
        print(f"[civitai] rel_path={rel_path}", flush=True)
        print(f"[civitai] abs_path={str(abs_path)}", flush=True)
    with contextlib.suppress(Exception):
        logger.info(f"[civitai] local_dir={rel_dir.as_posix()}")
        logger.info(f"[civitai] rel_path={rel_path}")
        logger.info(f"[civitai] abs_path={str(abs_path)}")

    # 共享：若有 sha256 且命中 index，尝试硬链/软链直接复用
    if sha256 and filename.lower().endswith(DOWNLOAD_SHARE_EXTS):
        sha_index.load()
        # 自清理：如果索引里记录的路径已经不存在，应该把这条脏数据删掉（否则索引会长期积累无效路径）
        # 只清理当前 sha，代价很小；并在有变更时落盘。
        try:
            removed = sha_index.prune_missing(sha256)
            if removed:
                await sha_index.save()
        except Exception:
            pass
        existing = sha_index.find_one(sha256)
        if existing:
            ok = sha_index.try_link(existing, abs_path)
            if ok:
                await send_log([f"[civitai] reused via link: sha256={sha256} -> {rel_path}"], 1)
                await send_status("completed", "reused via link", "", rel_path)
                with contextlib.suppress(Exception):
                    print(f"[civitai] reused via link -> {rel_path}", flush=True)
                    print("[civitai] skip download_url fetch (reused via link)", flush=True)
                with contextlib.suppress(Exception):
                    logger.info("[civitai] skip download_url fetch (reused via link)")
                sha_index.add(sha256, rel_path)
                await sha_index.save()
                return rel_path

    await send_log([f"[civitai] downloading {download_url} -> {rel_path}"], 1)
    with contextlib.suppress(Exception):
        print(f"[civitai] downloading -> {rel_path}", flush=True)
        logger.info(f"[civitai] downloading {download_url} -> {rel_path}")

    # 进度节流：worker 侧已是 3s；这里也对齐，减少 await/send_status 频率
    last_emit_t = 0.0
    last_percent = -1
    # 控制台：必须“>= 前端信息”。因此每次发送 status 时也打印一条（3s 节流 + percent 变化）
    last_console_emit_t = 0.0
    # 控制台：下载速率（基于最近一个窗口的增量，避免抖动过大）
    last_rate_t = 0.0
    last_rate_bytes = 0
    # 额外的低频桶打印（10% 一次），避免完全无输出（但不能替代 status 同步打印）
    last_print_bucket = -1

    tmp_path = abs_path.with_suffix(abs_path.suffix + ".part")
    # 断点续传：保留 .part，若服务端支持 Range 则从已下载的字节继续；否则回退为全量重下
    resume_from = 0
    if tmp_path.exists():
        with contextlib.suppress(Exception):
            resume_from = int(tmp_path.stat().st_size)
    if resume_from > 0:
        with contextlib.suppress(Exception):
            print(f"[civitai] resume_from={resume_from} bytes (part file exists)", flush=True)
            logger.info(f"[civitai] resume_from={resume_from} bytes (part file exists)")

    headers2 = {}
    if civitai_token:
        headers2["Authorization"] = f"Bearer {civitai_token}"
    if resume_from > 0:
        headers2["Range"] = f"bytes={resume_from}-"

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), trust_env=True) as session:
        async with session.get(download_url, headers=headers2) as resp:
            # 断点续传：Range 请求应返回 206；如果返回 200 表示不支持/未生效，回退重下
            if resp.status == 416 and resume_from > 0:
                # 本地 part 可能比远端还大（或远端文件变了），重置
                with contextlib.suppress(Exception):
                    tmp_path.unlink()
                resume_from = 0
                raise RuntimeError("civitai resume failed: HTTP 416 (range not satisfiable); part file cleared, please retry")

            if resp.status not in (200, 206):
                text = await resp.text()
                with contextlib.suppress(Exception):
                    print(f"[civitai] download failed: HTTP {resp.status}: {text[:200]}", flush=True)
                raise RuntimeError(f"Download failed: HTTP {resp.status}: {text[:200]}")

            # 若请求了 Range 但服务端回 200，则说明没续上：清理 part 并从头写
            if resume_from > 0 and resp.status == 200:
                with contextlib.suppress(Exception):
                    print("[civitai] WARNING: server did not honor Range (got 200). Restarting from 0...", flush=True)
                    logger.warning("[civitai] server did not honor Range (got 200). Restarting from 0...")
                with contextlib.suppress(Exception):
                    tmp_path.unlink()
                resume_from = 0

            # total：对 206，用 Content-Range 推出总大小；否则用 content_length
            total = 0
            if resp.status == 206:
                cr = resp.headers.get("Content-Range", "") or resp.headers.get("content-range", "")
                # e.g. "bytes 123-456/789"
                m = re.search(r"/(\d+)\s*$", cr)
                if m:
                    with contextlib.suppress(Exception):
                        total = int(m.group(1))
            if not total:
                total = resp.content_length or 0

            got = int(resume_from or 0)
            # 速率统计初始化
            last_rate_t = time.monotonic()
            last_rate_bytes = int(got)
            with contextlib.suppress(Exception):
                if total:
                    print(f"[civitai] content_length={int(total)} bytes", flush=True)
            mode = "ab" if got > 0 else "wb"
            with open(tmp_path, mode) as f:
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    if cancel_ev.is_set():
                        raise asyncio.CancelledError()
                    f.write(chunk)
                    got += len(chunk)
                    now = time.monotonic()
                    percent = int((got / total) * 100) if total else 0
                    # 3 秒节流（避免刷屏）
                    if (now - last_emit_t) >= 3.0 and percent != last_percent:
                        last_emit_t = now
                        last_percent = percent
                        await send_status(
                            "downloading",
                            f"{got}/{total} bytes ({percent}%)" if total else f"{got} bytes ({percent}%)",
                            "",
                            "",
                            got,
                            int(total or 0),
                            percent,
                        )
                        # 控制台同步：至少和前端一样“看得到进度”
                        if (now - last_console_emit_t) >= 3.0:
                            last_console_emit_t = now
                            dt = now - (last_rate_t or now)
                            db = got - int(last_rate_bytes or 0)
                            rate_bps = (db / dt) if dt > 0 else 0.0
                            last_rate_t = now
                            last_rate_bytes = int(got)
                            rate_mbps = rate_bps / (1024 * 1024)
                            with contextlib.suppress(Exception):
                                print(
                                    f"[civitai] progress: {got}/{int(total or 0)} bytes ({percent}%) "
                                    f"rate={rate_mbps:.2f} MB/s",
                                    flush=True,
                                )
                    # 终端：按 10% 桶打印一次（更直观、更低频）
                    if total:
                        bucket = int(percent // 10)
                        if bucket != last_print_bucket:
                            last_print_bucket = bucket
                            with contextlib.suppress(Exception):
                                dt = now - (last_rate_t or now)
                                db = got - int(last_rate_bytes or 0)
                                rate_bps = (db / dt) if dt > 0 else 0.0
                                rate_mbps = rate_bps / (1024 * 1024)
                                print(
                                    f"[civitai] progress: {got}/{int(total)} bytes ({percent}%) "
                                    f"rate={rate_mbps:.2f} MB/s",
                                    flush=True,
                                )

    if sha256:
        actual = file_sha256(tmp_path)
        if actual.lower() != sha256.lower():
            raise RuntimeError(f"sha256 mismatch: expected {sha256}, got {actual}")

    tmp_path.replace(abs_path)
    with contextlib.suppress(Exception):
        print(f"[civitai] downloaded -> {rel_path}", flush=True)

    # 增量更新 sha 索引（仅对大权重）
    try:
        if abs_path.stat().st_size >= DOWNLOAD_SHARE_MIN_SIZE_BYTES and abs_path.name.lower().endswith(DOWNLOAD_SHARE_EXTS):
            sha_index.load()
            h = sha256 or file_sha256(abs_path)
            sha_index.add(h, rel_path)
            await sha_index.save()
    except Exception:
        pass

    await send_status("completed", "downloaded", "", rel_path)
    return rel_path


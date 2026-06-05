"""
DownloadWorker：接收后端广播的 download/download_cancel，抢占锁并执行下载。

当前实现优先支持 civitai（通过 model-versions/{versionId} 获取下载链接与 sha256）。
HF/MS 的具体文件清单/下载策略可后续接入（预留钩子）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from common.logger import get_logger
from common.message_queue import MessageQueue
from common.ws_client import WebSocketClient
from common.config_loader import StartupConfig

from .sha_index import ShaIndex, file_sha256
from download.providers.civitai_provider import download_civitai
from download.providers.modelscope_provider import download_modelscope
from download.providers.huggingface_provider import download_huggingface

logger = get_logger(__name__)


@dataclass
class DownloadJob:
    model_key: str
    source: Dict[str, Any]
    asset_type: str = "checkpoint"

    @property
    def provider(self) -> str:
        return str((self.source or {}).get("provider") or "").strip().lower()

    @property
    def repo_id(self) -> str:
        return str((self.source or {}).get("repo_id") or "").strip()


class DownloadWorker:
    def __init__(
        self,
        *,
        service_id: str,
        startup: StartupConfig,
        ws_client: WebSocketClient,
        message_queue: MessageQueue,
    ):
        self.service_id = service_id
        self.startup = startup
        self.ws = ws_client
        self.q = message_queue

        self.models_dir = Path(self.startup.inference_config.models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / ".locks").mkdir(parents=True, exist_ok=True)

        # LoRA 目录：asset_type=lora 时使用
        self.loras_dir = Path(self.startup.inference_config.loras_dir).resolve()
        self.loras_dir.mkdir(parents=True, exist_ok=True)
        (self.loras_dir / ".locks").mkdir(parents=True, exist_ok=True)

        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}

        # sha 索引（models_sha256.json）：按 base_dir 分开维护
        self._sha_index_by_base: Dict[str, ShaIndex] = {
            str(self.models_dir): ShaIndex(self.models_dir),
            str(self.loras_dir): ShaIndex(self.loras_dir),
        }

        # 统一节流：避免 ws/前端刷屏
        self._last_status_emit_at: Dict[str, float] = {}
        self._last_log_emit_at: Dict[str, float] = {}
        self._last_log_pending: Dict[str, str] = {}

        # civitai token 从 service config 中读取（允许缺省）
        self.civitai_token = None
        try:
            self.civitai_token = (self.startup.config or {}).get("civitai_token") or (self.startup.config or {}).get("civitaiToken")
        except Exception:
            self.civitai_token = None

    async def run_forever(self):
        logger.info(f"DownloadWorker started: service_id={self.service_id}, models_dir={self.models_dir}")
        while True:
            # MessageQueue 底层是 queue.Queue（阻塞）；用 to_thread 避免卡住事件循环
            msg = await asyncio.to_thread(self.q.get, 0.5)
            if not msg:
                await asyncio.sleep(0.05)
                continue

            # 诊断：打印下载器侧“收到的原始消息”（不做过滤）
            # 说明：这里的 msg 来自 ws_client->message_queue 的入队结果，基本等同后端广播内容
            try:
                raw = json.dumps(msg, ensure_ascii=False, sort_keys=True)
            except Exception:
                raw = str(msg)
            with contextlib.suppress(Exception):
                print(f"[download_worker] received_message={raw}", flush=True)
            with contextlib.suppress(Exception):
                logger.info(f"[download_worker] received_message={raw}")

            mtype = msg.get("type")
            if mtype == "download":
                job = DownloadJob(
                    model_key=str(msg.get("model_key")),
                    source=dict(msg.get("source") or {}),
                    asset_type=str(msg.get("asset_type") or "checkpoint"),
                )
                await self._handle_download(job)
            elif mtype == "download_cancel":
                model_key = str(msg.get("model_key"))
                await self._handle_cancel(model_key)
            else:
                # 兼容：忽略非 download 消息
                continue

    async def _handle_download(self, job: DownloadJob):
        if not job.model_key or not job.provider or not job.repo_id:
            return

        # 若已有任务在跑，忽略重复广播
        if job.model_key in self._running_jobs and not self._running_jobs[job.model_key].done():
            return

        # 单例运行：不做抢占锁（简化逻辑）
        logger.info(f"Download start: model_key={job.model_key} service_id={self.service_id}")

        cancel_ev = asyncio.Event()
        self._cancel_events[job.model_key] = cancel_ev

        async def _runner():
            try:
                await self._send_status(job, status="downloading", progress_text="starting...", error_text="")
                await self._send_log(job, [f"[worker] runner started: service_id={self.service_id}"], seq=0)
                await self._run_download(job, cancel_ev)
            except asyncio.CancelledError:
                await self._send_status(job, status="canceled", progress_text="", error_text="canceled")
                raise
            except Exception as e:
                await self._send_status(job, status="failed", progress_text="", error_text=str(e))
            finally:
                self._cancel_events.pop(job.model_key, None)

        t = asyncio.create_task(_runner(), name=f"download:{job.model_key}")
        self._running_jobs[job.model_key] = t

    async def _handle_cancel(self, model_key: str):
        ev = self._cancel_events.get(model_key)
        if ev:
            ev.set()

    async def _send_status(
        self,
        job: DownloadJob,
        *,
        status: str,
        progress_text: str,
        error_text: str,
        local_path: str = "",
        bytes_downloaded: int = 0,
        bytes_total: int = 0,
        progress: int = 0,
    ):
        # WS：download_status 节流（默认 3s 一次）；最终态强制发
        now = time.monotonic()
        st = str(status or "").strip().lower()
        is_final = st in {"completed", "failed", "canceled"}
        last_t = self._last_status_emit_at.get(job.model_key, 0.0)
        if (not is_final) and (now - last_t) < 3.0:
            return
        self._last_status_emit_at[job.model_key] = now

        # 本地：只在最终态/错误时打一条，避免刷屏（原始 CLI 输出由 provider 负责原样输出）
        try:
            if is_final or str(error_text or "").strip():
                logger.info(
                    f"[download_status] model_key={job.model_key} status={st or '-'} "
                    f'progress_text="{str(progress_text or "").strip()}" '
                    f'error="{str(error_text or "").strip()}"'
                )
        except Exception:
            pass
        msg = {
            "type": "download_status",
            "model_key": job.model_key,
            "source": dict(job.source or {}),
            "status": status,
            "progress_text": progress_text,
            "error_text": error_text,
            "local_path": local_path,
            "bytes_downloaded": int(bytes_downloaded or 0),
            "bytes_total": int(bytes_total or 0),
            "progress": int(progress or 0),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ok = await self.ws.send_message(msg)
        if not ok:
            logger.warning(f"Failed to send download_status: model_key={job.model_key} status={status}")

    async def _send_log(self, job: DownloadJob, lines: List[str], seq: int):
        # WS：download_log 每秒最多发一次；取最新一行，1 秒内只保留最新 pending
        picked = ""
        for ln in (lines or []):
            s = str(ln if ln is not None else "").strip()
            if s:
                picked = s
        if not picked:
            return

        now = time.monotonic()
        last_t = self._last_log_emit_at.get(job.model_key, 0.0)
        if (now - last_t) < 1.0:
            self._last_log_pending[job.model_key] = picked
            return
        self._last_log_emit_at[job.model_key] = now
        picked2 = self._last_log_pending.pop(job.model_key, "") or picked

        msg = {
            "type": "download_log",
            "model_key": job.model_key,
            "source": dict(job.source or {}),
            "seq": int(seq),
            "lines": [picked2],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ok = await self.ws.send_message(msg)
        if not ok:
            logger.warning(f"Failed to send download_log: model_key={job.model_key} seq={seq}")

    async def _run_download(self, job: DownloadJob, cancel_ev: asyncio.Event):
        asset_type = str(job.asset_type or "").strip().lower()
        base_dir = self.loras_dir if asset_type == "lora" else self.models_dir
        sha_index = self._sha_index_by_base.get(str(base_dir)) or ShaIndex(base_dir)

        provider = job.provider
        if provider == "civitai":
            local_path = await download_civitai(
                version_id=job.repo_id,
                models_dir=base_dir,
                sha_index=sha_index,
                cancel_ev=cancel_ev,
                civitai_token=self.civitai_token,
                send_log=lambda lines, seq: self._send_log(job, lines, seq),
                # 兼容：provider 可选携带 bytes/progress（用于前端实时进度）
                send_status=lambda status, progress_text, error_text, local_path2, bytes_downloaded=0, bytes_total=0, progress=0: self._send_status(
                    job,
                    status=status,
                    progress_text=progress_text,
                    error_text=error_text,
                    local_path=local_path2,
                    bytes_downloaded=bytes_downloaded,
                    bytes_total=bytes_total,
                    progress=progress,
                ),
            )
            # provider 内部已经在 completed 时回传；这里不重复
            return
        if provider == "modelscope":
            local_path = await download_modelscope(
                repo_id=job.repo_id,
                models_dir=base_dir,
                sha_index=sha_index,
                cancel_ev=cancel_ev,
                default_excludes=["*.gguf"],
                send_log=lambda lines, seq: self._send_log(job, lines, seq),
                send_status=lambda status, progress_text, error_text, local_path2, bytes_downloaded=0, bytes_total=0, progress=0: self._send_status(
                    job,
                    status=status,
                    progress_text=progress_text,
                    error_text=error_text,
                    local_path=local_path2,
                    bytes_downloaded=bytes_downloaded,
                    bytes_total=bytes_total,
                    progress=progress,
                ),
            )
            await self._send_status(job, status="completed", progress_text="downloaded", error_text="", local_path=local_path)
            return
        if provider == "huggingface":
            # HF 的默认 excludes 可通过你给的规则扩展（例如 text_encoder/* 等）
            local_path = await download_huggingface(
                repo_id=job.repo_id,
                models_dir=base_dir,
                sha_index=sha_index,
                cancel_ev=cancel_ev,
                default_excludes=[],
                send_log=lambda lines, seq: self._send_log(job, lines, seq),
                send_status=lambda status, progress_text, error_text, local_path2, bytes_downloaded=0, bytes_total=0, progress=0: self._send_status(
                    job,
                    status=status,
                    progress_text=progress_text,
                    error_text=error_text,
                    local_path=local_path2,
                    bytes_downloaded=bytes_downloaded,
                    bytes_total=bytes_total,
                    progress=progress,
                ),
            )
            await self._send_status(job, status="completed", progress_text="downloaded", error_text="", local_path=local_path)
            return
        # 预留：huggingface / modelscope（先做到“可观测”：一定回传日志+失败）
        await self._send_log(
            job,
            [
                f"[{provider}] download not implemented yet.",
                "This is expected before file-list API spec is integrated.",
            ],
            seq=0,
        )
        raise RuntimeError(f"source.provider={provider} not implemented yet")

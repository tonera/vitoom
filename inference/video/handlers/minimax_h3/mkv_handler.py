"""MiniMax-H3 的 MKV / INP 入口。

没有图为 t2va；有首帧（url）或尾帧（image_file2）为 fl2va；MKV 带 tpl_list / ref_video / ref_audio 为 ref2va。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from common.config_loader import InferenceConfig
from common.logger import get_logger
from common.result_handler import ResultHandler
from common.task_cancel import TaskCancelledError
from schemas import InferenceRequestParams

from video.handlers.minimax_h3.models import H3Models
from video.handlers.minimax_h3.release import release_h3_models
from video.handlers.minimax_h3.references import load_references
from video.handlers.minimax_h3.request import H3Request, prepare_h3_request, reference_sources
from video.handlers.minimax_h3.runner import load_h3_models, run_h3
from video.runtime.io_utils import load_image_from_url_or_path

logger = get_logger(__name__)


class MiniMaxH3MkvHandler:
    def __init__(
        self,
        *,
        inference_config: InferenceConfig,
        result_handler: ResultHandler,
        service_id: str,
        logger=logger,
        run_blocking: Optional[Callable[..., Any]] = None,
        check_cancelled: Optional[Callable[[str], Any]] = None,
        is_task_cancelled: Optional[Callable[[], bool]] = None,
        pipeline_cache=None,
    ):
        self.inference_config = inference_config
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled
        self.is_task_cancelled = is_task_cancelled
        self.pipeline_cache = pipeline_cache

    async def run(self, params: InferenceRequestParams, task_id: str) -> None:
        sources = reference_sources(params)
        request = prepare_h3_request(
            params,
            models_dir=self.inference_config.models_dir,
            image=await _load_keyframe(params.url),
            last_image=await _load_keyframe(params.image_file2),
            references=await load_references(sources) if sources else None,
            log=self.logger.info,
        )
        cache_key = f"h3|{request.checkpoints.root}|{request.checkpoints.tokenizer_dir}"
        cached = self.pipeline_cache is not None and self.pipeline_cache.enabled()
        models: H3Models | None = None
        workspace = Path(tempfile.mkdtemp(prefix=f"h3_{task_id}_"))
        started = time.time()
        try:
            if await self._cancelled(task_id, "before encode"):
                return
            models = await self._acquire(request, cache_key)
            paths = [workspace / f"{task_id}_{index}.mp4" for index in range(len(request.seeds))]
            outputs, stats = await self._blocking(
                run_h3,
                request,
                models,
                paths,
                task_id=task_id,
                is_cancelled=self.is_task_cancelled,
                log=self.logger.info,
            )
            self.logger.info(
                f"[h3] task={task_id} peak={stats.peak_bytes / 2**30:.2f} GiB "
                f"allocated={stats.allocated_bytes / 2**30:.2f} GiB videos={len(outputs)}"
            )
            elapsed = time.time() - started
            for index, output in enumerate(outputs):
                if await self._cancelled(task_id, "before upload"):
                    return
                await self.result_handler.process_single_result(
                    file_data=output.path.read_bytes(),
                    request_params=params,
                    generate_time=elapsed,
                    service_id=self.service_id,
                    file_seed=output.seed,
                    index=index,
                    total=len(outputs),
                )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
            if cached:
                await self.pipeline_cache.release_use(key=cache_key)
            elif models is not None:
                await self._blocking(release_h3_models, models)

    async def _acquire(self, request: H3Request, cache_key: str) -> H3Models:
        """经过 PipelineCache 取分词器：key 变化时它会先驱逐常驻的 Wan2 / Turbo pipeline。"""
        create = lambda: self._blocking(load_h3_models, request, self.logger.info)
        if self.pipeline_cache is None:
            return await create()
        models, _hit = await self.pipeline_cache.acquire(key=cache_key, create_fn=create)
        return models

    async def _blocking(self, func, *args, **kwargs):
        if self.run_blocking is not None:
            return await self.run_blocking(lambda: func(*args, **kwargs))
        return func(*args, **kwargs)

    async def _cancelled(self, task_id: str, stage: str) -> bool:
        if self.is_task_cancelled is not None and self.is_task_cancelled():
            raise TaskCancelledError(task_id, stage)
        if self.check_cancelled is not None and await self.check_cancelled(stage):
            return True
        return False


async def _load_keyframe(url_or_path: str | None):
    """原图比例保留到画布解析，缩放和裁剪交给 diffusers 的关键帧步骤。"""
    if not str(url_or_path or "").strip():
        return None
    return await load_image_from_url_or_path(str(url_or_path).strip(), size=None)

"""
图片推理器主类（重写版）
按设备规划 / 模型定位 / 模式处理 分层编排
"""
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from common.base_inferrer import BaseInferrer
from common.logger import get_logger
from common.result_handler import ResultHandler
from common.pipeline_detector import PipelineDetector
from common.config_loader import load_inference_config
from common.message_cache import MessageCache
from common.Constant import (
    JT_ED,
    JT_RBG,
    JT_SR,
    JT_FS,
    JT_ID,
    JT_POSE,
)
from image.runtime.params_preprocessor import preprocess_inference_params
from image.handlers.diffusion_handler import DiffusionHandler
from image.handlers.editor_handler import EditorHandler
from image.handlers.id_handler import IdHandler
from image.handlers.rbg_sr_fs_handler import RbgSrFsHandler
from image.runtime.device_planner import DevicePlanner
from image.runtime.model_locator import ModelLocator
from common.pipeline_cache import PipelineCache
from image.runtime.pipeline_lifecycle import PipelineLifecycle
from image.runtime.pipeline_release import force_release_pipeline
from image.runtime.seed_manager import SeedManager
from schemas import InferenceRequestParams

logger = get_logger(__name__)

class ImageInferrer(BaseInferrer):
    """图片推理器"""

    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.result_handler: Optional[ResultHandler] = None
        self.message_cache: Optional[MessageCache] = None
        self.detector = PipelineDetector()
        self.inference_config = load_inference_config(service_id=service_id)
        self.device_planner = DevicePlanner()
        self.model_locator = ModelLocator(self.inference_config.models_dir)
        self.seed_manager = SeedManager()
        # LRU=1 pipeline 缓存（TTL 由 inference.yaml 控制）
        self.pipeline_cache: PipelineCache = PipelineCache(
            ttl_seconds=getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0),
            logger=logger,
            # 关键：驱逐释放也放到 run_blocking 的 worker 线程执行（与推理线程一致），并做二次清理
            release_fn=self._release_cached_pipeline,
        )

    async def _release_cached_pipeline(self, pipe: Any) -> None:
        # 与推理线程一致的释放路径，避免“TTL 驱逐日志有了但显存不掉”的观感
        await force_release_pipeline(
            pipe,
            logger=logger,
            run_blocking=self.run_blocking,
            # cache evict 场景更激进：先 to(cpu) 再释放，避免残留引用导致显存顽固占用
            aggressive_cpu=True,
        )

    async def initialize(self):
        """初始化推理器（加载配置、初始化组件）"""
        await super().initialize()
        self.result_handler = ResultHandler(
            ws_client=self.ws_client,
            storage_base_path=self.inference_config.outputs_dir,
            inference_config=self.inference_config,
        )
        # 启动缓存驱逐循环（若 ttl>0）
        try:
            self.pipeline_cache.start()
        except Exception:
            pass
        logger.info("Image inferrer initialized")

    async def cleanup(self):
        """清理资源（信号处理器回调）"""
        try:
            await self.pipeline_cache.stop()
        except Exception:
            pass
        await super().cleanup()

    async def _check_cancelled(self, task_id: str, stage: str) -> bool:
        if self.task_processor.is_task_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled {stage}")
            await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return True
        return False

    async def _cleanup_cache(self, task_id: str) -> None:
        """清理与任务关联的临时消息缓存"""
        if hasattr(self, "message_cache") and self.message_cache:
            cache_file = self.message_cache.get_cache_file_by_task_id(task_id)
            if cache_file:
                await self.message_cache.delete_message(cache_file)
                logger.debug(f"Deleted cache file for completed task: {task_id}")

    async def _send_task_status(
        self, task_id: str, status: str, error: Optional[str] = None
    ) -> None:
        """统一封装任务状态上报"""
        payload = {"task_id": task_id, "status": status}
        if status == "completed":
            payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        if error:
            payload["error"] = error
        await self.ws_client.send_task_status(**payload)

    @asynccontextmanager
    async def _task_context(self, task_id: str):
        """
        统一的任务生命周期包装：清理缓存 + 状态上报
        取消场景直接退出（状态已由取消检查发送）
        """
        try:
            yield
        except Exception as exc:
            await self._cleanup_cache(task_id)
            if not self.task_processor.is_task_cancelled(task_id):
                await self._send_task_status(task_id, "failed", error=str(exc))
            raise
        else:
            await self._cleanup_cache(task_id)
            if not self.task_processor.is_task_cancelled(task_id):
                await self._send_task_status(task_id, "completed")

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        """任务入口：调度模式、执行迭代、上报状态"""
        task_id = params.task_id
        cancel_event = self.task_processor.get_task_cancel_event(task_id)
        logger.info(f"Starting image inference for task: {task_id}")
        logger.info(f"params: {params}")

        if await self._check_cancelled(task_id, "before inference"):
            return

        async with self._task_context(task_id):
            # 统一的扁平预处理入口（集中管理，避免调用长链）
            params = await preprocess_inference_params(
                params,
                detector=self.detector,
                inference_config=self.inference_config,
                logger=logger,
                run_blocking=self.run_blocking,
            )
            logger.warning(f"推理Prompt: {params.prompt}")

            # lifecycle & handlers
            lifecycle = PipelineLifecycle(
                detector=self.detector,
                device_planner=self.device_planner,
                model_locator=self.model_locator,
                inference_config=self.inference_config,
                logger=logger,
                pipeline_cache=self.pipeline_cache,
                run_blocking=self.run_blocking,
            )

            if params.job_type == JT_ID:
                handler = IdHandler(
                    inference_config=self.inference_config,
                    device_planner=self.device_planner,
                    seed_manager=self.seed_manager,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set,
                    pipeline_cache=self.pipeline_cache,
                    lifecycle=lifecycle,
                )
                inference_start_time = time.time()
                await handler.run(params, task_id=task_id)
                inference_time = time.time() - inference_start_time
                logger.info(f"Inference completed in {inference_time:.2f}s")
            elif params.job_type in (JT_ED, JT_POSE):
                handler = EditorHandler(
                    inference_config=self.inference_config,
                    lifecycle=lifecycle,
                    seed_manager=self.seed_manager,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set,
                )
                inference_start_time = time.time()
                await handler.run(params, task_id=task_id)
                inference_time = time.time() - inference_start_time
                logger.info(f"Inference completed in {inference_time:.2f}s")
            elif params.job_type in (JT_RBG, JT_SR, JT_FS):
                # 非 pipeline 任务自身也可能加载 GPU 模型，先释放缓存的 diffusers pipeline 避免显存叠加。
                await self.pipeline_cache.evict(force=True)
                handler = RbgSrFsHandler(
                    inference_config=self.inference_config,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                )
                await handler.run(params)
            else:
                handler = DiffusionHandler(
                    lifecycle=lifecycle,
                    seed_manager=self.seed_manager,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set,
                )
                inference_start_time = time.time()
                await handler.run(params, task_id=task_id)
                inference_time = time.time() - inference_start_time
                logger.info(f"Inference completed in {inference_time:.2f}s")

        logger.info(f"Task {task_id} completed successfully")


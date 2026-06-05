"""
视频推理器主类（框架版）
复用 common/BaseInferrer 的 WS 连接、消息队列、任务消费与取消机制。
"""

import time
from typing import Any, Optional

from common.base_inferrer import BaseInferrer
from common.logger import get_logger, print_info
from common.result_handler import ResultHandler
from common.config_loader import load_inference_config
from common.pipeline_cache import PipelineCache
from common.Constant import JT_MKV, JT_S2V, JT_INP, JT_CCV
from common.task_cancel import TaskCancelledError
from schemas import InferenceRequestParams

from video.handlers.wan2.mkv_handler import Wan2MkvHandler
from video.handlers.wan2.s2v_handler import Wan2S2vHandler
from video.handlers.wan2.inp_handler import Wan2InpHandler
from video.handlers.wan2.ccv_handler import Wan2CcvHandler
from video.handlers.turbodiffusion.mkv_handler import TurboDiffusionMkvHandler
from video.runtime.wan2_pipeline_release import release_wan2_pipeline_twice_async

logger = get_logger(__name__)


class VideoInferrer(BaseInferrer):
    """视频推理器（入口：TaskProcessor -> inference_callback）"""

    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.inference_config = load_inference_config(service_id=service_id)
        self.result_handler: Optional[ResultHandler] = None
        # LRU=1 pipeline 缓存（TTL 由 inference.yaml 控制）
        self.pipeline_cache: PipelineCache = PipelineCache(
            ttl_seconds=getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0),
            logger=logger,
            # 关键：驱逐释放也放到 run_blocking 的 worker 线程执行（与推理线程一致），并做二次清理
            release_fn=self._release_cached_pipeline,
        )

    async def _release_cached_pipeline(self, obj: Any) -> None:
        """
        PipelineCache 的统一驱逐释放入口：
        - Wan2: release_wan2_pipeline_twice_async
        - TurboDiffusion: release_turbo_models_twice_async
        """
        # 与推理线程一致的释放路径，避免“TTL 驱逐日志有了但显存不掉”的观感
        try:
            name = getattr(getattr(obj, "__class__", None), "__name__", "")
            mod = getattr(getattr(obj, "__class__", None), "__module__", "")
            if name == "TurboModels" or ("turbodiffusion" in str(mod)):
                from video.handlers.turbodiffusion.release import release_turbo_models_twice_async

                await release_turbo_models_twice_async(
                    obj,
                    log=logger,
                    run_blocking=self.run_blocking,
                    aggressive_cpu=True,
                )
                return
        except Exception:
            pass

        await release_wan2_pipeline_twice_async(
            obj,
            log=logger,
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
        logger.info("Video inferrer initialized")

    async def cleanup(self):
        """清理资源（信号处理器回调）"""
        try:
            await self.pipeline_cache.stop()
        except Exception:
            pass
        await super().cleanup()

    async def _check_cancelled(self, task_id: str, stage: str) -> bool:
        if self.task_processor and self.task_processor.is_task_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled {stage}")
            if self.ws_client:
                await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return True
        return False

    async def _send_task_status(self, task_id: str, status: str, error: Optional[str] = None) -> None:
        if not self.ws_client:
            return
        payload = {"task_id": task_id, "status": status}
        if status == "completed":
            payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        if error:
            payload["error"] = error
        await self.ws_client.send_task_status(**payload)

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        """
        视频任务入口：
        - 取消检查
        - Handler 分发（占位）
        - 回传 completed/failed（TaskProcessor 不会自动发 completed）
        """
        task_id = params.task_id
        cancel_event = self.task_processor.get_task_cancel_event(task_id) if self.task_processor else None
        logger.info(f"Starting video inference for task: {task_id}")
        # 每个推理任务开始前，打印一次推理参数（调试用）
        try:
            data = params.model_dump() if hasattr(params, "model_dump") else params  # pydantic v2
        except Exception:
            data = params
        print_info(data, prefix=f"[video][task_id={task_id}] ")

        if await self._check_cancelled(task_id, "before inference"):
            return None

        if not self.result_handler:
            raise RuntimeError("ResultHandler not initialized")

        try:
            # 当前实现：Wan2 专用 handler 路由（未来可扩展为 family registry）
            jt = (params.job_type or "").strip().upper()
            handler: Any
            if jt == JT_MKV:
                # TurboDiffusion 路由：使用既有 fast_mode 作为“加速开关”，并结合 load_name 识别 Turbo 模型
                use_turbo = False
                try:
                    mn = str(params.load_name or "").strip().lower()
                    use_turbo = bool(getattr(params, "fast_mode", False)) and (mn.startswith("turbo") or "turbowan" in mn)
                except Exception:
                    use_turbo = False

                handler_cls = TurboDiffusionMkvHandler if use_turbo else Wan2MkvHandler
                kwargs = dict(
                    inference_config=self.inference_config,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set if cancel_event is not None else (lambda: False),
                )
                kwargs["pipeline_cache"] = self.pipeline_cache
                handler = handler_cls(**kwargs)
            elif jt == JT_S2V:
                handler = Wan2S2vHandler(
                    inference_config=self.inference_config,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set if cancel_event is not None else (lambda: False),
                    pipeline_cache=self.pipeline_cache,
                )
            elif jt == JT_INP:
                handler = Wan2InpHandler(
                    inference_config=self.inference_config,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set if cancel_event is not None else (lambda: False),
                    pipeline_cache=self.pipeline_cache,
                )
            elif jt == JT_CCV:
                handler = Wan2CcvHandler(
                    inference_config=self.inference_config,
                    result_handler=self.result_handler,
                    service_id=self.service_id,
                    logger=logger,
                    run_blocking=self.run_blocking,
                    check_cancelled=lambda stage: self._check_cancelled(task_id, stage),
                    is_task_cancelled=cancel_event.is_set if cancel_event is not None else (lambda: False),
                    pipeline_cache=self.pipeline_cache,
                )
            else:
                raise ValueError(f"Unsupported video job_type={params.job_type} (expected one of MKV/S2V/INP/CCV)")

            await handler.run(params, task_id=task_id)

            if not (self.task_processor and self.task_processor.is_task_cancelled(task_id)):
                await self._send_task_status(task_id, "completed")
        except TaskCancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in video inference for task {task_id}: {e}", exc_info=True)
            if not (self.task_processor and self.task_processor.is_task_cancelled(task_id)):
                await self._send_task_status(task_id, "failed", error=str(e))
            raise


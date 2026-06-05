"""
MiniInferrer：mini 推理服务的主编排器

职责：
1. 继承 BaseInferrer，完成 WS / ingress / task_processor 等通用初始化
2. 维护一个 LRU=1 + TTL 的 bundle 缓存（跨 handler 通用）
   - key 由 handler 自己决定（一般是 runtime+model_ref+policy 的复合值）
   - release 时把真正的释放逻辑交给 bundle 自带的 shutdown 钩子
3. 按 family 分发到对应 handler
   - 这点和 text 服务的 runtime_resolver 一致：handler 选择由"模型身份"决定
   - job_type 仅作为任务语义标签（OCR/RERANK/…），不参与 handler 选择
4. 收到任务后调用 handler.handle(params)；handler 自己负责
   - 下载/解析输入文件
   - 从 cache 获取 bundle（miss 时自己 load）
   - 运行推理
   - 走 ResultHandler 回传结果（落盘 + 内联文本）

设计要点：
- LRU=1 是"以时间换显存"的刻意选择：小模型重载成本低（秒级），但显存很贵
- bundle 的 release 走 run_blocking 线程，和推理线程同一路径，避免显存残留
- 单 worker executor（BaseInferrer 默认 max_workers=1）天然串行，无需额外并发保护
- models 表统一管理：模型元信息（family、local_path、is_local_model）由 backend 在
  /v1/tasks 处查库回填，推理端仅使用 request.load_name + request.family 做决策
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from common.base_inferrer import BaseInferrer
from common.config_loader import load_inference_config
from common.logger import get_logger
from common.message_cache import MessageCache
from common.pipeline_cache import PipelineCache
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

from mini.handlers.ocr_handler import OcrHandler

logger = get_logger(__name__)


# family → Handler 工厂映射
# 新增小模型时在这里注册一行即可；handler 内部对同一类不同 bridge 的分支（同家族不同
# 厂商）可以再自行收敛。现在 GLM-OCR 是 OCR 类唯一模型；未来 Nougat 等加入时，
# 可以：
#   a) 也映射到 OcrHandler，由 OcrHandler 内部按 family 选 bridge；或
#   b) 独立建 NougatHandler，在这里多加一行映射——取决于模型间差异大小。
_HANDLER_REGISTRY: Dict[str, str] = {
    "GLM-OCR": "ocr",
    # 未来示例：
    # "BGE-rerank": "rerank",
    # "BGE-embed":  "embed",
}


class MiniInferrer(BaseInferrer):
    def __init__(self, service_id: str):
        super().__init__(service_id)
        self.result_handler: Optional[ResultHandler] = None
        self.message_cache: Optional[MessageCache] = None
        self.inference_config = load_inference_config(service_id=service_id)

        # TTL 从 inference.yaml / service 配置读取（与 image 服务一致）。
        # LRU=1：同一时刻只驻留一个 bundle；切换 job_type 或模型会驱逐旧 bundle。
        self.bundle_cache: PipelineCache = PipelineCache(
            ttl_seconds=getattr(self.inference_config, "pipeline_cache_ttl_seconds", 0),
            logger=logger,
            release_fn=self._release_bundle,
        )

        self._handlers: Dict[str, Any] = {}

    async def _release_bundle(self, bundle: Any) -> None:
        """bundle 驱逐时的真正释放：调用 bundle 自带的 shutdown 钩子。"""
        if bundle is None:
            return
        shutdown = getattr(bundle, "shutdown", None)
        if callable(shutdown):
            try:
                # 绝大多数 shutdown 都是同步阻塞的（调 torch 释放 / vLLM engine stop）
                await self.run_blocking(shutdown)
            except Exception as e:
                logger.warning(f"bundle shutdown raised: {e}", exc_info=True)
        else:
            logger.debug("bundle has no shutdown(), skip")

    async def initialize(self):
        await super().initialize()

        self.result_handler = ResultHandler(
            ws_client=self.ws_client,
            storage_base_path=self.inference_config.outputs_dir,
            inference_config=self.inference_config,
        )

        # 启动 TTL 驱逐循环（若 ttl>0）
        try:
            self.bundle_cache.start()
        except Exception:
            logger.warning("bundle_cache.start failed", exc_info=True)

        # 实例化所有已知 handler 实例；按"handler 内部 id"保存，
        # 再通过 _HANDLER_REGISTRY(family -> handler_id) 做查找。
        self._handlers = {
            "ocr": OcrHandler(
                inference_config=self.inference_config,
                bundle_cache=self.bundle_cache,
                result_handler=self.result_handler,
                service_id=self.service_id,
                logger=logger,
                run_blocking=self.run_blocking,
                check_cancelled=self._check_cancelled,
                service_model_cfg=self._service_model_cfg(),
                ws_client=self.ws_client,
            ),
        }

        logger.info(
            "MiniInferrer initialized. registry=%s handlers=%s",
            _HANDLER_REGISTRY,
            list(self._handlers.keys()),
        )

    def _service_model_cfg(self) -> Dict[str, Any]:
        """读取 service 配置里的 runtime / model 信息（用于 handler 兜底默认值）。"""
        if not self.config or not isinstance(self.config.config, dict):
            return {}
        return dict(self.config.config)

    def _check_cancelled(self, task_id: str) -> bool:
        if self.task_processor is None:
            return False
        try:
            return bool(self.task_processor.is_task_cancelled(task_id))
        except Exception:
            return False

    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        task_id = params.task_id
        job_type = str(getattr(params, "job_type", "") or "").strip().upper()
        family = str(getattr(params, "family", "") or "").strip()
        load_name = str(getattr(params, "load_name", "") or "").strip()

        logger.info(
            "[mini] task received task_id=%s job_type=%s family=%s load_name=%s",
            task_id, job_type, family, load_name,
        )

        if self._check_cancelled(task_id):
            if self.ws_client:
                await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return None

        if not family:
            raise ValueError(
                "mini task requires family (comes from models.family via backend). "
                f"Got empty family for task_id={task_id} load_name={load_name!r}."
            )

        handler_id = _HANDLER_REGISTRY.get(family)
        if handler_id is None:
            raise ValueError(
                f"mini service does not support family={family!r}; "
                f"registered classes: {sorted(_HANDLER_REGISTRY.keys())}. "
                f"Register it in inferrer._HANDLER_REGISTRY if this is a new small model."
            )
        handler = self._handlers.get(handler_id)
        if handler is None:
            raise RuntimeError(
                f"handler_id={handler_id!r} registered in _HANDLER_REGISTRY but not instantiated; "
                "check MiniInferrer.initialize."
            )

        try:
            result = await handler.handle(params)
        except Exception:
            # handler 内部失败一律上抛，交给 BaseInferrer/task_processor 统一走失败回传链路
            raise

        if self._check_cancelled(task_id):
            if self.ws_client:
                await self.ws_client.send_task_status(task_id=task_id, status="cancelled")
            return None

        if self.ws_client:
            try:
                await self.ws_client.send_task_status(task_id=task_id, status="completed")
            except Exception:
                logger.warning(f"send_task_status completed failed for task={task_id}", exc_info=True)

        return result

    async def stop(self):
        # 停止 TTL 驱逐循环并强制释放缓存中的 bundle
        try:
            await self.bundle_cache.stop()
        except Exception:
            logger.warning("bundle_cache.stop failed", exc_info=True)
        await super().stop()

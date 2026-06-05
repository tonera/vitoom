"""
视频推理 Handler（占位框架）

后续接入具体视频模型（例如 Wan-Video / HunyuanVideo / Wan2.2 等）时，
建议在这里实现：
- 参数校验与预处理
- 通过 run_blocking 将重型推理放到线程执行
- 产物用 ResultHandler.process_single_result() 保存/上传并通过 WS 回传
"""

import time
from typing import Any, Callable, Optional

from common.config_loader import InferenceConfig
from common.result_handler import ResultHandler
from common.logger import get_logger
from schemas import InferenceRequestParams

logger = get_logger(__name__)


class VideoHandler:
    def __init__(
        self,
        *,
        inference_config: InferenceConfig,
        result_handler: ResultHandler,
        service_id: str,
        logger=logger,
        run_blocking: Optional[Callable[..., Any]] = None,
        check_cancelled: Optional[Callable[[str], Any]] = None,
    ):
        self.inference_config = inference_config
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.run_blocking = run_blocking
        self.check_cancelled = check_cancelled

    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        """
        视频推理主入口（当前为占位实现）。

        约定：
        - params.type 必须为 "video"
        - 输出建议使用 params.file_type（默认 mp4）
        """
        if params.type != "video":
            raise ValueError(f"VideoHandler expected params.type='video', got '{params.type}'")

        if self.check_cancelled and await self.check_cancelled("at handler entry"):
            return

        start = time.time()

        # TODO: 在此处接入真实 video pipeline，并产出 mp4/webm 等 bytes 或本地文件
        # 产出后示例（假设 video_bytes 是 mp4 文件字节）：
        # await self.result_handler.process_single_result(
        #     file_data=video_bytes,
        #     request_params=params,
        #     generate_time=time.time() - start,
        #     service_id=self.service_id,
        #     index=0,
        #     total=1,
        # )

        elapsed = time.time() - start
        raise NotImplementedError(
            "Video inference not implemented yet. "
            f"job_type={params.job_type}, load_name={params.load_name}, "
            f"duration={params.duration}s, elapsed={elapsed:.2f}s"
        )


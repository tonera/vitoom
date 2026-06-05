"""
推理器主程序模板
展示如何使用BaseInferrer基类
"""
import asyncio
import sys
from pathlib import Path

# 添加inference目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_inferrer import BaseInferrer
from common.logger import get_logger
from schemas import InferenceRequestParams
from typing import Any
from datetime import datetime

logger = get_logger(__name__)


class ExampleInferrer(BaseInferrer):
    """示例推理器（子类需要实现inference_callback）"""
    
    async def inference_callback(self, params: InferenceRequestParams) -> Any:
        """
        推理回调函数
        
        这是子类必须实现的方法，用于执行实际的推理逻辑
        
        Args:
            params: 推理请求参数
        
        Returns:
            推理结果
        """
        task_id = params.task_id
        logger.info(f"Processing inference for task: {task_id}")
        
        # TODO: 在这里实现实际的推理逻辑
        # 例如：
        # - 加载模型
        # - 执行推理
        # - 保存结果
        # - 更新任务状态
        
        # 示例：检查任务是否被取消
        if self.task_processor.is_task_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled, aborting inference")
            # 通过WS发送状态更新（不再直接更新数据库）
            await self.ws_client.send_task_status(
                task_id=task_id,
                status="cancelled"
            )
            return
        
        # 示例：执行推理（伪代码）
        try:
            # result = await self._do_inference(params)
            # await self._save_result(task_id, result)
            # 通过WS发送任务完成状态（不再直接更新数据库）
            # await self.ws_client.send_task_status(
            #     task_id=task_id,
            #     status="completed",
            #     completed_at=datetime.utcnow().isoformat()
            # )
            pass
        except Exception as e:
            logger.error(f"Inference failed for task {task_id}: {e}", exc_info=True)
            # 通过WS发送任务失败状态（不再直接更新数据库）
            await self.ws_client.send_task_status(
                task_id=task_id,
                status="failed",
                error=str(e)
            )
            raise


async def main():
    """主函数"""
    # 从命令行参数获取service_id
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <service_id>")
        sys.exit(1)
    
    service_id = sys.argv[1]
    
    # 创建推理器实例
    inferrer = ExampleInferrer(service_id=service_id)
    
    try:
        # 运行推理器
        await inferrer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


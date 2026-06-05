"""
图片推理器主程序入口
"""
import asyncio
import sys
from pathlib import Path

# 添加inference目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from image.inferrer import ImageInferrer
from common.config_loader import load_startup_config
from common.logger import get_logger

logger = get_logger(__name__)


async def main():
    """主函数"""
    # 从命令行参数获取service_id
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <service_id>")
        sys.exit(1)
    
    service_id = sys.argv[1]
    startup_config = load_startup_config(service_id)
    service_type = str(getattr(startup_config, "service_type", "") or "").strip().lower()
    if service_type != "image":
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{service_id}.yaml"
        recommended_cmd = f"python inference/{service_type or '<type>'}/main.py {service_id}"
        logger.error(
            f"Service '{service_id}' is configured as service_type='{service_type or '<empty>'}' in "
            f"'{config_path}', but this entrypoint only supports 'image'. "
            f"Recommended command: {recommended_cmd}"
        )
        sys.exit(1)
    
    # 创建推理器实例
    inferrer = ImageInferrer(service_id=service_id)
    
    try:
        # 运行推理器
        await inferrer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


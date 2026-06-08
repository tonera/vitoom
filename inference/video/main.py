"""
视频推理器主程序入口
"""

import asyncio
import sys
from pathlib import Path

# 添加 inference 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_loader import load_startup_config
from video.inferrer import VideoInferrer
from common.logger import get_logger

logger = get_logger(__name__)


async def main() -> None:
    """主函数"""
    if len(sys.argv) < 2:
        logger.error("Usage: python inference/video/main.py <service_id>")
        sys.exit(1)

    service_id = sys.argv[1]
    startup_config = load_startup_config(service_id)
    service_type = str(getattr(startup_config, "service_type", "") or "").strip().lower()
    if service_type != "video":
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{service_id}.yaml"
        recommended_cmd = f"python inference/{service_type or '<type>'}/main.py {service_id}"
        logger.error(
            f"Service '{service_id}' is configured as service_type='{service_type or '<empty>'}' in "
            f"'{config_path}', but this entrypoint only supports 'video'. "
            f"Recommended command: {recommended_cmd}"
        )
        sys.exit(1)

    inferrer = VideoInferrer(service_id=service_id)

    try:
        await inferrer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


"""
Mini 推理器主程序入口

用法：
    python inference/mini/main.py <service_id>

其中 <service_id> 对应 inference/config/<service_id>.yaml；
该配置必须包含 service_type: "mini"。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 关键：必须在 import torch 之前运行，确保 pip 装的 nvrtc 运行时库能被 loader 找到
# （torch cu13 wheel 不再自带 libnvrtc-builtins.so.13.0；若系统只装了老 CUDA toolkit
# 也会缺这个 .so。此函数负责探测并一次性把路径前置到 LD_LIBRARY_PATH 并 re-exec。
# 用户只需 `pip install nvidia-cuda-nvrtc` 即可，无需手动设环境变量。）
from common.cuda_libs_bootstrap import ensure_cuda_runtime_libs  # noqa: E402
ensure_cuda_runtime_libs()

from common.config_loader import load_startup_config  # noqa: E402
from common.logger import get_logger  # noqa: E402
from mini.inferrer import MiniInferrer  # noqa: E402

logger = get_logger(__name__)


async def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python inference/mini/main.py <service_id>")
        sys.exit(1)

    service_id = sys.argv[1]
    startup_config = load_startup_config(service_id)
    service_type = str(getattr(startup_config, "service_type", "") or "").strip().lower()
    if service_type != "mini":
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{service_id}.yaml"
        recommended_cmd = f"python inference/{service_type or '<type>'}/main.py {service_id}"
        logger.error(
            f"Service '{service_id}' is configured as service_type='{service_type or '<empty>'}' in "
            f"'{config_path}', but this entrypoint only supports 'mini'. "
            f"Recommended command: {recommended_cmd}"
        )
        sys.exit(1)

    inferrer = MiniInferrer(service_id=service_id)
    try:
        await inferrer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

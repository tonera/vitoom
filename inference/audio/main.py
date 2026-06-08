"""
音频推理器主程序入口
"""

import asyncio
import sys
from pathlib import Path

_inference_root = Path(__file__).parent.parent
sys.path.insert(0, str(_inference_root))
_third_party = _inference_root / "third_party"
if str(_third_party) not in sys.path:
    sys.path.append(str(_third_party))

# 必须在 import torch / vllm 相关模块之前运行。vLLM wheel 可能链接到 pip
# 安装的 CUDA runtime（例如 libcudart.so.12），Linux 动态链接器需要在进程
# 启动早期通过 LD_LIBRARY_PATH 看到这些目录。
from common.cuda_libs_bootstrap import ensure_cuda_runtime_libs  # noqa: E402
ensure_cuda_runtime_libs(require_nvrtc=False, required_sonames=("libcudart.so.12",))

from audio.inferrer import AudioInferrer  # noqa: E402
from common.config_loader import load_startup_config  # noqa: E402
from common.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


async def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python inference/audio/main.py <service_id>")
        sys.exit(1)

    service_id = sys.argv[1]
    startup_config = load_startup_config(service_id)
    service_type = str(getattr(startup_config, "service_type", "") or "").strip().lower()
    if service_type != "audio":
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{service_id}.yaml"
        recommended_cmd = f"python inference/{service_type or '<type>'}/main.py {service_id}"
        logger.error(
            f"Service '{service_id}' is configured as service_type='{service_type or '<empty>'}' in "
            f"'{config_path}', but this entrypoint only supports 'audio'. "
            f"Recommended command: {recommended_cmd}"
        )
        sys.exit(1)

    inferrer = AudioInferrer(service_id=service_id)
    try:
        await inferrer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

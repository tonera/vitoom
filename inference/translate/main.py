"""
翻译推理器主程序入口

用法：
    python inference/translate/main.py <service_id>
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.cuda_libs_bootstrap import ensure_cuda_runtime_libs  # noqa: E402
ensure_cuda_runtime_libs(require_nvrtc=False, required_sonames=("libcudart.so",))

from common.config_loader import load_startup_config  # noqa: E402
from common.logger import get_logger  # noqa: E402
from translate.inferrer import TranslateInferrer  # noqa: E402

logger = get_logger(__name__)


async def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python inference/translate/main.py <service_id>")
        sys.exit(1)

    service_id = sys.argv[1]
    startup_config = load_startup_config(service_id)
    service_type = str(getattr(startup_config, "service_type", "") or "").strip().lower()
    if service_type != "translate":
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{service_id}.yaml"
        recommended_cmd = f"python inference/{service_type or '<type>'}/main.py {service_id}"
        logger.error(
            f"Service '{service_id}' is configured as service_type='{service_type or '<empty>'}' in "
            f"'{config_path}', but this entrypoint only supports 'translate'. "
            f"Recommended command: {recommended_cmd}"
        )
        sys.exit(1)

    inferrer = TranslateInferrer(service_id=service_id)
    try:
        await inferrer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

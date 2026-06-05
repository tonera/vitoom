"""数字人 sidecar 主入口（avatar 类型）。

启动方式：
    python inference/avatar/main.py <service_id>

例如：
    python inference/avatar/main.py livetalking

会读取 ``inference/config/<service_id>.yaml``，按 yaml 里的 ``host`` / ``port`` /
``config.model`` / ``config.avatar_id`` 等字段拉起独立 aiohttp 进程。

为什么不复用 inference/audio/main.py 的 AudioInferrer 体系：
    AudioInferrer 是 task-based（消费 task message 出结果）的形态；LiveTalking
    是 always-on WebRTC + WS PCM 流式输入服务，模型不一致。这里走自己的
    sidecar runner（``inference/avatar/<backend>/main.py``）。
"""

import asyncio
import importlib
import sys
from pathlib import Path

_inference_root = Path(__file__).parent.parent
sys.path.insert(0, str(_inference_root))

from common.config_loader import load_startup_config  # noqa: E402
from common.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# service_id → 该 sidecar 的 main module 相对路径
# 加新 backend（例如未来接 SadTalker / EchoMimic）时在此登记。
_SIDECAR_RUNNERS = {
    "livetalking": "avatar.livetalking.main",
}


async def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python inference/avatar/main.py <service_id>")
        logger.error("Available service_ids: %s", list(_SIDECAR_RUNNERS.keys()))
        sys.exit(1)

    service_id = sys.argv[1]
    startup_config = load_startup_config(service_id)
    service_type = str(getattr(startup_config, "service_type", "") or "").strip().lower()
    if service_type != "avatar":
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{service_id}.yaml"
        recommended_cmd = f"python inference/{service_type or '<type>'}/main.py {service_id}"
        logger.error(
            "Service '%s' is configured as service_type='%s' in '%s', but this entrypoint "
            "only supports 'avatar'. Recommended command: %s",
            service_id, service_type or "<empty>", config_path, recommended_cmd,
        )
        sys.exit(1)

    runner_module = _SIDECAR_RUNNERS.get(service_id)
    if not runner_module:
        logger.error(
            "No avatar sidecar registered for service_id='%s'. Registered: %s",
            service_id, list(_SIDECAR_RUNNERS.keys()),
        )
        sys.exit(1)

    module = importlib.import_module(runner_module)
    if not hasattr(module, "run"):
        logger.error(
            "Avatar sidecar module '%s' missing required `async def run(startup_config) -> None`",
            runner_module,
        )
        sys.exit(1)

    try:
        await module.run(startup_config)
    except Exception as exc:
        logger.error("Avatar sidecar fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

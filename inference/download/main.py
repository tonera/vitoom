"""
下载服务入口

启动示例：
python inference/download/main.py download_nas

说明：
- service_id 对应 inference/config/{service_id}.yaml（包含 civitai token 等）
- 全局配置来自 inference/config/inference.yaml（包含 models_dir、outputs_dir、storage 等）
- 连接与传输相关（api_base_url/ws_url/transport）来自 inference/config/{service_id}.yaml
"""

import asyncio
import os
import sys
from pathlib import Path

# 兼容“从源码直接运行”：
# 允许在项目根目录执行：python inference/download/main.py download_200
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config_loader import load_startup_config
from common.api_client import APIClient
from common.message_queue import MessageQueue
from common.ws_client import WebSocketClient

from download.worker import DownloadWorker
from download.providers.subprocess_utils import cleanup_stale_download_cli_procs


async def _amain(service_id: str) -> int:
    startup = load_startup_config(service_id)
    # 让 providers 能把 registry 条目打上 service_id 标签（跨平台的“只清理自己”）
    os.environ["VITOOM_SERVICE_ID"] = startup.service_id

    # 启动兜底：清理上一次崩溃/被强杀遗留的 modelscope/hf 子进程（只针对本 service_id 的记录）
    # 避免再次运行时 CLI 自身锁死/占用。
    try:
        cleanup_stale_download_cli_procs(models_dir=Path(startup.inference_config.models_dir), service_id=startup.service_id)
    except Exception:
        pass

    api = APIClient(startup.api_base_url)
    start_payload = {
        "name": startup.name,
        "type": startup.type or "download",
        "service_type": startup.service_type or "download",
        "program_name": startup.service_id,
        **(startup.config or {}),
    }
    if startup.inference_config.supervisor_url:
        start_payload["supervisor_url"] = startup.inference_config.supervisor_url

    async def _notify_start_best_effort():
        try:
            await api.notify_start(
                service_id=startup.service_id,
                host=startup.host,
                port=startup.port,
                config=start_payload,
            )
        except Exception:
            # APIClient 内部已记录日志，这里只做 best-effort
            pass

    # 2) 连接 ws 并启动 worker
    queue = MessageQueue(maxsize=2000)
    ws = WebSocketClient(
        ws_url=startup.ws_url,
        message_queue=queue,
        service_id=startup.service_id,
        # 后端重启后，下载器会自动重连 WS，但后端会把 services 状态重置为 stopped；
        # 因此 WS 重连成功时必须重新上报 start，避免后端认为“无可用下载服务”。
        on_reconnect=_notify_start_best_effort,
    )

    # 启动前先尝试一次上报（不阻塞）
    await _notify_start_best_effort()
    await ws.connect()

    # WS 连接成功后再上报一次（避免“第一次上报失败但 ws 已连上”的窗口期）
    await _notify_start_best_effort()

    # 周期性重上报：覆盖“后端重启后未触发 ws 重连但状态被重置”的场景
    async def _register_loop():
        while True:
            try:
                # 仅在 ws 处于连接状态时上报，避免无意义刷请求
                if ws.is_connected():
                    await _notify_start_best_effort()
            except Exception:
                pass
            await asyncio.sleep(15)

    reg_task = asyncio.create_task(_register_loop(), name=f"download-register:{startup.service_id}")

    worker = DownloadWorker(
        service_id=startup.service_id,
        startup=startup,
        ws_client=ws,
        message_queue=queue,
    )

    try:
        await worker.run_forever()
    finally:
        try:
            reg_task.cancel()
        except Exception:
            pass
        try:
            await ws.disconnect()
        except Exception:
            pass
        try:
            await api.notify_stop(startup.service_id)
        except Exception:
            pass
        try:
            await api.close()
        except Exception:
            pass
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference/download/main.py <service_id>")
        raise SystemExit(2)
    service_id = str(sys.argv[1]).strip()
    if not service_id:
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_amain(service_id)))


if __name__ == "__main__":
    main()


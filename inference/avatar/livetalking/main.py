"""LiveTalking sidecar 进程入口。

由 ``inference/avatar/main.py`` 通过 importlib 加载并调用 ``run(startup_config)``。

配置来源：``inference/config/livetalking.yaml``，关键字段：

* ``host`` / ``port`` —— sidecar 绑定地址（一律 ``0.0.0.0``）
* ``config.model`` —— ``musetalk`` / ``wav2lip``
* ``config.avatar_id`` —— ``resources/models/livetalking/avatars/`` 下的子目录名
* ``config.fps`` —— 渲染帧率

服务注册（与其它推理服务一致：HTTP /start upsert + WS 长连接保活 + 自动重连）：

1. 启动 aiohttp **之前** 调 ``POST /api/inference/services/{service_id}/start``
   把 ``host`` / ``port`` / ``config`` 写到 ``inference_services`` 表（status=running）。
   后端 WS 端点 ``/ws/inference/{service_id}`` 在 accept 之前会校验"表里有
   service_id 这条记录"，没记录直接 1008 拒绝；所以 WS connect 必须排在 HTTP /start 之后。
2. **WS 长连接保活**：复用 ``inference.common.ws_client.WebSocketClient``。
   - 应用层心跳：每 20s 一次 ``heartbeat`` 帧（client → server）。
   - watchdog：每 5s 检查连接状态，未连上立即指数退避重连（最大 10s）。
   - 后端启动时 ``reset_all_status_on_startup()`` 会把所有 ``inference_services``
     的 status 全部重置为 stopped；此时 sidecar 这条 WS 也会被服务端关闭，watchdog
     会在最长 5s+10s 内重连成功 → ``_on_reconnect`` 回调里再发一次 ``service_register``
     → 后端 ``sync_service_registration`` 把 status 改回 running。
   - 这是 vitoom 所有推理服务统一的可用性维护策略；sidecar 不接 task，但保活/重连
     的逻辑跟其它推理器完全一致，没必要另起一套 HTTP 心跳。
3. 收到 ``SIGTERM`` / ``SIGINT`` 时：先 ``ws_client.disconnect()`` → 再
   ``POST /api/inference/services/{service_id}/stop`` 置 ``status=stopped``
   → 最后优雅停 aiohttp。
4. **不参与 vitoom 后端的 task 派发**：``service_register`` 消息里 ``supports_task=False``，
   后端永远不会通过这条 WS 发 task / cancel 给 sidecar。业务流量走 sidecar 自己的
   ``POST /offer`` + ``WS /avatar_stream``（前端直连）以及后端推 PCM 的 HTTP/WS 接口。

不再使用 ``VITOOM_LIVETALKING_ENABLED`` 环境变量做开关：sidecar 是否在线
完全由本进程是否启动 + 与后端的 WS 长连接是否在线决定。前端通过 ``GET
/api/avatar/status`` 查询。
"""

from __future__ import annotations

import asyncio
import signal
from datetime import datetime
from typing import Any, Dict, Optional

from common.api_client import APIClient  # type: ignore[import-not-found]
from common.logger import get_logger  # type: ignore[import-not-found]
from common.message_queue import MessageQueue  # type: ignore[import-not-found]
from common.ws_client import WebSocketClient  # type: ignore[import-not-found]

from .server import create_app, serve

logger = get_logger(__name__)


def _build_service_register_message(
    *, service_type: str, model: str
) -> Dict[str, Any]:
    """生成 service_register 帧。

    与 ``inference/audio/session_runtime.py::register_service`` 字段对齐，
    后端 ``sync_service_registration`` 直接消费。

    - ``supports_task=False``：sidecar 不接 task 派发，业务流量走 ``/offer``。
    - ``capabilities=["avatar"]``：方便后续按能力筛选可用推理器。
    """
    return {
        "type": "service_register",
        "service_type": service_type,
        "supports_task": False,
        "supported_models": [model] if model else [],
        "capabilities": ["avatar"],
        "fixed_model": model,
        "fixed_family": "avatar",
        "timestamp": datetime.utcnow().isoformat(),
    }


async def _safe_notify_start(
    api_client: APIClient,
    *,
    service_id: str,
    host: str,
    port: int,
    config: Dict[str, Any],
    timeout: float = 5.0,
) -> bool:
    """对 ``api_client.notify_start`` 加超时 + swallow，统一供"首次启动"和"WS 重连"复用。

    后端不可达不应阻塞 sidecar 主流程：返回值仅供日志打点。
    """
    try:
        return bool(
            await asyncio.wait_for(
                api_client.notify_start(
                    service_id=service_id, host=host, port=port, config=config,
                ),
                timeout=timeout,
            )
        )
    except asyncio.TimeoutError:
        logger.warning(
            "notify_start timed out (>%.1fs) service_id=%s; backend may be unreachable",
            timeout, service_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify_start failed service_id=%s err=%s", service_id, exc)
    return False


async def run(startup_config: Any) -> None:
    """sidecar 入口。``startup_config`` 是 ``inference.common.config_loader.StartupConfig``。"""
    cfg = startup_config.config or {}
    host = str(getattr(startup_config, "host", "127.0.0.1"))
    try:
        port = int(getattr(startup_config, "port", 0) or 0)
    except (TypeError, ValueError):
        port = 0
    if port <= 0:
        raise ValueError(
            "inference/config/livetalking.yaml must define a positive `port`"
        )

    model = str(cfg.get("model") or "musetalk")
    avatar_id = str(cfg.get("avatar_id") or "musetalk_avatar1")
    try:
        fps = int(cfg.get("fps") or 25)
    except (TypeError, ValueError):
        fps = 25

    service_id = str(getattr(startup_config, "service_id", "") or "").strip()
    if not service_id:
        raise ValueError(
            "inference/config/livetalking.yaml must define `service_id` "
            "(used to register with backend's inference_services table)"
        )

    api_base_url = str(getattr(startup_config, "api_base_url", "") or "").strip()
    if not api_base_url:
        raise ValueError(
            "inference/config/inference.yaml must define `api_base_url` "
            "(sidecar uses it to call POST /api/inference/services/{id}/start)"
        )

    ws_url = str(getattr(startup_config, "ws_url", "") or "").strip()
    if not ws_url:
        raise ValueError(
            "inference/config/inference.yaml must define `ws_url` "
            "(sidecar holds a long-lived WS to backend for liveness + reconnect)"
        )

    api_client = APIClient(api_base_url)

    # 上报到 inference_services 表的 config 字段；sync_service_start 是 upsert，
    # 表里没记录会自动 create，已有则只更新 status / host / port / config。
    register_config: Dict[str, Any] = {
        "name": str(getattr(startup_config, "name", "LiveTalking Avatar Sidecar")),
        "type": str(getattr(startup_config, "type", "avatar")),
        "service_type": "avatar",
        "model": model,
        "avatar_id": avatar_id,
        "fps": fps,
    }
    register_message = _build_service_register_message(
        service_type="avatar", model=model,
    )

    # ---------- 1. 首次 HTTP /start：必须先于 WS connect ----------
    logger.info(
        "Registering LiveTalking sidecar with backend: api=%s service_id=%s host=%s port=%d",
        api_base_url, service_id, host, port,
    )
    initial_registered = await _safe_notify_start(
        api_client,
        service_id=service_id, host=host, port=port, config=register_config,
    )
    if not initial_registered:
        # 首次注册失败不立即 fail：aiohttp 仍然起，watchdog 会持续重试 WS 连接，
        # 期间前端 ``/api/avatar/status`` 会返回 unavailable。后端可达后整套链路自愈。
        logger.warning(
            "Initial notify_start failed; sidecar will start aiohttp anyway and "
            "WS watchdog will keep retrying registration once backend is reachable.",
        )

    # ---------- 2. 准备 WS 长连接（保活 + 自动重连） ----------
    # MessageQueue: WebSocketClient 强制要求；sidecar 不接 task，给个最小队列即可。
    # 即使后端误发 task 进来（理论上不会，因为 supports_task=False），队列满了直接
    # 丢弃，不会影响 WebRTC 主链路。
    msg_queue = MessageQueue(maxsize=8)

    async def _send_register() -> None:
        """发 service_register 到后端：把 inference_services.status 切成 running。"""
        try:
            ok = await ws_client.send_message(register_message)
            if not ok:
                logger.warning(
                    "send service_register returned False (ws not connected?); "
                    "watchdog will retry on next reconnect",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("send service_register failed: %s", exc)

    async def _on_reconnect() -> None:
        """WS 重连成功后回调（被 watchdog 调用）。

        1. 重新 HTTP upsert：兜底"管理员手动从 inference_services 表删了行"的极端情况。
           没这一步的话，后端 WS 端点会拒掉重连（"Service not found"）—— 但那种情况
           下 connect() 本身就会失败，根本走不到 _on_reconnect。这里调一次纯属保险，
           更主要的目的是把 host/port/config 这些元数据再 upsert 一遍。
        2. 重发 service_register：把后端 reset_all 后置为 stopped 的 status 改回 running。
        """
        logger.info("WS reconnected; re-registering with backend...")
        await _safe_notify_start(
            api_client,
            service_id=service_id, host=host, port=port, config=register_config,
        )
        await _send_register()

    async def _on_disconnect(reason: str) -> None:
        """WS 断开时只记日志：watchdog 会自动重连，不手工干预。"""
        logger.info(
            "WS to backend disconnected (%s); watchdog will auto-reconnect", reason,
        )

    ws_client = WebSocketClient(
        ws_url=ws_url,
        message_queue=msg_queue,
        service_id=service_id,
        on_reconnect=_on_reconnect,
        on_disconnect=_on_disconnect,
    )

    ws_initial_connected = await ws_client.connect()
    if ws_initial_connected:
        await _send_register()
    else:
        logger.warning(
            "Initial WS connect to %s failed; watchdog will keep retrying. "
            "Frontend will see avatar as unavailable until reconnect succeeds.",
            ws_url,
        )

    # ---------- 3. 信号处理：SIGTERM/SIGINT 触发优雅退出 ----------
    stop_event = asyncio.Event()

    def _request_stop(sig_name: str) -> None:
        if stop_event.is_set():
            return
        logger.info("Received %s, shutting down LiveTalking sidecar...", sig_name)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except NotImplementedError:
            # Windows event loop 不支持 add_signal_handler；Ctrl+C 走 KeyboardInterrupt
            # 路径，由 asyncio.run 抛出后进入 finally 注销。
            pass

    # ---------- 4. 启动 aiohttp 业务面 ----------
    # AV-sync mode 探测：方案 A 由环境变量 ``VITOOM_LIVETALKING_AV_SYNC_MODE``
    # 控制；这里只是把当前模式 log 出来，便于排障"为什么前端没出声"。
    # 真正的判定逻辑在 ``avatar_session.av_sync_aligned_enabled``；运行中变更
    # env 不会重读（部署级配置）。
    from .avatar_session import av_sync_aligned_enabled
    av_sync_label = "aligned (方案 A: AV co-flow via WebRTC)" if av_sync_aligned_enabled() else "decorative (D 方案：本地音频权威源)"
    logger.info(
        "Starting LiveTalking sidecar: host=%s port=%d model=%s avatar_id=%s fps=%d av_sync=%s",
        host, port, model, avatar_id, fps, av_sync_label,
    )
    app = create_app(model=model, avatar_id=avatar_id, fps=fps)
    serve_task = asyncio.create_task(
        serve(app, host=host, port=port), name="livetalking-aiohttp",
    )
    try:
        # 等 stop_event 或 serve 异常退出，二选一先到为准
        done, _pending = await asyncio.wait(
            {serve_task, asyncio.create_task(stop_event.wait(), name="livetalking-stop")},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if serve_task in done:
            # serve 异常 → 抛出来让外层日志看到
            serve_task.result()
    finally:
        # 5.1 先关 WS：避免 disconnect/notify_stop 的顺序里 WS 又重连一次刚被
        #     stop 掉的 service_id。disconnect() 会 cancel watchdog/heartbeat/receive。
        try:
            await asyncio.wait_for(ws_client.disconnect(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("ws_client.disconnect timed out (>3s); proceeding shutdown")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_client.disconnect failed: %s", exc)

        # 5.2 通知后端 stop（best-effort，不能因为后端不可达就阻塞退出）
        try:
            await asyncio.wait_for(api_client.notify_stop(service_id), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("notify_stop timed out (>2s) for service_id=%s", service_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("notify_stop failed service_id=%s err=%s", service_id, exc)
        try:
            await api_client.close()
        except Exception:
            pass

        # 5.3 取消 serve task（serve() 内部 while True sleep）
        if not serve_task.done():
            serve_task.cancel()
            try:
                await serve_task
            except (asyncio.CancelledError, Exception):
                pass


__all__ = ["run"]

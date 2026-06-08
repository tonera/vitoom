"""LiveTalking sidecar 配置访问层。

事实源由两块拼合：

1. **是否在线**：``inference_services`` 表 ``id=livetalking`` 的行
   ``status == "running"``。sidecar 启动 / 退出时会调
   ``POST /api/inference/services/livetalking/{start,stop}`` 更新这一字段，
   与 vitoom 其它推理服务（vllm / qwen_tts / voxcpm 等）走同一注册协议。

2. **对外可达地址**：``config/app.yaml`` 的 ``server.livetalking_url``。
   - 浏览器走 ``{livetalking_url}/offer`` 直连 sidecar 做 WebRTC 信令
   - 后端 ``livetalking_client`` 走 ``ws://{host}:{port}/avatar_stream`` 推 PCM
   - 为什么不从 ``inference_services.host`` 拿：sidecar 现在统一 bind
     ``0.0.0.0``（既要让浏览器连也要让后端连），表里上报的 host 没有
     "对外可达"语义；网络拓扑应由部署方在后端 ``app.yaml`` 显式声明。

热路径性能：``push_pcm`` 是亚毫秒级 SLA，绝不能每条 chunk 都查 DB / 解析
config；这里加 **TTL=2s 的进程内缓存**。sidecar 启停或 app.yaml 变更后
最多 2s 后端感知到，对前端 toggle / 反向代理流量来说完全可接受
（重启 backend 立即生效）。

设计不变量：

* 单一来源：可用性 = 表行 status==running，地址 = app.yaml；任一缺失即 disabled
* 失败安全：DB 不可达 / 行不存在 / status != running / app.yaml 字段为空 /
  url 格式错 → 返回 ``enabled=False`` 占位 + 具体 ``reason``
* 缓存可失效：单测和管理路径通过 ``reset_settings_cache()`` 清缓存
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from backend.core.config import get_config
from backend.core.logger import get_app_logger
from backend.database.models import InferenceService

logger = get_app_logger(__name__)

# 默认 service_id；目前后端只对接一个 livetalking sidecar，未来若要按 avatar
# 类型多实例，再把这个字段抬到上层调用方传入。
LIVETALKING_SERVICE_ID = "livetalking"

# 表里 service_type 字段的期望值（区分 image/video/audio/text/avatar）
_EXPECTED_CONTENT_SERVICE_TYPE = "avatar"

# 进程内缓存 TTL：2 秒。sidecar 启 / 停后至多 2s 才感知，对前端 toggle 完全够用，
# 而 push_pcm 热路径单 session 100Hz 数量级也只是每 ~200 次查一次 DB / 解析 yaml。
_CACHE_TTL_SECONDS = 2.0


class LiveTalkingConfigError(Exception):
    """注册行字段缺失或格式错误时抛出（仅记录日志，不向上传播）。"""


@dataclass(frozen=True)
class LiveTalkingSettings:
    """运行时快照。``enabled=False`` 时其它字段对调用方无意义。"""

    enabled: bool
    # sidecar 对外可达 host:port —— 前端 WebRTC 直连 + 后端 PCM WS 都用这一组
    host: str
    port: int
    # http(s) scheme 由 app.yaml 直接声明（前端可能跑 https，sidecar 也得 https）
    scheme: str
    model: str
    avatar_id: str
    fps: int
    # 当 enabled=False 时填充，前端可拿来展示具体不可用原因
    reason: str = ""

    @property
    def base_url(self) -> str:
        """sidecar 对外可达 base URL（``http://host:port``）。"""
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def offer_url(self) -> str:
        """sidecar 的 ``POST /offer`` 入口；前端 fetch 这个做 WebRTC 信令。"""
        return f"{self.base_url}/offer"

    @property
    def avatar_stream_url(self) -> str:
        """sidecar 的 ``WS /avatar_stream`` 入口（后端 PCM 推流用）。

        scheme 自动 http→ws / https→wss。
        """
        ws_scheme = "wss" if self.scheme == "https" else "ws"
        return f"{ws_scheme}://{self.host}:{self.port}/avatar_stream"


# ----------------------------------------------------------------------
# 内部：DB 行 + app.yaml → LiveTalkingSettings
# ----------------------------------------------------------------------


def _disabled(reason: str) -> LiveTalkingSettings:
    """构造 enabled=False 占位，便于调用方统一短路。"""
    return LiveTalkingSettings(
        enabled=False, host="", port=0, scheme="http",
        model="", avatar_id="", fps=0, reason=reason,
    )


def _check_registry_row() -> Optional[str]:
    """查 ``inference_services`` 表，返回错误描述；返回 None 表示行 OK 且 running。"""
    try:
        row = InferenceService.get_by_id(LIVETALKING_SERVICE_ID)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to query inference_services row for service_id=%s: %s",
            LIVETALKING_SERVICE_ID, exc,
        )
        return f"db query failed: {exc}"

    if not row:
        return (
            f"sidecar not registered (no inference_services row for "
            f"service_id={LIVETALKING_SERVICE_ID!r})"
        )

    status = str(row.get("status") or "").strip().lower()
    if status != "running":
        return f"sidecar status={status or 'unknown'}"

    content_service_type = str(row.get("service_type") or "").strip().lower()
    if content_service_type and content_service_type != _EXPECTED_CONTENT_SERVICE_TYPE:
        # 同 service_id 在表里被改成了别的 service_type（比如人为改成 audio），
        # 安全起见拒绝路由 avatar 流量到非 avatar 服务。
        return (
            f"sidecar service_type={content_service_type!r}, expected "
            f"{_EXPECTED_CONTENT_SERVICE_TYPE!r}"
        )
    return None


def _read_avatar_meta_from_row() -> dict:
    """从 inference_services 行里拿 avatar 元数据（model/avatar_id/fps），
    用于前端 status 接口展示。失败返回默认值。"""
    try:
        row = InferenceService.get_by_id(LIVETALKING_SERVICE_ID)
    except Exception:
        return {"model": "musetalk", "avatar_id": "musetalk_avatar1", "fps": 25}
    if not row:
        return {"model": "musetalk", "avatar_id": "musetalk_avatar1", "fps": 25}
    config = row.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    try:
        fps = int(config.get("fps") or 25)
    except (TypeError, ValueError):
        fps = 25
    return {
        "model": str(config.get("model") or "musetalk").strip().lower(),
        "avatar_id": str(config.get("avatar_id") or "musetalk_avatar1").strip(),
        "fps": fps,
    }


def _build_settings_from_app_yaml() -> LiveTalkingSettings:
    """主入口：拼合"在线判定"和"对外地址"，输出 LiveTalkingSettings。"""
    err = _check_registry_row()
    if err is not None:
        return _disabled(err)

    raw_url = str(get_config("server.livetalking_url", "") or "").strip()
    if not raw_url:
        return _disabled(
            "config/app.yaml `server.livetalking_url` is empty; "
            "set it to sidecar's externally reachable URL "
            "(e.g. http://192.168.31.17:8014) so the frontend can connect"
        )

    try:
        parsed = urlparse(raw_url)
    except Exception as exc:
        return _disabled(f"invalid livetalking_url={raw_url!r}: {exc}")

    scheme = (parsed.scheme or "").strip().lower()
    if scheme not in ("http", "https"):
        return _disabled(
            f"livetalking_url scheme must be http/https, got {scheme!r} "
            f"(url={raw_url!r})"
        )
    host = (parsed.hostname or "").strip()
    if not host:
        return _disabled(f"livetalking_url missing host: {raw_url!r}")
    port = parsed.port
    if port is None:
        # 没显式带 port 时按 scheme 默认（http=80, https=443）
        port = 443 if scheme == "https" else 80
    if port <= 0 or port >= 65536:
        return _disabled(f"livetalking_url port out of range: {port}")

    meta = _read_avatar_meta_from_row()

    return LiveTalkingSettings(
        enabled=True,
        host=host,
        port=int(port),
        scheme=scheme,
        model=meta["model"],
        avatar_id=meta["avatar_id"],
        fps=meta["fps"],
        reason="",
    )


# ----------------------------------------------------------------------
# 进程内 TTL 缓存（线程安全）
# ----------------------------------------------------------------------


@dataclass
class _CacheBox:
    value: Optional[LiveTalkingSettings] = None
    expires_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_cache = _CacheBox()


def get_livetalking_settings() -> LiveTalkingSettings:
    """返回当前 LiveTalking 注册状态快照。

    线程模型：FastAPI 在多个 worker 线程上调用本函数；用一把进程级 lock
    保护 TTL 检查 + 刷新。``InferenceService.get_by_id`` 内部走自己的
    DB session 是线程安全的，这里 lock 仅是避免缓存被多线程同时刷新带来的
    短暂浪费。
    """
    now = time.monotonic()
    with _cache.lock:
        if _cache.value is not None and now < _cache.expires_at:
            return _cache.value
        snapshot = _build_settings_from_app_yaml()
        _cache.value = snapshot
        _cache.expires_at = now + _CACHE_TTL_SECONDS
        return snapshot


def reset_settings_cache() -> None:
    """单测 / 管理路径用：清缓存，下次调用立刻重新读 DB + app.yaml。"""
    with _cache.lock:
        _cache.value = None
        _cache.expires_at = 0.0


__all__ = [
    "LIVETALKING_SERVICE_ID",
    "LiveTalkingConfigError",
    "LiveTalkingSettings",
    "get_livetalking_settings",
    "reset_settings_cache",
]

"""数字人副链路相关后端 HTTP API。

仅保留 ``GET /api/avatar/status``：

* 前端 panel mount / 用户点"重新检测"时调用一次
* 返回 ``{available, reason?, model?, avatar_id?, fps?, webrtc_offer_url?}``
* ``webrtc_offer_url`` 是 sidecar 对外可达的 WebRTC 信令入口（来自
  ``config/app.yaml`` 的 ``server.livetalking_url``）。前端拿到这个 URL 后
  **直接 POST 到 sidecar** 做 WebRTC 信令握手，**不再经过本后端反向代理**。
  少一跳后端 SDP 中转，延迟更低，部署形态与 LiveTalking 官方 demo 一致。

为什么删除反向代理 ``POST /api/avatar/offer``：

* sidecar 的 ``/avatar_stream`` PCM 通道本来就要求 sidecar 暴露给后端访问，
  /offer 同源暴露给前端额外成本极低（多一条 CORS allow header 即可）
* 浏览器走反向代理拿不到任何额外鉴权 / 安全收益（sidecar 本来就只跑在
  内网或私有云，外网根本路由不到），反而引入 502/504 中转故障面
* 真正需要后端做的"是否可用"判断、运维"sidecar 在哪"声明，仍由本接口
  统一处理，对前端零暴露 sidecar 拓扑细节
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.auth import get_current_user_id
from backend.core.logger import get_app_logger
from backend.services.chat.avatar.livetalking_config import (
    LIVETALKING_SERVICE_ID,
    get_livetalking_settings,
)

logger = get_app_logger(__name__)

router = APIRouter(prefix="/api/avatar", tags=["Avatar"])


@router.get("/status")
async def get_avatar_status(
    _user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """查询数字人 sidecar 当前是否可用 + 返回 WebRTC 信令入口。

    返回 ``200`` 永不报错，由 ``available`` 字段表达可用性：

    * ``available=true`` —— 一并返回 ``webrtc_offer_url`` /
      ``model`` / ``avatar_id`` / ``fps``，前端用 ``webrtc_offer_url``
      直接 POST 做 WebRTC 信令握手
    * ``available=false`` —— ``reason`` 说明原因（``sidecar_not_registered``
      / ``sidecar_stopped`` / ``livetalking_url_not_configured`` 等），
      前端按 UX 规则置灰按钮 + 提示

    安全考量：``webrtc_offer_url`` 本身是部署方在 ``app.yaml`` 显式声明的
    地址，意味着它一定是希望对客户端可达的；本接口需要鉴权（任意已登录
    用户）即可拿到 URL，没有进一步访问控制——sidecar 的 ``/offer`` 端点
    自身设计为对内网开放、不校验 token。
    """
    settings = get_livetalking_settings()
    if settings.enabled:
        return JSONResponse(
            status_code=200,
            content={
                "available": True,
                "service_id": LIVETALKING_SERVICE_ID,
                "model": settings.model,
                "avatar_id": settings.avatar_id,
                "fps": settings.fps,
                "webrtc_offer_url": settings.offer_url,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "available": False,
            "service_id": LIVETALKING_SERVICE_ID,
            "reason": settings.reason or "sidecar_unavailable",
        },
    )


__all__ = ["router"]

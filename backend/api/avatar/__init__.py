"""数字人 HTTP 反向代理路由（仅 ``/api/avatar/offer``）。

详见 ``.cursor/plans/livetalking_装饰接入_*.plan.md`` 关键决策小节：
前端不直接接触 sidecar 端口，所有 WebRTC 信令通过 vitoom 后端 8888 反向
代理转发到 ``inference/config/livetalking.yaml`` 配置的 sidecar host:port。
"""

from .routes import router

__all__ = ["router"]

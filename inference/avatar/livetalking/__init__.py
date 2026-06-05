"""LiveTalking sidecar 实现。

对外接口（仅供 vitoom 后端访问，不暴露给浏览器直连）：

* ``POST /offer`` —— 标准 WebRTC 信令，由 vitoom 后端 ``/api/avatar/offer``
  反向代理转发。前端不应直接连此端口。
* ``WS /avatar_stream`` —— vitoom 后端 ``livetalking_client`` 推 PCM 用。
  入口契约：16k mono pcm_s16le，其它一律 400 拒绝。

代码组织：

* ``musetalk/`` —— vendored 的 MuseTalk 推理内核（vae/unet/audio2feature
  + ``MuseTalkRuntime``），完全不依赖 ``inference/third_party/livetalking``
* ``protocol.py`` —— WS 消息 schema + 严格校验
* ``avatar_session.py`` —— 协议层 ↔ ``MuseTalkRuntime`` 的桥
* ``video_track.py`` —— ``MuseTalkRuntime`` ↔ aiortc ``VideoStreamTrack`` 适配
* ``server.py`` —— aiohttp app（``/offer`` + ``/avatar_stream`` + ``/health``）
* ``main.py`` —— sidecar 进程入口

注意：本 ``__init__.py`` 故意不 import 任何运行时模块（避免 ``protocol`` 单测
意外拉起 torch / aiortc 等重依赖）。需要的子模块按 ``from .protocol import ...``
显式 import。
"""

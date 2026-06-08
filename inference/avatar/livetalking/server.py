"""aiohttp server：``POST /offer`` + ``WS /avatar_stream`` + ``GET /health``。

* ``POST /offer`` —— 浏览器 WebRTC 信令入口。**前端直接 POST 这里**，
  不经过 vitoom 后端反向代理。CORS 默认 ``Access-Control-Allow-Origin: *``，
  让任意 vitoom 前端 origin 都能跨域调用。生产部署若想收紧，可改成只允许
  特定 origin（环境变量 ``VITOOM_LIVETALKING_CORS_ORIGINS``，逗号分隔）。
* ``WS /avatar_stream`` —— vitoom 后端 ``livetalking_client`` 推 PCM 用，
  绑定到同一端口，复用 sidecar bind ``0.0.0.0`` 的对外可达地址。

bind 一律 ``0.0.0.0``：sidecar 既要被同机后端连（PCM WS）也要被浏览器
直连（WebRTC 信令）。具体对外地址由部署方在 vitoom 后端 ``app.yaml``
``server.livetalking_url`` 声明。

WebRTC 信令实现取决于是否安装了 ``aiortc``（LiveTalking 上游依赖）。未装时
``/offer`` 返 503，但 ``/avatar_stream`` 仍可工作以联调 PCM 链路。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Iterable, Optional

from aiohttp import WSMsgType, web
from common.logger import get_logger  # type: ignore[import-not-found]

from .avatar_session import AvatarSession
from .protocol import (
    ProtocolError,
    parse_audio_chunk,
    parse_audio_flush,
    parse_interrupt,
    parse_open,
    validate_pcm_bytes,
)

logger = get_logger(__name__)

# 浏览器跨 origin 直连 sidecar 的预检 + 实际请求需要 CORS 头；默认放开
# 任意 origin（数字人是装饰功能，sidecar 本身只跑在内网），生产可通过
# VITOOM_LIVETALKING_CORS_ORIGINS=https://a.example.com,https://b.example.com
# 收紧。
_CORS_ALLOW_HEADERS = "Content-Type, Authorization"
_CORS_ALLOW_METHODS = "POST, GET, OPTIONS"


def _allowed_origins() -> Iterable[str]:
    raw = os.environ.get("VITOOM_LIVETALKING_CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    return [s.strip() for s in raw.split(",") if s.strip()]


def _resolve_origin_header(request: web.Request) -> str:
    """根据 request.Origin + 白名单决定要回的 ACAO 头。"""
    allowed = list(_allowed_origins())
    if "*" in allowed:
        # 注意：开放 * 时不可同时回 Allow-Credentials，浏览器会拒；
        # 我们这里本来就不依赖 cookie 鉴权，无需 credentials。
        return "*"
    origin = request.headers.get("Origin", "")
    if origin and origin in allowed:
        return origin
    # Origin 不在白名单：回 first allowed origin 让浏览器自己拒；不暴露多余信息
    return allowed[0]


@web.middleware
async def _cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        # 预检直接 200 + headers，不进 handler
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as exc:
            # HTTPException 也要带 CORS 头，否则浏览器侧拿不到 status
            resp = exc
    resp.headers["Access-Control-Allow-Origin"] = _resolve_origin_header(request)
    resp.headers["Access-Control-Allow-Headers"] = _CORS_ALLOW_HEADERS
    resp.headers["Access-Control-Allow-Methods"] = _CORS_ALLOW_METHODS
    resp.headers["Access-Control-Max-Age"] = "600"
    return resp


def create_app(*, model: str, avatar_id: str, fps: int) -> web.Application:
    """构造 aiohttp app。

    app 字段：
    * ``model`` / ``avatar_id`` / ``fps`` —— sidecar 启动时固定的元数据
    * ``sessions`` —— ``session_id → AvatarSession``（PCM WS 与 /offer 共享）
    * ``pcs`` —— 所有活跃 ``aiortc.RTCPeerConnection``，sidecar 退出时统一关闭，
      避免 ``runner.cleanup()`` 默认 ``shutdown_timeout=60s`` 卡住 Ctrl+C 退出
    """
    app = web.Application(
        client_max_size=4 * 1024 * 1024,  # 4MB body 上限（SDP 远小于此）
        middlewares=[_cors_middleware],
    )
    app["model"] = model
    app["avatar_id"] = avatar_id
    app["fps"] = fps
    app["sessions"] = {}  # session_id → AvatarSession
    app["pcs"] = set()    # type: set[Any]  active aiortc RTCPeerConnection

    app.router.add_post("/offer", _handle_offer)
    # OPTIONS 预检由 _cors_middleware 短路掉，但仍要注册路由让 router match 成功
    app.router.add_options("/offer", _handle_offer)
    app.router.add_get("/avatar_stream", _handle_avatar_stream)
    app.router.add_get("/health", _handle_health)

    # shutdown 钩子：sidecar 收 SIGINT/SIGTERM 时主动关掉所有 pc + session，
    # 避免 aiohttp 默认 60s shutdown_timeout 等浏览器/后端自己 close WebRTC/WS
    app.on_shutdown.append(_on_shutdown)
    return app


async def _on_shutdown(app: web.Application) -> None:
    """优雅关停：关所有 RTCPeerConnection，停所有 MuseTalkRuntime 线程。

    并行 close 所有 pc，避免顺序 await 串成 N*timeout；任何单点失败都
    swallow，不能挡住后续 cleanup（sidecar 进程马上要退）。
    """
    pcs = list(app.get("pcs", []))
    sessions = list(app.get("sessions", {}).values())
    logger.info(
        "sidecar shutdown: closing %d RTCPeerConnection(s) and %d AvatarSession(s)",
        len(pcs), len(sessions),
    )
    if pcs:
        await asyncio.gather(*[_safe_close_pc(pc) for pc in pcs], return_exceptions=True)
    app.get("pcs", set()).clear()
    for sess in sessions:
        try:
            sess.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AvatarSession.shutdown failed: %s", exc)
    app.get("sessions", {}).clear()


async def _safe_close_pc(pc: Any) -> None:
    try:
        await asyncio.wait_for(pc.close(), timeout=1.0)
    except asyncio.TimeoutError:
        logger.warning("pc.close() timed out (>1s); proceeding")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pc.close() failed: %s", exc)


async def _handle_health(request: web.Request) -> web.Response:
    app = request.app
    return web.json_response({
        "status": "ok",
        "model": app["model"],
        "avatar_id": app["avatar_id"],
        "active_sessions": len(app["sessions"]),
    })


async def _handle_offer(request: web.Request) -> web.Response:
    """WebRTC 信令端点，仅供 vitoom 后端反向代理访问。

    实现路径：
    1. 取出 SDP body
    2. 创建/复用 ``AvatarSession`` 实例
    3. 用 aiortc 建 ``RTCPeerConnection``，addtrack(avatar_session.get_video_track())
    4. setRemoteDescription / createAnswer / setLocalDescription
    5. 返回 SDP answer

    aiortc 依赖未装时返回 503 + reason，方便 dev 环境只跑协议联调。
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.json_response(
            {"reason": "invalid_json", "detail": str(exc)}, status=400,
        )

    sdp = body.get("sdp") if isinstance(body, dict) else None
    sdp_type = body.get("type") if isinstance(body, dict) else None
    if not sdp or not sdp_type:
        return web.json_response(
            {"reason": "missing_sdp_fields", "detail": "expect {sdp, type}"},
            status=400,
        )

    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore[import-not-found]
    except ImportError:
        return web.json_response(
            {
                "reason": "aiortc_not_installed",
                "detail": (
                    "aiortc is required for WebRTC signaling. "
                    "Install with: pip install aiortc"
                ),
            },
            status=503,
        )

    app = request.app
    session_id = str(body.get("session_id") or "default").strip()
    avatar_session = app["sessions"].get(session_id)
    if avatar_session is None:
        try:
            avatar_session = AvatarSession(
                model=app["model"], avatar_id=app["avatar_id"], fps=app["fps"],
            )
        except Exception as exc:  # noqa: BLE001
            # 模型 / avatar 资源缺失或加载失败：直接 500（不要 503，因为重试也没用）
            logger.error(
                "AvatarSession init failed session_id=%s: %s",
                session_id, exc, exc_info=True,
            )
            return web.json_response(
                {"reason": "avatar_init_failed", "detail": str(exc)}, status=500,
            )
        app["sessions"][session_id] = avatar_session

    pc = RTCPeerConnection()
    app["pcs"].add(pc)  # 跟踪：sidecar shutdown 时统一关，否则 aiohttp cleanup 会等 60s
    video_track = avatar_session.get_video_track()
    if video_track is None:
        # MuseTalkRuntime 已就绪，但 aiortc / av 缺失 → 协议层兜底，便于 dev 联调
        app["pcs"].discard(pc)
        return web.json_response(
            {
                "reason": "avatar_video_track_unavailable",
                "detail": (
                    "Failed to construct WebRTC video track (aiortc / av missing). "
                    "Install with: pip install aiortc av"
                ),
            },
            status=503,
        )
    pc.addTrack(video_track)
    # 方案 A 音视频对齐：env ``VITOOM_LIVETALKING_AV_SYNC_MODE=aligned`` 时
    # AvatarSession.get_audio_track() 返回 MuseTalkAudioTrack；同 PeerConnection
    # 上加 audio track 后浏览器用 RTP 时间戳天然 lip-sync。decorative 模式下
    # 返回 None → 不 addTrack，前端依然按 D 方案走本地 useAudioPlayback。
    audio_track = avatar_session.get_audio_track()
    if audio_track is not None:
        pc.addTrack(audio_track)
        logger.info(
            "RTCPeerConnection session_id=%s av-aligned mode: audio track added",
            session_id,
        )

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(
            "RTCPeerConnection session_id=%s state=%s", session_id, pc.connectionState,
        )
        if pc.connectionState in ("failed", "closed"):
            app["pcs"].discard(pc)
            await pc.close()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
    })


class _StreamState:
    """单条 ``WS /avatar_stream`` 连接的本地状态。

    把 session_id / active_request_id / avatar_session / pending_chunk 这些
    短生命周期变量都集中到一个对象里，避免 handler 内闭包变量赋值带来的
    "nonlocal 还是中转 dict" 之类的纠结。
    """

    __slots__ = ("session_id", "active_request_id", "avatar_session", "pending_chunk")

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.active_request_id: Optional[str] = None
        self.avatar_session: Optional[AvatarSession] = None
        # JSON meta + binary frame 配对：收到 audio_chunk meta 后存这里，
        # 下一个 binary frame 到达时 pop 出来配对，再清零。
        self.pending_chunk: Optional[Dict[str, Any]] = None


async def _handle_avatar_stream(request: web.Request) -> web.WebSocketResponse:
    """vitoom 后端 livetalking_client 推 PCM 用。

    协议见 ``protocol.py``。任何 ``ProtocolError`` 都立刻关闭连接（``code=1008``）
    + 一条 ``error`` 消息回执，绝不做兜底转换。
    """
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    app = request.app
    state = _StreamState()

    async def _close_with_error(code: str, message: str, status: int = 1008) -> None:
        try:
            await ws.send_json({"type": "error", "code": code, "message": message})
        except Exception:
            pass
        await ws.close(code=status, message=message.encode("utf-8")[:120])

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError as exc:
                    await _close_with_error("invalid_json", f"json decode failed: {exc}")
                    return ws
                if not isinstance(payload, dict):
                    await _close_with_error(
                        "invalid_json", "top-level json must be object",
                    )
                    return ws
                mtype = str(payload.get("type") or "").strip()
                try:
                    _dispatch_text(mtype=mtype, payload=payload, app=app, state=state)
                except ProtocolError as exc:
                    await _close_with_error(exc.code, exc.message)
                    return ws
            elif msg.type == WSMsgType.BINARY:
                if state.pending_chunk is None:
                    await _close_with_error(
                        "unexpected_binary",
                        "binary frame received without preceding audio_chunk meta",
                    )
                    return ws
                try:
                    validate_pcm_bytes(msg.data)
                except ProtocolError as exc:
                    await _close_with_error(exc.code, exc.message)
                    return ws
                if state.avatar_session is not None:
                    state.avatar_session.put_pcm16(
                        request_id=str(state.pending_chunk.get("request_id") or ""),
                        pcm_bytes=bytes(msg.data),
                    )
                state.pending_chunk = None
            elif msg.type == WSMsgType.ERROR:
                logger.warning(
                    "WS error session_id=%s exc=%s", state.session_id, ws.exception(),
                )
                break
            elif msg.type == WSMsgType.CLOSE:
                break
    finally:
        if state.session_id:
            logger.info("avatar_stream WS closed session_id=%s", state.session_id)
    return ws


def _dispatch_text(
    *,
    mtype: str,
    payload: Dict[str, Any],
    app: web.Application,
    state: "_StreamState",
) -> None:
    if mtype == "open":
        meta = parse_open(payload)
        avatar_session = app["sessions"].get(meta.session_id)
        if avatar_session is None:
            avatar_session = AvatarSession(
                model=app["model"], avatar_id=app["avatar_id"], fps=app["fps"],
            )
            app["sessions"][meta.session_id] = avatar_session
        state.session_id = meta.session_id
        state.active_request_id = meta.request_id
        state.avatar_session = avatar_session
        logger.info(
            "avatar_stream open session_id=%s request_id=%s model=%s avatar=%s",
            meta.session_id, meta.request_id, app["model"], app["avatar_id"],
        )
    elif mtype == "audio_chunk":
        if state.session_id is None:
            raise ProtocolError(
                "open_required", "audio_chunk before open is not allowed",
            )
        meta = parse_audio_chunk(payload)
        state.active_request_id = meta.request_id
        state.pending_chunk = {"request_id": meta.request_id, "seq": meta.seq}
    elif mtype == "audio_flush":
        if state.session_id is None:
            raise ProtocolError(
                "open_required", "audio_flush before open is not allowed",
            )
        meta = parse_audio_flush(payload)
        if state.avatar_session is not None:
            state.avatar_session.flush_request(meta.request_id)
    elif mtype == "interrupt":
        meta = parse_interrupt(payload)
        target = app["sessions"].get(meta.session_id) or state.avatar_session
        if target is not None:
            try:
                target.interrupt()
            except Exception as exc:
                # plan 阶段 5：interrupt 异常 swallow，不影响主链路
                logger.warning(
                    "avatar_session.interrupt failed session_id=%s err=%s",
                    meta.session_id, exc,
                )
    elif mtype == "close":
        if state.session_id:
            sess = app["sessions"].pop(state.session_id, None)
            if sess is not None:
                try:
                    sess.shutdown()
                except Exception as exc:  # noqa: BLE001
                    # close 是优雅关闭，吞掉异常防止 ws handler 崩溃
                    logger.warning(
                        "AvatarSession.shutdown failed session_id=%s err=%s",
                        state.session_id, exc,
                    )
        state.session_id = None
        state.active_request_id = None
        state.avatar_session = None
        state.pending_chunk = None
    else:
        raise ProtocolError("unknown_type", f"unsupported message type: {mtype}")


async def serve(app: web.Application, *, host: str, port: int) -> None:
    """启动 aiohttp server，跑到信号关闭。"""
    # shutdown_timeout 默认 60s（用于等所有 in-flight WS / 请求自然退出）。
    # 数字人 sidecar 的 PCM WS 是 long-running 的，浏览器/后端不会主动断 →
    # 默认 60s 会让 Ctrl+C 后等一分钟才退。我们已经在 _on_shutdown 里把所有
    # pc + session 主动关掉，再给 1.5s 给 WS handler 走完 finally 即可。
    runner = web.AppRunner(app, shutdown_timeout=1.5)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    logger.info("LiveTalking sidecar listening on http://%s:%d", host, port)
    try:
        # 长跑：直到外部信号
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


__all__ = ["create_app", "serve"]

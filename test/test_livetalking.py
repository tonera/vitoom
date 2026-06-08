"""LiveTalking 装饰性数字人接入的核心单测。

覆盖 ``.cursor/plans/livetalking_装饰接入_*.plan.md`` 阶段 6 的必填用例：

后端 (backend/services/chat/avatar)
  1. sidecar 未注册（``inference_services`` 表无行）→ 所有公共 API 同步 no-op
  2. ``push_pcm`` 非阻塞 + 入队（不 await sidecar 写）
  3. resample 24k mono → 16k mono 长度准确
  4. resample stereo → mono 平均（反相完全相消）
  5. bounded queue + drop oldest（满了不阻塞 producer）
  6. sidecar 不可达 → consumer 标 failed + 不重连
  7. interrupt 200ms 超时 swallow，不 raise

Sidecar (inference/avatar/livetalking/protocol.py)
  8. ``parse_open`` 严格拒绝错误 sample_rate / format / channels
  9. ``validate_pcm_bytes`` 拒绝空 / 奇数字节

status 接口 (backend/api/avatar/routes.py)
 10. ``GET /status`` running + app.yaml livetalking_url 配置 → available=true
       并返回 webrtc_offer_url（前端直连 sidecar 用）
 11. ``GET /status`` sidecar 未注册 → available=false，reason 含 not registered
 12. ``GET /status`` sidecar status=stopped → available=false，reason 含 stopped
 13. ``GET /status`` app.yaml livetalking_url 为空 → available=false（即使
     sidecar 注册成功；防止后端"知道有服务但不知道前端怎么连"）

注册行为 + 配置融合 (backend/services/chat/avatar/livetalking_config.py)
 14. status=running + livetalking_url 配置 → enabled=True，offer/avatar_stream URL
     都从 app.yaml 解析（**不再用注册表里的 host=0.0.0.0**）
 15. status=stopped → enabled=False
 16. service_type 不匹配 → 拒绝路由
 17. ``reset_settings_cache`` 后立刻重读 DB + app.yaml
 18. http→ws / https→wss 自动转换

MuseTalk vendored 内核 (inference/avatar/livetalking/musetalk)
 19. ``paths`` 反推 vitoom 项目根 + 子路径拼接正确
 20. ``FeatureBuffer.push_pcm`` 严格按 320-sample 切片入队，末尾不足丢弃
 21. ``FeatureBuffer.step`` warm_up 后 1 次 step 出 1 个 batch_size 个 feat 块
 22. ``FeatureBuffer`` 输入空 → step 自动 silence 补齐 + audio_frame 标 type=1
 23. ``FeatureBuffer.flush`` 清空输入和未消费 feature 但保留窗口

CORS / sidecar
 24. sidecar OPTIONS /offer 预检返回 CORS 头

Sidecar 服务注册 + WS 长连接保活 (inference/avatar/livetalking/main.py)
 25. ``run()`` 启动顺序：先 HTTP /start upsert → 再 WS connect → 连上后立刻
     发一帧 ``service_register{supports_task=False}``
 26. WS watchdog 触发 ``_on_reconnect`` → 再次 HTTP upsert + 重发 service_register
     （后端重启 reset_all 后 sidecar 自愈把 status 切回 running 的核心路径）
 27. ``run()`` 退出（serve_task 自然结束）→ 先 ws.disconnect → 再 notify_stop
     → 再 close apiclient（确保后端先把 status 改 stopped 再丢失 WS 通知通道）
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ----------------------------------------------------------------------
# 公共：mock InferenceService.get_by_id 注入到 livetalking_config 模块
# ----------------------------------------------------------------------

# livetalking sidecar 在 inference_services 表里的 service_id（与生产配置一致）
_SERVICE_ID = "livetalking"


def _make_running_row(
    *,
    host: str = "127.0.0.1",
    port: int = 1,
    model: str = "musetalk",
    avatar_id: str = "musetalk_avatar1",
    fps: int = 25,
    service_type: str = "avatar",
) -> Dict[str, Any]:
    """构造一行 ``inference_services``  表 dict（运行中的 sidecar）。"""
    return {
        "id": _SERVICE_ID,
        "service_type": service_type,  # 内容类型：image/video/audio/text/avatar
        "type": "avatar",  # 引擎类型
        "name": "LiveTalking Avatar Sidecar",
        "status": "running",
        "host": host,
        "port": port,
        "config": {
            "model": model,
            "avatar_id": avatar_id,
            "fps": fps,
            "service_type": service_type,
        },
    }


def _patch_inference_service_row(monkeypatch, row: Optional[Dict[str, Any]]) -> None:
    """统一 patch ``backend.services.chat.avatar.livetalking_config.InferenceService``。

    传入 ``None`` 模拟"sidecar 未注册"；传入 dict 模拟"已注册"。
    每次都会 reset_settings_cache，确保下一次 ``get_livetalking_settings``
    立刻命中新 mock。
    """
    import backend.services.chat.avatar.livetalking_config as cfg_mod

    class _FakeInferenceService:
        @staticmethod
        def get_by_id(service_id: str) -> Optional[Dict[str, Any]]:
            if service_id != _SERVICE_ID:
                return None
            return None if row is None else dict(row)

    monkeypatch.setattr(cfg_mod, "InferenceService", _FakeInferenceService)
    cfg_mod.reset_settings_cache()


def _patch_app_yaml_livetalking_url(monkeypatch, url: str) -> None:
    """patch ``backend.core.config.get_config`` 的 ``server.livetalking_url`` 字段。

    传入空串模拟"app.yaml 未配置"，传入完整 URL 模拟"已声明 sidecar 对外地址"。
    其它配置 key 仍走真正的 get_config（避免 import-time 副作用）。
    """
    import backend.services.chat.avatar.livetalking_config as cfg_mod

    real_get_config = cfg_mod.get_config

    def fake_get_config(key: str, default: Any = None) -> Any:
        if key == "server.livetalking_url":
            return url
        return real_get_config(key, default)

    monkeypatch.setattr(cfg_mod, "get_config", fake_get_config)
    cfg_mod.reset_settings_cache()


def _setup_running_with_url(
    monkeypatch,
    *,
    url: str = "http://192.168.31.17:8014",
    row_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """便捷 helper：一行代码 mock"sidecar 已注册 + app.yaml 已配地址"。"""
    row = _make_running_row()
    if row_overrides:
        row.update(row_overrides)
    _patch_inference_service_row(monkeypatch, row=row)
    _patch_app_yaml_livetalking_url(monkeypatch, url=url)


# ======================================================================
# 1. sidecar 未注册 → 公共 API 同步 no-op
# ======================================================================
def test_unavailable_all_apis_are_noop(monkeypatch):
    _patch_inference_service_row(monkeypatch, row=None)
    _patch_app_yaml_livetalking_url(monkeypatch, url="")

    from backend.services.chat.avatar.livetalking_client import (
        get_livetalking_client,
        reset_client_for_test,
    )

    reset_client_for_test()
    client = get_livetalking_client()

    # 全部入口都不应抛异常 / 改任何状态
    client.set_enabled("sess", True)
    client.push_pcm("sess", "rid", b"\x00" * 320, sample_rate=24000, channels=1)
    client.flush("sess", "rid")

    async def _async_part():
        await client.interrupt("sess")
        await client.close("sess")
        await client.shutdown()

    asyncio.run(_async_part())
    # sidecar 未注册时甚至不会创建 session state
    assert client._sessions == {}


# ======================================================================
# 2. push_pcm 非阻塞：调用立即返回，sidecar 写在 consumer task 后台
# ======================================================================
def test_push_pcm_is_non_blocking(monkeypatch):
    # sidecar 已注册 + app.yaml 配的端口必然连不上 → consumer task 卡在 ws connect
    _setup_running_with_url(monkeypatch, url="http://127.0.0.1:1")

    from backend.services.chat.avatar.livetalking_client import (
        get_livetalking_client,
        reset_client_for_test,
    )

    reset_client_for_test()

    async def _async_part():
        client = get_livetalking_client()
        client.set_enabled("sess", True)
        chunk = b"\x00" * 640  # 20ms 16k mono
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            client.push_pcm("sess", "rid", chunk, sample_rate=16000, channels=1)
            latencies.append(time.perf_counter_ns() - t0)
        # 单次 push 必须远小于 5ms（resample + put_nowait 都是 in-memory）
        assert max(latencies) < 5_000_000, (
            f"push_pcm slowest call took {max(latencies)/1e6:.2f}ms; "
            f"non-blocking SLA violated"
        )
        await asyncio.sleep(0.05)
        await client.shutdown()

    asyncio.run(_async_part())


# ======================================================================
# 3. resample 24k mono → 16k mono 长度准确
# ======================================================================
def test_resample_24k_mono_to_16k():
    from backend.services.chat.avatar._resampler import Resampler

    r = Resampler()
    sine = (np.sin(2 * np.pi * 440 * np.arange(2400) / 24000) * 8000).astype(np.int16)
    out_bytes = r.process(sine.tobytes(), source_sr=24000, source_channels=1)
    out_samples = len(out_bytes) // 2
    # 24000 → 16000：2400 in → 1600 out (允许 ±32 误差兼容线性插值 fallback)
    assert 1568 <= out_samples <= 1632, f"got {out_samples} samples"


# ======================================================================
# 4. resample stereo 反相 → mono 完全相消
# ======================================================================
def test_resample_stereo_anti_phase_cancels():
    from backend.services.chat.avatar._resampler import Resampler

    r = Resampler()
    sine_l = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000) * 0.3
    stereo = np.column_stack([sine_l, -sine_l]).flatten()
    int16 = (stereo * 32767).astype(np.int16).tobytes()
    out = r.process(int16, source_sr=16000, source_channels=2)
    arr = np.frombuffer(out, dtype=np.int16)
    assert arr.size > 0
    assert abs(arr).max() <= 2, f"stereo cancellation leaked, max={abs(arr).max()}"


# ======================================================================
# 5. bounded queue 满了 → drop oldest，producer 永不 block
# ======================================================================
def test_bounded_queue_drops_oldest_under_backpressure(monkeypatch):
    _setup_running_with_url(monkeypatch, url="http://127.0.0.1:1")

    from backend.services.chat.avatar.livetalking_client import (
        _QUEUE_MAXSIZE,
        get_livetalking_client,
        reset_client_for_test,
    )

    reset_client_for_test()

    async def _async_part():
        client = get_livetalking_client()
        client.set_enabled("sess", True)
        state = client._sessions["sess"]

        # consumer task lazy 起：先 push 一次让它启动，再 cancel 模拟"完全写不动"
        chunk = b"\x00" * 320
        client.push_pcm("sess", "rid", chunk, sample_rate=16000, channels=1)
        await asyncio.sleep(0)
        if state.consumer_task is not None:
            state.consumer_task.cancel()
            try:
                await state.consumer_task
            except (asyncio.CancelledError, Exception):
                pass

        for _ in range(_QUEUE_MAXSIZE * 3):
            client.push_pcm("sess", "rid", chunk, sample_rate=16000, channels=1)
        assert state.queue.qsize() <= _QUEUE_MAXSIZE
        await client.shutdown()

    asyncio.run(_async_part())


# ======================================================================
# 6. sidecar 不可达 → consumer 标 failed + 不重连
# ======================================================================
def test_sidecar_unreachable_marks_failed_and_no_reconnect(monkeypatch):
    _setup_running_with_url(monkeypatch, url="http://127.0.0.1:1")

    from backend.services.chat.avatar.livetalking_client import (
        get_livetalking_client,
        reset_client_for_test,
    )

    reset_client_for_test()

    async def _async_part():
        client = get_livetalking_client()
        client.set_enabled("sess", True)
        state = client._sessions["sess"]
        client.push_pcm("sess", "rid", b"\x00" * 320, sample_rate=16000, channels=1)
        # 给 consumer 时间走完 ws connect 失败 → mark failed
        for _ in range(20):
            await asyncio.sleep(0.05)
            if state.failed:
                break
        assert state.failed, "expected sidecar unreachable to mark session failed"

        before_qsize = state.queue.qsize()
        client.push_pcm("sess", "rid", b"\x00" * 320, sample_rate=16000, channels=1)
        # failed=True 时 push 应直接 return，不入队
        assert state.queue.qsize() == before_qsize
        await client.shutdown()

    asyncio.run(_async_part())


# ======================================================================
# 7. interrupt 200ms 超时 swallow（不抛异常给上层）
# ======================================================================
def test_interrupt_swallows_timeout_and_does_not_raise(monkeypatch):
    _setup_running_with_url(monkeypatch, url="http://127.0.0.1:1")

    from backend.services.chat.avatar.livetalking_client import (
        _SessionState,
        get_livetalking_client,
        reset_client_for_test,
    )

    reset_client_for_test()

    class _StuckWs:
        async def send(self, _msg):
            await asyncio.sleep(10.0)

        async def close(self):
            pass

    async def _async_part():
        client = get_livetalking_client()
        client.set_enabled("sess", True)
        state: _SessionState = client._sessions["sess"]
        state.ws = _StuckWs()

        t0 = time.perf_counter()
        await client.interrupt("sess")
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"interrupt took {elapsed:.3f}s, expected <0.5s"
        await client.shutdown()

    asyncio.run(_async_part())


# ======================================================================
# 8. sidecar parse_open 严格拒绝错误 sample_rate / format / channels
# ======================================================================
def test_sidecar_parse_open_rejects_wrong_format():
    from inference.avatar.livetalking.protocol import (
        ProtocolError,
        parse_open,
    )

    base = {
        "session_id": "s",
        "request_id": "r",
        "sample_rate": 16000,
        "format": "pcm_s16le",
        "channels": 1,
    }
    parse_open(base)

    for k, bad in [
        ("sample_rate", 24000),
        ("format", "wav"),
        ("channels", 2),
        ("session_id", ""),
        ("request_id", ""),
    ]:
        d = dict(base)
        d[k] = bad
        with pytest.raises(ProtocolError):
            parse_open(d)


# ======================================================================
# 9. sidecar validate_pcm_bytes 拒绝空 / 奇数字节
# ======================================================================
def test_sidecar_validate_pcm_bytes():
    from inference.avatar.livetalking.protocol import (
        ProtocolError,
        validate_pcm_bytes,
    )

    validate_pcm_bytes(b"\x00\x00")

    with pytest.raises(ProtocolError):
        validate_pcm_bytes(b"")
    with pytest.raises(ProtocolError):
        validate_pcm_bytes(b"\x00\x00\x00")
    with pytest.raises(ProtocolError):
        validate_pcm_bytes("not bytes")  # type: ignore[arg-type]


# ======================================================================
# Helpers：FastAPI TestClient + 鉴权 override
# ======================================================================
def _make_test_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.avatar.routes import router
    from backend.auth import get_current_user_id

    app = FastAPI()
    app.dependency_overrides[get_current_user_id] = lambda: "test-user"
    app.include_router(router)
    return TestClient(app)


# ======================================================================
# 10. GET /status running + livetalking_url 配置 → available=true + webrtc_offer_url
# ======================================================================
def test_status_endpoint_returns_webrtc_offer_url_when_available(monkeypatch):
    _setup_running_with_url(
        monkeypatch,
        url="http://192.168.31.17:8014",
        row_overrides={
            "config": {
                "model": "musetalk",
                "avatar_id": "musetalk_avatar1",
                "fps": 25,
                "service_type": "avatar",
            },
        },
    )
    with _make_test_client() as cli:
        resp = cli.get("/api/avatar/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("available") is True
        # 关键：前端拿到的是 sidecar 对外可达 URL，不是注册表里的 0.0.0.0
        assert body.get("webrtc_offer_url") == "http://192.168.31.17:8014/offer"
        assert body.get("model") == "musetalk"
        assert body.get("fps") == 25
        assert body.get("service_id") == _SERVICE_ID
        # 不直接暴露 host/port（前端只用 webrtc_offer_url）
        assert "host" not in body and "port" not in body


# ======================================================================
# 11. GET /status sidecar 未注册 → available=false
# ======================================================================
def test_status_endpoint_returns_unavailable_when_not_registered(monkeypatch):
    _patch_inference_service_row(monkeypatch, row=None)
    _patch_app_yaml_livetalking_url(monkeypatch, url="http://192.168.31.17:8014")
    with _make_test_client() as cli:
        resp = cli.get("/api/avatar/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("available") is False
        assert "not registered" in body.get("reason", "").lower()
        assert "webrtc_offer_url" not in body


# ======================================================================
# 12. GET /status sidecar status=stopped → available=false
# ======================================================================
def test_status_endpoint_returns_unavailable_when_stopped(monkeypatch):
    row = _make_running_row()
    row["status"] = "stopped"
    _patch_inference_service_row(monkeypatch, row=row)
    _patch_app_yaml_livetalking_url(monkeypatch, url="http://192.168.31.17:8014")
    with _make_test_client() as cli:
        resp = cli.get("/api/avatar/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("available") is False
        assert "stopped" in body.get("reason", "").lower()


# ======================================================================
# 13. GET /status app.yaml livetalking_url 为空 → available=false
#     即使 sidecar 注册成功；后端"知道有服务但不知道前端怎么连"等价于不可用
# ======================================================================
def test_status_endpoint_returns_unavailable_when_app_yaml_missing_url(monkeypatch):
    _patch_inference_service_row(monkeypatch, row=_make_running_row())
    _patch_app_yaml_livetalking_url(monkeypatch, url="")
    with _make_test_client() as cli:
        resp = cli.get("/api/avatar/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("available") is False
        assert "livetalking_url" in body.get("reason", "").lower()


# ======================================================================
# 14. settings.enabled=True：URL 都从 app.yaml 解析（不是注册表 host=0.0.0.0）
# ======================================================================
def test_settings_url_comes_from_app_yaml_not_registry_row(monkeypatch):
    """关键回归：sidecar bind 0.0.0.0 后，注册表里 host=0.0.0.0 是绑定地址，
    后端必须用 app.yaml 里的对外地址。不能误用 0.0.0.0 去连。"""
    _setup_running_with_url(
        monkeypatch,
        url="http://192.168.31.17:8014",
        row_overrides={"host": "0.0.0.0"},  # sidecar 实际上报的就是 0.0.0.0
    )
    from backend.services.chat.avatar.livetalking_config import (
        get_livetalking_settings,
    )

    s = get_livetalking_settings()
    assert s.enabled is True
    assert s.host == "192.168.31.17"  # 不是 0.0.0.0！
    assert s.port == 8014
    assert s.offer_url == "http://192.168.31.17:8014/offer"
    assert s.avatar_stream_url == "ws://192.168.31.17:8014/avatar_stream"
    assert s.reason == ""


# ======================================================================
# 15. status=stopped → enabled=False，reason 非空
# ======================================================================
def test_settings_disabled_when_row_stopped(monkeypatch):
    row = _make_running_row()
    row["status"] = "stopped"
    _patch_inference_service_row(monkeypatch, row=row)
    _patch_app_yaml_livetalking_url(monkeypatch, url="http://192.168.31.17:8014")

    from backend.services.chat.avatar.livetalking_config import (
        get_livetalking_settings,
    )

    s = get_livetalking_settings()
    assert s.enabled is False
    assert s.reason


# ======================================================================
# 16. service_type 不匹配 → 拒绝路由（防误派）
# ======================================================================
def test_settings_rejects_mismatched_service_type(monkeypatch):
    """有人手动把 inference_services 里 service_id=livetalking 的 service_type 改成
    了别的（例如 audio），后端绝不能把 avatar 流量打过去。"""
    row = _make_running_row()
    row["service_type"] = "audio"
    row["config"]["service_type"] = "audio"
    _patch_inference_service_row(monkeypatch, row=row)
    _patch_app_yaml_livetalking_url(monkeypatch, url="http://192.168.31.17:8014")

    from backend.services.chat.avatar.livetalking_config import (
        get_livetalking_settings,
    )

    s = get_livetalking_settings()
    assert s.enabled is False
    assert "avatar" in s.reason.lower()


# ======================================================================
# 17. reset_settings_cache 后立刻重读 DB + app.yaml
# ======================================================================
def test_settings_cache_reset_picks_up_new_state(monkeypatch):
    _patch_inference_service_row(monkeypatch, row=None)
    _patch_app_yaml_livetalking_url(monkeypatch, url="http://192.168.31.17:8014")
    from backend.services.chat.avatar.livetalking_config import (
        get_livetalking_settings,
        reset_settings_cache,
    )

    s1 = get_livetalking_settings()
    assert s1.enabled is False

    # 把 sidecar 状态切到 running（_setup_running_with_url 内部 reset 缓存）
    _setup_running_with_url(monkeypatch, url="http://192.168.31.17:8014")
    s2 = get_livetalking_settings()
    assert s2.enabled is True

    reset_settings_cache()
    s3 = get_livetalking_settings()
    assert s3.enabled is True


# ======================================================================
# 18. https livetalking_url → wss avatar_stream_url
# ======================================================================
def test_settings_https_url_yields_wss_for_avatar_stream(monkeypatch):
    """生产部署若整套走 https，sidecar 也会被 https 反向代理（如 nginx），
    PCM WS 必须自动用 wss。"""
    _setup_running_with_url(monkeypatch, url="https://avatar.example.com/")
    from backend.services.chat.avatar.livetalking_config import (
        get_livetalking_settings,
    )

    s = get_livetalking_settings()
    assert s.enabled is True
    assert s.scheme == "https"
    assert s.host == "avatar.example.com"
    assert s.port == 443  # 没显式带 port，按 https default
    assert s.offer_url == "https://avatar.example.com:443/offer"
    assert s.avatar_stream_url == "wss://avatar.example.com:443/avatar_stream"


# ======================================================================
# 19. MuseTalk vendored paths：反推 vitoom 项目根 + 子路径拼接
# ======================================================================
def test_musetalk_paths_resolve_under_vitoom_root():
    from inference.avatar.livetalking.musetalk import paths

    root = Path(__file__).resolve().parent.parent
    expected_resources = root / "resources" / "models" / "livetalking"
    assert Path(paths.LIVETALKING_RESOURCES_ROOT) == expected_resources
    assert Path(paths.MUSETALK_V15_UNET) == expected_resources / "musetalkV15" / "unet.pth"
    assert Path(paths.WHISPER_DIR) == expected_resources / "whisper"
    assert Path(paths.SD_VAE_DIR) == expected_resources / "sd-vae"
    # vitoom 侧拼写为 face-parse-bisenet（含 t），不是上游 face-parse-bisent
    assert Path(paths.FACE_PARSE_DIR).name == "face-parse-bisenet"
    assert Path(paths.avatar_dir("musetalk_avatar1")) == expected_resources / "avatars" / "musetalk_avatar1"


# ======================================================================
# 24. sidecar OPTIONS /offer 预检返回 CORS 头
# ======================================================================
def test_sidecar_cors_preflight_returns_headers():
    """浏览器跨 origin POST sidecar /offer 之前会先发 OPTIONS 预检；
    middleware 必须在 204 响应里带 Access-Control-Allow-* 头。"""
    from aiohttp.test_utils import TestClient, TestServer

    from inference.avatar.livetalking.server import create_app

    async def _run():
        app = create_app(model="musetalk", avatar_id="musetalk_avatar1", fps=25)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/offer",
                headers={
                    "Origin": "http://example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
            assert "Content-Type" in resp.headers.get("Access-Control-Allow-Headers", "")

    asyncio.run(_run())


def test_sidecar_on_shutdown_closes_pcs_and_sessions():
    """aiohttp ``runner.cleanup()`` 默认 ``shutdown_timeout=60s`` 等所有 in-flight WS
    自然关闭。装饰性数字人的 PCM WS 是 long-running 的 → Ctrl+C 后会卡 ~1min。

    修复策略：``create_app`` 注册 ``on_shutdown`` 钩子，主动并行 close 所有 pc 并
    shutdown 所有 ``AvatarSession``。本用例验证：
      1. ``app['pcs']`` 中所有 pc 都被调用 ``close()`` 且集合被清空
      2. ``app['sessions']`` 中所有 session 都被调用 ``shutdown()`` 且字典被清空
      3. 即使某个 pc.close() 抛异常，剩下的 pc / session 仍要被正确处理
    """
    import asyncio as _asyncio

    from inference.avatar.livetalking.server import _on_shutdown, create_app

    closed_pcs: list = []
    shutdown_sessions: list = []

    class _FakePC:
        def __init__(self, fail: bool = False):
            self._fail = fail

        async def close(self):
            closed_pcs.append(self)
            if self._fail:
                raise RuntimeError("boom")

    class _FakeSession:
        def __init__(self, fail: bool = False):
            self._fail = fail

        def shutdown(self):
            shutdown_sessions.append(self)
            if self._fail:
                raise RuntimeError("boom")

    async def _run():
        app = create_app(model="musetalk", avatar_id="musetalk_avatar1", fps=25)
        # 注入 fake pc / session（含一个会抛异常的，验证 swallow 不影响其它）
        pc_ok = _FakePC()
        pc_bad = _FakePC(fail=True)
        sess_ok = _FakeSession()
        sess_bad = _FakeSession(fail=True)
        app["pcs"].update({pc_ok, pc_bad})
        app["sessions"]["sid-1"] = sess_ok
        app["sessions"]["sid-2"] = sess_bad

        await _on_shutdown(app)

        assert {pc_ok, pc_bad} == set(closed_pcs), \
            f"expected both pcs closed, got {closed_pcs}"
        assert {sess_ok, sess_bad} == set(shutdown_sessions), \
            f"expected both sessions shutdown, got {shutdown_sessions}"
        assert len(app["pcs"]) == 0, "pcs set should be cleared after shutdown"
        assert len(app["sessions"]) == 0, "sessions dict should be cleared after shutdown"

    _asyncio.run(_run())


def test_sidecar_serve_uses_short_shutdown_timeout():
    """``serve()`` 创建 ``AppRunner`` 时必须传 short ``shutdown_timeout``，
    否则 aiohttp 默认 60s 会让 Ctrl+C 卡一分钟才退。

    我们 patch ``AppRunner`` 捕获构造参数，断言 ``shutdown_timeout < 5``。
    """
    import asyncio as _asyncio

    from inference.avatar.livetalking import server as server_mod

    captured: dict = {}

    class _FakeRunner:
        def __init__(self, app, **kwargs):
            captured["kwargs"] = dict(kwargs)
            captured["app"] = app

        async def setup(self):
            return None

        async def cleanup(self):
            return None

    class _FakeSite:
        def __init__(self, runner, host, port):
            captured["host"] = host
            captured["port"] = port

        async def start(self):
            return None

    async def _run():
        from aiohttp import web as _web
        app = _web.Application()
        # 启 serve 后立即 cancel，让它跑到 finally 退出（不真正 sleep 3600s）
        task = _asyncio.create_task(server_mod.serve(app, host="127.0.0.1", port=0))
        await _asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except _asyncio.CancelledError:
            pass

    import unittest.mock as _mock
    with _mock.patch.object(server_mod.web, "AppRunner", _FakeRunner), \
         _mock.patch.object(server_mod.web, "TCPSite", _FakeSite):
        _asyncio.run(_run())

    assert "shutdown_timeout" in captured["kwargs"], \
        f"AppRunner must be constructed with shutdown_timeout, got {captured['kwargs']}"
    assert captured["kwargs"]["shutdown_timeout"] < 5.0, \
        f"shutdown_timeout should be small (<5s), got {captured['kwargs']['shutdown_timeout']}"


# ======================================================================
# Helpers for FeatureBuffer：mock Audio2Feature
# ======================================================================
class _StubAudio2Feature:
    """fake Audio2Feature：audio2feat 返回 (T, K, 384) 的伪 hidden states。"""

    def __init__(self):
        self.calls = 0

    def audio2feat(self, wav_data):
        self.calls += 1
        T = max(50, wav_data.size // 320)
        return np.random.randn(T, 5, 384).astype(np.float32)


def test_feature_buffer_push_pcm_chunks_at_320_samples():
    from inference.avatar.livetalking.musetalk.feature_buffer import FeatureBuffer

    buf = FeatureBuffer(_StubAudio2Feature())
    out = buf.push_pcm(np.zeros(1000, dtype=np.float32))
    assert out == 3
    assert buf._input_queue.qsize() == 3

    assert buf.push_pcm(np.zeros(0, dtype=np.float32)) == 0


def test_feature_buffer_step_warm_then_emit_batch():
    """warm_up + 一次 step 应产出 1 个 batch_size 长度的 feature 块列表。"""
    from inference.avatar.livetalking.musetalk.feature_buffer import (
        FRAME_TYPE_SPEAK,
        FeatureBuffer,
    )

    audio = _StubAudio2Feature()
    buf = FeatureBuffer(audio, batch_size=8, stride_left=10, stride_right=10)
    buf.warm_up()
    assert buf._output_queue.qsize() == buf.stride_right
    assert len(buf._frames) == buf.stride_left + buf.stride_right

    pcm = np.ones(buf.batch_size * 2 * buf.chunk_samples, dtype=np.float32) * 0.1
    n = buf.push_pcm(pcm)
    assert n == buf.batch_size * 2

    emitted = buf.step()
    assert emitted is True
    assert audio.calls == 1
    feats = buf.get_feat_batch(timeout=0.1)
    assert len(feats) == buf.batch_size
    assert feats[0].shape[1] == 384

    queued = [buf.get_audio_frame() for _ in range(buf.stride_right + buf.batch_size * 2)]
    head_silence = queued[:buf.stride_right]
    tail_speak = queued[buf.stride_right:]
    assert all(af.type != FRAME_TYPE_SPEAK for af in head_silence)
    assert all(af.type == FRAME_TYPE_SPEAK for af in tail_speak)


def test_feature_buffer_step_silence_when_input_empty():
    """输入队列空时 step 应自动用 silence chunk 补齐，audio_frame 标 type=1。"""
    from inference.avatar.livetalking.musetalk.feature_buffer import (
        FRAME_TYPE_SILENCE,
        FeatureBuffer,
    )

    buf = FeatureBuffer(_StubAudio2Feature(), batch_size=4, stride_left=4, stride_right=4)
    buf.warm_up()

    emitted = buf.step()
    assert emitted is True
    new_frames = [buf.get_audio_frame() for _ in range(buf.batch_size * 2)]
    assert all(af.type == FRAME_TYPE_SILENCE for af in new_frames)
    assert all(np.all(af.data == 0) for af in new_frames)


def test_feature_buffer_flush_drains_queues_but_keeps_window():
    """flush 应清空 input 和 feat 队列，但保留滑动窗口（_frames）。"""
    from inference.avatar.livetalking.musetalk.feature_buffer import FeatureBuffer

    buf = FeatureBuffer(_StubAudio2Feature(), batch_size=4, stride_left=4, stride_right=4)
    buf.warm_up()
    buf.push_pcm(np.zeros(buf.chunk_samples * buf.batch_size * 2, dtype=np.float32))
    buf.step()

    assert buf._input_queue.qsize() == 0
    assert buf._feat_queue.qsize() == 1
    snapshot_window = len(buf._frames)
    assert snapshot_window > 0

    buf.push_pcm(np.zeros(buf.chunk_samples * 5, dtype=np.float32))
    assert buf._input_queue.qsize() == 5

    buf.flush()
    assert buf._input_queue.qsize() == 0
    assert buf._feat_queue.qsize() == 0
    assert len(buf._frames) == snapshot_window


# ======================================================================
# 25-27. sidecar run() 启动 / 重连 / 退出三联（覆盖 WS 长连接保活）
# ======================================================================

def _ensure_inference_root_on_path() -> None:
    """让 ``inference/avatar/livetalking/main.py`` 的 ``from common.*`` import 生效。

    生产里 ``inference/avatar/main.py`` 启动时会把 ``inference/`` 加到 sys.path；
    单测直接 import sidecar main 时要复刻这一步。
    """
    inference_root = Path(__file__).parent.parent / "inference"
    p = str(inference_root)
    if p not in sys.path:
        sys.path.insert(0, p)


class _SidecarStartupConfig:
    """模拟 ``inference.common.config_loader.StartupConfig`` 的最小字段集。"""

    service_id = "livetalking"
    host = "0.0.0.0"
    port = 18014
    api_base_url = "http://test-backend:8888"
    ws_url = "ws://test-backend:8888"
    name = "Test LT Sidecar"
    type = "avatar"
    config = {"model": "musetalk", "avatar_id": "musetalk_avatar1", "fps": 25}


def _setup_sidecar_main_with_fakes(monkeypatch):
    """把 sidecar main 模块里的 APIClient / WebSocketClient / create_app / serve
    全部替换成可观测 fake，返回 (sidecar_main, captures)。

    captures 字典里收集所有调用，供 assert 用。
    """
    _ensure_inference_root_on_path()
    import inference.avatar.livetalking.main as sidecar_main

    captures: Dict[str, Any] = {
        "notify_start_calls": [],
        "notify_stop_calls": [],
        "close_calls": 0,
        "send_message_calls": [],
        "disconnect_calls": 0,
        "on_reconnect_holder": {},
        "serve_started": asyncio.Event(),
        "serve_done": asyncio.Event(),
    }

    class FakeAPIClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        async def notify_start(self, *, service_id, host, port, config):
            captures["notify_start_calls"].append(
                {"service_id": service_id, "host": host, "port": port, "config": dict(config)}
            )
            return True

        async def notify_stop(self, service_id):
            captures["notify_stop_calls"].append(service_id)
            return True

        async def close(self):
            captures["close_calls"] += 1

    class FakeWebSocketClient:
        def __init__(self, *, ws_url, message_queue, service_id, on_reconnect=None, on_disconnect=None):
            captures["on_reconnect_holder"]["ws_url"] = ws_url
            captures["on_reconnect_holder"]["service_id"] = service_id
            captures["on_reconnect_holder"]["on_reconnect"] = on_reconnect
            captures["on_reconnect_holder"]["on_disconnect"] = on_disconnect

        async def connect(self):
            return True

        async def send_message(self, msg):
            captures["send_message_calls"].append(dict(msg))
            return True

        async def disconnect(self):
            captures["disconnect_calls"] += 1

    def fake_create_app(**_kw):
        return object()

    async def fake_serve(_app, *, host, port):  # noqa: ARG001
        captures["serve_started"].set()
        # 等测试主控显式 set serve_done 后正常返回，让 main.run 走 finally 优雅路径
        await captures["serve_done"].wait()

    monkeypatch.setattr(sidecar_main, "APIClient", FakeAPIClient)
    monkeypatch.setattr(sidecar_main, "WebSocketClient", FakeWebSocketClient)
    monkeypatch.setattr(sidecar_main, "create_app", fake_create_app)
    monkeypatch.setattr(sidecar_main, "serve", fake_serve)

    return sidecar_main, captures


def test_sidecar_run_first_register_via_http_then_ws_send(monkeypatch):
    """启动顺序：HTTP /start upsert → WS connect → service_register。"""
    sidecar_main, captures = _setup_sidecar_main_with_fakes(monkeypatch)

    async def _scenario():
        run_task = asyncio.create_task(sidecar_main.run(_SidecarStartupConfig()))
        try:
            # 等 fake_serve 进入 wait（说明前置注册流程已跑完）
            await asyncio.wait_for(captures["serve_started"].wait(), timeout=2.0)

            # 1. 首次 HTTP notify_start 一次（且字段正确）
            assert len(captures["notify_start_calls"]) == 1
            first = captures["notify_start_calls"][0]
            assert first["service_id"] == "livetalking"
            assert first["host"] == "0.0.0.0"
            assert first["port"] == 18014
            assert first["config"]["model"] == "musetalk"
            assert first["config"]["service_type"] == "avatar"

            # 2. WS connect 后立刻发了一帧 service_register
            assert len(captures["send_message_calls"]) == 1
            register = captures["send_message_calls"][0]
            assert register["type"] == "service_register"
            assert register["service_type"] == "avatar"
            assert register["supports_task"] is False
            assert register["supported_models"] == ["musetalk"]
            assert "avatar" in register["capabilities"]
            assert register["fixed_model"] == "musetalk"
        finally:
            captures["serve_done"].set()
            await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_scenario())


def test_sidecar_ws_reconnect_re_registers(monkeypatch):
    """watchdog 触发 _on_reconnect → 再次 HTTP upsert + 再次发 service_register。

    这是后端重启 reset_all 后数字人自愈的核心路径，必须有单测兜住。
    """
    sidecar_main, captures = _setup_sidecar_main_with_fakes(monkeypatch)

    async def _scenario():
        run_task = asyncio.create_task(sidecar_main.run(_SidecarStartupConfig()))
        try:
            await asyncio.wait_for(captures["serve_started"].wait(), timeout=2.0)

            # 基线：首次启动后 1 次 notify_start + 1 次 service_register
            assert len(captures["notify_start_calls"]) == 1
            assert len(captures["send_message_calls"]) == 1

            # 模拟 WS 断了又连上：watchdog 在 reconnect 成功后会调 _on_reconnect
            on_reconnect = captures["on_reconnect_holder"]["on_reconnect"]
            assert on_reconnect is not None, "WebSocketClient 必须被传入 on_reconnect 回调"
            await on_reconnect()

            # 重连后应该：再 notify_start 一次（HTTP upsert 兜底）+ 再发一帧 service_register
            assert len(captures["notify_start_calls"]) == 2
            assert len(captures["send_message_calls"]) == 2
            assert captures["send_message_calls"][1]["type"] == "service_register"
            # 字段与首次保持一致（避免重连时改 supports_task/capabilities 这种致命退化）
            assert captures["send_message_calls"][1]["supports_task"] is False
            assert captures["send_message_calls"][1]["supported_models"] == ["musetalk"]

            # 再连一次 → 应该再多 1 + 1 次（说明回调本身可重入，不是一次性闭包）
            await on_reconnect()
            assert len(captures["notify_start_calls"]) == 3
            assert len(captures["send_message_calls"]) == 3
        finally:
            captures["serve_done"].set()
            await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_scenario())


def test_sidecar_run_shutdown_disconnects_ws_then_notify_stop(monkeypatch):
    """退出顺序：先 ws.disconnect → 再 notify_stop → 再 api.close。

    顺序不能反 —— 必须先关闭 WS，否则 disconnect 期间 watchdog 可能看见 ws 不通
    立刻去重连，重新把 status 切回 running，导致 notify_stop 之后 status 又被
    上一波 reconnect 覆盖。
    """
    sidecar_main, captures = _setup_sidecar_main_with_fakes(monkeypatch)
    call_order: list[str] = []

    # 在 capture fakes 上叠加一层"记录调用顺序"的 wrapper
    real_disconnect_count_holder = {"n": 0}

    class OrderedFakeWS:
        def __init__(self, *, ws_url, message_queue, service_id, on_reconnect=None, on_disconnect=None):
            captures["on_reconnect_holder"]["on_reconnect"] = on_reconnect

        async def connect(self):
            return True

        async def send_message(self, msg):
            captures["send_message_calls"].append(dict(msg))
            return True

        async def disconnect(self):
            real_disconnect_count_holder["n"] += 1
            call_order.append("ws_disconnect")

    class OrderedFakeAPI:
        def __init__(self, base_url):
            pass

        async def notify_start(self, *, service_id, host, port, config):
            captures["notify_start_calls"].append({"service_id": service_id})
            return True

        async def notify_stop(self, service_id):
            call_order.append("notify_stop")
            captures["notify_stop_calls"].append(service_id)
            return True

        async def close(self):
            call_order.append("api_close")
            captures["close_calls"] += 1

    monkeypatch.setattr(sidecar_main, "APIClient", OrderedFakeAPI)
    monkeypatch.setattr(sidecar_main, "WebSocketClient", OrderedFakeWS)

    async def _scenario():
        run_task = asyncio.create_task(sidecar_main.run(_SidecarStartupConfig()))
        await asyncio.wait_for(captures["serve_started"].wait(), timeout=2.0)
        # 让 fake_serve 自然返回 → main.run 走 finally
        captures["serve_done"].set()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_scenario())

    assert real_disconnect_count_holder["n"] == 1, "退出时必须 disconnect WS 一次"
    assert captures["notify_stop_calls"] == ["livetalking"]
    assert captures["close_calls"] == 1
    # 关键断言：顺序是 ws_disconnect → notify_stop → api_close
    assert call_order == ["ws_disconnect", "notify_stop", "api_close"]

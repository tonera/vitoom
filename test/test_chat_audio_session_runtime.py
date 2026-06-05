from pathlib import Path
import sys
import asyncio
import base64
import array

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.chat.session import InputMode, SessionRuntime, SessionState
import backend.services.chat.session.audio_turn as audio_turn_module
import backend.services.chat.session.interrupt as interrupt_module
import backend.services.chat.session.runtime as session_runtime_module
import backend.services.chat.session.transcript as transcript_module
from backend.services.chat.inference_session import RoleSpec
from backend.services.chat.master_runtime import MasterAgentRuntime
import backend.services.chat.master_runtime as master_runtime_module
from backend.services.chat.streaming_vad import StreamingVadDetector
from backend.services.chat.voice_reply import strip_markdown_for_tts


def _pcm_audio_chunk(b64: str, seq: int = 0) -> dict:
    raw = base64.b64decode(b64)
    return {
        "type": "audio_chunk",
        "binary_bytes": raw,
        "payload": {"seq": seq, "bytes_len": len(raw), "mime": "audio/pcm;rate=16000"},
    }


def _pcm_bytes_chunk(raw: bytes, seq: int = 0) -> dict:
    return {
        "type": "audio_chunk",
        "binary_bytes": raw,
        "payload": {"seq": seq, "bytes_len": len(raw), "mime": "audio/pcm;rate=16000"},
    }


def _tone_frame(value: int = 12000, samples: int = 320) -> bytes:
    return array.array("h", [value] * samples).tobytes()




def _patch_generate_uuid(monkeypatch, value_factory):
    monkeypatch.setattr(session_runtime_module, "generate_uuid", value_factory)
    monkeypatch.setattr(audio_turn_module, "generate_uuid", value_factory)


def _patch_append_message(monkeypatch, recorder):
    monkeypatch.setattr(session_runtime_module, "append_message", recorder)
    monkeypatch.setattr(audio_turn_module, "append_message", recorder)
    monkeypatch.setattr(transcript_module, "append_message", recorder)
    monkeypatch.setattr(interrupt_module, "append_message", recorder)


def _patch_agent_run(monkeypatch, fake_cls):
    monkeypatch.setattr(session_runtime_module, "AgentRun", fake_cls)
    monkeypatch.setattr(interrupt_module, "AgentRun", fake_cls)


def _patch_task(monkeypatch, fake_cls):
    monkeypatch.setattr(interrupt_module, "Task", fake_cls)


def test_strip_markdown_for_tts_removes_emoji_but_keeps_text():
    text = "我是人工智能助手，没有具体的年龄。😊\n请选择 1️⃣ 或 2️⃣，谢谢！"

    assert strip_markdown_for_tts(text, state={}) == "我是人工智能助手，没有具体的年龄。\n请选择 或 ，谢谢！"


class _FakeSileroEngine:
    """测试用 Silero 替身：detect(audio) 返回固定 probability，跳过 ONNX 推理 + 512-sample 强约束。

    需要支持自定义 probability 序列（每次 detect 取下一个），用于 streak / 多帧场景。
    """

    def __init__(self, *, probs=None, default_prob: float = 0.99):
        self._probs = list(probs or [])
        self._default = float(default_prob)
        self.calls = 0
        self.reset_calls = 0

    def detect(self, _audio):
        self.calls += 1
        if self._probs:
            return float(self._probs.pop(0))
        return self._default

    def reset(self):
        self.reset_calls += 1


def test_streaming_vad_energy_floor_filters_low_rms_speech():
    """Silero 时代的"能量地板"语义：Silero 高分判 speech，但 RMS 低于地板就软化为
    silence。挡掉电视/广播/远处对话这类"频谱像人声但近场能量低"的环境噪声。

    - 无地板：Silero speech → speech，loud / quiet 一视同仁；
    - 有地板（loud RMS ≥ floor）：speech → speech；
    - 有地板（quiet RMS < floor）：speech → silence（被过滤）。
    """
    async def main():
        loud = _tone_frame(value=12000)  # 约 0.36 RMS，远高于 floor 0.02
        quiet = _tone_frame(value=200)  # 约 0.006 RMS，远低于 floor

        # 无地板：Silero speech 直接 fire
        vad_no_floor = StreamingVadDetector(
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_speech_floor=0.0,
            model_factory=lambda _name: _FakeSileroEngine(default_prob=0.99),
        )
        events = await vad_no_floor.push(quiet)
        assert [e.type for e in events] == ["speech_start"], (
            "无地板时 Silero 高分应直接 fire，能量不进决策"
        )

        # 有地板 + loud：RMS 过地板，speech 放行
        vad_loud = StreamingVadDetector(
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_speech_floor=0.02,
            model_factory=lambda _name: _FakeSileroEngine(default_prob=0.99),
        )
        events = await vad_loud.push(loud)
        assert [e.type for e in events] == ["speech_start"]

        # 有地板 + quiet：Silero 仍判 speech，但能量地板过滤为 silence，不应 fire
        vad_quiet = StreamingVadDetector(
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_speech_floor=0.02,
            model_factory=lambda _name: _FakeSileroEngine(default_prob=0.99),
        )
        assert await vad_quiet.push(quiet) == [], (
            "Silero 高分但 RMS 低于地板必须被软化为 silence（典型电视/远处对话）"
        )
        snap = vad_quiet.diag_snapshot()
        assert snap["last_state"] == "silence"
        assert snap["last_prob"] >= 0.5
        assert snap["last_rms"] < 0.02

    asyncio.run(main())


def test_streaming_vad_energy_floor_requires_consecutive_streak():
    """固化"连续 N 帧都过地板才认 speech"行为：

    - 单帧高能量 ≠ speech：echo 漏过前端 echoGate / 远处突发响声常常是脉冲式高能量；
    - streak=3 时前两帧应被压成 silence，第 3 帧才 fire；
    - 中间插入一个 quiet 帧应重置 streak 计数。
    """
    async def main():
        loud = _tone_frame(value=12000)
        quiet = _tone_frame(value=200)

        vad = StreamingVadDetector(
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_speech_floor=0.02,
            energy_floor_min_streak=3,
            model_factory=lambda _name: _FakeSileroEngine(default_prob=0.99),
        )
        assert await vad.push(loud) == [], "streak=1，未达阈值，不应 fire"
        assert vad.diag_snapshot()["energy_floor_streak"] == 1
        assert await vad.push(loud) == [], "streak=2，未达阈值，不应 fire"
        assert vad.diag_snapshot()["energy_floor_streak"] == 2
        events = await vad.push(loud)
        assert [e.type for e in events] == ["speech_start"], "streak=3 才应 fire"
        assert vad.diag_snapshot()["energy_floor_streak"] == 3

        # streak 中途被 quiet 中断：只要不连续就不 fire
        vad2 = StreamingVadDetector(
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_speech_floor=0.02,
            energy_floor_min_streak=3,
            model_factory=lambda _name: _FakeSileroEngine(default_prob=0.99),
        )
        assert await vad2.push(loud) == []
        assert await vad2.push(loud) == []
        assert vad2.diag_snapshot()["energy_floor_streak"] == 2
        # quiet 帧（RMS < floor）应把 streak 清零
        assert await vad2.push(quiet) == []
        assert vad2.diag_snapshot()["energy_floor_streak"] == 0
        assert await vad2.push(loud) == [], "重新计数，streak=1"
        assert await vad2.push(loud) == [], "streak=2"
        events = await vad2.push(loud)
        assert [e.type for e in events] == ["speech_start"], "重新连续 3 帧后 fire"

    asyncio.run(main())


def test_streaming_vad_silero_silence_does_not_fire_even_on_loud_rms():
    """Silero 判 silence 时不应被高 RMS 强行翻转——这是 Silero 时代相对 fsmn 时代
    的关键改动：Silero 对人声 vs 非人声鲁棒，相信它的 silence 判决，避免脉冲式
    噪声（敲门/键盘）靠能量翻盘进 speech。
    """
    async def main():
        loud = _tone_frame(value=12000)  # RMS 高，但 fake silero 始终判 silence
        vad = StreamingVadDetector(
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_speech_floor=0.02,
            model_factory=lambda _name: _FakeSileroEngine(default_prob=0.05),
        )
        for _ in range(5):
            assert await vad.push(loud) == [], "Silero silence 一票否决，loud 也不应 fire"
        snap = vad.diag_snapshot()
        assert snap["last_state"] == "silence"
        assert snap["last_prob"] < 0.5

    asyncio.run(main())


def test_streaming_vad_energy_fallback_detects_start_and_end():
    """Silero 不可用（use_neural_vad=False）时回退到纯能量阈值，保留 speech_start /
    speech_end 契约——本地开发 / 未下载模型场景下不至于卡死。"""
    async def main():
        vad = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )
        speech = _tone_frame()
        silence = bytes(len(speech))

        events = await vad.push(speech)
        assert [event.type for event in events] == ["speech_start"]
        assert events[0].frames

        assert await vad.push(silence) == []
        events = await vad.push(silence)
        assert [event.type for event in events] == ["speech_end"]
        assert vad.speaking is False

    asyncio.run(main())


def test_streaming_vad_uses_models_storage_path_for_local_onnx(monkeypatch, tmp_path):
    """模型路径解析：``models.storage_path`` + safe(model_name) → onnx 文件。
    与原 fsmn 时代的目录结构对齐，运维统一管理。
    """
    monkeypatch.setattr(
        "backend.services.chat.streaming_vad.get_config",
        lambda key, default=None: str(tmp_path) if key == "models.storage_path" else default,
    )
    local_dir = tmp_path / "silero-vad"
    local_dir.mkdir()
    (local_dir / "silero_vad.onnx").write_bytes(b"x" * 16)

    captured = {}

    class _FakeIO:
        def __init__(self, name):
            self.name = name

    class _FakeOnnxSession:
        def get_inputs(self):
            return [_FakeIO("input"), _FakeIO("state"), _FakeIO("sr")]

        def get_outputs(self):
            return [_FakeIO("output"), _FakeIO("stateN")]

    import backend.services.chat.streaming_vad as svad

    monkeypatch.setattr(svad, "_SILERO_SESSION_CACHE", {}, raising=False)

    def fake_load(onnx_path):
        captured["onnx_path"] = onnx_path
        return _FakeOnnxSession()

    monkeypatch.setattr(svad, "_load_or_get_silero_session_sync", fake_load)

    vad = StreamingVadDetector()
    engine = asyncio.run(vad._ensure_engine_async())
    assert engine is not None
    assert captured["onnx_path"] == str(local_dir / "silero_vad.onnx")


def test_streaming_vad_warmup_runs_one_silence_inference_and_resets_state():
    """warmup 必须真的跑一次 detect 触发 onnxruntime 内部 cache，且重置 hidden
    state，避免 warm 用过的 state 污染下次真实推理。
    """
    fake_engine = _FakeSileroEngine(default_prob=0.05)

    async def main():
        vad = StreamingVadDetector(
            model_factory=lambda _name: fake_engine,
            chunk_ms=20,
            min_speech_ms=20,
        )
        ok = await vad.warmup()
        assert ok is True
        assert fake_engine.calls == 1, "warmup 必须真的跑一次 silence detect"
        assert fake_engine.reset_calls == 1, "warmup 后 hidden state 必须重置"

    asyncio.run(main())


def test_streaming_vad_warmup_returns_false_when_neural_vad_disabled():
    """use_neural_vad=False 时 warmup 必须直接返回 False，不分配任何资源。"""
    async def main():
        vad = StreamingVadDetector(
            model_factory=lambda _name: object(),
            use_neural_vad=False,
        )
        assert await vad.warmup() is False

    asyncio.run(main())


class _FakeInferenceSession:
    """仅实现 SessionRuntime 侧用到的 InferenceSessionManager 子集。"""

    def __init__(self):
        self.opened_roles: list[RoleSpec] = []
        self.asr_chunks: list[tuple[bytes, object]] = []
        self.asr_commits: list[object] = []
        self.close_all_called = False
        self.tts_requests: list[dict] = []
        self.tts_cancelled: list[str | None] = []
        self._active_tts_request_id: str | None = None
        self._tts_waiters: dict[str, asyncio.Future[None]] = {}

    @property
    def active_tts_request_id(self):
        return self._active_tts_request_id

    async def open(self, spec: RoleSpec) -> bool:
        self.opened_roles.append(spec)
        return True

    async def close_all(self) -> None:
        self.close_all_called = True
        for fut in list(self._tts_waiters.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("closed"))
        self._tts_waiters.clear()

    async def asr_chunk(self, pcm_bytes: bytes, seq) -> bool:
        self.asr_chunks.append((pcm_bytes, seq))
        return True

    async def asr_commit(self, seq=None) -> bool:
        self.asr_commits.append(seq)
        return True

    async def tts_request(self, *, text, voice_cfg, request_id=None, metadata=None):
        rid = request_id or f"rid-{len(self.tts_requests) + 1}"
        self.tts_requests.append(
            {"text": text, "voice_cfg": dict(voice_cfg or {}), "request_id": rid, "metadata": dict(metadata or {})}
        )
        self._active_tts_request_id = rid
        self._tts_waiters[rid] = asyncio.get_event_loop().create_future()
        return rid

    async def tts_cancel(self, request_id=None) -> bool:
        self.tts_cancelled.append(request_id)
        if self._active_tts_request_id and (request_id is None or request_id == self._active_tts_request_id):
            self._active_tts_request_id = None
        return True

    async def await_tts_finish(self, request_id: str, *, timeout: float = 240.0) -> None:
        fut = self._tts_waiters.get(request_id)
        if fut is None:
            return
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._tts_waiters.pop(request_id, None)
            if self._active_tts_request_id == request_id:
                self._active_tts_request_id = None

    def resolve_tts_waiter(self, request_id: str, *, error=None) -> None:
        fut = self._tts_waiters.get(request_id)
        if fut is None or fut.done():
            return
        if error:
            fut.set_exception(RuntimeError(error))
        else:
            fut.set_result(None)


def test_audio_runtime_open_eagerly_opens_inference_session():
    emitted = []
    fake = _FakeInferenceSession()

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        raise AssertionError("master_run should not be called in this test")

    async def main():
        runtime = SessionRuntime(
            session_id="session-audio-open",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={
                "audio_input": {
                    "load_name": "Qwen3-ASR-1.7B",
                    "family": "Qwen-asr",
                },
            },
            inference_session=fake,
        )

        await runtime.open()

        assert runtime.state == SessionState.READY
        assert any(e[0]["type"] == "session_ready" for e in emitted)
        assert any(spec.role == "asr" and spec.load_name == "Qwen3-ASR-1.7B" for spec in fake.opened_roles)

    asyncio.run(main())


def test_audio_session_open_fires_background_vad_warmup(monkeypatch):
    """audio session 必须在 session_ready 发出后立刻 fire-and-forget VAD warmup。

    动机：funasr 模型 loading ~1s + 首次 generate jit warmup ~3s 合计 ~4s。
    若不预热，用户首次开口讲完话后要等 ~4s 才有 ASR 反馈，体感是"卡死了"，
    用户会刷新页面 → backend 收到 `User WebSocket disconnected`。
    open() 期间 fire 后台 task 把这 4s 提前消化掉，用户开口时模型已热好。

    关键不变量：
      - session_ready 必须先 emit（不能被 warmup 阻塞——前端 readyTimer 5s）
      - warmup 必须真的被调到（否则等于没改）
      - warmup 失败必须不影响 session 主流程（fire-and-forget 不传播异常）
    """
    warmup_calls = []

    async def fake_warmup(self):
        warmup_calls.append(self)
        return True

    monkeypatch.setattr(
        "backend.services.chat.streaming_vad.StreamingVadDetector.warmup",
        fake_warmup,
        raising=False,
    )

    emitted = []
    fake = _FakeInferenceSession()

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        return None

    async def main():
        runtime = SessionRuntime(
            session_id="session-warmup",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "fake-asr", "family": "Qwen-asr"},
                "audio_output": {"load_name": "fake-tts", "family": "VoxCpm"},
            },
            inference_session=fake,
        )
        await runtime.open()
        # session_ready 必须已经 emit（warmup 不阻塞 open）
        assert any(e[0]["type"] == "session_ready" for e in emitted)
        # 让 fire-and-forget warmup task 跑起来
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert warmup_calls, "audio session open 必须 fire VAD warmup task"

    asyncio.run(main())


def test_text_session_open_does_not_warm_up_vad(monkeypatch):
    """纯文本 session 不需要 VAD，不应浪费 CPU 跑 warmup。

    InputMode.TEXT 走 _is_audio_input_mode() == False 分支，不该触发后台
    warmup task。
    """
    warmup_calls = []

    async def fake_warmup(self):
        warmup_calls.append(self)
        return True

    monkeypatch.setattr(
        "backend.services.chat.streaming_vad.StreamingVadDetector.warmup",
        fake_warmup,
        raising=False,
    )

    emitted = []
    fake = _FakeInferenceSession()

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        return None

    async def main():
        runtime = SessionRuntime(
            session_id="session-text",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.TEXT,
            output_mode="text_stream",
            metadata={},
            inference_session=fake,
        )
        await runtime.open()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not warmup_calls, "text session 不应该触发 VAD warmup"

    asyncio.run(main())


def test_audio_chunk_silence_does_not_start_turn(monkeypatch):
    emitted = []
    fake = _FakeInferenceSession()

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        raise AssertionError("silence should not start a run")

    async def main():
        runtime = SessionRuntime(
            session_id="session-silence-gate",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={"audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"}},
            inference_session=fake,
        )
        runtime._audio._turn_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )
        await runtime.open()
        await runtime.on_client_message(_pcm_bytes_chunk(bytes(len(_tone_frame()))))

        assert runtime.state == SessionState.READY
        assert runtime.current_turn is None
        assert fake.asr_chunks == []

    asyncio.run(main())


def test_audio_commit_runs_master_after_final_transcript(monkeypatch):
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-audio-1", "run-audio-1"])

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    _patch_agent_run(monkeypatch, _FakeAgentRun)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(runtime, turn):
        assert turn.input_mode == InputMode.AUDIO_STREAM
        assert turn.user_text == "你好，帮我总结一下"
        await runtime.enter_streaming_output()
        await runtime.emit_message_started()
        await runtime.emit_message_delta("收到")
        await runtime.complete_run(assistant_text="收到")

    async def main():
        runtime = SessionRuntime(
            session_id="session-audio-p2",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={
                "audio_input": {
                    "load_name": "Qwen3-ASR-1.7B",
                    "family": "Qwen-asr",
                },
            },
            inference_session=fake,
        )
        runtime._audio._turn_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=2500,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )

        await runtime.open()
        speech = _tone_frame()
        await runtime.on_client_message(_pcm_bytes_chunk(speech))
        await runtime.on_client_message({"type": "session_commit", "payload": {}})
        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-audio-p2:asr",
                "role": "asr",
                "text": "你好，帮我总结一下",
            }
        )
        for _ in range(40):
            if any(e[0]["type"] == "message_completed" for e in emitted):
                break
            await asyncio.sleep(0.02)

        assert fake.asr_chunks and fake.asr_chunks[0][0] == speech
        assert fake.asr_commits, "session_commit 应当触发 asr_commit"
        assert any(
            e[0]["type"] == "transcript_delta"
            and e[0]["payload"]["text"] == "你好，帮我总结一下"
            and e[0]["payload"]["is_final"] is True
            for e in emitted
        )
        assert any(e[0]["type"] == "message_started" for e in emitted)
        assert any(e[0]["type"] == "message_delta" and e[0]["payload"]["delta"] == "收到" for e in emitted)
        assert any(e[0]["type"] == "message_completed" and e[0]["payload"]["content"] == "收到" for e in emitted)
        assert runtime.state == SessionState.READY
        assert runtime.current_turn is None
        assert [item["role"] for item in persisted] == ["user", "assistant"]
        assert persisted[0]["content"] == "你好，帮我总结一下"
        assert persisted[1]["content"] == "收到"

    asyncio.run(main())


def test_audio_turn_auto_commits_after_vad_endpoint(monkeypatch):
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-auto", "run-auto"])

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    _patch_agent_run(monkeypatch, _FakeAgentRun)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(runtime, turn):
        assert turn.user_text == "这是一整句话"
        await runtime.emit_message_started()
        await runtime.emit_message_delta("已收到")
        await runtime.complete_run(assistant_text="已收到")

    async def main():
        runtime = SessionRuntime(
            session_id="session-auto-endpoint",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={"audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"}},
            inference_session=fake,
        )
        runtime._audio._turn_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )
        await runtime.open()
        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))

        assert fake.asr_commits, "VAD endpoint should auto commit the audio turn"
        assert runtime.state == SessionState.STREAMING_OUTPUT
        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-auto-endpoint:asr",
                "role": "asr",
                "text": "这是一整句话",
            }
        )
        for _ in range(40):
            if any(e[0]["type"] == "message_completed" for e in emitted):
                break
            await asyncio.sleep(0.02)

        assert any(e[0]["type"] == "message_delta" and e[0]["payload"]["delta"] == "已收到" for e in emitted)
        assert [item["role"] for item in persisted] == ["user", "assistant"]

    asyncio.run(main())


def test_late_audio_chunks_after_auto_commit_are_ignored():
    emitted = []
    fake = _FakeInferenceSession()

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        raise AssertionError("ASR final was not sent in this test")

    async def main():
        runtime = SessionRuntime(
            session_id="session-late-audio-drop",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={"audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"}},
            inference_session=fake,
        )
        runtime._audio._turn_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )
        await runtime.open()
        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))
        chunks_before_late_frame = len(fake.asr_chunks)

        assert runtime.state == SessionState.STREAMING_OUTPUT
        assert fake.asr_commits

        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=4))

        assert len(fake.asr_chunks) == chunks_before_late_frame
        assert not any(
            event["type"] == "error" and "refuses audio_chunk" in event["payload"]["message"]
            for event, _binary in emitted
        )

    asyncio.run(main())


def test_barge_in_listening_does_not_open_asr_before_confirmation(monkeypatch):
    emitted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-old", "run-old"])
    old_run_started = {"event": None}

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    class _FakeTask:
        @staticmethod
        def list_by_agent_run_id(*_args, **_kwargs):
            return []

    _patch_agent_run(monkeypatch, _FakeAgentRun)
    _patch_task(monkeypatch, _FakeTask)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(runtime, turn):
        assert turn.turn_id == "turn-old"
        await runtime.enter_streaming_output()
        await runtime.emit_message_started()
        old_run_started["event"].set()
        await asyncio.sleep(60)

    async def main():
        old_run_started["event"] = asyncio.Event()
        runtime = SessionRuntime(
            session_id="session-barge-probe",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "audio_output": {"load_name": "Qwen3-TTS", "family": "Qwen-tts"},
            },
            inference_session=fake,
        )
        runtime._audio._barge_in_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=60,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )

        await runtime.open()
        await runtime.on_client_message({"type": "user_message", "payload": {"text": "讲个故事"}})
        await asyncio.wait_for(old_run_started["event"].wait(), timeout=1.0)
        await runtime.on_client_message(_pcm_bytes_chunk(_tone_frame(), seq=1))

        assert runtime.state == SessionState.STREAMING_OUTPUT
        assert fake.asr_chunks == []
        assert not any(
            e[0]["type"] == "message_completed"
            and e[0]["payload"].get("interrupt_reason") == "user_interrupt"
            for e in emitted
        )
        await runtime.close(reason="test_done")

    asyncio.run(main())


def test_barge_in_stop_command_interrupts_without_starting_new_run(monkeypatch):
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-old", "run-old", "turn-stop"])
    old_run_started = {"event": None}

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    class _FakeTask:
        @staticmethod
        def list_by_agent_run_id(*_args, **_kwargs):
            return []

    _patch_agent_run(monkeypatch, _FakeAgentRun)
    _patch_task(monkeypatch, _FakeTask)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(runtime, turn):
        if turn.turn_id == "turn-old":
            await runtime.enter_streaming_output()
            await runtime.emit_message_started()
            await runtime.emit_message_delta("旧回复")
            old_run_started["event"].set()
            await asyncio.sleep(60)
            return
        raise AssertionError("stop-only barge-in should not start a new LLM run")

    async def main():
        old_run_started["event"] = asyncio.Event()
        runtime = SessionRuntime(
            session_id="session-barge-stop",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "audio_output": {"load_name": "Qwen3-TTS", "family": "Qwen-tts"},
            },
            inference_session=fake,
        )
        runtime._audio._barge_in_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )

        await runtime.open()
        await runtime.on_client_message({"type": "user_message", "payload": {"text": "讲个故事"}})
        await asyncio.wait_for(old_run_started["event"].wait(), timeout=1.0)
        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))
        assert fake.asr_commits

        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-barge-stop:asr",
                "role": "asr",
                "text": "停停停停停我去",
            }
        )
        await asyncio.sleep(0.05)

        assert runtime.state == SessionState.READY
        assert any(
            e[0]["type"] == "message_completed"
            and e[0]["turn_id"] == "turn-old"
            and e[0]["payload"]["interrupt_reason"] == "user_interrupt"
            for e in emitted
        )
        assert not any(e[0]["type"] == "message_started" and e[0]["turn_id"] == "turn-stop" for e in emitted)
        assert [item["role"] for item in persisted] == ["user", "assistant", "user"]
        # 关键不变量：barge-in control 命中也必须落库**用户原话**，不能用匹配的
        # control prefix 覆盖。曾经把 content 写成 "停"（prefix）→ 历史消息只剩
        # 一个字，原话"停停停停停我去"丢失。
        assert persisted[2]["content"] == "停停停停停我去"
        assert persisted[2]["metadata"]["barge_in_control"] is True
        # control 类别（哪个 prefix / exact 命中）放 metadata 标签字段，不上 content
        assert persisted[2]["metadata"]["barge_in_control_text"] == "停"
        # 同步：emit 给前端的 final transcript_delta 也必须是原话，不能是 prefix —
        # 否则前端 ASR partial 累计显示完整原文后，final 把气泡覆盖成单字 "停"。
        final_transcripts = [
            e[0] for e in emitted
            if e[0]["type"] == "transcript_delta"
            and e[0]["payload"].get("is_final") is True
            and e[0]["turn_id"] == "turn-stop"
        ]
        assert final_transcripts, "barge-in control 必须 emit 一条 final transcript_delta"
        assert final_transcripts[-1]["payload"]["text"] == "停停停停停我去", (
            "control 命中的 final transcript_delta 必须用原话，不能用 prefix 覆盖"
        )

    asyncio.run(main())


def test_inference_audio_chunk_after_tts_cancel_is_dropped(monkeypatch):
    """复现"残音"bug：barge-in 触发 tts_cancel 之后，推理侧那批已经在管道里的
    `session.audio.chunk` 仍然会按队列到达后端。修复后这些 chunk 必须被丢弃，
    不能再 emit_audio_delta 给前端，否则前端 `status_changed{interrupted}` 到达前
    就把它们入队继续播。
    """
    emitted = []
    fake = _FakeInferenceSession()

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        raise AssertionError("此用例无需启动 master_run")

    async def main():
        runtime = SessionRuntime(
            session_id="session-stale-audio",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "audio_output": {"load_name": "Qwen3-TTS", "family": "Qwen-tts"},
            },
            inference_session=fake,
        )
        await runtime.open()

        # 模拟一次正常 TTS：tts_request 注册 active=rid
        rid = await fake.tts_request(text="hello", voice_cfg={"load_name": "Qwen3-TTS"})
        assert fake.active_tts_request_id == rid
        chunk_payload = {
            "type": "session.audio.chunk",
            "session_id": "session-stale-audio:tts",
            "role": "tts",
            "request_id": rid,
            "bytes_len": 4,
            "binary_bytes": b"\x01\x02\x03\x04",
            "mime": "audio/pcm",
            "sample_rate": 24000,
            "is_final": False,
        }
        # 第一个 chunk 是"活跃 request"，正常 emit
        await runtime.on_inference_session_event(chunk_payload)
        assert any(
            e[0]["type"] == "audio_delta" and e[0]["payload"]["bytes_len"] == 4
            for e in emitted
        )

        # 模拟 barge-in：tts_cancel 把 active 置为 None
        emitted.clear()
        await fake.tts_cancel(rid)
        assert fake.active_tts_request_id is None

        # 推理侧管道里的"在途 chunk"继续到达——必须被后端丢弃，不再 emit
        await runtime.on_inference_session_event(chunk_payload)
        assert not any(e[0]["type"] == "audio_delta" for e in emitted), (
            "tts_cancel 之后 in-flight 的 audio chunk 应当被整个丢掉，避免前端继续播放残音"
        )

        # session.audio.end 也来自老 rid：同样不要 emit；waiter 已不在，resolve 是 no-op。
        end_payload = dict(chunk_payload)
        end_payload["type"] = "session.audio.end"
        end_payload["is_final"] = True
        end_payload.pop("binary_bytes", None)
        end_payload["bytes_len"] = 0
        await runtime.on_inference_session_event(end_payload)
        assert not any(e[0]["type"] == "audio_delta" for e in emitted)

    asyncio.run(main())


def test_is_likely_noise_transcript_classification():
    """固化 _is_likely_noise_transcript 的判定边界。

    判定语义（normalize 后判定，normalize 会剥常见中英标点 + 引号 + 破折号 +
    《》〈〉¿¡«»等"装饰性"符号）：

    - normalize 后空 → 噪声（"《》" / "..." 等纯标点直接清掉）
    - normalize 后单字符：在 NOISE_FILLER 里 / 单个拉丁字母 → 噪声；
                         单字汉字 / 单个数字 → 保留（"对" / "是" / "好" / "不"
                         是用户真实简短回复）
    - normalize 后整体在 NOISE_FILLER → 噪声（"oh" / "uh" / "嗯啊"）
    - normalize 后单一字符重复 + 在 NOISE_FILLER → 噪声（"嗯嗯嗯"）
    - normalize 后不含 CJK 且长度 ≤ 4 → 噪声（"Cómo" / "What" / "OK"），
      中文 chat 场景下 ASR 凑词高发；用户真要英文回复一般是完整短语 ≥ 5 字符
    - 其他 → 保留，让 LLM 处理（"好的" / "讲个故事" / "hello world"）
    """
    is_noise = transcript_module._is_likely_noise_transcript

    # 噪声 —— NOISE_FILLER 单字
    assert is_noise("嗯。") is True
    assert is_noise("嗯") is True
    assert is_noise("啊") is True
    assert is_noise("oh") is True
    assert is_noise("UH ") is True  # 大小写 + 标点归一化
    # 噪声 —— NOISE_FILLER 重复
    assert is_noise("嗯嗯嗯") is True
    assert is_noise("啊啊啊") is True
    # 噪声 —— 单个拉丁字母
    assert is_noise("E") is True
    assert is_noise("a.") is True
    assert is_noise("El....") is True  # normalize 后 "el" len=2 不含 CJK ≤ 4 → 噪声
    # 噪声 —— 纯装饰性标点
    assert is_noise("《》") is True
    assert is_noise("...") is True
    assert is_noise("「」") is True
    assert is_noise("...???") is True
    # 噪声 —— 中文 chat 场景下纯西文短词（≤ 4）
    assert is_noise("¿Cómo") is True
    assert is_noise("Cómo") is True
    assert is_noise("What") is True
    assert is_noise("OK") is True
    assert is_noise("Hi!") is True
    # 不是噪声 —— 用户真实单字汉字简短回复
    assert is_noise("对") is False
    assert is_noise("是") is False
    assert is_noise("好。") is False
    assert is_noise("不") is False
    assert is_noise("行") is False
    # 不是噪声 —— 多字真实输入
    assert is_noise("好的") is False
    assert is_noise("讲个故事") is False
    assert is_noise("中文长一点") is False
    # 不是噪声 —— 含 CJK 即使整体短也保留
    assert is_noise("好 ok") is False  # normalize 后 "好ok" 含 CJK
    # 不是噪声 —— 西文长短语 ≥ 5
    assert is_noise("hello") is False
    assert is_noise("hello world") is False
    # 严格空 / 纯空白 —— 与"纯标点"语义对齐都判 True；下游 _handle_transcript_final
    # 用 text.strip() 决定 has_meaningful_text，这里 True/False 不会影响行为分歧。
    assert is_noise("") is True
    assert is_noise("   ") is True


def test_barge_in_vad_profile_defaults_kept_responsive():
    """钉死 barge_in profile 默认参数 —— 这些值直接决定语音打断响应速度。

    历史上曾经把 min_speech_ms=900 / 500 → 用户说"停"/"打住"/"别说了"这种 300-
    500ms 的常用短打断词在讲完之前 detector 都来不及 fire → ASR 拿不到 →
    control words 永远命中不了 → TTS 一直播完。当前 300ms 是经过反复调试
    得到的"短打断词能在用户讲完前触发 + streak=2 仍能拦 echo 单帧脉冲"折中点。

    任何调整都要同步更新 docs §3.6 的延迟公式表 + 触发延迟应当满足
    delay_worst ≤ 600ms。
    """
    async def _build_cfg():
        async def emit(*a, **kw):
            return None

        async def master_run(*a, **kw):
            return None

        runtime = SessionRuntime(
            session_id="session-barge-cfg",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={},
            inference_session=_FakeInferenceSession(),
        )
        return runtime._audio._vad_config("barge_in")

    cfg = asyncio.run(_build_cfg())

    # 短打断词响应能力的关键参数
    assert cfg["min_speech_ms"] == 300, (
        "min_speech_ms 必须 ≤ 300ms，否则 '停'/'打住' 这种 300-500ms 短打断词"
        "在用户讲完之前 detector fire 不出来"
    )
    assert cfg["energy_floor_min_streak"] == 2, (
        "streak=2 是 RMS 兜底路径下拦 echo 单帧脉冲的关键防御层，调到 1 会出现"
        "TTS 播放期间 echo 偶尔脉冲触发假打断"
    )
    assert cfg["energy_speech_floor"] == 0.02, (
        "AEC=true 下 floor=0.02 让大部分人声 chunk 能过；调到 0.05 是 AEC=false"
        "时代的旧值，重开 AEC 后会挡掉大部分人声 → 'TTS 生成阶段无法打断'"
    )
    # pre_roll 必须远超 worst delay = (2-1+⌈300/200⌉)*200 = 600ms，留余量盖住
    # "用户讲话不完全连续 / streak 中途被低能量帧重置" 等情况
    assert cfg["pre_roll_ms"] >= 1500, (
        "pre_roll 不能小于触发延迟 + 600ms 余量，否则 ASR 拿到的会截掉用户开口"
        "前段（曾经 pre_roll=500 + delay=1400ms 时把 '好的，停停停，别再说了' 截"
        "成 '再说了'）"
    )


def test_barge_in_control_text_recognizes_acknowledgments():
    """固化附和词在 barge-in turn 上被识别为 control word（cancel TTS 但不触发 LLM）。

    barge-in 触发延迟 ~600ms 后，用户在 TTS 中途说"对的"/"知道了"/"OK"基本上是
    "我懂了你别再讲了"而不是想开新话题；让它走 control word 捷径直接 cancel 了
    事，不再触发 LLM 回一通"了解，那您还有别的问题吗"。
    """
    detect = transcript_module._barge_in_control_text

    # 新增的附和词
    assert detect("对") == "对"
    assert detect("对的") == "对的"
    assert detect("是") == "是"
    assert detect("是的") == "是的"
    assert detect("明白") == "明白"
    assert detect("明白了") == "明白了"
    assert detect("知道") == "知道"
    assert detect("知道了") == "知道了"
    assert detect("OK") == "ok"  # normalize 到小写
    assert detect("Okay") == "okay"
    # 标点归一化
    assert detect("对的。") == "对的"
    assert detect("OK!") == "ok"
    # 原有的"停"族仍生效（不要回归）
    assert detect("停") == "停"
    assert detect("停一下") == "停一下"
    # negative：长指令不应被误识别为 control
    assert detect("讲个故事") is None
    assert detect("帮我查一下天气") is None


def test_empty_transcript_does_not_emit_final_transcript_delta(monkeypatch):
    """复现 bug：环境噪声触发 audio turn → ASR 返回空 final transcript →
    旧代码会 emit `transcript_delta(text="", is_final=True)`，前端把 in-flight 气泡
    锁成空白蓝点。修复后空文本场景只 emit `error{empty_transcript}`，前端可以靠
    error 自行清理 in-flight 气泡。
    """
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-noise"])

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        raise AssertionError("空 transcript 不应该启动 LLM run")

    test_vad_cfg = {
        "use_neural_vad": False,
        "chunk_ms": 20,
        "min_speech_ms": 20,
        "silence_ms": 40,
        "pre_roll_ms": 20,
        "energy_threshold": 0.01,
    }

    async def main():
        runtime = SessionRuntime(
            session_id="session-empty-transcript",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "vad": test_vad_cfg,
            },
            inference_session=fake,
        )

        await runtime.open()
        # 模拟噪声：speech 一帧 + silence 两帧 → VAD speech_start → speech_end → auto_commit
        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        assert runtime.state == SessionState.TURN_BUFFERING
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))
        assert fake.asr_commits

        # ASR 判定本段实为噪声 / 空段
        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-empty-transcript:asr",
                "role": "asr",
                "text": "   ",  # 空白文本，模拟噪声
            }
        )
        await asyncio.sleep(0.02)

        # 关键断言：不要 emit 任何 final transcript_delta（包括 text=""）
        assert not any(
            e[0]["type"] == "transcript_delta" and e[0]["payload"]["is_final"] is True
            for e in emitted
        ), "空 final transcript 不应该再 emit transcript_delta"
        # 必须 emit error{empty_transcript}
        empty_errors = [
            e[0] for e in emitted
            if e[0]["type"] == "error" and e[0]["payload"]["code"] == "empty_transcript"
        ]
        assert empty_errors and empty_errors[0]["payload"]["recoverable"] is True
        # 关键：text="   " 严格空必须走 "empty" message 路径，不能拼出
        # "transcript is likely noise: ''" 这种自相矛盾字样（曾经 noise filter
        # 改 normalize 后空也算 noise → text=""/"   " 也被分到 noise 路径）
        empty_msg = empty_errors[0]["payload"]["message"]
        assert "empty" in empty_msg
        assert "likely noise" not in empty_msg, (
            f"严格空 transcript 不应走 likely noise 文案，但拿到: {empty_msg!r}"
        )
        # 必须配套 emit transcript_canceled 行为指令事件（前端按事件类型路由清气泡，
        # 不再依赖 error 事件做业务过滤）
        canceled_events = [
            e[0] for e in emitted if e[0]["type"] == "transcript_canceled"
        ]
        assert canceled_events, "空 transcript 必须 emit transcript_canceled 让前端清气泡"
        assert canceled_events[0]["payload"]["reason"] == "empty"
        # state 直接回 ready，不落任何 user 消息
        assert runtime.state == SessionState.READY
        assert runtime.current_turn is None
        assert persisted == []

    asyncio.run(main())


def test_noise_transcript_skips_llm_run_like_empty(monkeypatch):
    """复现：环境噪声触发 audio turn → ASR 把噪声识别成"嗯。"或"El...." →
    必须按 empty_transcript 等价处理（不 emit transcript_delta、不 persist user
    消息、不触发 LLM run、emit error{empty_transcript} 让前端清空 in-flight 气泡）。
    """
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-noise"])

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(_runtime, _turn):
        raise AssertionError("噪声 transcript 不应该启动 LLM run")

    test_vad_cfg = {
        "use_neural_vad": False,
        "chunk_ms": 20,
        "min_speech_ms": 20,
        "silence_ms": 40,
        "pre_roll_ms": 20,
        "energy_threshold": 0.01,
    }

    async def main():
        runtime = SessionRuntime(
            session_id="session-noise-transcript",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="text_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "vad": test_vad_cfg,
            },
            inference_session=fake,
        )

        await runtime.open()
        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        assert runtime.state == SessionState.TURN_BUFFERING
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))
        assert fake.asr_commits

        # ASR 识别为典型的"噪声误识别"短词
        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-noise-transcript:asr",
                "role": "asr",
                "text": "嗯。",
            }
        )
        await asyncio.sleep(0.02)

        # 不应 emit final transcript_delta（有 text 但 likely noise）
        assert not any(
            e[0]["type"] == "transcript_delta" and e[0]["payload"]["is_final"] is True
            for e in emitted
        ), "likely noise transcript 不应该 emit final transcript_delta"
        # 必须 emit error{empty_transcript}（用户/开发者可见的事件卡片信息）
        noise_errors = [
            e[0] for e in emitted
            if e[0]["type"] == "error" and e[0]["payload"]["code"] == "empty_transcript"
        ]
        assert noise_errors, "likely noise transcript 必须 emit error{empty_transcript}"
        # message 里应当带原文方便排障
        assert "嗯。" in noise_errors[0]["payload"]["message"]
        # 必须配套 emit transcript_canceled 行为指令事件，且 reason="noise" 与 error
        # 区分开（前端按事件类型路由清气泡，不再依赖 error code 做业务过滤）
        canceled_events = [
            e[0] for e in emitted if e[0]["type"] == "transcript_canceled"
        ]
        assert canceled_events, "likely noise 必须 emit transcript_canceled 让前端清气泡"
        assert canceled_events[0]["payload"]["reason"] == "noise"
        assert canceled_events[0]["payload"]["text"] == "嗯。"
        # 不应 persist user 消息
        assert persisted == []
        # state 直接回 ready
        assert runtime.state == SessionState.READY
        assert runtime.current_turn is None

    asyncio.run(main())


def test_audio_chunk_after_complete_within_playback_window_marks_turn_as_barge_in(monkeypatch):
    """复现 bug：助手 TTS 已经全部 emit 完、后端进入 ready，但前端音频队列仍在播。
    此时用户喊"停"应被识别为 barge-in 控制词（不再触发 LLM），且 status_changed
    必须带 ``vad_state=speech_start`` 与 ``barge_in=true``，给前端取消播放的信号。
    """
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-old", "run-old", "turn-stop"])
    old_run_started = {"event": None}

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    _patch_agent_run(monkeypatch, _FakeAgentRun)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(runtime, turn):
        if turn.turn_id == "turn-old":
            await runtime.enter_streaming_output()
            await runtime.emit_message_started()
            await runtime.emit_message_delta("旧回复")
            # 模拟向前端送 ~5 秒 24kHz PCM（5 * 24000 * 2 = 240000 字节）；
            # 之后 `complete_run` 把 session 切回 ready，但 _pending_audio_playback_until_ts
            # 仍指向未来 ~5 秒，进入"前端尚未播完"的窗口。
            big_pcm = bytes(240000)
            await runtime.emit_audio_delta(
                pcm_bytes=big_pcm,
                mime="audio/pcm;rate=24000",
                is_final=False,
                sample_rate=24000,
            )
            await runtime.emit_audio_delta(
                pcm_bytes=None,
                mime="audio/pcm;rate=24000",
                is_final=True,
                sample_rate=24000,
            )
            await runtime.complete_run(assistant_text="旧回复")
            old_run_started["event"].set()
            return
        raise AssertionError("控制词 barge-in 不应该再启 LLM run")

    # complete_run 会重置 vad detector，这里通过 metadata.vad 注入"测试用"的极小阈值
    # 配置，让 _build_vad_detector 自动用上。
    test_vad_cfg = {
        "use_neural_vad": False,
        "chunk_ms": 20,
        "min_speech_ms": 20,
        "silence_ms": 40,
        "pre_roll_ms": 20,
        "energy_threshold": 0.01,
    }

    async def main():
        old_run_started["event"] = asyncio.Event()
        runtime = SessionRuntime(
            session_id="session-post-complete-barge",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "audio_output": {"load_name": "Qwen3-TTS", "family": "Qwen-tts"},
                "vad": test_vad_cfg,
            },
            inference_session=fake,
        )

        await runtime.open()
        await runtime.on_client_message({"type": "user_message", "payload": {"text": "讲个故事"}})
        await asyncio.wait_for(old_run_started["event"].wait(), timeout=1.0)
        # complete_run 之后 session 已经回到 ready，但播放估算仍指向未来。
        assert runtime.state == SessionState.READY
        assert runtime._tts.pending_audio_playback_until_ts is not None

        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        # speech_start 触发后开了一个新 audio turn，且应被标 barge_in=True
        assert runtime.state == SessionState.TURN_BUFFERING
        assert runtime.current_turn is not None
        assert runtime.current_turn.barge_in is True
        # 估算窗口在 _begin_audio_turn 之后被清零，避免影响下一轮
        assert runtime._tts.pending_audio_playback_until_ts is None
        # 前端取消播放靠这个事件：status_changed{turn_buffering, vad_state=speech_start, barge_in=true}
        assert any(
            e[0]["type"] == "status_changed"
            and e[0]["payload"].get("state") == "turn_buffering"
            and e[0]["payload"].get("vad_state") == "speech_start"
            and e[0]["payload"].get("barge_in") is True
            for e in emitted
        )

        # 用户继续说控制词文本（auto VAD silence 后自动 commit ASR）
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))
        assert fake.asr_commits

        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-post-complete-barge:asr",
                "role": "asr",
                "text": "停停停停停停停停停，不要再说了",
            }
        )
        await asyncio.sleep(0.05)

        # 控制词捷径生效：state 直接回 ready；不再有第二轮 LLM run。
        # user 文本必须落**原话**（"停停停停停停停停停，不要再说了"），不能落
        # 匹配的 prefix"停"——见上面 test_barge_in_stop_word_cancels_streaming
        # 的同款断言注释。
        assert runtime.state == SessionState.READY
        assert any(
            item["role"] == "user"
            and item["content"] == "停停停停停停停停停，不要再说了"
            and item["metadata"].get("barge_in_control") is True
            and item["metadata"].get("barge_in_control_text") == "停"
            for item in persisted
        )
        # 对比：旧"等一下，换个说法"测试里会有 turn-stop 的 message_started/delta；
        # 这里不应该出现，因为 LLM 没启
        assert not any(
            e[0]["type"] == "message_started" and e[0]["turn_id"] == "turn-stop" for e in emitted
        )

    asyncio.run(main())


def test_audio_chunk_after_complete_outside_playback_window_starts_normal_turn(monkeypatch):
    """对照：播放估算窗口已过，新 turn 仍按普通新 turn 处理（barge_in=False、走 LLM）。"""
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-old", "run-old", "turn-new", "run-new"])
    old_run_started = {"event": None}

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    _patch_agent_run(monkeypatch, _FakeAgentRun)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    new_run_started = {"event": None}

    async def master_run(runtime, turn):
        if turn.turn_id == "turn-old":
            await runtime.enter_streaming_output()
            await runtime.emit_message_started()
            await runtime.emit_message_delta("旧回复")
            await runtime.complete_run(assistant_text="旧回复")
            old_run_started["event"].set()
            return
        assert turn.turn_id == "turn-new"
        assert turn.barge_in is False
        await runtime.emit_message_started()
        await runtime.emit_message_delta("新回复")
        await runtime.complete_run(assistant_text="新回复")
        new_run_started["event"].set()

    test_vad_cfg = {
        "use_neural_vad": False,
        "chunk_ms": 20,
        "min_speech_ms": 20,
        "silence_ms": 40,
        "pre_roll_ms": 20,
        "energy_threshold": 0.01,
    }

    async def main():
        old_run_started["event"] = asyncio.Event()
        new_run_started["event"] = asyncio.Event()
        runtime = SessionRuntime(
            session_id="session-post-complete-normal",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "audio_output": {"load_name": "Qwen3-TTS", "family": "Qwen-tts"},
                "vad": test_vad_cfg,
            },
            inference_session=fake,
        )

        await runtime.open()
        await runtime.on_client_message({"type": "user_message", "payload": {"text": "讲个故事"}})
        await asyncio.wait_for(old_run_started["event"].wait(), timeout=1.0)
        # 旧 turn 没有产出 audio_delta（output 走纯文本），估算窗口为 None。
        assert runtime._tts.pending_audio_playback_until_ts is None
        assert runtime.state == SessionState.READY

        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        assert runtime.state == SessionState.TURN_BUFFERING
        assert runtime.current_turn is not None
        assert runtime.current_turn.barge_in is False
        # 没有 barge_in 标，前端不会因为 vad_state=speech_start 误以为是打断
        assert any(
            e[0]["type"] == "status_changed"
            and e[0]["payload"].get("state") == "turn_buffering"
            and e[0]["payload"].get("vad_state") == "speech_start"
            and e[0]["payload"].get("barge_in") is None
            for e in emitted
        )

        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))

        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-post-complete-normal:asr",
                "role": "asr",
                "text": "再讲一遍",
            }
        )
        await asyncio.wait_for(new_run_started["event"].wait(), timeout=1.0)
        assert any(
            e[0]["type"] == "message_delta" and e[0]["payload"]["delta"] == "新回复" for e in emitted
        )

    asyncio.run(main())


def test_audio_chunk_during_output_barge_in_interrupts_and_starts_new_turn(monkeypatch):
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-old", "run-old", "turn-barge", "run-barge"])
    old_run_started = {"event": None}

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    class _FakeTask:
        @staticmethod
        def list_by_agent_run_id(*_args, **_kwargs):
            return []

    _patch_agent_run(monkeypatch, _FakeAgentRun)
    _patch_task(monkeypatch, _FakeTask)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def master_run(runtime, turn):
        if turn.turn_id == "turn-old":
            await runtime.enter_streaming_output()
            await runtime.emit_message_started()
            await runtime.emit_message_delta("旧回复")
            old_run_started["event"].set()
            await asyncio.sleep(60)
            return
        assert turn.turn_id == "turn-barge"
        assert turn.user_text == "等一下，换个说法"
        await runtime.emit_message_started()
        await runtime.emit_message_delta("新回复")
        await runtime.complete_run(assistant_text="新回复")

    async def main():
        old_run_started["event"] = asyncio.Event()
        runtime = SessionRuntime(
            session_id="session-barge-in",
            user_id="user-1",
            emit=emit,
            master_run=master_run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {"load_name": "Qwen3-ASR-1.7B", "family": "Qwen-asr"},
                "audio_output": {"load_name": "Qwen3-TTS", "family": "Qwen-tts"},
            },
            inference_session=fake,
        )
        runtime._audio._turn_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )
        runtime._audio._barge_in_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=40,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )

        await runtime.open()
        await runtime.on_client_message({"type": "user_message", "payload": {"text": "讲个故事"}})
        await asyncio.wait_for(old_run_started["event"].wait(), timeout=1.0)
        assert runtime.state == SessionState.STREAMING_OUTPUT

        speech = _tone_frame()
        silence = bytes(len(speech))
        await runtime.on_client_message(_pcm_bytes_chunk(speech, seq=1))
        assert runtime.state == SessionState.TURN_BUFFERING
        assert fake.asr_chunks, "barge-in speech_start 应当把 pre-roll 音频送入 ASR"
        assert any(
            e[0]["type"] == "message_completed"
            and e[0]["turn_id"] == "turn-old"
            and e[0]["payload"]["interrupt_reason"] == "user_interrupt"
            for e in emitted
        )

        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=2))
        await runtime.on_client_message(_pcm_bytes_chunk(silence, seq=3))
        assert fake.asr_commits, "speech_end 应当自动触发 asr_commit"

        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-barge-in:asr",
                "role": "asr",
                "text": "等一下，换个说法",
            }
        )
        for _ in range(40):
            if any(e[0]["type"] == "message_completed" and e[0]["turn_id"] == "turn-barge" for e in emitted):
                break
            await asyncio.sleep(0.02)

        assert any(e[0]["type"] == "message_delta" and e[0]["payload"]["delta"] == "新回复" for e in emitted)
        assert runtime.state == SessionState.READY
        assert [item["role"] for item in persisted] == ["user", "assistant", "user", "assistant"]
        assert persisted[2]["content"] == "等一下，换个说法"

    asyncio.run(main())


def test_audio_commit_flows_to_master_and_tts(monkeypatch):
    """PR3 切换后：`_synthesize_voice_reply` 走 `runtime.submit_tts`。

    不再有 `audio_tts.synthesize_audio_sync` 调用；本用例模拟推理侧对
    ``session.tts.request`` 回流 ``session.audio.chunk`` / ``session.audio.end``，
    验证 master 链路最终仍能产生 ``audio_delta`` 与 ``message_completed``。
    """
    emitted = []
    persisted = []
    fake = _FakeInferenceSession()
    generated_ids = iter(["turn-audio-2", "run-audio-2"])

    _patch_generate_uuid(monkeypatch, lambda: next(generated_ids))
    _patch_append_message(monkeypatch, lambda **kwargs: persisted.append(kwargs))

    class _FakeAgentRun:
        @staticmethod
        def create(**kwargs):
            return {"id": kwargs["id"]}

        @staticmethod
        def update(*_args, **_kwargs):
            return True

    _patch_agent_run(monkeypatch, _FakeAgentRun)
    monkeypatch.setattr(master_runtime_module, "ensure_default_agent_presets", lambda: None)
    monkeypatch.setattr(master_runtime_module, "get_master_preset_agent_id", lambda: "preset-master")
    monkeypatch.setattr(
        master_runtime_module,
        "build_prompt_with_history",
        lambda conversation_id, new_message: new_message,
    )
    monkeypatch.setattr(master_runtime_module, "record_run_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(master_runtime_module, "record_run_completed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(master_runtime_module, "record_run_failed", lambda *_args, **_kwargs: None)

    class _FakeAgent:
        @staticmethod
        def get_by_id(_agent_id):
            return {"id": "preset-master", "config": {}}

    class _FakeConversation:
        @staticmethod
        def get_by_id(_conversation_id):
            return {"metadata": {}}

    monkeypatch.setattr(master_runtime_module, "Agent", _FakeAgent)
    monkeypatch.setattr(master_runtime_module, "Conversation", _FakeConversation)

    def _fake_run_crew_blocking(**_kwargs):
        return "这是助手回复"

    monkeypatch.setattr(master_runtime_module, "_run_crew_blocking", _fake_run_crew_blocking)

    async def emit(event, *, binary=None):
        emitted.append((event, binary))

    async def main():
        runtime = SessionRuntime(
            session_id="session-audio-voice-loop",
            user_id="user-1",
            emit=emit,
            master_run=MasterAgentRuntime(ws_manager=None).run,
            input_mode=InputMode.AUDIO_STREAM,
            output_mode="audio_stream",
            metadata={
                "audio_input": {
                    "load_name": "Qwen3-ASR-1.7B",
                    "family": "Qwen-asr",
                },
                "audio_output": {
                    "load_name": "Qwen3-TTS",
                    "family": "Qwen-tts",
                    "tts_mode": "custom_voice",
                    "speaker_name": "xiaoyun",
                },
            },
            inference_session=fake,
        )
        runtime._audio._turn_vad_detector = StreamingVadDetector(
            use_neural_vad=False,
            chunk_ms=20,
            min_speech_ms=20,
            silence_ms=2500,
            pre_roll_ms=20,
            energy_threshold=0.01,
        )

        await runtime.open()
        await runtime.on_client_message(_pcm_bytes_chunk(_tone_frame()))
        await runtime.on_client_message({"type": "session_commit", "payload": {}})
        await runtime.on_inference_session_event(
            {
                "type": "session.transcript.final",
                "session_id": "session-audio-voice-loop:asr",
                "role": "asr",
                "text": "请用语音回复我",
            }
        )

        # 等 submit_tts 把 tts_request 送进 fake inference session
        for _ in range(100):
            if fake.tts_requests:
                break
            await asyncio.sleep(0.02)
        assert fake.tts_requests, "submit_tts 应当调用 inference_session.tts_request"
        tts_req = fake.tts_requests[0]
        assert tts_req["text"] == "这是助手回复"
        assert tts_req["voice_cfg"]["load_name"] == "Qwen3-TTS"
        assert tts_req["voice_cfg"]["speaker_name"] == "xiaoyun"
        rid = tts_req["request_id"]
        sid = "session-audio-voice-loop:tts"
        tts_pcm = base64.b64decode("YXVkaW8=")

        # 模拟推理侧回流 session.audio.chunk / session.audio.end
        await runtime.on_inference_session_event(
            {
                "type": "session.audio.chunk",
                "session_id": sid,
                "role": "tts",
                "request_id": rid,
                "bytes_len": len(tts_pcm),
                "mime": "audio/wav",
                "sample_rate": 24000,
                "is_final": False,
                "binary_bytes": tts_pcm,
            }
        )
        await runtime.on_inference_session_event(
            {
                "type": "session.audio.end",
                "session_id": sid,
                "role": "tts",
                "request_id": rid,
                "sample_rate": 24000,
                "is_final": True,
            }
        )

        for _ in range(100):
            if any(e[0]["type"] == "message_completed" for e in emitted):
                break
            await asyncio.sleep(0.02)

        assert fake.asr_commits, "session_commit 应当触发 asr_commit"
        assert any(
            spec.role == "tts" and spec.load_name == "Qwen3-TTS"
            for spec in fake.opened_roles
        ), "open() 时 TTS inference session 应当已被打开"
        assert any(e[0]["type"] == "message_started" for e in emitted)
        assert any(e[0]["type"] == "message_delta" and e[0]["payload"]["delta"] == "这是助手回复" for e in emitted)
        assert any(
            e[0]["type"] == "audio_delta"
            and e[0]["payload"]["bytes_len"] == len(tts_pcm)
            and e[0]["payload"]["is_final"] is False
            and e[1] == tts_pcm
            for e in emitted
        )
        assert any(
            e[0]["type"] == "audio_delta"
            and e[0]["payload"]["is_final"] is True
            and e[0]["payload"]["sample_rate"] == 24000
            for e in emitted
        )
        assert any(e[0]["type"] == "message_completed" and e[0]["payload"]["content"] == "这是助手回复" for e in emitted)

        # 顺序契约：final audio_delta 必须严格早于 message_completed
        final_idx = next(
            i
            for i, e in enumerate(emitted)
            if e[0]["type"] == "audio_delta" and e[0]["payload"]["is_final"] is True
        )
        completed_idx = next(i for i, e in enumerate(emitted) if e[0]["type"] == "message_completed")
        assert final_idx < completed_idx, (
            f"final audio_delta ({final_idx}) 必须早于 message_completed ({completed_idx})"
        )

        assert persisted[0]["content"] == "请用语音回复我"
        assert persisted[1]["content"] == "这是助手回复"

    asyncio.run(main())


def test_task_event_adapter_drops_stale_events_after_interrupt():
    """回归：interrupt 之后从工作线程漂回主 loop 的 task_event 不能改状态。

    复现链路：
      master_runtime 用 ``asyncio.run_coroutine_threadsafe`` 把工作线程的 task event
      投递到主 loop。``interrupt.execute()`` cancel 主 task 后，已 schedule 但未执行的
      ``task_bound`` 协程仍会被 loop 调度执行；handle_task_event 进而调
      ``runtime.enter_waiting_task()`` 把 state 从 READY 改回 WAITING_TASK。
      用户随后点 mic 发 ``audio_chunk`` 就会拿到 ``state=waiting_task refuses
      audio_chunk`` 的 busy 报错（因为 current_turn=None，barge-in / TURN_BUFFERING
      分支都无法走通）。

    守卫契约：
      - ``current_turn is not turn``：finalize_turn 后或新 turn 启动后丢弃；
      - ``interrupt_requested``：interrupt 已发起、finalize_turn 还没跑的窗口期丢弃。
    """
    from backend.services.chat.session._models import InputMode, SessionState, Turn
    from backend.services.chat.task_event_adapter import handle_task_event

    class _StubRuntime:
        """最小 SessionRuntime 替身，仅暴露 task_event_adapter 用到的接口。"""

        def __init__(self, current_turn, interrupt_requested=False):
            self.current_turn = current_turn
            self.interrupt_requested = interrupt_requested
            self.state = SessionState.READY
            self.calls = []

        async def enter_waiting_task(self):
            self.calls.append("enter_waiting_task")
            self.state = SessionState.WAITING_TASK

        async def emit_task_status(self, *, payload):
            self.calls.append(("emit_task_status", payload))

    async def main():
        old_turn = Turn(turn_id="turn-old", input_mode=InputMode.TEXT)

        # 场景 1：interrupt 已 finalize_turn → current_turn=None。
        # 漂回来的 task_bound 必须被识别为 stale 并丢弃，state 保持 READY。
        rt_finalized = _StubRuntime(current_turn=None)
        await handle_task_event(
            runtime=rt_finalized,
            turn=old_turn,
            event={"type": "task_bound", "task_id": "t-1"},
        )
        assert rt_finalized.state == SessionState.READY, (
            "finalize_turn 之后漂回的 task_bound 必须被守卫丢弃，state 不应被改回 waiting_task"
        )
        assert rt_finalized.calls == [], (
            "stale task_bound 不应触发 enter_waiting_task / emit_task_status"
        )

        # 场景 2：用户已开新 turn → current_turn 是 new_turn 但事件来自 old_turn。
        new_turn = Turn(turn_id="turn-new", input_mode=InputMode.TEXT)
        rt_new_turn = _StubRuntime(current_turn=new_turn)
        await handle_task_event(
            runtime=rt_new_turn,
            turn=old_turn,
            event={"type": "task_bound", "task_id": "t-1"},
        )
        assert rt_new_turn.state == SessionState.READY
        assert rt_new_turn.calls == []

        # 场景 3：interrupt 已发起、finalize_turn 还没跑的窗口期（current_turn 仍为 turn）。
        # interrupt_requested 守卫挡住，避免用户再发起 audio_chunk 前 state 被回改。
        rt_interrupting = _StubRuntime(current_turn=old_turn, interrupt_requested=True)
        await handle_task_event(
            runtime=rt_interrupting,
            turn=old_turn,
            event={"type": "task_bound", "task_id": "t-1"},
        )
        assert rt_interrupting.state == SessionState.READY
        assert rt_interrupting.calls == []

        # 场景 4：正常活跃 turn（无 interrupt）→ 守卫不应误伤，task_bound 正常切到 waiting_task。
        rt_active = _StubRuntime(current_turn=old_turn, interrupt_requested=False)
        await handle_task_event(
            runtime=rt_active,
            turn=old_turn,
            event={"type": "task_bound", "task_id": "t-1"},
        )
        assert rt_active.state == SessionState.WAITING_TASK
        assert rt_active.calls[0] == "enter_waiting_task"
        # 同时也 emit 了 task_status 给前端（task_bound 路径既切状态又广播状态）
        assert any(
            isinstance(c, tuple) and c[0] == "emit_task_status" for c in rt_active.calls
        ), "活跃 turn 的 task_bound 必须正常 emit task_status"

    asyncio.run(main())

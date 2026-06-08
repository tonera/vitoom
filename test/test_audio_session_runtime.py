from pathlib import Path
import sys
import asyncio
import base64
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

import numpy as np

from audio.session_runtime import AudioSessionRuntime
from audio.engines.tts_engine import AudioChunk, VoiceConfig


def test_audio_session_runtime_asr_ready_delta_final_closed():
    sent = []

    async def sender(message, *, binary=None):
        sent.append((message, binary))
        return True

    class _FakeStreamingSession:
        def __init__(self):
            self.calls = 0

        def push_chunk(self, _chunk):
            self.calls += 1
            return {
                "text": f"partial-{self.calls}",
                "delta": f"partial-{self.calls}",
                "language": "English",
                "is_final": False,
            }

        def finish(self):
            return {
                "text": "final-text",
                "delta": "final-text",
                "language": "English",
                "is_final": True,
            }

    async def main():
        async def factory(_state):
            return _FakeStreamingSession()

        runtime = AudioSessionRuntime(sender=sender, backend="vllm", streaming_session_factory=factory)

        assert await runtime.handle_message(
            {
                "type": "session.open",
                "session_id": "s1:asr",
                "role": "asr",
                "seq": 1,
            }
        ) is True
        assert sent[-1][0]["type"] == "session.ready"
        assert sent[-1][0]["role"] == "asr"

        raw = base64.b64decode("AAABAA==")
        assert await runtime.handle_message(
            {
                "type": "session.asr.chunk",
                "session_id": "s1:asr",
                "role": "asr",
                "seq": 2,
                "bytes_len": len(raw),
                "binary_bytes": raw,
            }
        ) is True
        assert sent[-1][0]["type"] == "session.transcript.delta"
        assert sent[-1][0]["role"] == "asr"
        assert sent[-1][0]["chunk_count"] == 1
        assert sent[-1][0]["text"] == "partial-1"

        assert await runtime.handle_message(
            {
                "type": "session.asr.commit",
                "session_id": "s1:asr",
                "role": "asr",
                "seq": 3,
            }
        ) is True
        assert sent[-1][0]["type"] == "session.transcript.final"
        assert sent[-1][0]["text"] == "final-text"

        assert await runtime.handle_message(
            {
                "type": "session.close",
                "session_id": "s1:asr",
                "role": "asr",
                "seq": 4,
            }
        ) is True
        assert sent[-1][0]["type"] == "session.closed"
        assert sent[-1][0]["role"] == "asr"

    asyncio.run(main())


def test_audio_session_runtime_ignores_non_session_messages():
    sent = []

    async def sender(message, *, binary=None):
        sent.append(message)
        return True

    async def main():
        runtime = AudioSessionRuntime(sender=sender)
        ok = await runtime.handle_message({"type": "foo", "session_id": "s1"})
        assert ok is False
        ok = await runtime.handle_message({"type": "session.open"})
        assert ok is False

    asyncio.run(main())


def test_audio_session_runtime_register_service():
    sent = []

    async def sender(message, *, binary=None):
        sent.append(message)
        return True

    async def main():
        runtime = AudioSessionRuntime(sender=sender)
        ok = await runtime.register_service(
            service_type="audio",
            supported_models=["qwen-asr", "qwen-tts"],
            capabilities=["tts"],
            fixed_model="qwen-tts",
            fixed_family="Qwen-tts",
        )

        assert ok is True
        assert sent[0]["type"] == "service_register"
        assert sent[0]["service_type"] == "audio"
        assert sent[0]["supports_task"] is True
        assert sent[0]["supported_models"] == ["qwen-asr", "qwen-tts"]
        assert sent[0]["capabilities"] == ["tts"]
        assert sent[0]["fixed_model"] == "qwen-tts"
        assert sent[0]["fixed_family"] == "Qwen-tts"

    asyncio.run(main())


class _RecordingTtsEngine:
    """记录 synthesize_stream 收到的 voice_cfg，并产出可控 PCM 流。"""

    def __init__(self, *, sample_rate: int = 24000, samples_per_chunk: int = 4800, num_chunks: int = 2, raise_on_call: bool = False):
        self._sample_rate = sample_rate
        self._samples_per_chunk = samples_per_chunk
        self._num_chunks = num_chunks
        self._raise = raise_on_call
        self.calls = []  # 每次调用记录 voice_cfg 的关键字段

    async def synthesize_stream(self, *, text, voice_cfg, cancel_check=None, stream_mode=True):
        self.calls.append(
            {
                "text": text,
                "prompt_wav_path": getattr(voice_cfg, "prompt_wav_path", None),
                "prompt_text": getattr(voice_cfg, "prompt_text", None),
                "speaker_name": getattr(voice_cfg, "speaker_name", None),
                "tts_mode": getattr(voice_cfg, "tts_mode", None),
                "design_instruct": getattr(voice_cfg, "design_instruct", None),
            }
        )
        if self._raise:
            raise RuntimeError("forced failure for test")
        for idx in range(self._num_chunks):
            pcm = np.full((self._samples_per_chunk,), 0.1 * (idx + 1), dtype=np.float32)
            yield AudioChunk(pcm=pcm, sample_rate=self._sample_rate, is_final=False)
        yield AudioChunk(pcm=None, sample_rate=self._sample_rate, is_final=True)


def _build_runtime_with_engine(engine):
    sent = []

    async def sender(message, *, binary=None):
        sent.append((dict(message), binary))
        return True

    async def factory(_state):
        return engine

    runtime = AudioSessionRuntime(sender=sender, backend="vllm", tts_engine_factory=factory)
    return runtime, sent


def _open_tts_session(runtime):
    return runtime.handle_message(
        {
            "type": "session.open",
            "session_id": "chat-1:tts",
            "role": "tts",
            "seq": 1,
            "model": {"name": "voxcpm", "family": "voxcpm-tts"},
        }
    )


async def _drive_tts_request(runtime, *, text, voice_cfg_dict, request_id, seq, engine):
    await runtime.handle_message(
        {
            "type": "session.tts.request",
            "session_id": "chat-1:tts",
            "role": "tts",
            "seq": seq,
            "request_id": request_id,
            "text": text,
            "voice_cfg": voice_cfg_dict,
        }
    )
    # _run_tts_stream 是后台 task，等它跑完
    state = runtime._sessions["chat-1:tts"]  # type: ignore[attr-defined]
    if state.active_tts_task is not None:
        try:
            await state.active_tts_task
        except Exception:
            pass


def test_audio_session_runtime_chained_prompt_inject_and_purge():
    """方案 D：第一段建 cache → 第二段自动注入 → set_chat_voice 切音色后清缓存。

    覆盖：
    1. 第 1 次合成：voice_cfg 不含 prompt_wav_path，cache 为空；
    2. 第 2 次合成：runtime 自动把上一段写好的 wav 注入到 voice_cfg；
    3. 切换 speaker（音色签名不一致）：旧 cache 被清空，第 3 次合成又是裸的；
    4. session.close：临时 wav 文件被删掉。
    """

    engine = _RecordingTtsEngine()
    runtime, _sent = _build_runtime_with_engine(engine)

    async def main():
        await _open_tts_session(runtime)

        await _drive_tts_request(
            runtime,
            text="第一段文本",
            voice_cfg_dict={"tts_mode": "custom_voice", "speaker_name": "anchen"},
            request_id="rid-1",
            seq=2,
            engine=engine,
        )

        state = runtime._sessions["chat-1:tts"]
        assert state.tts_prompt_wav_path is not None, "第 1 段成功后必须建 chained prompt cache"
        assert os.path.exists(state.tts_prompt_wav_path)
        assert state.tts_prompt_text == "第一段文本"
        first_path = state.tts_prompt_wav_path

        assert engine.calls[0]["prompt_wav_path"] is None
        assert engine.calls[0]["prompt_text"] is None

        await _drive_tts_request(
            runtime,
            text="第二段文本",
            voice_cfg_dict={"tts_mode": "custom_voice", "speaker_name": "anchen"},
            request_id="rid-2",
            seq=3,
            engine=engine,
        )

        assert engine.calls[1]["prompt_wav_path"] == first_path, "同音色第二段必须续上一段 prompt"
        assert engine.calls[1]["prompt_text"] == "第一段文本"
        # cache 被刷新成第 2 段
        assert state.tts_prompt_text == "第二段文本"
        assert state.tts_prompt_wav_path is not None
        assert os.path.exists(state.tts_prompt_wav_path)
        # 旧文件应被清掉
        assert not os.path.exists(first_path), "更新 cache 时应清掉旧的临时 wav"

        second_path = state.tts_prompt_wav_path

        # 第 3 段切换 speaker：签名不一致，runtime 必须丢掉旧 cache
        await _drive_tts_request(
            runtime,
            text="切换之后的第一段",
            voice_cfg_dict={"tts_mode": "custom_voice", "speaker_name": "linda"},
            request_id="rid-3",
            seq=4,
            engine=engine,
        )

        assert engine.calls[2]["prompt_wav_path"] is None, "音色切换后第一段必须不带 prompt"
        assert engine.calls[2]["prompt_text"] is None
        assert not os.path.exists(second_path), "音色切换时应清掉旧的临时 wav"
        # 切换后第 3 段又落地了一份新 cache
        assert state.tts_prompt_wav_path is not None
        assert os.path.exists(state.tts_prompt_wav_path)
        assert state.tts_prompt_text == "切换之后的第一段"
        third_path = state.tts_prompt_wav_path

        # session.close 必须把临时文件删干净
        await runtime.handle_message(
            {
                "type": "session.close",
                "session_id": "chat-1:tts",
                "role": "tts",
                "seq": 5,
            }
        )
        assert not os.path.exists(third_path), "session.close 后临时 wav 应被删除"

    asyncio.run(main())


def test_audio_session_runtime_chained_prompt_no_update_on_failure():
    """合成失败 / 取消时不更新 cache，避免把残缺音频喂给下一段。"""

    # 第 1 段先成功建 cache
    engine_ok = _RecordingTtsEngine()
    runtime, _sent = _build_runtime_with_engine(engine_ok)

    async def main():
        await _open_tts_session(runtime)
        await _drive_tts_request(
            runtime,
            text="正常段",
            voice_cfg_dict={"tts_mode": "custom_voice", "speaker_name": "anchen"},
            request_id="rid-1",
            seq=2,
            engine=engine_ok,
        )

        state = runtime._sessions["chat-1:tts"]
        good_path = state.tts_prompt_wav_path
        assert good_path is not None and os.path.exists(good_path)

        # 第 2 段强制失败：cache 必须保留为第 1 段的内容（不能被覆盖、不能被清空）
        # 通过替换 engine 模拟失败
        async def failing_factory(_state):
            return _RecordingTtsEngine(raise_on_call=True)

        # 覆盖 factory（直接替私有属性，单元测试范围内可接受）
        state.tts_engine = None
        runtime._tts_engine_factory = failing_factory  # type: ignore[attr-defined]

        await _drive_tts_request(
            runtime,
            text="失败段",
            voice_cfg_dict={"tts_mode": "custom_voice", "speaker_name": "anchen"},
            request_id="rid-2",
            seq=3,
            engine=None,
        )

        assert state.tts_prompt_wav_path == good_path, "合成失败不能覆盖原 cache"
        assert state.tts_prompt_text == "正常段", "合成失败不能覆盖原 cache 的 prompt_text"
        assert os.path.exists(good_path)

        # 收尾
        await runtime.handle_message(
            {
                "type": "session.close",
                "session_id": "chat-1:tts",
                "role": "tts",
                "seq": 4,
            }
        )

    asyncio.run(main())


def test_audio_session_runtime_chained_prompt_skips_voice_design():
    """voice_design 模式靠 instruct 文本驱动音色，不应该被注入 chained prompt。"""

    engine = _RecordingTtsEngine()
    runtime, _sent = _build_runtime_with_engine(engine)

    async def main():
        await _open_tts_session(runtime)

        # 先用 custom_voice 建好 cache
        await _drive_tts_request(
            runtime,
            text="先垫一段",
            voice_cfg_dict={"tts_mode": "custom_voice", "speaker_name": "anchen"},
            request_id="rid-1",
            seq=2,
            engine=engine,
        )
        state = runtime._sessions["chat-1:tts"]
        assert state.tts_prompt_wav_path is not None

        # 切到 voice_design，此时不应注入 chained prompt
        await _drive_tts_request(
            runtime,
            text="voice design 段",
            voice_cfg_dict={
                "tts_mode": "voice_design",
                "design_instruct": "沧桑沙哑的男性",
            },
            request_id="rid-2",
            seq=3,
            engine=engine,
        )

        assert engine.calls[1]["tts_mode"] == "voice_design"
        assert engine.calls[1]["prompt_wav_path"] is None
        assert engine.calls[1]["prompt_text"] is None

        # 收尾
        await runtime.handle_message(
            {
                "type": "session.close",
                "session_id": "chat-1:tts",
                "role": "tts",
                "seq": 4,
            }
        )

    asyncio.run(main())

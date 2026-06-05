"""``VoxCPMTtsHandler._run_drama`` 的 chain prompt 行为单测。

聚焦方案 D 在 drama 路径上的扩展：
- 同 character 第二段开始走 chain 模式（``continuation_prompt=True``）；
- chain 模式 reference 锚定在该 character 的 ``seed_path``
  （``continuation_reference_wav_path``），而 ``prompt_wav_path`` 指向上一段
  合成结果，``prompt_text`` 严格对应那一段的文本；
- 不同 character 的 chain state 互不污染；
- 长段（> 18s）跳过 cache 更新，下一段仍用上一段健康缓存；
- 临时 wav 在 ``finally`` 里被清理（包括 seed + chain）。
"""

import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))


def _ensure_video_runtime_stub() -> None:
    if "video.runtime.io_utils" in sys.modules:
        return
    video_pkg = sys.modules.setdefault("video", types.ModuleType("video"))
    video_pkg.__path__ = []  # type: ignore[attr-defined]
    runtime_pkg = sys.modules.setdefault("video.runtime", types.ModuleType("video.runtime"))
    runtime_pkg.__path__ = []  # type: ignore[attr-defined]
    io_utils = types.ModuleType("video.runtime.io_utils")

    async def _stub_download(url, *_args, **_kwargs):
        return str(url)

    io_utils.download_url_to_tempfile = _stub_download  # type: ignore[attr-defined]
    sys.modules["video.runtime.io_utils"] = io_utils


_ensure_video_runtime_stub()

from audio.engines.tts_engine import AudioChunk  # noqa: E402
from audio.handlers.voxcpm_tts_handler import VoxCPMTtsHandler  # noqa: E402


class _FakeEngine:
    """记录每次 ``synthesize_stream`` 调用的 voice_cfg 关键字段，并产出可控音频。

    第 N 次调用的输出长度由 ``durations_seconds[N]`` 决定（默认 2 秒），
    配合长段保护测试。
    """

    def __init__(self, *, sample_rate: int = 24000, durations_seconds: List[float] | None = None):
        self._sample_rate = sample_rate
        self._durations = list(durations_seconds or [])
        self.calls: List[Dict[str, Any]] = []

    async def synthesize_stream(self, *, text, voice_cfg, cancel_check=None, stream_mode=False):
        idx = len(self.calls)
        self.calls.append(
            {
                "text": text,
                "tts_mode": getattr(voice_cfg, "tts_mode", None),
                "speaker_name": getattr(voice_cfg, "speaker_name", None),
                "prompt_wav_path": getattr(voice_cfg, "prompt_wav_path", None),
                "prompt_text": getattr(voice_cfg, "prompt_text", None),
                "continuation_prompt": getattr(voice_cfg, "continuation_prompt", False),
                "continuation_reference_wav_path": getattr(voice_cfg, "continuation_reference_wav_path", None),
                "design_instruct": getattr(voice_cfg, "design_instruct", None),
                "instruct": getattr(voice_cfg, "instruct", None),
            }
        )
        seconds = self._durations[idx] if idx < len(self._durations) else 2.0
        samples = max(1, int(seconds * self._sample_rate))
        # 不同段用不同 amplitude，便于在断言中区分（但不参与 chain prompt 内容判断）
        amp = 0.05 + 0.01 * (idx % 8)
        pcm = np.full((samples,), amp, dtype=np.float32)
        yield AudioChunk(pcm=pcm, sample_rate=self._sample_rate, is_final=False)
        yield AudioChunk(pcm=np.zeros(0, dtype=np.float32), sample_rate=self._sample_rate, is_final=True)


class _FakeResultHandler:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def process_single_result(self, **kwargs):
        self.calls.append(kwargs)


def _build_handler(engine: _FakeEngine) -> Tuple[VoxCPMTtsHandler, _FakeResultHandler]:
    result_handler = _FakeResultHandler()

    async def _stream_sender(_event: Dict[str, Any]) -> bool:  # pragma: no cover
        return True

    async def _bundle_loader(_mode: str) -> Dict[str, Any]:  # pragma: no cover - 不会被调用
        return {}

    handler = VoxCPMTtsHandler(
        audio_mode="tts",
        result_handler=result_handler,  # type: ignore[arg-type]
        service_id="svc-test",
        bundle_loader=_bundle_loader,
        stream_sender=_stream_sender,
        check_cancelled=None,
        speaker_presets={},
        default_speaker=None,
    )
    # 替换真实 engine 为 fake，绕过 model bundle 加载。
    handler._engine = engine  # type: ignore[assignment]
    return handler, result_handler


def _make_params(*, characters: List[Dict[str, Any]], dialogues: List[Dict[str, Any]]):
    return SimpleNamespace(
        type="audio",
        stream=False,
        file_type="mp3",
        drama={"characters": characters, "dialogues": dialogues},
    )


def _base_voice_cfg():
    from audio.engines.tts_engine import VoiceConfig

    return VoiceConfig(tts_mode="custom_voice")


def test_drama_chain_inject_after_first_block_per_character():
    """A→B→A→B：每个 character 自己的第二段开始走 chain 模式，
    chain 状态在不同 character 间互不污染。"""

    engine = _FakeEngine()
    handler, result_handler = _build_handler(engine)
    # 注：pause_after_ms ≥ 800 是为了禁用 _build_dialogue_blocks 的同 speaker 合并优化，
    # 这样每个 dialogue line 都独立成 block，便于精确断言每段的 chain 注入行为。
    params = _make_params(
        characters=[
            {"id": "A", "voice_mode": "voice_design", "instruct": "温柔女声"},
            {"id": "B", "voice_mode": "voice_design", "instruct": "沉稳男声"},
        ],
        dialogues=[
            {"speaker_id": "A", "text": "A 第一句", "pause_after_ms": 900},
            {"speaker_id": "B", "text": "B 第一句", "pause_after_ms": 900},
            {"speaker_id": "A", "text": "A 第二句", "pause_after_ms": 900},
            {"speaker_id": "B", "text": "B 第二句", "pause_after_ms": 900},
            {"speaker_id": "A", "text": "A 第三句", "pause_after_ms": 900},
        ],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t1",
            voice_cfg=_base_voice_cfg(),
            started_at=0.0,
        )
    )

    # 调用顺序：seed A、seed B、A1、B1、A2、B2、A3 共 7 次
    assert len(engine.calls) == 7

    # seed 阶段：voice_design，无 chain 字段
    seed_a, seed_b = engine.calls[0], engine.calls[1]
    assert seed_a["tts_mode"] == "voice_design"
    assert seed_b["tts_mode"] == "voice_design"
    assert seed_a["continuation_prompt"] is False
    assert seed_b["continuation_prompt"] is False

    # 每个 character 第一段：Controllable，prompt_wav=seed，prompt_text=None
    a1, b1, a2, b2, a3 = engine.calls[2], engine.calls[3], engine.calls[4], engine.calls[5], engine.calls[6]

    assert a1["text"].endswith("A 第一句")
    assert a1["continuation_prompt"] is False
    assert a1["prompt_text"] is None
    assert a1["prompt_wav_path"] is not None  # = seed_path_A
    seed_a_path = a1["prompt_wav_path"]

    assert b1["text"].endswith("B 第一句")
    assert b1["continuation_prompt"] is False
    assert b1["prompt_text"] is None
    assert b1["prompt_wav_path"] is not None
    seed_b_path = b1["prompt_wav_path"]
    assert seed_a_path != seed_b_path, "不同 character 的 seed 必须分开"

    # A 第二段：chain 模式，reference 锚 seed_a，prompt 续 A 第一段
    assert a2["continuation_prompt"] is True, "A 第二段必须走 chain 模式"
    assert a2["continuation_reference_wav_path"] == seed_a_path, "A chain reference 必须锚 seed_a"
    assert a2["prompt_text"] == "A 第一句", "A chain prompt_text 必须严格对应 A 第一段文本"
    a1_chain_wav = a2["prompt_wav_path"]
    assert a1_chain_wav is not None and a1_chain_wav != seed_a_path, (
        "A chain prompt_wav 必须是上一段合成结果，不能等于 seed"
    )

    # B 第二段：B 自己的 chain 模式，与 A chain 完全隔离
    assert b2["continuation_prompt"] is True
    assert b2["continuation_reference_wav_path"] == seed_b_path, "B chain reference 必须锚 seed_b"
    assert b2["prompt_text"] == "B 第一句"
    b1_chain_wav = b2["prompt_wav_path"]
    assert b1_chain_wav is not None and b1_chain_wav != seed_b_path
    assert b1_chain_wav != a1_chain_wav, "A 与 B 的 chain wav 必须分开"

    # A 第三段：chain 续 A 第二段（不是第一段），证明 cache 被刷新
    assert a3["continuation_prompt"] is True
    assert a3["continuation_reference_wav_path"] == seed_a_path
    assert a3["prompt_text"] == "A 第二句", "A 第三段 chain prompt 必须续 A 第二段，不是 A 第一段"
    assert a3["prompt_wav_path"] != a1_chain_wav, "A chain wav 必须被刷新成 A 第二段"

    # 收尾：临时文件应已被 finally 清理（seed_a / seed_b / chain wav 全删）
    for path in (seed_a_path, seed_b_path, a1_chain_wav, b1_chain_wav, a3["prompt_wav_path"]):
        assert not os.path.exists(path), f"临时 wav 必须被清理：{path}"

    # result_handler 应被调用一次
    assert len(result_handler.calls) == 1


def test_drama_chain_skips_long_block():
    """单段时长 > 18s 时跳过 cache 更新；下一段仍用上一段健康缓存。"""

    # durations 按调用顺序：seed-A(2s), A1(2s), A2(20s 越界), A3(2s)
    engine = _FakeEngine(durations_seconds=[2.0, 2.0, 20.0, 2.0])
    handler, _ = _build_handler(engine)

    params = _make_params(
        characters=[{"id": "A", "voice_mode": "voice_design", "instruct": "温柔女声"}],
        dialogues=[
            {"speaker_id": "A", "text": "A 第一句", "pause_after_ms": 900},
            {"speaker_id": "A", "text": "A 第二句很长", "pause_after_ms": 900},
            {"speaker_id": "A", "text": "A 第三句", "pause_after_ms": 900},
        ],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t2",
            voice_cfg=_base_voice_cfg(),
            started_at=0.0,
        )
    )

    # 调用：seed-A、A1、A2、A3 共 4 次
    assert len(engine.calls) == 4
    a1, a2, a3 = engine.calls[1], engine.calls[2], engine.calls[3]

    # A2 走 chain，prompt 续 A1（正常）
    assert a2["continuation_prompt"] is True
    assert a2["prompt_text"] == "A 第一句"
    a1_chain_wav = a2["prompt_wav_path"]

    # A2 时长 > 18s → cache 不更新；A3 还应该续 A1，而不是 A2
    assert a3["continuation_prompt"] is True
    assert a3["prompt_text"] == "A 第一句", (
        "长段 A2 不应覆盖 cache；A3 必须仍用 A1 作 chain prompt"
    )
    assert a3["prompt_wav_path"] == a1_chain_wav, (
        "长段 A2 不应覆盖 cache wav；A3 必须仍用 A1 chain wav"
    )


def test_drama_chain_temp_paths_cleaned_on_failure():
    """合成中途抛错时，已写入磁盘的 chain wav 也必须在 finally 里被清理。"""

    class _BoomEngine(_FakeEngine):
        def __init__(self):
            super().__init__()
            self._fail_at = 3  # seed-A, seed-B, A1 之后 B1 抛错

        async def synthesize_stream(self, *, text, voice_cfg, cancel_check=None, stream_mode=False):
            idx = len(self.calls)
            if idx == self._fail_at:
                # 仍要 record，便于取 chain 路径做断言
                self.calls.append(
                    {
                        "text": text,
                        "tts_mode": getattr(voice_cfg, "tts_mode", None),
                        "prompt_wav_path": getattr(voice_cfg, "prompt_wav_path", None),
                        "prompt_text": getattr(voice_cfg, "prompt_text", None),
                        "continuation_prompt": getattr(voice_cfg, "continuation_prompt", False),
                        "continuation_reference_wav_path": getattr(
                            voice_cfg, "continuation_reference_wav_path", None
                        ),
                    }
                )
                raise RuntimeError("forced failure mid-drama")
            async for c in super().synthesize_stream(
                text=text, voice_cfg=voice_cfg, cancel_check=cancel_check, stream_mode=stream_mode
            ):
                yield c

    engine = _BoomEngine()
    handler, _ = _build_handler(engine)
    params = _make_params(
        characters=[
            {"id": "A", "voice_mode": "voice_design", "instruct": "温柔女声"},
            {"id": "B", "voice_mode": "voice_design", "instruct": "沉稳男声"},
        ],
        dialogues=[
            {"speaker_id": "A", "text": "A 第一句", "pause_after_ms": 900},
            {"speaker_id": "B", "text": "B 第一句（这次会失败）", "pause_after_ms": 900},
        ],
    )

    raised = None
    try:
        asyncio.run(
            handler._run_drama(  # type: ignore[arg-type]
                params=params,
                task_id="t3",
                voice_cfg=_base_voice_cfg(),
                started_at=0.0,
            )
        )
    except RuntimeError as exc:
        raised = exc

    assert raised is not None and "forced failure" in str(raised)

    # 取已被记录的 chain wav 路径并验证它们都被清理了（A1 写完后才挂掉）
    a1 = engine.calls[2]
    a1_chain_wav = a1["prompt_wav_path"]
    seed_a = engine.calls[2]["prompt_wav_path"]  # A1 的 prompt_wav 即 seed_a
    # 注意：A1 是首段（continuation_prompt=False），其 prompt_wav 就是 seed_a。
    # 真正的 chain 写盘发生在 A1 合成完之后；此时还没轮到 A 续段，所以 chain_temp_paths
    # 至少有 1 个元素（A1 的合成结果）。我们无法直接拿到那个 path，但 finally 应清理掉所有临时
    # 文件——通过断言"任何 voxcpm-drama-chain-* / voxcpm-drama-seed-* 临时文件都不残留"
    # 来反向证明清理生效。
    import glob
    import tempfile

    leftover = glob.glob(os.path.join(tempfile.gettempdir(), "voxcpm-drama-chain-*.wav"))
    leftover += glob.glob(os.path.join(tempfile.gettempdir(), "voxcpm-drama-seed-*.wav"))
    assert leftover == [], f"drama 抛错后临时 wav 必须被清理，残留：{leftover}"
    # 同时确保我们读出的 seed/chain 路径已经不存在
    assert not os.path.exists(seed_a)
    assert not os.path.exists(a1_chain_wav)

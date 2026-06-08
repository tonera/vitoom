"""``VoxCPMTtsEngine._build_generation_kwargs`` 的纯逻辑单测。

聚焦"chained prompt"语义（方案 D）：
    - task 通道经典语义：``prompt_wav_path`` 与 ``reference_wav`` 复用同一份用户音频；
    - chat 实时 chained prompt：``continuation_prompt=True`` 时两者必须分开
      （reference 锁住 speaker preset 的音色，prompt_wav 用上一段合成结果续韵律），
      避免连续多段后音色累积漂移。
"""

import logging
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))


# voxcpm_tts_engine 顶层 import `video.runtime.io_utils.download_url_to_tempfile`，
# 而 `video` 包链路会拉入 diffsynth → einops 等推理重依赖（在纯 Python 测试机上不安装）。
# 这里仅替换最小依赖项，让 module-level import 通过；engine._build_generation_kwargs
# 不会触发 download_url_to_tempfile，所以 stub 实现可保持空体。
def _ensure_video_runtime_stub() -> None:
    if "video.runtime.io_utils" in sys.modules:
        return
    video_pkg = sys.modules.setdefault("video", types.ModuleType("video"))
    video_pkg.__path__ = []  # type: ignore[attr-defined]
    runtime_pkg = sys.modules.setdefault("video.runtime", types.ModuleType("video.runtime"))
    runtime_pkg.__path__ = []  # type: ignore[attr-defined]
    io_utils = types.ModuleType("video.runtime.io_utils")

    async def _stub_download(url, *_args, **_kwargs):
        # 测试不需要真实下载/落盘：本地绝对路径直接透传，
        # 这样 _resolve_reference_wav_path 末尾的统一落盘步骤不会抛异常。
        return str(url)

    io_utils.download_url_to_tempfile = _stub_download  # type: ignore[attr-defined]
    sys.modules["video.runtime.io_utils"] = io_utils


_ensure_video_runtime_stub()

from audio.engines.tts_engine import VoiceConfig  # noqa: E402
from audio.engines.voxcpm_tts_engine import VoxCPMTtsEngine  # noqa: E402


def _make_engine() -> VoxCPMTtsEngine:
    return VoxCPMTtsEngine(
        audio_mode="realtime_tts",
        bundle_loader=lambda mode: None,  # 不会被 _build_generation_kwargs 触发
        logger=logging.getLogger("test.voxcpm"),
        speaker_presets={"anchen": "third_party/voices/anchen.wav"},
        default_speaker="anchen",
    )


def test_build_generation_kwargs_classic_ultimate_cloning_shares_reference_and_prompt():
    """task 通道经典语义：用户传 prompt_wav_path + prompt_text，
    reference 与 prompt_wav 复用同一份音频（保持向后兼容）。"""

    engine = _make_engine()
    voice_cfg = VoiceConfig(
        tts_mode="custom_voice",
        speaker_name="anchen",
        prompt_wav_path="user_audio.wav",
        prompt_text="用户自带的参考文本",
        # continuation_prompt 默认 False
    )
    # 假设 reference 已被解析成 user_audio.wav 的绝对路径
    resolved_reference = "/tmp/resolved_user_audio.wav"

    kwargs = engine._build_generation_kwargs(
        text="目标文本",
        voice_cfg=voice_cfg,
        reference_wav_path=resolved_reference,
    )

    assert kwargs["reference_wav_path"] == resolved_reference
    assert kwargs["prompt_wav_path"] == resolved_reference, (
        "经典语义下 prompt_wav 与 reference 必须复用同一份用户音频，保持向后兼容"
    )
    assert kwargs["prompt_text"] == "用户自带的参考文本"


def test_build_generation_kwargs_chain_mode_separates_reference_and_prompt(tmp_path):
    """chain 模式：``continuation_prompt=True`` 时 reference 与 prompt_wav 分开。

    reference 由调用方按 speaker preset 解析得到，
    prompt_wav 由 voice_cfg.prompt_wav_path（上一段合成结果）独立提供。
    """

    engine = _make_engine()
    chain_wav = tmp_path / "last_block.wav"
    chain_wav.write_bytes(b"")  # 占位：测试只在乎路径解析，不读内容

    voice_cfg = VoiceConfig(
        tts_mode="custom_voice",
        speaker_name="anchen",
        prompt_wav_path=str(chain_wav),
        prompt_text="上一段的真实文本",
        continuation_prompt=True,
    )
    # session_runtime 期望此时 reference 走 speaker preset 解析
    speaker_reference = "/abs/path/to/anchen.wav"

    kwargs = engine._build_generation_kwargs(
        text="新一段文本",
        voice_cfg=voice_cfg,
        reference_wav_path=speaker_reference,
    )

    assert kwargs["reference_wav_path"] == speaker_reference, (
        "chain 模式 reference_wav 必须保持 speaker preset，不能被上一段合成结果替换"
    )
    assert kwargs["prompt_wav_path"] == str(chain_wav), (
        "chain 模式 prompt_wav 必须是 voice_cfg.prompt_wav_path 指向的上一段合成结果"
    )
    assert kwargs["prompt_text"] == "上一段的真实文本"
    assert kwargs["reference_wav_path"] != kwargs["prompt_wav_path"], (
        "chain 模式核心约束：reference 与 prompt_wav 必须分开，否则音色会累积漂移"
    )


def test_build_generation_kwargs_chain_mode_falls_back_when_prompt_wav_missing():
    """``continuation_prompt=True`` 但 prompt_wav_path 为空时，退回纯 Controllable Voice Cloning
    （只有 reference，没有 prompt_wav/prompt_text）——避免 prompt_text 半残留。"""

    engine = _make_engine()
    voice_cfg = VoiceConfig(
        tts_mode="custom_voice",
        speaker_name="anchen",
        prompt_wav_path=None,
        prompt_text="",  # 空字符串，进不了 Ultimate Cloning 分支
        continuation_prompt=True,
    )
    speaker_reference = "/abs/path/to/anchen.wav"

    kwargs = engine._build_generation_kwargs(
        text="文本",
        voice_cfg=voice_cfg,
        reference_wav_path=speaker_reference,
    )

    assert kwargs["reference_wav_path"] == speaker_reference
    assert "prompt_wav_path" not in kwargs
    assert "prompt_text" not in kwargs


def test_resolve_reference_chain_mode_uses_continuation_reference_wav_path(tmp_path):
    """drama 路径：chain 模式下 reference 必须走 ``continuation_reference_wav_path``，
    而不是 prompt_wav_path（上一段合成结果）也不是 speaker_name preset
    （drama 内 character 的 reference 不在 speaker_presets 字典里）。

    这是 drama 扩展方案 D 的核心约束：reference 必须锚定在 character seed_path。
    """

    import asyncio

    engine = _make_engine()
    seed_wav = tmp_path / "character_seed.wav"
    seed_wav.write_bytes(b"")
    chain_wav = tmp_path / "previous_block.wav"
    chain_wav.write_bytes(b"")

    voice_cfg = VoiceConfig(
        tts_mode="custom_voice",
        speaker_name=None,  # drama 路径无 speaker_name
        prompt_wav_path=str(chain_wav),
        prompt_text="上一段对白",
        continuation_prompt=True,
        continuation_reference_wav_path=str(seed_wav),
    )

    resolved = asyncio.run(engine._resolve_reference_wav_path(voice_cfg))

    assert resolved == str(seed_wav), (
        "chain 模式下 reference 必须取 continuation_reference_wav_path，"
        "不能被 prompt_wav_path 抢占，也不能落到 default_speaker"
    )


def test_resolve_reference_classic_mode_ignores_continuation_reference():
    """非 chain 模式下 ``continuation_reference_wav_path`` 不应起作用，
    避免误传字段污染 task 通道经典语义。"""

    import asyncio

    engine = _make_engine()
    voice_cfg = VoiceConfig(
        tts_mode="custom_voice",
        speaker_name="anchen",
        prompt_wav_path=None,
        prompt_text=None,
        continuation_prompt=False,  # 非 chain 模式
        continuation_reference_wav_path="/should/be/ignored.wav",
    )
    resolved = asyncio.run(engine._resolve_reference_wav_path(voice_cfg))
    # 应走 speaker preset
    assert resolved is not None and resolved.endswith("anchen.wav")


def test_build_generation_kwargs_voice_design_unaffected():
    """voice_design 路径不依赖 reference / prompt，行为不应被 chained prompt 改动影响。"""

    engine = _make_engine()
    voice_cfg = VoiceConfig(
        tts_mode="voice_design",
        design_instruct="温柔的女性声音",
    )
    kwargs = engine._build_generation_kwargs(
        text="hi",
        voice_cfg=voice_cfg,
        reference_wav_path=None,
    )
    assert kwargs["text"] == "(温柔的女性声音)hi"
    assert "reference_wav_path" not in kwargs
    assert "prompt_wav_path" not in kwargs

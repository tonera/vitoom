"""``QwenTtsHandler`` 在新架构下的核心行为单测。

聚焦 Qwen-TTS 后端 drama + 入口默认规则：
- 单段 TTS 入口规则：``model_name`` 为空时按 ``instruct`` 决定权重 + ``tts_mode``；
- drama per-character 决定 design 阶段权重（VoiceDesign / CustomVoice），按权重分组后
  各自一次 swap；最后切到 Base 走 voice_clone；
- 同 character 紧邻 dialogue 走 batch；
- ``bundle_loader`` 调用序列锁住"权重 swap 顺序"，借此证明同时刻只持一份 Qwen-TTS 权重；
- 工具层 ``_normalize_characters`` 的 family-aware 行为保持不变。
"""

import asyncio
import io
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

from audio.engines.tts_engine import VoiceConfig  # noqa: E402
from audio.engines.qwen_tts_engine import (  # noqa: E402
    DEFAULT_QWEN_BASE_MODEL,
    DEFAULT_QWEN_CUSTOM_VOICE_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_MODEL,
    QwenTtsEngine,
)
from audio.handlers.qwen_tts_handler import QwenTtsHandler  # noqa: E402


class _FakeDesignModel:
    """承接 ``generate_voice_design``；返回固定 ramp 音频，方便区分 character。"""

    def __init__(self, *, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.calls: List[Dict[str, Any]] = []

    def generate_voice_design(self, **kwargs):
        self.calls.append(dict(kwargs))
        idx = len(self.calls)
        ref = np.full((self.sample_rate,), 0.05 * idx, dtype=np.float32)
        return [ref], self.sample_rate


class _FakeCustomVoiceModel:
    """承接 ``generate_custom_voice``；返回固定 ramp 音频，标记 speaker 来源。"""

    def __init__(self, *, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.calls: List[Dict[str, Any]] = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(dict(kwargs))
        idx = len(self.calls)
        # 跟 design 模型用不同区段的振幅，便于断言 seed 来自哪条链路。
        ref = np.full((self.sample_rate,), -0.05 * idx, dtype=np.float32)
        return [ref], self.sample_rate


class _FakeBaseModel:
    """承接 ``create_voice_clone_prompt`` / ``generate_voice_clone``。"""

    def __init__(self, *, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.prompt_calls: List[Dict[str, Any]] = []
        self.clone_calls: List[Dict[str, Any]] = []
        self._next_prompt_id = 0

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(dict(kwargs))
        self._next_prompt_id += 1
        return {"prompt_id": self._next_prompt_id}

    def generate_voice_clone(self, **kwargs):
        self.clone_calls.append(dict(kwargs))
        text = kwargs["text"]
        if isinstance(text, list):
            outputs = [
                np.full((self.sample_rate // 2,), 0.1 * (i + 1), dtype=np.float32)
                for i in range(len(text))
            ]
            return outputs, self.sample_rate
        audio = np.full((self.sample_rate // 2,), 0.2, dtype=np.float32)
        return [audio], self.sample_rate


class _FakeResultHandler:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def process_single_result(self, **kwargs):
        self.calls.append(kwargs)


def _build_handler(
    *,
    design_model: _FakeDesignModel,
    base_model: _FakeBaseModel,
    custom_voice_model: _FakeCustomVoiceModel | None = None,
) -> Tuple[QwenTtsHandler, _FakeResultHandler, "List[Dict[str, Any]]"]:
    """构造 handler，让 fake bundle_loader 按 ``model_name`` 切换三种 bundle。

    bundle_loader 同时记录调用序列，用例据此断言权重切换顺序——这是唯一能在
    单元测试里证明"同时刻只持一份权重"的方式（真实 LRU 由 PipelineCache 保证）。
    """
    result_handler = _FakeResultHandler()
    custom_voice_model = custom_voice_model or _FakeCustomVoiceModel()

    design_bundle = {
        "model": design_model,
        "model_ref": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "capabilities": {"voice_design": True, "create_voice_clone_prompt": False, "voice_clone": False, "custom_voice": False},
        "sample_rate": design_model.sample_rate,
        "supported_languages": ["Chinese", "English"],
        "supported_speakers": ["Vivian", "Serena"],
        "streaming_variant": False,
    }
    custom_voice_bundle = {
        "model": custom_voice_model,
        "model_ref": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "capabilities": {"voice_design": False, "create_voice_clone_prompt": False, "voice_clone": False, "custom_voice": True},
        "sample_rate": custom_voice_model.sample_rate,
        "supported_languages": ["Chinese", "English"],
        "supported_speakers": ["Vivian", "Serena"],
        "streaming_variant": False,
    }
    base_bundle = {
        "model": base_model,
        "model_ref": "Qwen3-TTS-12Hz-1.7B-Base",
        "capabilities": {"voice_design": False, "create_voice_clone_prompt": True, "voice_clone": True, "custom_voice": False},
        "sample_rate": base_model.sample_rate,
        "supported_languages": ["Chinese", "English"],
        "supported_speakers": ["Vivian", "Serena"],
        "streaming_variant": False,
    }

    bundle_load_log: List[Dict[str, Any]] = []

    async def _bundle_loader(mode: str, *, model_name=None) -> Dict[str, Any]:
        bundle_load_log.append({"mode": mode, "model_name": model_name})
        name = str(model_name or "")
        if "Base" in name:
            return base_bundle
        if "CustomVoice" in name:
            return custom_voice_bundle
        return design_bundle

    async def _stream_sender(_event: Dict[str, Any]) -> bool:  # pragma: no cover
        return True

    handler = QwenTtsHandler(
        audio_mode="tts",
        result_handler=result_handler,  # type: ignore[arg-type]
        service_id="svc-test",
        bundle_loader=_bundle_loader,
        stream_sender=_stream_sender,
        check_cancelled=None,
    )
    return handler, result_handler, bundle_load_log


def _make_params(*, characters, dialogues, model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
    return SimpleNamespace(
        type="audio",
        stream=False,
        file_type="wav",
        prompt="",
        model_name=model_name,
        model_cfg={},
        drama={"characters": characters, "dialogues": dialogues},
    )


# =====================  drama 主流程  =====================


def test_qwen_drama_voice_design_only_loads_design_then_base():
    """只有 voice_design 角色：bundle 序列必须是 VoiceDesign → Base，全程持 1 份权重。"""

    design_model = _FakeDesignModel()
    base_model = _FakeBaseModel()
    handler, result_handler, bundle_load_log = _build_handler(
        design_model=design_model, base_model=base_model
    )
    params = _make_params(
        characters=[
            {"id": "A", "name": "小赵", "voice_mode": "voice_design", "instruct": "温柔女声", "language": "Chinese"},
            {"id": "B", "name": "Bob", "voice_mode": "voice_design", "instruct": "deep male voice", "language": "English"},
        ],
        dialogues=[
            {"speaker_id": "A", "text": "你好啊", "pause_after_ms": 900},
            {"speaker_id": "B", "text": "Hi there", "pause_after_ms": 900},
            {"speaker_id": "A", "text": "再见", "pause_after_ms": 0},
        ],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t-vd-only",
            voice_cfg=VoiceConfig(
                tts_mode="voice_design",
                model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                clone_base_model_name="Qwen3-TTS-12Hz-1.7B-Base",
            ),
            started_at=0.0,
        )
    )

    # design 阶段：每个 voice_design 角色一次 generate_voice_design，instruct 透传
    assert len(design_model.calls) == 2
    assert design_model.calls[0]["instruct"] == "温柔女声"
    assert design_model.calls[1]["instruct"] == "deep male voice"

    # 每个 character 对应一次 voice_clone_prompt（Base 层）
    assert len(base_model.prompt_calls) == 2
    seed_a_audio, seed_a_sr = base_model.prompt_calls[0]["ref_audio"]
    assert seed_a_sr == design_model.sample_rate
    assert seed_a_audio.shape == (design_model.sample_rate,)

    # clone 阶段：A 的两段 dialogue 因 pause=900ms 不会合并；B 一次
    assert len(base_model.clone_calls) == 3
    assert base_model.clone_calls[0]["voice_clone_prompt"] == {"prompt_id": 1}
    assert base_model.clone_calls[1]["voice_clone_prompt"] == {"prompt_id": 2}
    # A 的第二次 clone 必须复用 A 的 prompt（id=1），不能拿 B 的（id=2）
    assert base_model.clone_calls[2]["voice_clone_prompt"] == {"prompt_id": 1}

    # 必须落盘一次
    assert len(result_handler.calls) == 1
    assert "file_data" in result_handler.calls[0]

    # bundle_loader 序列：VoiceDesign（design 阶段）→ Base（clone 阶段），共 2 次。
    # 每次切换由 PipelineCache LRU=1 触发对前一份权重的 release_fn → VRAM 收回。
    assert len(bundle_load_log) == 2
    assert bundle_load_log[0]["model_name"] == "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert bundle_load_log[1]["model_name"] == "Qwen3-TTS-12Hz-1.7B-Base"


def test_qwen_drama_custom_voice_only_uses_custom_voice_weight():
    """只有 custom_voice 角色：bundle 序列必须是 CustomVoice → Base，不再走 VoiceDesign。

    覆盖用户的核心诉求："custom_voice + 预置说话人"应该走 CustomVoice 权重直接产 seed，
    不再像旧代码那样从 speaker meta 派生 instruct 再丢给 VoiceDesign。
    """

    design_model = _FakeDesignModel()
    custom_voice_model = _FakeCustomVoiceModel()
    base_model = _FakeBaseModel()
    handler, _, bundle_load_log = _build_handler(
        design_model=design_model,
        base_model=base_model,
        custom_voice_model=custom_voice_model,
    )
    params = _make_params(
        characters=[
            {"id": "v", "name": "薇薇安", "voice_mode": "custom_voice", "speaker_name": "Vivian"},
            {"id": "s", "name": "塞蕾娜", "voice_mode": "custom_voice", "speaker_name": "Serena"},
        ],
        dialogues=[
            {"speaker_id": "v", "text": "测试一下"},
            {"speaker_id": "s", "text": "你好"},
        ],
        model_name="",  # 让 voice_cfg.model_name 为空，验证默认权重选择
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t-cv-only",
            voice_cfg=VoiceConfig(
                tts_mode="custom_voice",
                clone_base_model_name="Qwen3-TTS-12Hz-1.7B-Base",
            ),
            started_at=0.0,
        )
    )

    # 关键断言：design 模型完全没被调用；CustomVoice 模型每个 character 各一次
    assert design_model.calls == []
    assert len(custom_voice_model.calls) == 2
    assert custom_voice_model.calls[0]["speaker"] == "Vivian"
    assert custom_voice_model.calls[1]["speaker"] == "Serena"

    # bundle 序列：CustomVoice → Base
    assert [item["model_name"] for item in bundle_load_log] == [
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-Base",
    ]


def test_qwen_drama_mixed_groups_loads_three_weights_in_order():
    """voice_design 角色 + custom_voice 角色混合：bundle 序列必须是
    VoiceDesign → CustomVoice → Base。三次 swap 之间 LRU 各 evict 一份权重，
    GPU 上同时刻最多 1 份。
    """

    design_model = _FakeDesignModel()
    custom_voice_model = _FakeCustomVoiceModel()
    base_model = _FakeBaseModel()
    handler, _, bundle_load_log = _build_handler(
        design_model=design_model,
        base_model=base_model,
        custom_voice_model=custom_voice_model,
    )
    params = _make_params(
        characters=[
            {"id": "a", "name": "甲", "voice_mode": "voice_design", "instruct": "沧桑男声"},
            {"id": "b", "name": "乙", "voice_mode": "custom_voice", "speaker_name": "Vivian"},
        ],
        dialogues=[
            {"speaker_id": "a", "text": "我是甲"},
            {"speaker_id": "b", "text": "我是乙"},
        ],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t-mixed",
            voice_cfg=VoiceConfig(
                tts_mode="voice_design",
                model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                clone_base_model_name="Qwen3-TTS-12Hz-1.7B-Base",
            ),
            started_at=0.0,
        )
    )

    # design 给甲一次；custom_voice 给乙一次
    assert len(design_model.calls) == 1
    assert design_model.calls[0]["instruct"] == "沧桑男声"
    assert len(custom_voice_model.calls) == 1
    assert custom_voice_model.calls[0]["speaker"] == "Vivian"

    # 两个 character 各自的 voice_clone_prompt + dialogue clone
    assert len(base_model.prompt_calls) == 2
    assert len(base_model.clone_calls) == 2

    # bundle 序列：VoiceDesign → CustomVoice → Base，3 次 swap。
    assert [item["model_name"] for item in bundle_load_log] == [
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-Base",
    ]


def test_qwen_drama_batches_consecutive_same_character_lines():
    """同 character 紧邻多句（pause < 800ms）会合并为单次 batch ``generate_voice_clone``。"""

    design_model = _FakeDesignModel()
    base_model = _FakeBaseModel()
    handler, _, _ = _build_handler(design_model=design_model, base_model=base_model)
    params = _make_params(
        characters=[
            {"id": "A", "name": "小赵", "voice_mode": "voice_design", "instruct": "温柔女声", "language": "Chinese"},
        ],
        dialogues=[
            {"speaker_id": "A", "text": "第一句", "pause_after_ms": 300},
            {"speaker_id": "A", "text": "第二句", "pause_after_ms": 300},
            {"speaker_id": "A", "text": "第三句", "pause_after_ms": 0},
        ],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t-batch",
            voice_cfg=VoiceConfig(
                tts_mode="voice_design",
                model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                clone_base_model_name="Qwen3-TTS-12Hz-1.7B-Base",
            ),
            started_at=0.0,
        )
    )

    assert len(base_model.clone_calls) == 1
    call = base_model.clone_calls[0]
    assert call["text"] == ["第一句", "第二句", "第三句"]
    assert call["language"] == ["Chinese", "Chinese", "Chinese"]
    assert call["voice_clone_prompt"] == {"prompt_id": 1}


def test_qwen_drama_loads_base_only_after_all_designs_complete():
    """显存切换断言：所有 design 调用必须发生在 Base 权重加载之前。"""

    design_model = _FakeDesignModel()
    base_model = _FakeBaseModel()
    handler, _, bundle_load_log = _build_handler(
        design_model=design_model, base_model=base_model
    )

    design_call_indices_at_call_time: List[int] = []
    original_design = design_model.generate_voice_design

    def _spy_design(**kwargs):
        # 记录"调用 design 时已有几次 bundle 加载"，借此证明 design 全部发生在
        # Base 加载之前（否则索引会出现 == 2）。
        design_call_indices_at_call_time.append(len(bundle_load_log))
        return original_design(**kwargs)

    design_model.generate_voice_design = _spy_design  # type: ignore[assignment]

    params = _make_params(
        characters=[
            {"id": "A", "name": "甲", "voice_mode": "voice_design", "instruct": "女声"},
            {"id": "B", "name": "乙", "voice_mode": "voice_design", "instruct": "男声"},
            {"id": "C", "name": "丙", "voice_mode": "voice_design", "instruct": "童声"},
        ],
        dialogues=[
            {"speaker_id": "A", "text": "句一"},
            {"speaker_id": "B", "text": "句二"},
            {"speaker_id": "C", "text": "句三"},
        ],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t-swap-order",
            voice_cfg=VoiceConfig(
                tts_mode="voice_design",
                model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                clone_base_model_name="Qwen3-TTS-12Hz-1.7B-Base",
            ),
            started_at=0.0,
        )
    )

    assert design_call_indices_at_call_time == [1, 1, 1]
    assert [item["model_name"] for item in bundle_load_log] == [
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen3-TTS-12Hz-1.7B-Base",
    ]


def test_qwen_drama_defaults_clone_base_when_voice_cfg_missing_field():
    """缺省 ``clone_base_model_name`` 时，handler 自带 Qwen3-TTS-12Hz-1.7B-Base 兜底。

    覆盖"工具层不再补默认，全部下沉到 handler"的新边界——session/cli 路径不经过
    audio_drama_tts 工具层时也得有可用的 Base 名。
    """

    design_model = _FakeDesignModel()
    base_model = _FakeBaseModel()
    handler, _, bundle_load_log = _build_handler(
        design_model=design_model, base_model=base_model
    )
    params = _make_params(
        characters=[
            {"id": "A", "name": "甲", "voice_mode": "voice_design", "instruct": "温柔女声"},
        ],
        dialogues=[{"speaker_id": "A", "text": "你好"}],
    )

    asyncio.run(
        handler._run_drama(  # type: ignore[arg-type]
            params=params,
            task_id="t-default-base",
            voice_cfg=VoiceConfig(
                tts_mode="voice_design",
                model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                clone_base_model_name=None,  # ← 关键：模拟未传字段
            ),
            started_at=0.0,
        )
    )

    assert bundle_load_log[-1]["model_name"] == "Qwen3-TTS-12Hz-1.7B-Base"


def test_qwen_drama_rejects_character_without_voice_definition():
    """character 既没有 instruct 也没有 speaker_name → 在分组阶段就 ValueError，
    不会发生任何 bundle 加载。
    """

    design_model = _FakeDesignModel()
    custom_voice_model = _FakeCustomVoiceModel()
    base_model = _FakeBaseModel()
    handler, _, bundle_load_log = _build_handler(
        design_model=design_model,
        base_model=base_model,
        custom_voice_model=custom_voice_model,
    )
    params = _make_params(
        characters=[
            {"id": "x", "name": "未知", "voice_mode": "", "speaker_name": "", "instruct": ""},
        ],
        dialogues=[{"speaker_id": "x", "text": "should fail"}],
    )

    raised = None
    try:
        asyncio.run(
            handler._run_drama(  # type: ignore[arg-type]
                params=params,
                task_id="t-bad",
                voice_cfg=VoiceConfig(
                    tts_mode="custom_voice",
                    clone_base_model_name="Qwen3-TTS-12Hz-1.7B-Base",
                ),
                started_at=0.0,
            )
        )
    except ValueError as exc:
        raised = exc
    assert raised is not None
    assert "instruct" in str(raised) and "speaker_name" in str(raised)
    assert design_model.calls == []
    assert custom_voice_model.calls == []
    assert base_model.clone_calls == []
    # 在分组阶段就报错，bundle_loader 一次都不应被调
    assert bundle_load_log == []


# =====================  engine 单段合成的"按 voice_cfg 选权重"规则  =====================
#
# task 与 session 两条路径共用 ``QwenTtsEngine._resolve_weight_and_mode``：
# voice_cfg.model_name 显式 → 用之；否则按 tts_mode / instruct 选默认权重并对齐 mode。


def _engine() -> QwenTtsEngine:
    async def _loader(_mode, *, model_name=None):  # pragma: no cover
        return {}

    import logging

    return QwenTtsEngine(bundle_loader=_loader, logger=logging.getLogger("test"))


def test_engine_resolves_explicit_model_name_passthrough():
    weight, mode = _engine()._resolve_weight_and_mode(
        VoiceConfig(model_name="Qwen3-TTS-12Hz-0.6B-Base", tts_mode="voice_clone")
    )
    assert weight == "Qwen3-TTS-12Hz-0.6B-Base"
    assert mode == "voice_clone"


def test_engine_resolves_default_by_tts_mode_when_explicit_model_empty():
    weight, mode = _engine()._resolve_weight_and_mode(
        VoiceConfig(tts_mode="voice_design", instruct="温柔女声")
    )
    assert weight == DEFAULT_QWEN_VOICE_DESIGN_MODEL and mode == "voice_design"

    weight, mode = _engine()._resolve_weight_and_mode(VoiceConfig(tts_mode="voice_clone"))
    assert weight == DEFAULT_QWEN_BASE_MODEL and mode == "voice_clone"


def test_engine_upgrades_default_mode_to_voice_design_when_instruct_present():
    """tts_mode 是默认 'custom_voice' + 有 instruct → 整体升档到 voice_design，
    避免 instruct 在 CustomVoice 路径里被降格成 tone tweak。"""
    weight, mode = _engine()._resolve_weight_and_mode(
        VoiceConfig(tts_mode="custom_voice", instruct="温柔女声")
    )
    assert weight == DEFAULT_QWEN_VOICE_DESIGN_MODEL
    assert mode == "voice_design"


def test_engine_resolves_custom_voice_when_no_instruct():
    weight, mode = _engine()._resolve_weight_and_mode(VoiceConfig(tts_mode="custom_voice"))
    assert weight == DEFAULT_QWEN_CUSTOM_VOICE_MODEL and mode == "custom_voice"


# =====================  audio_drama_tts.py 工具层 family normalization  =====================


def test_audio_drama_tts_normalize_keeps_qwen_speaker_for_qwen_family():
    """family=qwen 时，``custom_voice + Vivian`` 原样保留，让 qwen handler 走 CustomVoice 链路。"""

    from backend.services.agent.tools.builtin.audio_drama_tts import (
        _normalize_characters,
        _resolve_target_family,
    )

    family = _resolve_target_family("Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    assert family == "qwen"

    normalized = _normalize_characters(
        [
            {
                "id": "v",
                "name": "薇薇安",
                "voice_mode": "custom_voice",
                "speaker_name": "Vivian",
                "instruct": "",
            }
        ],
        target_family=family,
    )
    assert len(normalized) == 1
    char = normalized[0]
    assert char["voice_mode"] == "custom_voice"
    assert char["speaker_name"] == "Vivian"


def test_audio_drama_tts_normalize_converts_qwen_speaker_for_voxcpm_family():
    """family=voxcpm 时，``Vivian`` 这种 qwen 预置仍按旧链路转 voice_design + 派生 instruct。"""

    from backend.services.agent.tools.builtin.audio_drama_tts import (
        _normalize_characters,
        _resolve_target_family,
    )

    family = _resolve_target_family("VoxCPM2")
    assert family == "voxcpm"

    normalized = _normalize_characters(
        [
            {
                "id": "v",
                "name": "薇薇安",
                "voice_mode": "custom_voice",
                "speaker_name": "Vivian",
                "instruct": "",
            }
        ],
        target_family=family,
    )
    assert len(normalized) == 1
    char = normalized[0]
    assert char["voice_mode"] == "voice_design"
    assert char["speaker_name"] is None
    assert "Vivian" in (char["instruct"] or "")


def test_audio_drama_tts_voice_source_validates_against_family():
    """family=qwen 路径下用 voxcpm 预置（如 luoli）必须报 unknown speaker，不能放过。"""

    from backend.services.agent.tools.builtin.audio_drama_tts import (
        _character_voice_source_error,
    )

    err = _character_voice_source_error(
        [
            {
                "id": "x",
                "name": "X",
                "voice_mode": "custom_voice",
                "speaker_name": "luoli",
                "instruct": "",
            }
        ],
        target_family="qwen",
    )
    assert err is not None
    assert "luoli" in err
    assert "qwen" in err

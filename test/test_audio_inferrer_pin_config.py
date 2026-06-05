"""``AudioInferrer`` 的 fixed_model / fixed_family 三态语义单测。

聚焦"仅 fixed_family"模式（用户的 voice chat 场景）：
- 启动校验放过：``fixed_model`` 空 + ``fixed_family`` 非空合法；
- ``_coerce_params_for_pin``：缺 ``family`` 时回填，不动 ``model_name``；
- pin 模式（两者都设）的硬覆盖语义不被破坏。

为了避开 AudioInferrer 真实 __init__（要 load_inference_config / 起 ws / 起 cache），
这里用 ``__new__`` 跳过 init，只手动把校验/coerce 用到的字段塞进去。
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))


def _ensure_video_runtime_stub() -> None:
    """audio.inferrer 通过 voxcpm/qwen_asr handler 间接 import video.runtime.io_utils；
    后者所属的 video/__init__.py 会拉 diffsynth → einops（pytest 环境无 einops）。
    在 import audio.inferrer 之前把 video 系列包桩成空 module，避免触发真包加载。
    """
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

import pytest  # noqa: E402

from audio.inferrer import AudioInferrer  # noqa: E402


def _make_inferrer(
    *,
    fixed_model: str | None = None,
    fixed_family: str | None = None,
    supported_models: list[str] | None = None,
) -> AudioInferrer:
    """绕过真实 __init__，只设置校验/coerce 关心的字段。"""
    inf = AudioInferrer.__new__(AudioInferrer)
    inf.service_id = "svc-test"
    inf._fixed_model = fixed_model
    inf._fixed_family = fixed_family
    inf._supported_models = list(supported_models or [])
    return inf


# =====================  _enforce_fixed_model_consistency  =====================


def test_consistency_both_empty_is_legal():
    inf = _make_inferrer()
    inf._enforce_fixed_model_consistency()  # 不抛 = 通过


def test_consistency_only_fixed_family_is_legal():
    """用户的 voice chat 场景：仅声明服务的默认 family，不 pin 具体权重。"""
    inf = _make_inferrer(
        fixed_family="Qwen-tts",
        supported_models=["Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
    )
    inf._enforce_fixed_model_consistency()


def test_consistency_only_fixed_model_is_illegal():
    """仅设 fixed_model 不设 fixed_family——pin 模式没法决定 runtime，应在启动期报错。"""
    inf = _make_inferrer(
        fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
        supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
    )
    with pytest.raises(RuntimeError, match="requires fixed_family"):
        inf._enforce_fixed_model_consistency()


def test_consistency_pin_mode_requires_fixed_model_in_supported():
    inf = _make_inferrer(
        fixed_model="Qwen3-TTS-NotInList",
        fixed_family="Qwen-tts",
        supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
    )
    with pytest.raises(RuntimeError, match="not listed in supported_models"):
        inf._enforce_fixed_model_consistency()


def test_consistency_pin_mode_with_listed_fixed_model_passes():
    inf = _make_inferrer(
        fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
        fixed_family="Qwen-tts",
        supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
    )
    inf._enforce_fixed_model_consistency()


# =====================  _coerce_params_for_pin  =====================


def test_coerce_no_config_returns_params_untouched():
    inf = _make_inferrer()
    params = SimpleNamespace(model_name="user-model", family="user-class")
    out = inf._coerce_params_for_pin(params)
    assert out is params  # 同一对象，未拷贝


def test_coerce_only_class_fills_missing_family():
    """仅 fixed_family 模式：调用方没传 family → 用服务默认补上。"""
    inf = _make_inferrer(fixed_family="Qwen-tts")
    params = SimpleNamespace(model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign", family="")
    out = inf._coerce_params_for_pin(params)
    assert out is not params  # 拷贝后返回，避免污染原 params
    assert out.family == "Qwen-tts"
    # model_name 不被动——pin 没开，调用方/handler 默认规则继续掌控
    assert out.model_name == "Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def test_coerce_only_class_keeps_caller_model_name_when_empty():
    """仅 fixed_family 模式 + 调用方 model_name 也空：保留空——交给下游 handler 默认规则。"""
    inf = _make_inferrer(fixed_family="Qwen-tts")
    params = SimpleNamespace(model_name="", family="")
    out = inf._coerce_params_for_pin(params)
    assert out.family == "Qwen-tts"
    assert out.model_name == ""


def test_coerce_only_class_does_not_override_explicit_family():
    """仅 fixed_family 模式：调用方显式传了 family → 保留原值，不强制覆盖。

    这条规则保证调用方仍能跨 class 调用（虽然实际能不能跑得通要看 supported_models）。
    """
    inf = _make_inferrer(fixed_family="Qwen-tts")
    params = SimpleNamespace(model_name="weird-name", family="Voxcpm")
    out = inf._coerce_params_for_pin(params)
    assert out is params  # 没改任何字段，直接返回原对象
    assert out.family == "Voxcpm"


def test_coerce_pin_mode_hard_overrides_both_fields():
    """pin 模式：调用方传啥都被 fixed_model / fixed_family 覆盖——这是 pin 的承诺。"""
    inf = _make_inferrer(
        fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
        fixed_family="Qwen-tts",
    )
    params = SimpleNamespace(
        model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        family="Qwen-tts",
    )
    out = inf._coerce_params_for_pin(params)
    assert out.model_name == "Qwen3-TTS-12Hz-0.6B-Base"
    assert out.family == "Qwen-tts"


def test_coerce_pin_mode_no_op_when_already_correct():
    """pin 模式 + 调用方已经传了正确值：不必重新拷贝。"""
    inf = _make_inferrer(
        fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
        fixed_family="Qwen-tts",
    )
    params = SimpleNamespace(
        model_name="Qwen3-TTS-12Hz-0.6B-Base",
        family="Qwen-tts",
    )
    out = inf._coerce_params_for_pin(params)
    assert out is params

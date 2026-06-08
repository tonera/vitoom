"""``InferenceServiceManager.sync_service_registration`` 的 audio 服务注册校验单测。

聚焦 audio 服务的 fixed_model / fixed_family 三态语义——跟
``inference/audio/inferrer.py::_enforce_fixed_model_consistency`` 严格对齐：
- 都为空：合法；
- 仅 ``fixed_family``：合法（声明服务级默认 family，不 pin）；
- 都设置：pin 模式合法；
- 仅 ``fixed_model``：非法（pin 模式必须知道 runtime）。

外加：仅 ``fixed_family`` 模式下要把 class 真的写到 db.config，
不能像旧实现那样跟 fixed_model 一起被 pop 掉，否则 dispatch 看不到默认 class。
"""

import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from backend.services.inference import service as svc_module  # noqa: E402
from backend.services.inference.service import InferenceServiceManager  # noqa: E402


class _FakeInferenceServiceTable:
    """最小可工作的 InferenceService 桩。"""

    def __init__(self, initial: Dict[str, Any]):
        self.row = dict(initial)
        self.last_update_kwargs: Dict[str, Any] | None = None

    def get_by_id(self, service_id: str) -> Dict[str, Any] | None:
        if service_id != self.row["id"]:
            return None
        return dict(self.row)

    def update(self, service_id: str, **kwargs) -> Dict[str, Any] | None:
        if service_id != self.row["id"]:
            return None
        self.last_update_kwargs = dict(kwargs)
        for key, value in kwargs.items():
            self.row[key] = value
        return dict(self.row)


def _patch_table(monkeypatch: pytest.MonkeyPatch, fake: _FakeInferenceServiceTable) -> None:
    monkeypatch.setattr(svc_module.InferenceService, "get_by_id", fake.get_by_id)
    monkeypatch.setattr(svc_module.InferenceService, "update", fake.update)


def _audio_service_row(service_id: str = "service_qwen_tts") -> Dict[str, Any]:
    return {
        "id": service_id,
        "service_type": "audio",
        "config": {},
    }


# =====================  三态合法性  =====================


def test_register_no_pin_no_default_class_is_legal(monkeypatch):
    """都不设：完全无 pin、无默认 class。合法。"""

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    mgr.sync_service_registration(
        "service_qwen_tts",
        content_service_type="audio",
        supports_task=True,
        supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
        capabilities=["tts"],
        fixed_model=None,
        fixed_family=None,
    )
    cfg = fake.row["config"]
    assert cfg.get("supported_models") == ["Qwen3-TTS-12Hz-1.7B-VoiceDesign"]
    assert "fixed_model" not in cfg
    assert "fixed_family" not in cfg


def test_register_only_fixed_family_is_legal_and_persists(monkeypatch):
    """仅 fixed_family：必须合法，且要真的写入 db.config——这是用户的 voice chat 场景。

    旧实现在 ``if normalized_fixed_model:`` 分支把 fixed_family 也 pop 掉，
    会让 dispatch 完全看不到默认 class，inferrer 端的 fallback 形同虚设。本用例锁住
    "仅 class 也要持久化"的新契约。
    """

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    mgr.sync_service_registration(
        "service_qwen_tts",
        content_service_type="audio",
        supports_task=True,
        supported_models=[
            "Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ],
        capabilities=["tts"],
        fixed_model=None,
        fixed_family="Qwen-tts",
    )
    cfg = fake.row["config"]
    assert cfg.get("fixed_family") == "Qwen-tts"
    assert "fixed_model" not in cfg
    assert cfg.get("capabilities") == ["tts"]


def test_register_pin_mode_persists_both(monkeypatch):
    """pin 模式：fixed_model + fixed_family 同时写入。"""

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    mgr.sync_service_registration(
        "service_qwen_tts",
        content_service_type="audio",
        supports_task=True,
        supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
        capabilities=["tts"],
        fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
        fixed_family="Qwen-tts",
    )
    cfg = fake.row["config"]
    assert cfg.get("fixed_model") == "Qwen3-TTS-12Hz-0.6B-Base"
    assert cfg.get("fixed_family") == "Qwen-tts"


def test_register_only_fixed_model_is_illegal(monkeypatch):
    """仅 fixed_model 不带 fixed_family——pin 模式必须知道 runtime，注册期就要拒绝。"""

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    with pytest.raises(ValueError, match="requires fixed_family"):
        mgr.sync_service_registration(
            "service_qwen_tts",
            content_service_type="audio",
            supports_task=True,
            supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
            capabilities=["tts"],
            fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
            fixed_family=None,
        )


def test_register_pin_model_must_be_in_supported_models(monkeypatch):
    """pin 模式下 fixed_model 必须出现在 supported_models 中——这是配置一致性闸。"""

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    with pytest.raises(ValueError, match="not listed in supported_models"):
        mgr.sync_service_registration(
            "service_qwen_tts",
            content_service_type="audio",
            supports_task=True,
            supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
            capabilities=["tts"],
            fixed_model="Qwen3-TTS-12Hz-NotListed",
            fixed_family="Qwen-tts",
        )


def test_register_audio_requires_supported_models(monkeypatch):
    """audio 服务必填 supported_models——与原行为保持。"""

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    with pytest.raises(ValueError, match="non-empty supported_models"):
        mgr.sync_service_registration(
            "service_qwen_tts",
            content_service_type="audio",
            supports_task=True,
            supported_models=[],
            capabilities=["tts"],
        )


def test_register_audio_requires_capabilities(monkeypatch):
    """audio 服务必填 capabilities——与原行为保持。"""

    fake = _FakeInferenceServiceTable(_audio_service_row())
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    with pytest.raises(ValueError, match="non-empty capabilities"):
        mgr.sync_service_registration(
            "service_qwen_tts",
            content_service_type="audio",
            supports_task=True,
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=[],
        )


def test_register_only_class_clears_stale_fixed_model_in_db(monkeypatch):
    """切换：原本是 pin 模式的服务，重启时改成"仅 class"——db 里的旧 fixed_model 必须被清掉，
    fixed_family 保留。否则 dispatch 仍按 pin 收窄，行为不一致。
    """

    initial = _audio_service_row()
    initial["config"] = {
        "fixed_model": "Qwen3-TTS-12Hz-0.6B-Base",
        "fixed_family": "Qwen-tts",
        "supported_models": ["Qwen3-TTS-12Hz-0.6B-Base"],
        "capabilities": ["tts"],
    }
    fake = _FakeInferenceServiceTable(initial)
    _patch_table(monkeypatch, fake)
    mgr = InferenceServiceManager()

    mgr.sync_service_registration(
        "service_qwen_tts",
        content_service_type="audio",
        supports_task=True,
        supported_models=[
            "Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ],
        capabilities=["tts"],
        fixed_model=None,
        fixed_family="Qwen-tts",
    )
    cfg = fake.row["config"]
    assert "fixed_model" not in cfg
    assert cfg.get("fixed_family") == "Qwen-tts"

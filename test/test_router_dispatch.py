"""``DispatchRouter.list_services`` 的关键路径单测。

聚焦 TTS 路径在用户新需求下的"放宽匹配"行为：

- ``model_name`` 命中时优先返回严格匹配的服务（preserve 旧行为）；
- ``model_name`` 不命中时回落到任意 tts-capable 在线服务（用户的核心诉求：
  "后端不用管哪个 tts 模型在运行，哪个在线就用哪个"）；
- ``model_name`` 为空时优先 pinned，否则回落任意 tts-capable；
- ASR 路径仍保持严格匹配，不受放宽影响。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from backend.services.chat import router as router_module
from backend.services.chat.router import DispatchRouter, DispatchSpec


def _svc(
    *,
    service_id: str,
    service_type: str = "audio",
    status: str = "running",
    supports_task: bool = True,
    supported_models: List[str] | None = None,
    fixed_model: str | None = None,
    capabilities: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "id": service_id,
        "service_type": service_type,
        "status": status,
        "supports_task": supports_task,
        "config": {
            "supported_models": list(supported_models or []),
            "fixed_model": fixed_model,
            "capabilities": list(capabilities or []),
        },
    }


def _patch_services(monkeypatch: pytest.MonkeyPatch, services: List[Dict[str, Any]]) -> None:
    monkeypatch.setattr(
        router_module.InferenceService,
        "list_all",
        staticmethod(lambda: list(services)),
    )


def test_dispatch_tts_strict_match_when_model_name_served(monkeypatch):
    """``model_name`` 被服务声明时，仍走严格匹配；放宽逻辑不会回落。"""

    services = [
        _svc(
            service_id="qwen-tts",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
        ),
        _svc(
            service_id="voxcpm",
            supported_models=["VoxCPM2"],
            capabilities=["tts"],
        ),
    ]
    _patch_services(monkeypatch, services)

    router = DispatchRouter()
    spec = DispatchSpec(
        service_type="audio",
        model_name="VoxCPM2",
        capability="tts",
        require_supports_task=True,
    )
    selected = router.list_services(spec, connected_service_ids=["qwen-tts", "voxcpm"])
    assert [s["id"] for s in selected] == ["voxcpm"]


def test_dispatch_tts_loose_fallback_when_model_name_not_served(monkeypatch):
    """``model_name`` 没人声明时回落到任意 tts-capable 服务——用户的核心诉求。

    场景：LLM 误填 ``VoxCPM2`` 但本机只有 qwen-tts 在线。旧行为是直接抛
    DispatchSelectionError；新行为应该返回 qwen-tts，让推理侧 handler 自己决定权重。
    """

    services = [
        _svc(
            service_id="qwen-tts",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
        ),
    ]
    _patch_services(monkeypatch, services)

    router = DispatchRouter()
    spec = DispatchSpec(
        service_type="audio",
        model_name="VoxCPM2",
        capability="tts",
        require_supports_task=True,
    )
    selected = router.list_services(spec, connected_service_ids=["qwen-tts"])
    assert [s["id"] for s in selected] == ["qwen-tts"]


def test_dispatch_tts_loose_fallback_when_model_name_empty_and_no_pinned(monkeypatch):
    """``model_name`` 为空且没有 pinned 服务时，也回落到任意 tts-capable 服务。

    旧行为要求至少有一个服务声明 ``fixed_model`` 才接收空 model_name 请求；新行为下
    handler 自己有"按 instruct 选权重"的默认规则，dispatch 不再卡这一道。
    """

    services = [
        _svc(
            service_id="qwen-tts",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
        ),
    ]
    _patch_services(monkeypatch, services)

    router = DispatchRouter()
    spec = DispatchSpec(
        service_type="audio",
        model_name="",
        capability="tts",
        require_supports_task=True,
    )
    selected = router.list_services(spec, connected_service_ids=["qwen-tts"])
    assert [s["id"] for s in selected] == ["qwen-tts"]


def test_dispatch_tts_empty_model_prefers_pinned_over_others(monkeypatch):
    """``model_name`` 空 + 多服务在线时，优先返回 pinned 服务（保留对 pin 的"默认偏好"）。"""

    services = [
        _svc(
            service_id="qwen-pinned",
            supported_models=["Qwen3-TTS-12Hz-0.6B-Base"],
            fixed_model="Qwen3-TTS-12Hz-0.6B-Base",
            capabilities=["tts"],
        ),
        _svc(
            service_id="qwen-free",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
        ),
    ]
    _patch_services(monkeypatch, services)

    router = DispatchRouter()
    spec = DispatchSpec(
        service_type="audio",
        model_name="",
        capability="tts",
        require_supports_task=True,
    )
    selected = router.list_services(
        spec, connected_service_ids=["qwen-pinned", "qwen-free"]
    )
    assert [s["id"] for s in selected] == ["qwen-pinned"]


def test_dispatch_asr_keeps_strict_matching(monkeypatch):
    """ASR 路径不走放宽：``model_name`` 不命中时返回空，让上游报错。"""

    services = [
        _svc(
            service_id="qwen-asr",
            supported_models=["Qwen3-ASR-12Hz-3B"],
            capabilities=["asr"],
        ),
    ]
    _patch_services(monkeypatch, services)

    router = DispatchRouter()
    spec = DispatchSpec(
        service_type="audio",
        model_name="totally-unknown-asr-model",
        capability="asr",
        require_supports_task=True,
    )
    selected = router.list_services(spec, connected_service_ids=["qwen-asr"])
    assert selected == []


def test_dispatch_tts_filters_by_capability_and_status(monkeypatch):
    """放宽 tts 兜底前仍要先过 capability + status + supports_task 三道闸——
    不能把 ASR 服务、stopped 服务、不接 task 的服务误判进 fallback。
    """

    services = [
        _svc(
            service_id="qwen-asr",
            supported_models=["Qwen3-ASR-12Hz-3B"],
            capabilities=["asr"],
        ),
        _svc(
            service_id="qwen-tts-stopped",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
            status="stopped",
        ),
        _svc(
            service_id="qwen-tts-no-task",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
            supports_task=False,
        ),
        _svc(
            service_id="qwen-tts-ok",
            supported_models=["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
            capabilities=["tts"],
        ),
    ]
    _patch_services(monkeypatch, services)

    router = DispatchRouter()
    spec = DispatchSpec(
        service_type="audio",
        model_name="VoxCPM2",  # 谁都不声明，触发 loose fallback
        capability="tts",
        require_supports_task=True,
    )
    selected = router.list_services(
        spec,
        connected_service_ids=[
            "qwen-asr",
            "qwen-tts-stopped",
            "qwen-tts-no-task",
            "qwen-tts-ok",
        ],
    )
    assert [s["id"] for s in selected] == ["qwen-tts-ok"]

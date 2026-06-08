from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.api.chat.routes as chat_routes
from backend.api.chat.routes import ChatSessionCreateRequest, _build_session_metadata


def test_build_session_metadata_allows_empty_load_name_for_audio_input(monkeypatch):
    """新协议：audio 输入场景 load_name 缺省时不再兜底到任何默认模型。

    dispatch 会把"空 load_name"任务路由到声明了 fixed_model 的 pinned 服务，
    由推理侧做 load_name/family 覆盖。
    """
    monkeypatch.setattr(chat_routes.Model, "get_by_load_name", staticmethod(lambda _name: None))

    request = ChatSessionCreateRequest(
        input_mode="audio_stream",
        output_mode="text_stream",
        metadata={"source": "test"},
    )

    metadata = _build_session_metadata(request)

    assert metadata["input_mode"] == "audio_stream"
    assert metadata["output_mode"] == "text_stream"
    assert metadata["audio_input"] == {
        "load_name": "",
        "family": "",
        "runtime_config": {},
    }
    assert metadata["source"] == "test"
    assert "load_name" not in metadata


def test_build_session_metadata_resolves_audio_input_model_when_given(monkeypatch):
    """显式传 load_name 时，按 catalog 回填 family / runtime_config。"""
    monkeypatch.setattr(chat_routes.Model, "get_by_load_name", staticmethod(lambda name: {
        "load_name": name,
        "family": "Qwen-asr",
        "runtime_config": {"runtime": {"backend": "vllm"}},
    }))

    request = ChatSessionCreateRequest(
        input_mode="audio_stream",
        output_mode="text_stream",
        load_name="Qwen3-ASR-1.7B",
    )

    metadata = _build_session_metadata(request)

    assert metadata["audio_input"] == {
        "load_name": "Qwen3-ASR-1.7B",
        "family": "Qwen-asr",
        "runtime_config": {"runtime": {"backend": "vllm"}},
    }


def test_build_session_metadata_includes_audio_output_preferences(monkeypatch):
    monkeypatch.setattr(chat_routes.Model, "get_by_load_name", staticmethod(lambda _name: None))

    request = ChatSessionCreateRequest(
        input_mode="audio_stream",
        output_mode="multimodal_result",
        audio_output={
            "tts_mode": "custom_voice",
            "speaker_name": "vivian",
            "language": "zh",
            "sample_rate": 24000,
            "file_type": "wav",
        },
    )

    metadata = _build_session_metadata(request)

    assert metadata["audio_output"] == {
        "tts_mode": "custom_voice",
        "speaker_name": "vivian",
        "language": "zh",
        "sample_rate": 24000,
        "file_type": "wav",
    }

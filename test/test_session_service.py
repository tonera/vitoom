from pathlib import Path
import sys
import types
from datetime import timezone

sys.path.insert(0, str(Path(__file__).parent.parent))


class _FakeUTC:
    @staticmethod
    def localize(dt):
        return dt.replace(tzinfo=timezone.utc)


sys.modules.setdefault(
    "pytz",
    types.SimpleNamespace(
        timezone=lambda _name: timezone.utc,
        UTC=_FakeUTC(),
    ),
)

import pytest

session_service_module = pytest.importorskip(
    "backend.services.session.service",
    reason="legacy session service removed after unified chat refactor",
)
SessionServiceManager = session_service_module.SessionServiceManager


def test_create_text_chat_session_selects_text_service(monkeypatch):
    manager = SessionServiceManager()

    monkeypatch.setattr(
        session_service_module.InferenceService,
        "list_all",
        staticmethod(
            lambda: [
                {
                    "id": "svc-text-1",
                    "status": "running",
                    "service_type": "text",
                    "supports_session": True,
                    "capabilities": [
                        "session_bind",
                        "text_input",
                        "text_output",
                        "llm_stream",
                        "openai_compatible",
                    ],
                }
            ]
        ),
    )
    monkeypatch.setattr(session_service_module, "generate_uuid", lambda: "session-text-1")

    def fake_create(**kwargs):
        return {
            "id": kwargs["id"],
            "user_id": kwargs["user_id"],
            "scene": kwargs["scene"],
            "status": kwargs["status"],
            "bound_audio_service_id": kwargs.get("bound_audio_service_id"),
            "bound_audio_output_service_id": kwargs.get("bound_audio_output_service_id"),
            "bound_text_service_id": kwargs.get("bound_text_service_id"),
            "metadata": kwargs.get("metadata") or {},
        }

    monkeypatch.setattr(
        session_service_module.RealtimeSession,
        "create",
        staticmethod(fake_create),
    )

    session = manager.create_session(user_id="user-1", scene="text_chat", metadata={"source": "test"})

    assert session["id"] == "session-text-1"
    assert session["scene"] == "text_chat"
    assert session["bound_text_service_id"] == "svc-text-1"
    assert session["bound_audio_service_id"] is None


def test_create_realtime_asr_session_requires_matching_service(monkeypatch):
    manager = SessionServiceManager()

    monkeypatch.setattr(
        session_service_module.InferenceService,
        "list_all",
        staticmethod(lambda: []),
    )

    with pytest.raises(RuntimeError):
        manager.create_session(user_id="user-1", scene="realtime_asr", metadata={})


def test_get_target_service_ids_for_voice_chat():
    manager = SessionServiceManager()
    session = {
        "bound_audio_service_id": "svc-audio-in",
        "bound_audio_output_service_id": "svc-audio-out",
        "bound_text_service_id": "svc-text",
    }

    assert manager.get_target_service_ids(session, "input_audio_chunk") == ["svc-audio-in"]
    assert manager.get_target_service_ids(session, "input_text") == ["svc-text"]
    assert manager.get_target_service_ids(session, "interrupt") == [
        "svc-audio-in",
        "svc-audio-out",
        "svc-text",
    ]

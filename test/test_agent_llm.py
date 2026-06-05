import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.llm import build_crewai_llm, resolve_agent_llm_model_name


def test_resolve_agent_llm_model_name_falls_back_to_active_text_model():
    mock_service = Mock()
    mock_service.list_models.side_effect = [
        ([{"name": "QwenText", "full_name": "QwenText-OpenAI", "type": "text"}], 1),
    ]

    with (
        patch("backend.services.agent.llm.get_agent_llm_model_name", return_value=""),
        patch("backend.services.agent.llm.get_model_service", return_value=mock_service),
    ):
        model_name = resolve_agent_llm_model_name()

    assert model_name == "QwenText-OpenAI"


def test_build_crewai_llm_does_not_require_openai_env_key_when_config_is_explicit():
    class _FakeLLM:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    with (
        patch("backend.services.agent.llm.resolve_agent_llm_model_name", return_value="dummy-model"),
        patch("backend.services.agent.llm.get_agent_llm_base_url", return_value="http://127.0.0.1:8888/v1"),
        patch("backend.services.agent.llm.get_agent_internal_auth_token", return_value="dummy-token"),
        patch("backend.services.agent.llm.get_agent_llm_timeout_seconds", return_value=30.0),
        patch("backend.services.agent.llm.get_agent_effective_user_header_name", return_value="X-Vitoom-Effective-User-Id"),
        patch.dict(sys.modules, {"crewai": types.SimpleNamespace(LLM=_FakeLLM)}),
    ):
        llm = build_crewai_llm(effective_user_id="user-123")

    assert getattr(llm, "model", None) == "dummy-model"
    assert getattr(llm, "base_url", None) == "http://127.0.0.1:8888/v1"
    assert getattr(llm, "default_headers", None) == {"X-Vitoom-Effective-User-Id": "user-123"}
    assert getattr(llm, "stream", None) is True
    assert getattr(llm, "extra_body", None) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }

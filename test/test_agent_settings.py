import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent import settings


def test_get_agent_secret_reads_process_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "env-value")

    assert settings.get_tavily_api_key() == "env-value"


def test_get_agent_secret_returns_default_when_env_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert settings.get_tavily_api_key() == ""

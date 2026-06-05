from __future__ import annotations

from typing import Dict, List, Optional

from backend.database import Agent

from .settings import is_agents_enabled, is_openclaw_enabled
from .runtime import AgentValidationError


def _is_agent_available(agent: Dict) -> bool:
    if not is_agents_enabled():
        return False
    agent_type = str(agent.get("type") or "").strip().lower()
    if agent_type == "openclaw" and not is_openclaw_enabled():
        return False
    return True


def get_agent_or_raise(agent_id: str) -> Dict:
    agent = Agent.get_by_id(agent_id)
    if not agent or not _is_agent_available(agent):
        raise AgentValidationError("Agent not found")
    return agent


def list_agents(*, status: Optional[str] = None, is_preset: Optional[bool] = None) -> List[Dict]:
    if not is_agents_enabled():
        return []
    agents = Agent.list_all(status=status, is_preset=is_preset)
    return [agent for agent in agents if _is_agent_available(agent)]

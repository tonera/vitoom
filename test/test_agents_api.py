import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.auth import create_access_token
from backend.database import Agent, User, init_db
from backend.queue.queue import TaskQueue


def _create_user_and_agent():
    init_db()
    user_id = str(uuid.uuid4())
    email = f"agent_test_{uuid.uuid4().hex[:8]}@example.com"
    user = User.create(
        id=user_id,
        email=email,
        password_hash="hashed_password",
        nickname="Agent Test User",
    )
    assert user is not None

    agent_id = str(uuid.uuid4())
    agent = Agent.create(
        id=agent_id,
        name="Test Agent",
        agent_type="general",
        config={
            "agent": {
                "role": "Test Assistant",
                "goal": "Handle user requests",
                "backstory": "A test agent used for API verification.",
                "tools": [],
                "verbose": False,
            },
            "tasks": [
                {
                    "id": "task_main",
                    "description": "Complete the user's request: {message}",
                    "expected_output": "A concise markdown answer.",
                }
            ],
        },
        description="API test agent",
        status="active",
    )
    assert agent is not None
    return user_id, agent_id


def test_agents_runs_create_list_cancel():
    app = create_app(enable_static_files=False)
    user_id, agent_id = _create_user_and_agent()
    token = create_access_token({"sub": user_id, "email": "test@example.com"})
    queue = TaskQueue(max_workers=1)

    with (
        patch("backend.workers.startup_agent_runtime", new=AsyncMock()),
        patch("backend.workers.shutdown_agent_runtime", new=AsyncMock()),
        patch("backend.services.agent.runtime.get_task_queue", return_value=queue),
    ):
        client = TestClient(app)

        create_resp = client.post(
            "/v1/agents/runs",
            json={
                "agent_id": agent_id,
                "message": "帮我总结一下今天的待办",
                "runtime_config": {"priority": 7, "process": "sequential"},
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert create_resp.status_code == 201, create_resp.text
        create_data = create_resp.json()["data"]
        assert create_data["status"] == "queued"
        assert create_data["run_id"]
        assert create_data["task_id"]

        get_resp = client.get(
            f"/v1/agents/runs/{create_data['run_id']}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_resp.status_code == 200, get_resp.text
        get_data = get_resp.json()["data"]
        assert get_data["id"] == create_data["run_id"]
        assert get_data["task_id"] == create_data["task_id"]
        assert get_data["status"] == "queued"

        list_resp = client.get(
            "/v1/agents/runs",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 200, list_resp.text
        runs = list_resp.json()["data"]["runs"]
        assert any(item["id"] == create_data["run_id"] for item in runs)

        cancel_resp = client.post(
            f"/v1/agents/runs/{create_data['run_id']}/cancel",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert cancel_resp.status_code == 200, cancel_resp.text
        cancel_data = cancel_resp.json()["data"]
        assert cancel_data["status"] == "cancelled"


def test_channel_ingress_accepts_explicit_user_id_without_auth():
    app = create_app(enable_static_files=False)
    user_id, agent_id = _create_user_and_agent()
    queue = TaskQueue(max_workers=1)

    with (
        patch("backend.workers.startup_agent_runtime", new=AsyncMock()),
        patch("backend.workers.shutdown_agent_runtime", new=AsyncMock()),
        patch("backend.services.agent.runtime.get_task_queue", return_value=queue),
    ):
        client = TestClient(app)

        response = client.post(
            "/v1/channel-ingress",
            json={
                "user_id": user_id,
                "agent_id": agent_id,
                "message": "从hook进入的任务",
                "source_type": "hook",
                "source_ref": "msg-001",
            },
        )
        assert response.status_code == 202, response.text
        data = response.json()["data"]
        assert data["run_id"]
        assert data["task_id"]
        assert data["status"] == "queued"


def test_agents_list_exposes_default_preset():
    app = create_app(enable_static_files=False)
    init_db()
    user_id = str(uuid.uuid4())
    email = f"preset_test_{uuid.uuid4().hex[:8]}@example.com"
    user = User.create(
        id=user_id,
        email=email,
        password_hash="hashed_password",
        nickname="Preset Test User",
    )
    assert user is not None

    hidden_openclaw_agent = Agent.get_by_id("preset-openclaw-agent") or Agent.create(
        id="preset-openclaw-agent",
        name="OpenClaw Hidden Preset",
        agent_type="openclaw",
        config={
            "agent": {
                "role": "OpenClaw Agent",
                "goal": "Use OpenClaw tools",
                "backstory": "Hidden when OpenClaw integration is disabled.",
                "tools": ["openclaw_sessions_list"],
            }
        },
        description="Should be hidden when OpenClaw is disabled",
        status="active",
        is_preset=True,
    )
    assert hidden_openclaw_agent is not None

    token = create_access_token({"sub": user_id, "email": email})

    with (
        patch("backend.workers.startup_agent_runtime", new=AsyncMock()),
        patch("backend.workers.shutdown_agent_runtime", new=AsyncMock()),
    ):
        client = TestClient(app)
        list_resp = client.get(
            "/v1/agents?is_preset=true",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 200, list_resp.text
        agents = list_resp.json()["data"]["agents"]
        assert any(agent["id"] == "preset-local-agent" for agent in agents)
        assert any(agent["id"] == "preset-travel-planner-agent" for agent in agents)
        assert all(agent["id"] != "preset-openclaw-agent" for agent in agents)

        preset_id = next(agent["id"] for agent in agents if agent["id"] == "preset-local-agent")
        detail_resp = client.get(
            f"/v1/agents/{preset_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert detail_resp.status_code == 200, detail_resp.text
        detail = detail_resp.json()["data"]
        tools = detail["config"]["agent"]["tools"]
        assert tools == []

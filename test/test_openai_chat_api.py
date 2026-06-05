import asyncio
import sys
from pathlib import Path
from datetime import timezone
import types

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
sys.modules.setdefault("aiofiles", types.SimpleNamespace())

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.api.openai import routes as openai_routes


class _FakeSessionManager:
    def __init__(self):
        self.created_with = None
        self.session = {
            "id": "session-openai-1",
            "bound_audio_service_id": None,
            "bound_audio_output_service_id": None,
            "bound_text_service_id": "svc-text-1",
        }

    def create_session(self, **kwargs):
        self.created_with = dict(kwargs)
        return dict(self.session)

    def get_target_service_ids(self, session, client_message_type):
        del session
        if client_message_type in {"session_open", "input_text", "session_close"}:
            return ["svc-text-1"]
        return []

    def close_session(self, session_id):
        return {"id": session_id, "status": "closed"}


class _FakeWebSocketManager:
    def __init__(self):
        self.queue = None
        self.last_open_message = None
        self.last_input_message = None

    async def register_session_subscriber(self, session_id):
        del session_id
        self.queue = asyncio.Queue()
        return self.queue

    async def unregister_session_subscriber(self, session_id, queue):
        del session_id, queue
        return None

    async def get_connected_inference_service_ids(self):
        return {"svc-text-1"}

    async def send_message_to_inference_service(self, service_id, message, binary=None):
        assert service_id == "svc-text-1"
        if message["type"] == "session_open":
            self.last_open_message = message
            await self.queue.put(
                {
                    "type": "session_ready",
                    "session_id": message["session_id"],
                }
            )
        elif message["type"] == "session_text_input":
            self.last_input_message = message
            assert message["messages"][0]["content"][0]["type"] == "image_url"
            await self.queue.put(
                {
                    "type": "llm_text_delta",
                    "session_id": message["session_id"],
                    "delta": "hello",
                    "is_final": False,
                }
            )
            await self.queue.put(
                {
                    "type": "llm_text_delta",
                    "session_id": message["session_id"],
                    "delta": " world",
                    "is_final": True,
                    "finish_reason": "stop",
                    "prompt_tokens": 12,
                    "output_tokens": 2,
                }
            )
        elif message["type"] == "session_close":
            await self.queue.put(
                {
                    "type": "session_closed",
                    "session_id": message["session_id"],
                }
            )
        return True


def test_chat_completions_stream_endpoint_supports_cursor_style_sse(monkeypatch):
    app = create_app(enable_static_files=False)
    app.dependency_overrides[openai_routes._get_openai_request_user_id] = lambda: "user-openai-1"

    fake_ws_manager = _FakeWebSocketManager()
    fake_dispatch_router = types.SimpleNamespace(
        pick_service=lambda spec, connected_service_ids=None: {"id": "svc-text-1"}
    )
    monkeypatch.setattr(openai_routes, "get_websocket_manager", lambda: fake_ws_manager)
    monkeypatch.setattr(openai_routes, "get_dispatch_router", lambda: fake_dispatch_router)
    monkeypatch.setattr(
        openai_routes.User,
        "get_by_id",
        staticmethod(lambda user_id: {"id": user_id}),
    )
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": load_name,
                "name": load_name,
                "family": "Qwen-text",
                "runtime_config": {},
            }
        ),
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer vitoom-agent-internal",
            "X-Vitoom-Effective-User-Id": "user-openai-1",
        },
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/demo.jpg"},
                        },
                        {
                            "type": "text",
                            "text": "请描述图片",
                        },
                    ],
                }
            ],
            "extra_body": {
                "mm_processor_kwargs": {"fps": 2},
            },
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert '"object": "chat.completion.chunk"' in body
    assert '"role": "assistant"' in body
    assert '"content": "hello"' in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body
    assert fake_ws_manager.last_open_message["load_name"] == "Qwen/Qwen3.5-35B-A3B"
    assert "model_name" not in fake_ws_manager.last_open_message
    assert fake_ws_manager.last_input_message["load_name"] == "Qwen/Qwen3.5-35B-A3B"
    assert "model_name" not in fake_ws_manager.last_input_message


def test_chat_completions_stream_endpoint_can_emit_usage_chunk(monkeypatch):
    app = create_app(enable_static_files=False)
    app.dependency_overrides[openai_routes._get_openai_request_user_id] = lambda: "user-openai-usage-1"

    fake_ws_manager = _FakeWebSocketManager()
    fake_dispatch_router = types.SimpleNamespace(
        pick_service=lambda spec, connected_service_ids=None: {"id": "svc-text-1"}
    )
    monkeypatch.setattr(openai_routes, "get_websocket_manager", lambda: fake_ws_manager)
    monkeypatch.setattr(openai_routes, "get_dispatch_router", lambda: fake_dispatch_router)
    monkeypatch.setattr(
        openai_routes.User,
        "get_by_id",
        staticmethod(lambda user_id: {"id": user_id}),
    )
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": load_name,
                "name": load_name,
                "family": "Qwen-text",
                "runtime_config": {},
            }
        ),
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer vitoom-agent-internal",
            "X-Vitoom-Effective-User-Id": "user-openai-usage-1",
        },
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/demo.jpg"},
                        },
                        {
                            "type": "text",
                            "text": "请描述图片",
                        },
                    ],
                }
            ],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert '"usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}' in body
    assert '"choices": []' in body
    assert "data: [DONE]" in body


def test_chat_completions_internal_agent_header_sets_effective_user(monkeypatch):
    app = create_app(enable_static_files=False)

    fake_ws_manager = _FakeWebSocketManager()
    fake_dispatch_router = types.SimpleNamespace(
        pick_service=lambda spec, connected_service_ids=None: {"id": "svc-text-1"}
    )
    monkeypatch.setattr(openai_routes, "get_websocket_manager", lambda: fake_ws_manager)
    monkeypatch.setattr(openai_routes, "get_dispatch_router", lambda: fake_dispatch_router)
    monkeypatch.setattr(
        openai_routes.User,
        "get_by_id",
        staticmethod(lambda user_id: {"id": user_id}),
    )
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": load_name,
                "name": load_name,
                "family": "Qwen-text",
                "runtime_config": {},
            }
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer vitoom-agent-internal",
            "X-Vitoom-Effective-User-Id": "user-agent-run-1",
        },
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/demo.jpg"},
                        },
                        {
                            "type": "text",
                            "text": "请描述图片",
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "hello world"
    assert fake_ws_manager.last_open_message["session_id"]
    assert fake_ws_manager.last_input_message["session_id"] == fake_ws_manager.last_open_message["session_id"]
    assert fake_ws_manager.last_open_message["load_name"] == "Qwen/Qwen3.5-35B-A3B"
    assert "model_name" not in fake_ws_manager.last_open_message
    assert fake_ws_manager.last_input_message["load_name"] == "Qwen/Qwen3.5-35B-A3B"
    assert "model_name" not in fake_ws_manager.last_input_message


class _FakeToolCallSessionWSManager:
    """发出一个带 tool_calls 的 final event 来验证 HTTP 层响应拼装。"""

    def __init__(self):
        self.queue = None
        self.last_input_message = None

    async def register_session_subscriber(self, session_id):
        del session_id
        self.queue = asyncio.Queue()
        return self.queue

    async def unregister_session_subscriber(self, session_id, queue):
        del session_id, queue
        return None

    async def get_connected_inference_service_ids(self):
        return {"svc-text-1"}

    async def send_message_to_inference_service(self, service_id, message, binary=None):
        assert service_id == "svc-text-1"
        if message["type"] == "session_open":
            # 校验 open 消息把 tools / tool_choice 真的带过来了。
            assert message["tools"][0]["function"]["name"] == "analyze_media"
            assert message["tool_choice"] == "auto"
            await self.queue.put(
                {"type": "session_ready", "session_id": message["session_id"]}
            )
        elif message["type"] == "session_text_input":
            self.last_input_message = message
            assert message["tools"][0]["function"]["name"] == "analyze_media"
            await self.queue.put(
                {
                    "type": "llm_text_delta",
                    "session_id": message["session_id"],
                    "delta": "",
                    "is_final": True,
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "analyze_media",
                                "arguments": '{"url": "http://x/a.jpg"}',
                            },
                        }
                    ],
                    "prompt_tokens": 12,
                    "output_tokens": 5,
                }
            )
        elif message["type"] == "session_close":
            await self.queue.put(
                {"type": "session_closed", "session_id": message["session_id"]}
            )
        return True


def test_chat_completions_returns_tool_calls_in_non_streaming_response(monkeypatch):
    app = create_app(enable_static_files=False)

    fake_ws_manager = _FakeToolCallSessionWSManager()
    fake_dispatch_router = types.SimpleNamespace(
        pick_service=lambda spec, connected_service_ids=None: {"id": "svc-text-1"}
    )
    monkeypatch.setattr(openai_routes, "get_websocket_manager", lambda: fake_ws_manager)
    monkeypatch.setattr(openai_routes, "get_dispatch_router", lambda: fake_dispatch_router)
    monkeypatch.setattr(
        openai_routes.User,
        "get_by_id",
        staticmethod(lambda user_id: {"id": user_id}),
    )
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": load_name,
                "name": load_name,
                "family": "Qwen-text",
                "runtime_config": {},
            }
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer vitoom-agent-internal",
            "X-Vitoom-Effective-User-Id": "user-openai-tools-1",
        },
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "messages": [{"role": "user", "content": "这张图是什么 http://x/a.jpg"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_media",
                        "description": "analyze image/video",
                        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] is None
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_abc"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "analyze_media"
    assert tool_calls[0]["function"]["arguments"] == '{"url": "http://x/a.jpg"}'
    # usage 仍然应正常
    assert payload["usage"]["prompt_tokens"] == 12
    assert payload["usage"]["completion_tokens"] == 5


def test_chat_completions_stream_endpoint_emits_final_tool_call_message_chunk(monkeypatch):
    app = create_app(enable_static_files=False)

    fake_ws_manager = _FakeToolCallSessionWSManager()
    fake_dispatch_router = types.SimpleNamespace(
        pick_service=lambda spec, connected_service_ids=None: {"id": "svc-text-1"}
    )
    monkeypatch.setattr(openai_routes, "get_websocket_manager", lambda: fake_ws_manager)
    monkeypatch.setattr(openai_routes, "get_dispatch_router", lambda: fake_dispatch_router)
    monkeypatch.setattr(
        openai_routes.User,
        "get_by_id",
        staticmethod(lambda user_id: {"id": user_id}),
    )
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": load_name,
                "name": load_name,
                "family": "Qwen-text",
                "runtime_config": {},
            }
        ),
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer vitoom-agent-internal",
            "X-Vitoom-Effective-User-Id": "user-openai-tools-stream-1",
        },
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "stream": True,
            "messages": [{"role": "user", "content": "这张图是什么 http://x/a.jpg"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_media",
                        "description": "analyze image/video",
                        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                    },
                }
            ],
            "tool_choice": "auto",
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert '"finish_reason": "tool_calls"' in body
    assert '"message": {"role": "assistant", "content": null, "tool_calls": [{' in body
    assert '"name": "analyze_media"' in body
    assert '"arguments": "{\\"url\\": \\"http://x/a.jpg\\"}"' in body
    assert "data: [DONE]" in body


def test_chat_completions_internal_agent_header_rejects_missing_effective_user(monkeypatch):
    app = create_app(enable_static_files=False)
    monkeypatch.setattr(
        openai_routes.User,
        "get_by_id",
        staticmethod(lambda _user_id: None),
    )
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": load_name,
                "name": load_name,
                "family": "Qwen-text",
                "runtime_config": {},
            }
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer vitoom-agent-internal",
            "X-Vitoom-Effective-User-Id": "missing-user",
        },
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["msg"] == "Invalid effective user id: missing-user"


def test_list_models_returns_openai_compatible_payload(monkeypatch):
    app = create_app(enable_static_files=False)
    app.dependency_overrides[openai_routes._get_openai_request_user_id] = lambda: "user-openai-models"

    class _FakeModelService:
        def list_models(self, **kwargs):
            del kwargs
            return (
                [
                    {
                        "load_name": "Qwen3.5-35B-A3B-GPTQ-Int4",
                        "name": "Qwen3.5-35B-A3B-GPTQ-Int4",
                        "created_at": "2026-01-01T00:00:00",
                    }
                ],
                1,
            )

    monkeypatch.setattr(openai_routes, "get_model_service", lambda: _FakeModelService())

    client = TestClient(app)
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer vitoom-agent-internal"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    item = payload["data"][0]
    assert item["id"] == "Qwen3.5-35B-A3B-GPTQ-Int4"
    assert item["object"] == "model"
    assert item["owned_by"] == "vitoom"
    assert isinstance(item.get("created"), int)
    assert item["permission"] == []
    assert item["root"] == "Qwen3.5-35B-A3B-GPTQ-Int4"


def test_get_model_returns_model_metadata(monkeypatch):
    app = create_app(enable_static_files=False)
    app.dependency_overrides[openai_routes._get_openai_request_user_id] = lambda: "user-openai-model-detail"
    monkeypatch.setattr(
        openai_routes.Model,
        "get_by_load_name",
        staticmethod(
            lambda load_name: {
                "load_name": "Qwen3.5-35B-A3B-GPTQ-Int4",
                "name": "Qwen3.5-35B-A3B-GPTQ-Int4",
                "created_at": "2026-01-01T00:00:00",
            }
            if load_name == "Qwen3.5-35B-A3B-GPTQ-Int4"
            else None
        ),
    )

    client = TestClient(app)
    response = client.get(
        "/v1/models/Qwen3.5-35B-A3B-GPTQ-Int4",
        headers={"Authorization": "Bearer vitoom-agent-internal"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "Qwen3.5-35B-A3B-GPTQ-Int4"
    assert payload["root"] == "Qwen3.5-35B-A3B-GPTQ-Int4"

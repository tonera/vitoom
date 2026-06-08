"""推理服务 /start 请求来源 IP 记录单测。"""

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from backend.services.inference import service as svc_module  # noqa: E402
from backend.services.inference.service import InferenceServiceManager  # noqa: E402
from backend.utils.http_utils import resolve_client_ip  # noqa: E402


def _make_request(
    *,
    client_host: str | None = "192.168.1.10",
    headers: dict[str, str] | None = None,
):
    request = MagicMock()
    request.headers = headers or {}
    if client_host is None:
        request.client = None
    else:
        request.client = MagicMock(host=client_host)
    return request


def test_resolve_client_ip_prefers_x_forwarded_for():
    request = _make_request(
        client_host="10.0.0.1",
        headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"},
    )
    assert resolve_client_ip(request) == "203.0.113.5"


def test_resolve_client_ip_uses_x_real_ip():
    request = _make_request(
        client_host="10.0.0.1",
        headers={"x-real-ip": "198.51.100.2"},
    )
    assert resolve_client_ip(request) == "198.51.100.2"


def test_resolve_client_ip_falls_back_to_direct_client():
    request = _make_request(client_host="172.16.0.8")
    assert resolve_client_ip(request) == "172.16.0.8"


class _FakeInferenceServiceTable:
    def __init__(self, initial: Dict[str, Any]):
        self.row = dict(initial)

    def get_by_id(self, service_id: str) -> Dict[str, Any] | None:
        if service_id != self.row["id"]:
            return None
        return dict(self.row)

    def update(self, service_id: str, **kwargs) -> Dict[str, Any] | None:
        if service_id != self.row["id"]:
            return None
        for key, value in kwargs.items():
            self.row[key] = value
        return dict(self.row)


def test_sync_service_start_persists_client_ip(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeInferenceServiceTable(
        {
            "id": "service_qwen_text",
            "name": "Text",
            "type": "vllm",
            "service_type": "text",
            "status": "stopped",
            "host": "127.0.0.1",
            "port": 8003,
            "config": {},
        }
    )
    monkeypatch.setattr(svc_module.InferenceService, "get_by_id", fake.get_by_id)
    monkeypatch.setattr(svc_module.InferenceService, "update", fake.update)

    mgr = InferenceServiceManager()
    updated = mgr.sync_service_start(
        "service_qwen_text",
        host="127.0.0.1",
        port=8003,
        client_ip="192.168.31.163",
    )

    assert updated["client_ip"] == "192.168.31.163"
    assert updated["status"] == "running"

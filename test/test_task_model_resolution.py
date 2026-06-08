import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.tasks.routes import TaskCreateRequest, _resolve_request_model
from backend.database import Model


def _catalog_row(*, model_key: str, load_name: str) -> dict:
    return {
        "model_key": model_key,
        "load_name": load_name,
        "family": "sdxl",
        "runtime_config": {"foo": "bar"},
        "storage_mode": "local",
    }


def test_resolve_request_model_prefers_model_key(monkeypatch):
    by_key = _catalog_row(model_key="key-a", load_name="catalog-load")

    def fake_get_by_model_key(value: str):
        assert value == "key-a"
        return by_key

    def fake_get_by_load_name(value: str):
        raise AssertionError(f"load_name lookup should not run, got {value}")

    monkeypatch.setattr(Model, "get_by_model_key", staticmethod(fake_get_by_model_key))
    monkeypatch.setattr(Model, "get_by_load_name", staticmethod(fake_get_by_load_name))

    request = TaskCreateRequest(
        task_type="image",
        model_key="key-a",
        load_name="stale-client-load",
        prompt="hello",
    )

    resolved = _resolve_request_model(request, task_type="image", requires_model=True)

    assert resolved == by_key
    assert request.model_key == "key-a"
    assert request.load_name == "catalog-load"
    assert request.family == "sdxl"


def test_resolve_request_model_uses_load_name_without_model_key(monkeypatch):
    by_load_name = _catalog_row(model_key="key-from-load", load_name="client-load")

    def fake_get_by_model_key(value: str):
        raise AssertionError(f"model_key lookup should not run, got {value}")

    def fake_get_by_load_name(value: str):
        assert value == "client-load"
        return by_load_name

    monkeypatch.setattr(Model, "get_by_model_key", staticmethod(fake_get_by_model_key))
    monkeypatch.setattr(Model, "get_by_load_name", staticmethod(fake_get_by_load_name))

    request = TaskCreateRequest(
        task_type="video",
        load_name="client-load",
        prompt="hello",
    )

    resolved = _resolve_request_model(request, task_type="video", requires_model=True)

    assert resolved == by_load_name
    assert request.model_key == "key-from-load"
    assert request.load_name == "client-load"
    assert request.family == "sdxl"

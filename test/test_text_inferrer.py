# pyright: reportMissingImports=false

import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

fake_common = sys.modules.setdefault("common", ModuleType("common"))
fake_common_base_inferrer = ModuleType("common.base_inferrer")
fake_common_config_loader = ModuleType("common.config_loader")
fake_common_logger = ModuleType("common.logger")
fake_schemas = ModuleType("schemas")


class _FakeBaseInferrer:
    async def run_blocking(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _FakeLogger:
    def __init__(self):
        self.records = []

    def info(self, *args, **kwargs):
        self.records.append(("info", args, kwargs))
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


fake_common_base_inferrer.BaseInferrer = _FakeBaseInferrer
fake_common_config_loader.load_inference_config = lambda *args, **kwargs: SimpleNamespace(models_dir="/tmp/models")
fake_common_logger.get_logger = lambda _name: _FakeLogger()
fake_schemas.InferenceRequestParams = object

setattr(fake_common, "base_inferrer", fake_common_base_inferrer)
setattr(fake_common, "config_loader", fake_common_config_loader)
setattr(fake_common, "logger", fake_common_logger)
sys.modules["common.base_inferrer"] = fake_common_base_inferrer
sys.modules["common.config_loader"] = fake_common_config_loader
sys.modules["common.logger"] = fake_common_logger
sys.modules["schemas"] = fake_schemas

from text.inferrer import TextInferrer
from text.runtime.runtime_resolver import TextRuntimePolicy


def test_get_bundle_uses_run_blocking_and_caches_result(monkeypatch):
    inferrer = TextInferrer.__new__(TextInferrer)
    inferrer.inference_config = SimpleNamespace(models_dir="/tmp/models")
    inferrer._model_cache = {}

    policy = TextRuntimePolicy(
        runtime="vllm",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
        enable_thinking=False,
        engine_kwargs={},
    )

    monkeypatch.setattr("text.inferrer.resolve_text_runtime", lambda spec: "vllm")
    monkeypatch.setattr("text.inferrer.resolve_text_model_ref", lambda spec, models_dir=None: "/tmp/models/demo")
    monkeypatch.setattr("text.inferrer.resolve_text_runtime_policy", lambda spec: policy)

    calls = []
    bundle = object()

    async def fake_run_blocking(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        assert func is not None
        return bundle

    inferrer.run_blocking = fake_run_blocking

    async def main():
        first = await inferrer._get_bundle(SimpleNamespace())
        second = await inferrer._get_bundle(SimpleNamespace())
        return first, second

    first, second = asyncio.run(main())

    assert first is bundle
    assert second is bundle
    assert len(calls) == 1
    assert calls[0][1] == ("/tmp/models/demo", policy)


def test_get_bundle_dispatches_to_transformers_loader(monkeypatch):
    inferrer = TextInferrer.__new__(TextInferrer)
    inferrer.inference_config = SimpleNamespace(models_dir="/tmp/models")
    inferrer._model_cache = {}

    policy = TextRuntimePolicy(
        runtime="transformers",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
        enable_thinking=False,
        engine_kwargs={},
        dtype="bfloat16",
        device_map="auto",
        model_kwargs={"attn_implementation": "sdpa"},
    )

    monkeypatch.setattr("text.inferrer.resolve_text_runtime", lambda spec: "transformers")
    monkeypatch.setattr("text.inferrer.resolve_text_model_ref", lambda spec, models_dir=None: "/tmp/models/demo")
    monkeypatch.setattr("text.inferrer.resolve_text_runtime_policy", lambda spec: policy)

    calls = []
    bundle = object()

    async def fake_run_blocking(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return bundle

    inferrer.run_blocking = fake_run_blocking

    async def main():
        return await inferrer._get_bundle(SimpleNamespace())

    resolved = asyncio.run(main())

    assert resolved is bundle
    assert len(calls) == 1
    assert calls[0][0] is not None
    assert calls[0][1] == ("/tmp/models/demo", policy)


def test_log_inference_request_reports_model_bytes_and_tool_names(monkeypatch):
    fake_logger = _FakeLogger()
    monkeypatch.setattr("text.inferrer.logger", fake_logger)

    inferrer = TextInferrer.__new__(TextInferrer)
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好，帮我总结一下这段文本"},
    ]
    tools = [
        {"type": "function", "function": {"name": "analyze_media"}},
        {"name": "travel_planner"},
    ]

    inferrer._log_inference_request(
        request_id="task:123",
        model_name="qwen3-32b",
        messages=messages,
        tools=tools,
    )

    assert len(fake_logger.records) == 1
    level, args, _kwargs = fake_logger.records[0]
    assert level == "info"
    assert args[0] == "Text inference request request_id=%s model_name=%s message_bytes=%s tool_names=%s"
    assert args[1] == "task:123"
    assert args[2] == "qwen3-32b"
    assert args[3] == TextInferrer._estimate_message_package_bytes(messages)
    assert args[4] == ["analyze_media", "travel_planner"]

# pyright: reportMissingImports=false

import asyncio
import queue
import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

fake_common = sys.modules.setdefault("common", ModuleType("common"))
fake_common_logger = ModuleType("common.logger")


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


fake_common_logger.get_logger = lambda _name: _FakeLogger()
setattr(fake_common, "logger", fake_common_logger)
sys.modules["common.logger"] = fake_common_logger

from text.runtime.runtime_resolver import TextRuntimePolicy
from text.runtime.transformers_bridge import (
    TransformersTextBundle,
    abort_chat_request,
    load_transformers_text_bundle,
    stream_chat_text,
)


class _FakeTensor:
    def __init__(self, token_count: int):
        self.shape = (1, token_count)

    def to(self, device):  # noqa: ANN001
        return self


class _FakeBatchEncoding(dict):
    def to(self, device):  # noqa: ANN001
        return self


class _FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=None):
        parts = []
        for item in messages:
            content = item["content"]
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part.get("text") or "")
                    else:
                        parts.append(f"<{part.get('type')}>")
            else:
                parts.append(str(content))
        return "\n".join(parts)

    def __call__(self, prompt, return_tensors="pt"):
        token_count = max(1, len(str(prompt or "").split()))
        return _FakeBatchEncoding({"input_ids": _FakeTensor(token_count)})

    def encode(self, text, add_special_tokens=False):
        text = str(text or "").strip()
        if not text:
            return []
        return list(range(len(text.split())))


class _FakeModel:
    def __init__(self):
        self.device = "cuda:0"

    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        stopping_criteria = kwargs.get("stopping_criteria") or []
        streamer.on_finalized_text("hello ", stream_end=False)
        time.sleep(0.01)
        if stopping_criteria and stopping_criteria[0](None, None):
            streamer.on_finalized_text("", stream_end=True)
            return None
        streamer.on_finalized_text("world", stream_end=True)
        return None

    def to(self, device):  # noqa: ANN001
        return self


class _FakeTextIteratorStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self._queue = queue.Queue()

    def on_finalized_text(self, text, stream_end=False):
        self._queue.put(text)
        if stream_end:
            self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item


class _FakeStoppingCriteria:
    def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
        return False


class _FakeStoppingCriteriaList(list):
    pass


@contextmanager
def _fake_inference_mode():
    yield


def _install_fake_runtime_modules(monkeypatch):
    fake_torch = types.SimpleNamespace(
        inference_mode=_fake_inference_mode,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    fake_generation = ModuleType("transformers.generation")
    fake_generation.StoppingCriteria = _FakeStoppingCriteria
    fake_generation.StoppingCriteriaList = _FakeStoppingCriteriaList
    fake_generation.TextIteratorStreamer = _FakeTextIteratorStreamer

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers.generation", fake_generation)


def _make_bundle() -> TransformersTextBundle:
    return TransformersTextBundle(
        model_ref="demo-model",
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        policy=TextRuntimePolicy(
            runtime="transformers",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={},
        ),
        chat_adapter=_FakeTokenizer(),
    )


def test_stream_chat_text_transformers_yields_deltas_and_stats(monkeypatch):
    _install_fake_runtime_modules(monkeypatch)
    bundle = _make_bundle()

    async def main():
        items = []
        async for item in stream_chat_text(
            bundle,
            messages=[{"role": "user", "content": "hi there"}],
            request_id="req-1",
            max_tokens=16,
            temperature=0.2,
        ):
            items.append(item)
        return items

    items = asyncio.run(main())

    assert items[0]["delta"] == "hello "
    assert items[0]["finished"] is False
    assert items[1]["delta"] == "world"
    assert items[1]["finished"] is False
    assert items[-1]["delta"] == ""
    assert items[-1]["finished"] is True
    assert items[-1]["finish_reason"] == "stop"
    assert items[-1]["prompt_tokens"] >= 1
    assert items[-1]["output_tokens"] == 2
    assert items[-1]["total_seconds"] >= 0


def test_abort_chat_request_sets_registered_stop_event():
    bundle = _make_bundle()
    stop_event = threading.Event()
    bundle.active_requests["req-abort"] = types.SimpleNamespace(stop_event=stop_event)

    asyncio.run(abort_chat_request(bundle, "req-abort"))

    assert stop_event.is_set() is True


def test_stream_chat_text_transformers_rejects_multimodal(monkeypatch):
    _install_fake_runtime_modules(monkeypatch)
    bundle = _make_bundle()

    async def main():
        async for _item in stream_chat_text(
            bundle,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            request_id="req-mm",
        ):
            pass

    with pytest.raises(RuntimeError, match="text-only chat"):
        asyncio.run(main())


def test_load_transformers_text_bundle_rejects_cpu_offload(monkeypatch):
    fake_transformers = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    class _OffloadedModel(_FakeModel):
        def __init__(self):
            super().__init__()
            self.hf_device_map = {"layer0": "cuda:0", "layer1": "cpu"}

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _OffloadedModel()

    fake_transformers.AutoTokenizer = _AutoTokenizer
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="CPU offload"):
        load_transformers_text_bundle(
            "demo-model",
            TextRuntimePolicy(
                runtime="transformers",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                trust_remote_code=True,
                enable_thinking=False,
                engine_kwargs={},
                device_map="auto",
            ),
        )


def test_load_transformers_text_bundle_allows_cpu_offload_when_enabled(monkeypatch):
    fake_transformers = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    class _OffloadedModel(_FakeModel):
        def __init__(self):
            super().__init__()
            self.hf_device_map = {"layer0": "cuda:0", "layer1": "cpu"}

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _OffloadedModel()

    fake_transformers.AutoTokenizer = _AutoTokenizer
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    bundle = load_transformers_text_bundle(
        "demo-model",
        TextRuntimePolicy(
            runtime="transformers",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            enable_thinking=False,
            allow_cpu_offload=True,
            engine_kwargs={},
            device_map="auto",
        ),
    )

    assert isinstance(bundle, TransformersTextBundle)

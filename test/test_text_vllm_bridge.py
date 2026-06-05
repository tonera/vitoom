# pyright: reportMissingImports=false

import asyncio
import json
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

sys.modules.setdefault("PIL", types.SimpleNamespace(Image=object()))

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
from text.runtime.vllm_bridge import (
    VllmTextBundle,
    abort_chat_request,
    load_vllm_text_bundle,
    stream_chat_text,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
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


class _FakeCompletion:
    def __init__(self, text, finish_reason=None, token_ids=None):
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = token_ids if token_ids is not None else []


class _FakeOutput:
    def __init__(
        self,
        text,
        finished,
        finish_reason=None,
        token_ids=None,
        prompt_token_ids=None,
        metrics=None,
    ):
        self.outputs = [_FakeCompletion(text, finish_reason=finish_reason, token_ids=token_ids)]
        self.finished = finished
        self.prompt_token_ids = prompt_token_ids if prompt_token_ids is not None else []
        self.metrics = metrics


class _FakeMetrics:
    def __init__(self, num_generation_tokens=None, first_token_latency=None):
        self.num_generation_tokens = num_generation_tokens
        self.first_token_latency = first_token_latency


class _FakeEngine:
    def __init__(self):
        self.aborted = []
        self.last_generate_kwargs = None

    async def generate(self, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        yield _FakeOutput("hello ", False, token_ids=[11], prompt_token_ids=[1, 2, 3])
        yield _FakeOutput(
            "world",
            True,
            finish_reason="stop",
            token_ids=[11, 12],
            prompt_token_ids=[1, 2, 3],
            metrics=_FakeMetrics(num_generation_tokens=2, first_token_latency=0.25),
        )

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def shutdown(self):
        return None


def test_stream_chat_text_yields_true_deltas(monkeypatch):
    class _Kind:
        DELTA = "delta"

    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (object(), object(), _Kind),
    )
    monkeypatch.setattr(
        "text.runtime.vllm_bridge._build_sampling_params",
        lambda **kwargs: kwargs,
    )

    bundle = VllmTextBundle(
        model_ref="demo-model",
        tokenizer=_FakeTokenizer(),
        engine=_FakeEngine(),
        policy=TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={},
        ),
    )

    async def main():
        items = []
        async for item in stream_chat_text(
            bundle,
            messages=[{"role": "user", "content": "hi"}],
            request_id="req-1",
            temperature=0.2,
            max_tokens=16,
        ):
            items.append(item)
        return items

    items = asyncio.run(main())
    assert items[0]["delta"] == "hello "
    assert items[-1]["delta"] == "world"
    assert items[-1]["finished"] is True
    assert items[-1]["prompt_tokens"] == 3
    assert items[-1]["output_tokens"] == 2
    assert items[-1]["ttft_seconds"] == 0.25
    assert items[-1]["tok_s_total"] > 0


def test_abort_chat_request_calls_engine_abort():
    bundle = VllmTextBundle(
        model_ref="demo-model",
        tokenizer=_FakeTokenizer(),
        engine=_FakeEngine(),
        policy=TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={},
        ),
    )

    async def main():
        await abort_chat_request(bundle, "req-abort")

    asyncio.run(main())
    assert bundle.engine.aborted == ["req-abort"]


def test_stream_chat_text_passes_multimodal_payload(monkeypatch):
    class _Kind:
        DELTA = "delta"

    captured_messages = {}

    def _fake_process_vision_info(messages, **kwargs):
        captured_messages["messages"] = messages
        captured_messages["kwargs"] = kwargs
        return (
            ["fake-image"],
            [("fake-video", {"fps": 1.5, "total_num_frames": 12})],
            {"do_sample_frames": False, "fps": [1.5]},
        )

    fake_qwen_vl_utils = ModuleType("qwen_vl_utils")
    fake_qwen_vl_utils.process_vision_info = _fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl_utils)
    monkeypatch.setitem(sys.modules, "torchcodec", ModuleType("torchcodec"))
    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (object(), object(), _Kind),
    )
    monkeypatch.setattr(
        "text.runtime.vllm_bridge._build_sampling_params",
        lambda **kwargs: kwargs,
    )

    engine = _FakeEngine()
    bundle = VllmTextBundle(
        model_ref="demo-model",
        tokenizer=_FakeTokenizer(),
        engine=engine,
        policy=TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={},
        ),
    )

    async def main():
        items = []
        async for item in stream_chat_text(
            bundle,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
                        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            request_id="req-mm",
            mm_processor_kwargs={"fps": 2},
        ):
            items.append(item)
        return items

    items = asyncio.run(main())
    assert items[-1]["finished"] is True
    assert engine.last_generate_kwargs["prompt"] == {
        "prompt": "<image>\n<video>\ndescribe",
        "multi_modal_data": {
            "image": ["fake-image"],
            "video": [("fake-video", {"fps": 1.5, "total_num_frames": 12})],
        },
        "mm_processor_kwargs": {"do_sample_frames": False, "fps": 2},
    }
    assert captured_messages["kwargs"] == {
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }
    adapted_content = captured_messages["messages"][0]["content"]
    assert adapted_content[0] == {"type": "image", "image": "https://example.com/a.jpg"}
    assert adapted_content[1] == {"type": "video", "video": "https://example.com/a.mp4"}


def test_stream_chat_text_video_requires_supported_decoder(monkeypatch):
    class _Kind:
        DELTA = "delta"

    fake_qwen_vl_utils = ModuleType("qwen_vl_utils")
    fake_qwen_vl_utils.process_vision_info = lambda messages, **kwargs: (None, ["fake-video"])
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl_utils)
    monkeypatch.delenv("FORCE_QWENVL_VIDEO_READER", raising=False)
    monkeypatch.setattr("text.runtime.vllm_bridge._module_available", lambda name: False)
    monkeypatch.setattr("text.runtime.vllm_bridge._torchvision_has_read_video", lambda: False)
    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (object(), object(), _Kind),
    )
    monkeypatch.setattr(
        "text.runtime.vllm_bridge._build_sampling_params",
        lambda **kwargs: kwargs,
    )

    engine = _FakeEngine()
    bundle = VllmTextBundle(
        model_ref="demo-model",
        tokenizer=_FakeTokenizer(),
        engine=engine,
        policy=TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={},
        ),
    )

    async def main():
        async for _item in stream_chat_text(
            bundle,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            request_id="req-video",
        ):
            pass

    with pytest.raises(RuntimeError, match="torchcodec"):
        asyncio.run(main())


def test_load_vllm_text_bundle_normalizes_legacy_rope_scaling(monkeypatch, tmp_path):
    fake_transformers = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    captured_kwargs = {}

    class _FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class _FakeAsyncEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            return {"engine_args": engine_args}

    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (_FakeAsyncEngine, _FakeAsyncEngineArgs, object()),
    )

    model_dir = tmp_path / "Gemma-4-31B-IT-NVFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"rope_scaling": {"type": "linear", "factor": 8.0}}),
        encoding="utf-8",
    )

    load_vllm_text_bundle(
        str(model_dir),
        TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={"hf_overrides": {"architectures": ["Gemma3ForCausalLM"]}},
        ),
    )

    assert captured_kwargs["hf_overrides"] == {
        "rope_scaling": {"rope_type": "linear", "factor": 8.0},
        "architectures": ["Gemma3ForCausalLM"],
    }


def test_load_vllm_text_bundle_normalizes_legacy_hf_override_rope_scaling(monkeypatch, tmp_path):
    fake_transformers = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    captured_kwargs = {}

    class _FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class _FakeAsyncEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            return {"engine_args": engine_args}

    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (_FakeAsyncEngine, _FakeAsyncEngineArgs, object()),
    )

    model_dir = tmp_path / "Gemma-4-31B-IT-NVFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    load_vllm_text_bundle(
        str(model_dir),
        TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            trust_remote_code=True,
            enable_thinking=False,
            engine_kwargs={
                "hf_overrides": {
                    "rope_scaling": {"type": "linear", "factor": 8.0},
                    "architectures": ["Gemma3ForCausalLM"],
                }
            },
        ),
    )

    assert captured_kwargs["hf_overrides"] == {
        "rope_scaling": {"rope_type": "linear", "factor": 8.0},
        "architectures": ["Gemma3ForCausalLM"],
    }


def test_load_vllm_text_bundle_applies_speculative_config(monkeypatch, tmp_path):
    fake_transformers = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    captured_kwargs = {}

    class _FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class _FakeAsyncEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            return {"engine_args": engine_args}

    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (_FakeAsyncEngine, _FakeAsyncEngineArgs, object()),
    )

    target_dir = tmp_path / "gemma-target"
    target_dir.mkdir()
    assistant_dir = tmp_path / "gemma-4-31B-it-assistant"
    assistant_dir.mkdir()

    load_vllm_text_bundle(
        str(target_dir),
        TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=8192,
            trust_remote_code=True,
            enable_thinking=False,
            speculative_config={
                "method": "mtp",
                "model": "gemma-4-31B-it-assistant",
                "num_speculative_tokens": 4,
            },
        ),
        models_dir=str(tmp_path),
    )

    assert captured_kwargs["speculative_config"]["method"] == "mtp"
    assert captured_kwargs["speculative_config"]["model"] == str(assistant_dir.resolve())
    assert captured_kwargs["speculative_config"]["num_speculative_tokens"] == 4


def test_load_vllm_text_bundle_skips_speculative_when_assistant_missing(monkeypatch, tmp_path):
    fake_transformers = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    captured_kwargs = {}

    class _FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class _FakeAsyncEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            return {"engine_args": engine_args}

    monkeypatch.setattr(
        "text.runtime.vllm_bridge._import_async_vllm_symbols",
        lambda: (_FakeAsyncEngine, _FakeAsyncEngineArgs, object()),
    )

    target_dir = tmp_path / "gemma-target"
    target_dir.mkdir()

    load_vllm_text_bundle(
        str(target_dir),
        TextRuntimePolicy(
            runtime="vllm",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=8192,
            trust_remote_code=True,
            enable_thinking=False,
            speculative_config={
                "method": "mtp",
                "model": "gemma-4-31B-it-assistant",
            },
        ),
        models_dir=str(tmp_path),
    )

    assert "speculative_config" not in captured_kwargs

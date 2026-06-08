import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

fake_common = sys.modules.setdefault("common", types.ModuleType("common"))
fake_common_logger = types.ModuleType("common.logger")


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


fake_common_logger.get_logger = lambda _name: _FakeLogger()
setattr(fake_common, "logger", fake_common_logger)
sys.modules["common.logger"] = fake_common_logger

fake_video = sys.modules.setdefault("video", types.ModuleType("video"))
fake_video_runtime = types.ModuleType("video.runtime")
fake_video_io_utils = types.ModuleType("video.runtime.io_utils")


async def _fake_download_url_to_tempfile(*args, **kwargs):
    raise AssertionError("download_url_to_tempfile should be monkeypatched in tests")


fake_video_io_utils.download_url_to_tempfile = _fake_download_url_to_tempfile
setattr(fake_video_runtime, "io_utils", fake_video_io_utils)
setattr(fake_video, "runtime", fake_video_runtime)
sys.modules["video.runtime"] = fake_video_runtime
sys.modules["video.runtime.io_utils"] = fake_video_io_utils

from text.runtime.ollama_bridge import (  # noqa: E402
    OllamaTextBundle,
    _load_or_generate_modelfile,
    load_ollama_text_bundle,
    _scan_gguf_dir,
    _to_ollama_messages,
    stream_chat_text,
)
from text.runtime.runtime_resolver import TextRuntimePolicy  # noqa: E402


def _policy(ollama_cfg=None) -> TextRuntimePolicy:
    return TextRuntimePolicy(
        runtime="ollama",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
        enable_thinking=False,
        ollama_cfg=dict(ollama_cfg or {}),
    )


def _bundle(*, multimodal_mode: str) -> OllamaTextBundle:
    return OllamaTextBundle(
        model_source="local_gguf",
        model_ref="/tmp/model",
        tag="demo:latest",
        main_gguf="/tmp/model.gguf",
        mmproj_path="/tmp/mmproj.gguf",
        modelfile_source="generated",
        client=object(),
        policy=_policy({"multimodal_mode": multimodal_mode}),
        multimodal_mode=multimodal_mode,
        vision_enabled=multimodal_mode != "off",
    )


def test_generated_modelfile_ignores_mmproj_when_multimodal_disabled(tmp_path):
    modelfile_text, source = _load_or_generate_modelfile(
        model_ref=str(tmp_path),
        main_gguf="/models/main.gguf",
        mmproj_path="/models/mmproj.gguf",
        extra_lines=[],
        chat_template=None,
        stop_tokens=(),
        vision_enabled=False,
    )
    assert source == "generated"
    assert modelfile_text.splitlines()[0] == "FROM /models/main.gguf"
    assert "mmproj" not in modelfile_text


def test_generated_modelfile_uses_directory_from_when_multimodal_enabled(tmp_path):
    modelfile_text, source = _load_or_generate_modelfile(
        model_ref=str(tmp_path),
        main_gguf="/models/main.gguf",
        mmproj_path="/models/mmproj.gguf",
        extra_lines=[],
        chat_template=None,
        stop_tokens=(),
        vision_enabled=True,
    )
    assert source == "generated"
    assert modelfile_text.splitlines()[0] == f"FROM {tmp_path.resolve()}"


def test_generated_modelfile_requires_mmproj_when_multimodal_enabled(tmp_path):
    with pytest.raises(RuntimeError, match="no mmproj"):
        _load_or_generate_modelfile(
            model_ref=str(tmp_path),
            main_gguf="/models/main.gguf",
            mmproj_path=None,
            extra_lines=[],
            chat_template=None,
            stop_tokens=(),
            vision_enabled=True,
        )


def test_scan_gguf_dir_rejects_invalid_mmproj_file(tmp_path):
    main = tmp_path / "model-Q4_K_M.gguf"
    main.write_bytes(b"GGUFdemo-main")
    mmproj = tmp_path / "mmproj-BF16.gguf"
    mmproj.write_bytes(b"NOTGdemo-mmproj")

    with pytest.raises(RuntimeError, match="mmproj GGUF"):
        _scan_gguf_dir(str(tmp_path))


def test_load_ollama_text_bundle_uses_tag_source_without_scanning(monkeypatch):
    calls = {}

    def _fail_scan(_model_ref):
        raise AssertionError("_scan_gguf_dir should not run for tag source")

    def _fake_ensure_tag_available(*, tag, host, auto_pull):
        calls["ensure"] = {"tag": tag, "host": host, "auto_pull": auto_pull}

    monkeypatch.setattr("text.runtime.ollama_bridge._scan_gguf_dir", _fail_scan)
    monkeypatch.setattr("text.runtime.ollama_bridge._ensure_tag_available", _fake_ensure_tag_available)
    monkeypatch.setattr("text.runtime.ollama_bridge._make_async_client", lambda host: {"host": host})

    bundle = load_ollama_text_bundle(
        "qwen3.6:35b",
        _policy({"model_source": "tag", "auto_pull": True, "base_url": "http://127.0.0.1:11434"}),
    )

    assert bundle.model_source == "tag"
    assert bundle.model_ref == "qwen3.6:35b"
    assert bundle.tag == "qwen3.6:35b"
    assert bundle.modelfile_source == "tag"
    assert bundle.main_gguf == ""
    assert bundle.mmproj_path is None
    assert calls["ensure"] == {
        "tag": "qwen3.6:35b",
        "host": "http://127.0.0.1:11434",
        "auto_pull": True,
    }


def test_stream_chat_text_returns_message_when_multimodal_disabled(monkeypatch):
    bundle = _bundle(multimodal_mode="off")

    async def _fail_call_chat_stream(**kwargs):
        raise AssertionError("_call_chat_stream should not run when multimodal is unsupported")

    monkeypatch.setattr("text.runtime.ollama_bridge._call_chat_stream", _fail_call_chat_stream)

    async def _main():
        items = []
        async for item in stream_chat_text(
            bundle,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}],
                }
            ],
            request_id="req-mm-disabled",
        ):
            items.append(item)
        return items

    items = asyncio.run(_main())
    assert items == [
        {
            "delta": "当前所选 Ollama 服务未开启多模态支持，暂时不能处理图片或视频输入。请切换到支持多模态的服务配置，或改用纯文本提问。",
            "finished": True,
            "finish_reason": "unsupported_multimodal",
        }
    ]


def test_to_ollama_messages_attaches_images(monkeypatch):
    bundle = _bundle(multimodal_mode="image")

    async def _fake_resolve_binary_input(url_or_path, *, default_suffix, max_bytes=None):
        return (f"blob:{url_or_path}".encode("utf-8"), None)

    monkeypatch.setattr("text.runtime.ollama_bridge._resolve_binary_input", _fake_resolve_binary_input)

    async def _main():
        return await _to_ollama_messages(
            bundle,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            mm_processor_kwargs=None,
        )

    converted = asyncio.run(_main())
    assert converted == [
        {
            "role": "user",
            "content": "describe",
            "images": [b"blob:https://example.com/cat.png"],
        }
    ]


def test_to_ollama_messages_samples_video_into_images(monkeypatch):
    bundle = _bundle(multimodal_mode="image_and_video_frames")
    captured = {}

    async def _fake_sample_video_frames(bundle_arg, video_ref, *, mm_processor_kwargs):
        captured["bundle"] = bundle_arg
        captured["video_ref"] = video_ref
        captured["mm_processor_kwargs"] = mm_processor_kwargs
        return [b"frame-1", b"frame-2", b"frame-3"]

    monkeypatch.setattr("text.runtime.ollama_bridge._sample_video_frames", _fake_sample_video_frames)

    async def _main():
        return await _to_ollama_messages(
            bundle,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": "/tmp/demo.mp4"}},
                        {"type": "text", "text": "summarize the clip"},
                    ],
                }
            ],
            mm_processor_kwargs={"fps": 2, "video_frame_count": 6},
        )

    converted = asyncio.run(_main())
    assert captured["bundle"] is bundle
    assert captured["video_ref"] == "/tmp/demo.mp4"
    assert captured["mm_processor_kwargs"] == {"fps": 2, "video_frame_count": 6}
    assert converted[0]["images"] == [b"frame-1", b"frame-2", b"frame-3"]
    assert "summarize the clip" in converted[0]["content"]
    assert "Video 1" in converted[0]["content"]


def test_stream_chat_text_passes_think_false_when_disabled(monkeypatch):
    bundle = _bundle(multimodal_mode="off")
    bundle.client = object()
    captured = {}

    async def _fake_call_chat_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStream(
            [
                {"message": {"content": "hello"}, "done": False},
                {"message": {"content": " world"}, "done": True, "done_reason": "stop"},
            ]
        )

    monkeypatch.setattr("text.runtime.ollama_bridge._call_chat_stream", _fake_call_chat_stream)

    async def _main():
        items = []
        async for item in __import__("text.runtime.ollama_bridge", fromlist=["stream_chat_text"]).stream_chat_text(
            bundle,
            messages=[{"role": "user", "content": "hi"}],
            request_id="req-think-off",
            enable_thinking=False,
        ):
            items.append(item)
        return items

    items = asyncio.run(_main())
    assert captured["think"] is False
    assert "".join(item["delta"] for item in items) == "hello world"


def test_stream_chat_text_surfaces_thinking_delta_when_enabled(monkeypatch):
    bundle = _bundle(multimodal_mode="off")
    bundle.client = object()

    async def _fake_call_chat_stream(**kwargs):
        return _FakeStream(
            [
                {"message": {"thinking": "step 1"}, "done": False},
                {"message": {"content": "final answer"}, "done": True, "done_reason": "stop"},
            ]
        )

    monkeypatch.setattr("text.runtime.ollama_bridge._call_chat_stream", _fake_call_chat_stream)

    async def _main():
        items = []
        async for item in __import__("text.runtime.ollama_bridge", fromlist=["stream_chat_text"]).stream_chat_text(
            bundle,
            messages=[{"role": "user", "content": "hi"}],
            request_id="req-think-on",
            enable_thinking=True,
        ):
            items.append(item)
        return items

    items = asyncio.run(_main())
    assert items[0]["delta"] == "step 1"
    assert items[0]["thinking_delta"] == "step 1"
    assert items[-1]["delta"] == "final answer"


def test_stream_chat_text_returns_message_when_video_mode_disabled(monkeypatch):
    bundle = _bundle(multimodal_mode="image")
    bundle.client = object()

    async def _fail_call_chat_stream(**kwargs):
        raise AssertionError("_call_chat_stream should not run when video is unsupported")

    monkeypatch.setattr("text.runtime.ollama_bridge._call_chat_stream", _fail_call_chat_stream)

    async def _main():
        items = []
        async for item in stream_chat_text(
            bundle,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "video_url", "video_url": {"url": "/tmp/demo.mp4"}}],
                }
            ],
            request_id="req-video-disabled",
        ):
            items.append(item)
        return items

    items = asyncio.run(_main())
    assert items == [
        {
            "delta": "当前所选 Ollama 服务未开启视频输入支持。请切换到支持视频抽帧输入的配置，或仅发送文本/图片。",
            "finished": True,
            "finish_reason": "unsupported_multimodal",
        }
    ]

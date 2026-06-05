# pyright: reportMissingImports=false

from pathlib import Path
import sys
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from text.runtime.common import chunk_text
from text.session_runtime import TextSessionRuntime


def test_chunk_text_helper():
    chunks = chunk_text("hello world from vitoom", chunk_size=10)
    assert len(chunks) >= 2


def test_text_session_runtime_emits_ready_delta_and_closed():
    sent = []
    requests = []

    async def sender(message):
        sent.append(message)
        return True

    async def stream_text(request):
        requests.append(request)
        yield {"delta": "hello ", "finished": False}
        yield {
            "delta": "from gemma",
            "finished": True,
            "prompt_tokens": 9,
            "output_tokens": 4,
            "ttft_seconds": 0.12,
            "tok_s_decode": 42.0,
        }

    async def main():
        runtime = TextSessionRuntime(sender=sender, stream_text=stream_text)

        assert await runtime.handle_message(
            {
                "type": "session_open",
                "session_id": "s1",
                "seq": 1,
                "model_name": "nvidia/Gemma-4-31B-IT-NVFP4",
                "family": "Gemma",
                "temperature": 0.2,
            }
        ) is True
        assert sent[-1]["type"] == "session_ready"

        assert await runtime.handle_message(
            {"type": "session_text_input", "session_id": "s1", "seq": 2, "text": "写一个 hello world"}
        ) is True
        assert requests[-1]["model_name"] == "nvidia/Gemma-4-31B-IT-NVFP4"
        assert requests[-1]["family"] == "Gemma"
        assert requests[-1]["messages"][-1]["content"] == "写一个 hello world"
        assert any(item["type"] == "llm_text_delta" for item in sent)
        final_delta = [item for item in sent if item["type"] == "llm_text_delta" and item.get("is_final")][-1]
        assert final_delta["prompt_tokens"] == 9
        assert final_delta["output_tokens"] == 4
        assert final_delta["ttft_seconds"] == 0.12
        assert final_delta["tok_s_decode"] == 42.0

        assert await runtime.handle_message({"type": "session_close", "session_id": "s1", "seq": 3}) is True
        assert sent[-1]["type"] == "session_closed"

    asyncio.run(main())


def test_text_session_runtime_uses_open_metadata_when_input_omits_model():
    sent = []

    async def sender(message):
        sent.append(message)
        return True

    async def stream_text(request):
        yield {"delta": f"reply for {request['family']}", "finished": True}

    async def main():
        runtime = TextSessionRuntime(sender=sender, stream_text=stream_text)

        await runtime.handle_message(
            {
                "type": "session_open",
                "session_id": "s2",
                "seq": 1,
                "model_name": "Qwen/Qwen3-8B",
                "family": "Qwen-text",
            }
        )
        await runtime.handle_message(
            {
                "type": "session_text_input",
                "session_id": "s2",
                "seq": 2,
                "text": "你好",
            }
        )

        assert any(item.get("delta") == "reply for Qwen-text" for item in sent if item["type"] == "llm_text_delta")

    asyncio.run(main())


def test_text_session_runtime_accepts_openai_messages_payload():
    sent = []
    requests = []

    async def sender(message):
        sent.append(message)
        return True

    async def stream_text(request):
        requests.append(request)
        yield {"delta": "multimodal reply", "finished": True}

    async def main():
        runtime = TextSessionRuntime(sender=sender, stream_text=stream_text)
        await runtime.handle_message(
            {
                "type": "session_open",
                "session_id": "s-mm",
                "seq": 1,
                "model_name": "Qwen/Qwen3.5-35B-A3B",
                "family": "Qwen-text",
            }
        )
        await runtime.handle_message(
            {
                "type": "session_text_input",
                "session_id": "s-mm",
                "seq": 2,
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
                                "text": "请描述图片内容",
                            },
                        ],
                    }
                ],
                "mm_processor_kwargs": {"fps": 2},
            }
        )

        assert requests[-1]["messages"][0]["content"][0]["type"] == "image_url"
        assert requests[-1]["messages"][0]["content"][1]["text"] == "请描述图片内容"
        assert requests[-1]["mm_processor_kwargs"] == {"fps": 2}
        assert any(item.get("delta") == "multimodal reply" for item in sent if item["type"] == "llm_text_delta")

    asyncio.run(main())


def test_text_session_runtime_forwards_tools_and_emits_tool_calls():
    """会话层需要透传 tools / tool_choice，并把模型 <tool_call> 输出重组成 tool_calls 事件。"""
    sent = []
    requests = []

    async def sender(message):
        sent.append(message)
        return True

    async def stream_text(request):
        requests.append(request)
        # 模拟模型先说一句自然语言，再发起一次 tool_call，随后结束。
        yield {"delta": "好的，我来调用工具：", "finished": False}
        yield {
            "delta": '<tool_call>\n{"name": "analyze_media", "arguments": {"url": "http://x/a.jpg"}}\n</tool_call>',
            "finished": False,
        }
        yield {
            "delta": "",
            "finished": True,
            "prompt_tokens": 10,
            "output_tokens": 7,
            "finish_reason": "stop",
        }

    async def main():
        runtime = TextSessionRuntime(sender=sender, stream_text=stream_text)
        await runtime.handle_message(
            {
                "type": "session_open",
                "session_id": "s-tool",
                "seq": 1,
                "model_name": "Qwen/Qwen3.5-35B-A3B",
                "family": "Qwen-text",
            }
        )
        await runtime.handle_message(
            {
                "type": "session_text_input",
                "session_id": "s-tool",
                "seq": 2,
                "messages": [{"role": "user", "content": "这张图是什么? http://x/a.jpg"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_media",
                            "description": "analyze an image",
                            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )

        assert requests[-1]["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "analyze_media",
                    "description": "analyze an image",
                    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                },
            }
        ]
        assert requests[-1]["tool_choice"] == "auto"

        deltas = [item for item in sent if item["type"] == "llm_text_delta"]
        # 自然语言那段保留在 delta.content
        assert any("好的" in (item.get("delta") or "") for item in deltas)
        # 任何一帧 content 里都不应出现 <tool_call> 片段
        assert not any("<tool_call>" in (item.get("delta") or "") for item in deltas)
        # 中间帧里应该出现 tool_calls_delta
        mid_tool_frames = [item for item in deltas if item.get("tool_calls_delta")]
        assert mid_tool_frames
        assert mid_tool_frames[0]["tool_calls_delta"][0]["function"]["name"] == "analyze_media"
        # 结束帧必须带完整 tool_calls 与 finish_reason=tool_calls
        final = [item for item in deltas if item.get("is_final")][-1]
        assert final["finish_reason"] == "tool_calls"
        assert final["tool_calls"][0]["function"]["name"] == "analyze_media"
        assert '"url": "http://x/a.jpg"' in final["tool_calls"][0]["function"]["arguments"]
        assert final["prompt_tokens"] == 10
        assert final["output_tokens"] == 7

    asyncio.run(main())


def test_text_session_runtime_handles_xml_style_tool_call_output():
    """Qwen3 / Hermes 风格的 XML <function=...> payload 也要被重组成 OpenAI tool_calls 事件。"""
    import json

    sent = []

    async def sender(message):
        sent.append(message)
        return True

    async def stream_text(request):
        # 分片模拟 token 流：头部、开 tag、函数名、参数、闭 tag、结束。
        yield {"delta": "我来调用工具：", "finished": False}
        yield {"delta": "<tool_call><function=analyze_media>", "finished": False}
        yield {"delta": "<parameter=url>\nhttp://x/a.jpg\n</parameter>", "finished": False}
        yield {"delta": "<parameter=question>\n这张图是什么？\n</parameter>", "finished": False}
        yield {"delta": "</function></tool_call>", "finished": False}
        yield {
            "delta": "",
            "finished": True,
            "prompt_tokens": 12,
            "output_tokens": 9,
            "finish_reason": "stop",
        }

    async def main():
        runtime = TextSessionRuntime(sender=sender, stream_text=stream_text)
        await runtime.handle_message(
            {
                "type": "session_open",
                "session_id": "s-xml",
                "seq": 1,
                "model_name": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
                "family": "Qwen-text",
            }
        )
        await runtime.handle_message(
            {
                "type": "session_text_input",
                "session_id": "s-xml",
                "seq": 2,
                "messages": [{"role": "user", "content": "这张图是什么? http://x/a.jpg"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_media",
                            "description": "analyze an image",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "question": {"type": "string"},
                                },
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )

        deltas = [item for item in sent if item["type"] == "llm_text_delta"]
        # plain content 要保留，但所有 XML 标签片段都不能泄漏给客户端
        assert any("我来调用工具" in (item.get("delta") or "") for item in deltas)
        for item in deltas:
            body = item.get("delta") or ""
            assert "<tool_call>" not in body
            assert "<function=" not in body
            assert "<parameter=" not in body
        # 最终帧带完整 tool_calls 与 finish_reason=tool_calls
        final = [item for item in deltas if item.get("is_final")][-1]
        assert final["finish_reason"] == "tool_calls"
        assert len(final["tool_calls"]) == 1
        call = final["tool_calls"][0]
        assert call["function"]["name"] == "analyze_media"
        args = json.loads(call["function"]["arguments"])
        assert args["url"] == "http://x/a.jpg"
        assert args["question"] == "这张图是什么？"

    asyncio.run(main())


def test_text_session_runtime_interrupt_aborts_active_request():
    sent = []
    aborted = []

    async def sender(message):
        sent.append(message)
        return True

    async def stream_text(request, *, started, release):
        started.set()
        await release.wait()
        yield {"delta": "late output", "finished": False}
        yield {"delta": "", "finished": True}

    async def abort_request(request):
        aborted.append(request)

    async def main():
        started = asyncio.Event()
        release = asyncio.Event()

        runtime = TextSessionRuntime(
            sender=sender,
            stream_text=lambda request: stream_text(request, started=started, release=release),
            abort_request=abort_request,
        )

        await runtime.handle_message(
            {
                "type": "session_open",
                "session_id": "s3",
                "seq": 1,
                "model_name": "nvidia/Gemma-4-31B-IT-NVFP4",
                "family": "Gemma",
            }
        )

        worker = asyncio.create_task(
            runtime.handle_message(
                {
                    "type": "session_text_input",
                    "session_id": "s3",
                    "seq": 2,
                    "text": "请开始输出",
                }
            )
        )
        await started.wait()
        await runtime.handle_message({"type": "session_interrupt", "session_id": "s3", "seq": 3})
        release.set()
        await worker

        assert aborted
        assert aborted[-1]["request_id"] == "session:s3:1"
        assert not any(item.get("delta") == "late output" for item in sent if item["type"] == "llm_text_delta")

    asyncio.run(main())

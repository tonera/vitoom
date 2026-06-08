import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.chat.master_runtime import _should_forward_llm_stream_chunk


def test_should_forward_llm_stream_chunk_rejects_tool_call_chunks():
    assert _should_forward_llm_stream_chunk(chunk='{"url":"http://x/a.jpg"}', call_type="tool_call") is False
    assert (
        _should_forward_llm_stream_chunk(
            chunk='{"url":"http://x/a.jpg"}',
            tool_call={"id": "call_1"},
        )
        is False
    )


def test_should_forward_llm_stream_chunk_accepts_normal_text_chunks():
    assert _should_forward_llm_stream_chunk(chunk="你好", call_type="llm_call") is True
    assert _should_forward_llm_stream_chunk(chunk="hello") is True

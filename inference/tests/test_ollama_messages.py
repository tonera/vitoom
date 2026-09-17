from text.runtime.ollama_messages import (
    fold_tool_roles_for_ollama,
    inline_tool_calls_and_fold,
    keep_openai_tool_pairing,
    wrap_qwen_tool_response,
)


def test_wrap_tool_response_is_idempotent() -> None:
    wrapped = wrap_qwen_tool_response('{"ok":true}')
    assert wrapped.startswith("<tool_response>")
    assert wrapped.endswith("</tool_response>")
    assert wrap_qwen_tool_response(wrapped) == wrapped


def test_fold_tool_roles_merges_consecutive_results() -> None:
    folded = fold_tool_roles_for_ollama(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plan"},
            {"role": "assistant", "content": "<tool_call>\n{}\n</tool_call>"},
            {"role": "tool", "content": '{"ok":true,"origin":"a"}'},
            {"role": "tool", "content": '{"ok":true,"origin":"b"}'},
        ]
    )
    assert [item["role"] for item in folded] == ["system", "user", "assistant", "user"]
    last = folded[-1]["content"]
    assert last.count("<tool_response>") == 2
    assert '"origin":"a"' in last
    assert '"origin":"b"' in last
    assert all(item["role"] != "tool" for item in folded)


def test_keep_openai_pairing_parses_argument_strings() -> None:
    converted = keep_openai_tool_pairing(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plan"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_route",
                            "arguments": '{"origin":"故宫","destination":"颐和园","mode":"transit"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok":false}',
            },
        ]
    )
    assert [item["role"] for item in converted] == ["system", "user", "assistant", "tool"]
    assistant = converted[2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == {
        "origin": "故宫",
        "destination": "颐和园",
        "mode": "transit",
    }
    assert converted[3]["tool_call_id"] == "call_1"


def test_inline_and_fold_drops_tool_role() -> None:
    folded = inline_tool_calls_and_fold(
        [
            {"role": "user", "content": "plan"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_route",
                            "arguments": '{"origin":"a","destination":"b","mode":"transit"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"ok":false}'},
        ]
    )
    assert [item["role"] for item in folded] == ["user", "assistant", "user"]
    assert "<tool_call>" in folded[1]["content"]
    assert "<tool_response>" in folded[2]["content"]
    assert "tool_calls" not in folded[1]

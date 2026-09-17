from text.runtime.ollama_messages import (
    fold_tool_roles_for_ollama,
    inject_qwen_tool_schema,
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


def test_inject_qwen_tool_schema_appends_to_system() -> None:
    tools = [{
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }]
    injected = inject_qwen_tool_schema(
        [{"role": "system", "content": "你是行程顾问。"}, {"role": "user", "content": "故宫"}],
        tools,
    )
    assert injected[0]["role"] == "system"
    assert "你是行程顾问。" in injected[0]["content"]
    assert "<tools>" in injected[0]["content"]
    assert "search_web" in injected[0]["content"]
    assert injected[1]["content"] == "故宫"
    again = inject_qwen_tool_schema(injected, tools)
    assert again[0]["content"].count("# Tools") == 1

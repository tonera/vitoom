#!/usr/bin/env python3
"""在推理器本机打 Ollama，一次只改一个变量，定位 500 EOF。

必须在 GPU 机器上跑，且绕过 HTTP_PROXY：

    python inference/text/diagnose_ollama_eof.py --model qwen3.8:27b
    python inference/text/diagnose_ollama_eof.py --model qwen3.8:27b --phase history
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_route",
        "description": "Look up distance and travel time",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "mode": {"type": "string", "enum": ["driving", "walking", "riding", "transit"]},
            },
            "required": ["origin", "destination", "mode"],
        },
    },
}

ROUTE_TOOL_STRICT = {
    "type": "function",
    "function": {
        "name": "lookup_route",
        "description": "Look up distance and travel time between two named places. Use driving, walking, riding, or transit.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Start place or town name."},
                "destination": {"type": "string", "description": "End place or town name."},
                "mode": {
                    "type": "string",
                    "enum": ["driving", "walking", "riding", "transit"],
                    "description": "driving=car, walking=on foot, riding=bicycle, transit=bus/rail.",
                },
            },
            "required": ["origin", "destination", "mode"],
            "additionalProperties": False,
        },
    },
}

PROD_TOOLS = [SEARCH_TOOL, ROUTE_TOOL]
ITINERARY_TOOLS = [SEARCH_TOOL, ROUTE_TOOL_STRICT]

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(url: str, body: dict[str, Any] | None, timeout: int) -> tuple[int, str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, raw
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def padded_user(nbytes: int) -> str:
    content = "只回复一个字：好"
    if nbytes <= 0:
        return content
    pad = "行程占位。"
    pad_bytes = pad.encode("utf-8")
    extra = max(0, nbytes - len(content.encode("utf-8")))
    content = content + "\n" + pad * (extra // len(pad_bytes) + 1)
    encoded = content.encode("utf-8")
    return encoded[:nbytes].decode("utf-8", errors="ignore")


def report(label: str, status: int, preview: str) -> None:
    ok = status == 200 and "EOF" not in preview
    snippet = preview.replace("\n", " ")[:160]
    print(f"{'PASS' if ok else 'FAIL'} {label}  http={status}  {snippet}", flush=True)


def post_chat(base: str, timeout: int, body: dict[str, Any], label: str) -> None:
    packed = json.dumps(body, ensure_ascii=False)
    nmsg = len(body.get("messages") or [])
    label = f"msgs={nmsg} bytes={len(packed.encode('utf-8'))} {label}"
    status, preview = request_json(base.rstrip("/") + "/api/chat", body, timeout)
    report(label, status, preview)


def chat_body(
    model: str,
    messages: list[dict[str, Any]],
    *,
    stream: bool,
    tools: list[dict[str, Any]] | None,
    num_ctx: int | None,
    think: Any,
    num_predict: int = 16,
    extra_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if extra_options:
        options.update(extra_options)
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": "5m",
        "options": options,
    }
    if tools:
        body["tools"] = tools
    if think != "<omit>":
        body["think"] = think
    return body


def run_case(base: str, model: str, timeout: int, *, user_bytes: int = 0, **flags: Any) -> None:
    content = padded_user(user_bytes)
    tools = [SEARCH_TOOL] if flags.get("tools") else None
    body = chat_body(
        model,
        [{"role": "user", "content": content}],
        stream=bool(flags["stream"]),
        tools=tools,
        num_ctx=flags.get("num_ctx"),
        think=flags.get("think", "<omit>"),
    )
    post_chat(
        base,
        timeout,
        body,
        (
            f"stream={body['stream']} tools={bool(tools)} "
            f"num_ctx={flags.get('num_ctx')} think={flags.get('think', '<omit>')}"
        ),
    )


def route_args(i: int) -> dict[str, str]:
    return {"origin": f"故宫门口{i}", "destination": f"颐和园东门{i}", "mode": "driving"}


def route_result(i: int) -> str:
    return json.dumps(
        {
            "ok": True,
            "origin": f"故宫门口{i}",
            "destination": f"颐和园东门{i}",
            "mode": "driving",
            "durationMinutes": 35 + i,
            "distanceMeters": 18000 + i * 100,
        },
        ensure_ascii=False,
    )


def xml_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def xml_tool_response(content: str) -> str:
    return f"<tool_response>\n{content}\n</tool_response>"


def history_inline_fold(rounds: int, *, parallel: int = 1) -> list[dict[str, Any]]:
    """推理器当前实际发出的形态：tool_calls 内联成文本，role=tool 折成 user。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你是行程规划助手。需要里程或用时时调用 lookup_route。"},
        {"role": "user", "content": "帮我规划一个北京5日游。只回复一个字：好。"},
    ]
    for r in range(rounds):
        calls = [xml_tool_call("lookup_route", route_args(r * parallel + j)) for j in range(parallel)]
        replies = [xml_tool_response(route_result(r * parallel + j)) for j in range(parallel)]
        messages.append({"role": "assistant", "content": "\n".join(calls)})
        messages.append({"role": "user", "content": "\n".join(replies)})
    return messages


def history_native(rounds: int, *, parallel: int = 1) -> list[dict[str, Any]]:
    """Ollama tools= 期望的 OpenAI 配对：assistant.tool_calls ↔ role=tool。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你是行程规划助手。需要里程或用时时调用 lookup_route。"},
        {"role": "user", "content": "帮我规划一个北京5日游。只回复一个字：好。"},
    ]
    for r in range(rounds):
        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_route",
                    "arguments": route_args(r * parallel + j),
                },
            }
            for j in range(parallel)
        ]
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for j in range(parallel):
            messages.append(
                {
                    "role": "tool",
                    "content": route_result(r * parallel + j),
                }
            )
    return messages


YAML_CTX = 60480
YAML_PREDICT = 60480


def run_yaml(base: str, model: str, timeout: int) -> None:
    """对齐 aigc_text_local_5090.yaml：num_ctx=max_model_len=60480，num_predict=max_tokens=60480。"""
    print("--- yaml aigc_text_local_5090 num_ctx=60480 think=false ---", flush=True)
    tiny = [{"role": "user", "content": "只回复一个字：好"}]
    fat = history_inline_fold(1, parallel=10)
    fat[0]["content"] = padded_user(18000)

    cases: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        (
            "tiny predict=16 tools",
            tiny,
            dict(stream=True, tools=PROD_TOOLS, num_ctx=YAML_CTX, think=False, num_predict=16),
        ),
        (
            "tiny predict=2048 tools",
            tiny,
            dict(stream=True, tools=PROD_TOOLS, num_ctx=YAML_CTX, think=False, num_predict=2048),
        ),
        (
            "tiny predict=60480 tools",
            tiny,
            dict(stream=True, tools=PROD_TOOLS, num_ctx=YAML_CTX, think=False, num_predict=YAML_PREDICT),
        ),
        (
            "tiny predict=60480 notools",
            tiny,
            dict(stream=True, tools=None, num_ctx=YAML_CTX, think=False, num_predict=YAML_PREDICT),
        ),
        (
            "tiny predict=60480 tools nostream",
            tiny,
            dict(stream=False, tools=PROD_TOOLS, num_ctx=YAML_CTX, think=False, num_predict=YAML_PREDICT),
        ),
        (
            "tiny predict=60480 tools num_gpu=-1",
            tiny,
            dict(
                stream=True,
                tools=PROD_TOOLS,
                num_ctx=YAML_CTX,
                think=False,
                num_predict=YAML_PREDICT,
                extra_options={"num_gpu": -1},
            ),
        ),
        (
            "fold10 predict=16 tools",
            fat,
            dict(stream=True, tools=PROD_TOOLS, num_ctx=YAML_CTX, think=False, num_predict=16),
        ),
        (
            "fold10 predict=60480 tools num_gpu=-1",
            fat,
            dict(
                stream=True,
                tools=PROD_TOOLS,
                num_ctx=YAML_CTX,
                think=False,
                num_predict=YAML_PREDICT,
                extra_options={"num_gpu": -1},
            ),
        ),
    ]
    for name, messages, kwargs in cases:
        body = chat_body(model, messages, **kwargs)
        opts = body["options"]
        post_chat(
            base,
            timeout,
            body,
            (
                f"{name} stream={body['stream']} tools={bool(kwargs.get('tools'))} "
                f"num_ctx={opts.get('num_ctx')} num_predict={opts.get('num_predict')} "
                f"num_gpu={opts.get('num_gpu', '<omit>')} think={kwargs.get('think')}"
            ),
        )


def run_history(base: str, model: str, timeout: int) -> None:
    lock = dict(stream=True, num_ctx=65536, think=False)
    print("--- history shape think=false num_ctx=65536 stream=True ---", flush=True)

    # 1 轮：先确认形态本身会不会炸
    for name, builder, tools in (
        ("inline_fold", history_inline_fold, PROD_TOOLS),
        ("native_pair", history_native, PROD_TOOLS),
        ("inline_fold_notools", history_inline_fold, None),
    ):
        messages = builder(1)
        post_chat(
            base,
            timeout,
            chat_body(model, messages, tools=tools, **lock),
            f"shape={name} rounds=1 parallel=1 tools={bool(tools)} stream=True",
        )

    # 多轮：对齐地点清单连搜
    for rounds in (4, 8):
        for name, builder in (("inline_fold", history_inline_fold), ("native_pair", history_native)):
            messages = builder(rounds)
            post_chat(
                base,
                timeout,
                chat_body(model, messages, tools=PROD_TOOLS, **lock),
                f"shape={name} rounds={rounds} parallel=1 tools=True stream=True",
            )

    # 一轮里 10 条查路结果：对齐串日那次 502
    for name, builder in (("inline_fold", history_inline_fold), ("native_pair", history_native)):
        messages = builder(1, parallel=10)
        post_chat(
            base,
            timeout,
            chat_body(model, messages, tools=PROD_TOOLS, **lock),
            f"shape={name} rounds=1 parallel=10 tools=True stream=True",
        )

    # 非流式对照：只打生产形态（inline_fold + 10 路）
    messages = history_inline_fold(1, parallel=10)
    post_chat(
        base,
        timeout,
        chat_body(model, messages, stream=False, tools=PROD_TOOLS, num_ctx=65536, think=False),
        "shape=inline_fold rounds=1 parallel=10 tools=True stream=False",
    )


def run_tiny(base: str, model: str, timeout: int) -> None:
    print("--- tiny ---", flush=True)
    run_case(base, model, timeout, stream=True, tools=False, num_ctx=None)
    run_case(base, model, timeout, stream=False, tools=False, num_ctx=None)
    run_case(base, model, timeout, stream=True, tools=True, num_ctx=None)
    run_case(base, model, timeout, stream=False, tools=True, num_ctx=None)
    run_case(base, model, timeout, stream=True, tools=True, num_ctx=4096)
    run_case(base, model, timeout, stream=True, tools=True, num_ctx=65536)
    run_case(base, model, timeout, stream=True, tools=True, num_ctx=None, think=False)
    run_case(base, model, timeout, stream=False, tools=True, num_ctx=None, think=False)


def run_size(base: str, model: str, timeout: int) -> None:
    print("--- production-shaped think=false num_ctx=65536 ---", flush=True)
    for nbytes in (500, 4500, 21000):
        for stream in (True, False):
            for tools in (False, True):
                run_case(
                    base,
                    model,
                    timeout,
                    user_bytes=nbytes,
                    stream=stream,
                    tools=tools,
                    num_ctx=65536,
                    think=False,
                )


def run_sdk(base: str, model: str, timeout: int) -> None:
    print("--- python ollama.Client same payload as yaml fold10 ---", flush=True)
    try:
        from ollama import Client
    except Exception as exc:
        print(f"SKIP sdk: cannot import ollama ({exc})", flush=True)
        return

    client = Client(host=base, trust_env=False)
    messages = history_inline_fold(1, parallel=10)
    messages[0]["content"] = padded_user(18000)
    native = history_native(1, parallel=10)
    cases = [
        ("sdk inline_fold urllib-twin", messages, PROD_TOOLS),
        ("sdk inline_fold itinerary-schema", messages, ITINERARY_TOOLS),
        ("sdk native_pair itinerary-schema", native, ITINERARY_TOOLS),
    ]
    for name, msgs, tools in cases:
        try:
            stream = client.chat(
                model=model,
                messages=msgs,
                stream=True,
                think=False,
                keep_alive="5m",
                options={"num_predict": 16, "num_ctx": YAML_CTX, "num_gpu": -1, "temperature": 0.2},
                tools=tools,
            )
            preview = ""
            for chunk in stream:
                dumped = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                preview = json.dumps(dumped, ensure_ascii=False)
                break
            packed = json.dumps({"messages": msgs, "tools": tools}, ensure_ascii=False)
            print(
                f"PASS msgs={len(msgs)} bytes={len(packed.encode('utf-8'))} {name}  {preview[:160]}",
                flush=True,
            )
        except Exception as exc:
            print(f"FAIL {name}  {type(exc).__name__}: {exc}", flush=True)


def replay_openai_request(base: str, model: str, timeout: int, path: str) -> None:
    print(f"--- openai-request {path} ---", flush=True)
    inference_root = str(Path(__file__).resolve().parents[1])
    if inference_root not in sys.path:
        sys.path.insert(0, inference_root)
    from text.runtime.ollama_messages import inline_tool_calls_and_fold, keep_openai_tool_pairing

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    body = payload.get("request") if isinstance(payload.get("request"), dict) else payload
    messages = list(body.get("messages") or [])
    tools = body.get("tools") or PROD_TOOLS
    lock = dict(stream=True, num_ctx=YAML_CTX, think=False, num_predict=16, extra_options={"num_gpu": -1})
    cases = [
        ("native pairing + tools", keep_openai_tool_pairing(messages), tools),
        ("inline+fold + tools  (old path)", inline_tool_calls_and_fold(messages), tools),
        ("native pairing no tools", keep_openai_tool_pairing(messages), None),
        ("inline+fold no tools", inline_tool_calls_and_fold(messages), None),
    ]
    for name, converted, case_tools in cases:
        post_chat(
            base,
            timeout,
            chat_body(model, converted, tools=case_tools, **lock),
            f"{name} roles={[item.get('role') for item in converted]}",
        )


def replay_dump(base: str, timeout: int, path: str) -> None:
    print(f"--- replay {path} ---", flush=True)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    kwargs = payload.get("kwargs") if isinstance(payload.get("kwargs"), dict) else payload
    body = {
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages") or [],
        "stream": bool(kwargs.get("stream", True)),
        "think": kwargs.get("think", False),
        "keep_alive": kwargs.get("keep_alive", "5m"),
        "options": kwargs.get("options") or {},
    }
    if kwargs.get("tools"):
        body["tools"] = kwargs["tools"]
    post_chat(base, timeout, body, "replay urllib")

    try:
        from ollama import Client
    except Exception as exc:
        print(f"SKIP sdk replay: cannot import ollama ({exc})", flush=True)
        return
    client = Client(host=base, trust_env=False)
    try:
        stream = client.chat(
            model=str(body["model"]),
            messages=list(body["messages"]),
            stream=bool(body["stream"]),
            think=body.get("think"),
            keep_alive=body.get("keep_alive"),
            options=body.get("options"),
            tools=body.get("tools"),
        )
        if body["stream"]:
            for chunk in stream:
                dumped = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                print(f"PASS replay sdk  {json.dumps(dumped, ensure_ascii=False)[:160]}", flush=True)
                break
        else:
            dumped = stream.model_dump() if hasattr(stream, "model_dump") else dict(stream)
            print(f"PASS replay sdk  {json.dumps(dumped, ensure_ascii=False)[:160]}", flush=True)
    except Exception as exc:
        print(f"FAIL replay sdk  {type(exc).__name__}: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--phase",
        default="sdk",
        choices=("tiny", "size", "history", "yaml", "sdk", "all"),
        help="默认对照 Python ollama.Client。",
    )
    parser.add_argument("--replay", default="", help="重放推理器落盘的 /tmp/vitoom-ollama-eof.json")
    parser.add_argument(
        "--openai-request",
        default="",
        help="重放 trace 落盘的 OpenAI 请求，对照 native pairing vs 旧 XML fold",
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    ver_status, ver_body = request_json(base + "/api/version", None, 10)
    tags_status, tags_body = request_json(base + "/api/tags", None, 10)
    print(f"version http={ver_status} {ver_body[:200]}", flush=True)
    if tags_status == 200:
        names = [item.get("name") for item in json.loads(tags_body).get("models", [])]
        print(f"tags http={tags_status} models={names}", flush=True)
    else:
        print(f"tags http={tags_status} {tags_body[:200]}", flush=True)
        return 1

    if args.replay:
        replay_dump(base, args.timeout, args.replay)
        return 0
    if args.openai_request:
        replay_openai_request(base, args.model, args.timeout, args.openai_request)
        return 0
    if args.phase in ("tiny", "all"):
        run_tiny(base, args.model, args.timeout)
    if args.phase in ("size", "all"):
        run_size(base, args.model, args.timeout)
    if args.phase in ("history", "all"):
        run_history(base, args.model, args.timeout)
    if args.phase in ("yaml", "all"):
        run_yaml(base, args.model, args.timeout)
    if args.phase in ("sdk", "all"):
        run_sdk(base, args.model, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

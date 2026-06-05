#!/usr/bin/env python3
"""Vitoom 统一会话命令行工具。

默认模式走新的统一 chat 入口：

1. `POST /v1/chat/sessions` 创建会话
2. `WS /ws/chat/{session_id}` 发送 `user_message`
3. `GET /v1/chat/sessions/{session_id}/messages` 查看历史

示例：
    python scripts/agent_chatbox.py
    python scripts/agent_chatbox.py -m "规划东京3日游"
    python scripts/agent_chatbox.py --resume <session_id>

调试开关：
    --force-agent <id>   创建新会话时显式绑定某个 agent_id
    --raw                直接调用 /v1/agents/runs 调试单次运行
    --token <jwt>        显式指定 JWT
    --email <addr>       自动挑用户时优先选这个邮箱

交互命令：
    /exit                退出
    /new                 新建会话并重连
    /resume <session_id> 切到已有会话
    /messages [N]        打印当前会话最近历史
    /help                显示帮助
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("This CLI requires `requests`. Install via: pip install requests")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("This CLI requires `websockets`. Install via: pip install websockets")
    sys.exit(1)


AGENT_ALIASES: Dict[str, str] = {
    "local": "preset-local-agent",
    "travel": "preset-travel-planner-agent",
    "openclaw": "preset-openclaw-agent",
    "master": "preset-master-agent",
}

DEFAULT_TOKEN_CACHE = Path.home() / ".vitoom" / "agent_cli_token.json"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
RAW_INFERENCE_NOISE = {
    "llm_text_delta",
    "text_stream_delta",
    "transcript_partial",
    "transcript_final",
    "transcript_segment",
    "audio_stream_start",
    "audio_stream_chunk",
    "audio_stream_end",
    "audio_chunk",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JWT_SECRET = "vitoom-default-secret-key-change-in-production"
DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE = 86400


def _print_err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def _unwrap(payload: Dict[str, Any]) -> Any:
    if isinstance(payload, dict) and "data" in payload and "code" in payload:
        return payload.get("data")
    return payload


def _load_cached_token(base_url: str) -> Optional[str]:
    if not DEFAULT_TOKEN_CACHE.exists():
        return None
    try:
        with DEFAULT_TOKEN_CACHE.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        return data.get(base_url)
    except Exception:
        return None


def _save_cached_token(base_url: str, token: str) -> None:
    try:
        DEFAULT_TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, str] = {}
        if DEFAULT_TOKEN_CACHE.exists():
            try:
                with DEFAULT_TOKEN_CACHE.open("r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
            except Exception:
                data = {}
        data[base_url] = token
        with DEFAULT_TOKEN_CACHE.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        try:
            os.chmod(DEFAULT_TOKEN_CACHE, 0o600)
        except Exception:
            pass
    except Exception as exc:
        _print_err(f"failed to cache token: {exc}")


def _resolve_agent_alias(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = raw.strip()
    if not key:
        return None
    return AGENT_ALIASES.get(key.lower(), key)


def _ws_base_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :].rstrip("/")
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :].rstrip("/")
    return base_url.rstrip("/")


def _format_usage_line(usage: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(usage, dict):
        return None

    def _fmt_int(value: Any) -> str:
        try:
            if value is None:
                return "-"
            return str(int(value))
        except Exception:
            return "-"

    def _fmt_float(value: Any, suffix: str = "") -> str:
        try:
            if value is None:
                return "-"
            return f"{float(value):.2f}{suffix}"
        except Exception:
            return "-"

    total = usage.get("total_tokens")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    elapsed = usage.get("elapsed_seconds")
    speed = usage.get("tokens_per_second")
    if all(v is None for v in (total, prompt, completion, elapsed, speed)):
        return None
    parts = [
        f"tokens={_fmt_int(total)}",
        f"(prompt={_fmt_int(prompt)}, completion={_fmt_int(completion)})",
    ]
    if elapsed is not None:
        parts.append(f"elapsed={_fmt_float(elapsed, 's')}")
    if speed is not None:
        parts.append(f"speed={_fmt_float(speed)} tok/s")
    return "[usage] " + "  ".join(parts)


def _load_security_settings() -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {}

    merged: Dict[str, Any] = {}
    for candidate in (PROJECT_ROOT / "config" / "default.yaml", PROJECT_ROOT / "config" / "app.yaml"):
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("security"), dict):
            merged.update(data["security"])
    return merged


def _create_access_token_local(payload: Dict[str, Any]) -> str:
    try:
        from jose import jwt
    except Exception as exc:
        raise RuntimeError(
            "需要安装 python-jose 才能本地签发 JWT；可 `pip install python-jose`，或改用 --token 登录"
        ) from exc

    security = _load_security_settings()
    jwt_cfg = security.get("jwt") if isinstance(security.get("jwt"), dict) else {}
    secret = str(jwt_cfg.get("secret_key") or DEFAULT_JWT_SECRET)
    algorithm = str(jwt_cfg.get("algorithm") or DEFAULT_JWT_ALGORITHM)
    access_token_expire = int(jwt_cfg.get("access_token_expire") or DEFAULT_ACCESS_TOKEN_EXPIRE)
    now = datetime.utcnow()
    body = dict(payload)
    body.update(
        {
            "exp": now + timedelta(seconds=access_token_expire),
            "iat": now,
            "type": "access",
        }
    )
    return jwt.encode(body, secret, algorithm=algorithm)


def _pick_existing_user(preferred_email: Optional[str] = None) -> Tuple[str, str]:
    try:
        from backend.database import User
    except Exception as exc:
        raise RuntimeError(
            "无法导入 backend.database.User 自动选用户；请 `conda activate vitoom` 后重试，或显式传 --token"
        ) from exc

    users = User.list_all(limit=50) or []
    if not users:
        raise RuntimeError("users 表里没有可用账号，请先创建账号，或用 --token 登录")

    if preferred_email:
        target = preferred_email.strip().lower()
        for user in users:
            if str(user.get("email") or "").strip().lower() == target:
                return str(user.get("id") or ""), str(user.get("email") or "")

    active = next((u for u in users if str(u.get("status") or "").lower() == "active"), users[0])
    return str(active.get("id") or ""), str(active.get("email") or "")


def _auto_issue_token(preferred_email: Optional[str]) -> Tuple[str, str, str]:
    user_id, email = _pick_existing_user(preferred_email)
    if not user_id:
        raise RuntimeError("挑到的用户缺 id，放弃自动签 token")
    token = _create_access_token_local({"sub": user_id, "email": email})
    return token, user_id, email


class VitoomClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def login(self, email: str, password: str) -> str:
        resp = self._session.post(
            f"{self.base_url}/api/auth/login",
            json={"email": email, "password": password},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = _unwrap(resp.json()) or {}
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"login response missing access_token: {resp.text}")
        self.token = token
        return token

    def list_agents(self) -> List[Dict[str, Any]]:
        resp = self._session.get(f"{self.base_url}/v1/agents", headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = _unwrap(resp.json()) or {}
        return list(data.get("agents") or [])

    def create_run(self, *, agent_id: str, message: str) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/v1/agents/runs",
            json={
                "agent_id": agent_id,
                "message": message,
                "source_type": "cli",
                "context": {},
                "runtime_config": {},
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _unwrap(resp.json()) or {}

    def get_run(self, run_id: str) -> Dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}/v1/agents/runs/{run_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _unwrap(resp.json()) or {}

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/v1/agents/runs/{run_id}/cancel",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _unwrap(resp.json()) or {}

    def create_chat_session(
        self,
        *,
        title: Optional[str] = None,
        agent_id: Optional[str] = None,
        load_name: Optional[str] = None,
        input_mode: str = "text",
        output_mode: str = "text_stream",
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "input_mode": input_mode,
            "output_mode": output_mode,
            "metadata": {"source": "agent_chatbox"},
        }
        if title:
            body["title"] = title
        if agent_id:
            body["agent_id"] = agent_id
        if load_name:
            body["load_name"] = load_name
        resp = self._session.post(
            f"{self.base_url}/v1/chat/sessions",
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = _unwrap(resp.json()) or {}
        if not isinstance(data, dict) or not data.get("id"):
            raise RuntimeError(f"unexpected create_chat_session response: {data}")
        return data

    def get_chat_session(self, session_id: str) -> Dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}/v1/chat/sessions/{session_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = _unwrap(resp.json()) or {}
        if not isinstance(data, dict) or not data.get("id"):
            raise RuntimeError(f"unexpected get_chat_session response: {data}")
        return data

    def list_chat_messages(self, session_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        resp = self._session.get(
            f"{self.base_url}/v1/chat/sessions/{session_id}/messages",
            headers=self._headers(),
            timeout=self.timeout,
            params={"limit": limit, "offset": 0},
        )
        resp.raise_for_status()
        data = _unwrap(resp.json()) or {}
        return list(data.get("items") or [])

    def wait_for_run(
        self,
        run_id: str,
        *,
        poll_interval: float = 1.5,
        max_seconds: float = 600.0,
        on_tick=None,
    ) -> Dict[str, Any]:
        deadline = time.time() + max_seconds
        while True:
            run = self.get_run(run_id)
            status = str(run.get("status") or "").strip().lower()
            if on_tick is not None:
                try:
                    on_tick(run)
                except Exception:
                    pass
            if status in TERMINAL_STATUSES:
                return run
            if time.time() > deadline:
                raise TimeoutError(f"agent run {run_id} did not complete within {max_seconds:.0f}s")
            time.sleep(poll_interval)


def _render_agent_row(index: int, agent: Dict[str, Any]) -> str:
    badge = []
    if agent.get("is_preset"):
        badge.append("preset")
    badge.append(str(agent.get("type") or "general"))
    badge_str = f" [{'/'.join(badge)}]" if badge else ""
    description = str(agent.get("description") or "").strip() or "(no description)"
    return f"  {index:>2}. {agent.get('name'):<20} {agent.get('id'):<36}{badge_str}\n      {description}"


def _pick_agent(client: VitoomClient, preferred: Optional[str]) -> Optional[Dict[str, Any]]:
    agents = client.list_agents()
    if not agents:
        _print_err("no agents available on this server")
        return None
    if preferred:
        normalized = AGENT_ALIASES.get(preferred.strip().lower(), preferred.strip())
        for agent in agents:
            if agent.get("id") == normalized:
                return agent
        _print_err(f"agent not found: {preferred} (resolved to {normalized})")
        return None
    print("\navailable agents:")
    for idx, agent in enumerate(agents, start=1):
        print(_render_agent_row(idx, agent))
    while True:
        raw = input("\npick an agent (number / id / alias): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(agents):
                return agents[i - 1]
            print("invalid number, try again")
            continue
        normalized = AGENT_ALIASES.get(raw.lower(), raw)
        for agent in agents:
            if agent.get("id") == normalized:
                return agent
        print("no agent matched, try again")


def _on_poll_tick(run: Dict[str, Any]) -> None:
    status = str(run.get("status") or "")
    if status in TERMINAL_STATUSES:
        return
    sys.stdout.write(f"\r[waiting] status={status} updated={run.get('updated_at') or ''}   ")
    sys.stdout.flush()


def _print_run_result(run: Dict[str, Any]) -> None:
    status = str(run.get("status") or "")
    summary = run.get("result_summary")
    error = str(run.get("error_message") or "").strip()
    print()
    print(f"--- run {run.get('id')} [{status}] ---")
    if error:
        print(f"[error] {error}")
    if isinstance(summary, dict):
        text = str(summary.get("text") or "").strip()
    else:
        text = str(summary or "").strip()
    if text:
        print(text)
    elif not error:
        print("(no output text captured; check server logs)")
    usage_line = _format_usage_line(run.get("usage_metrics"))
    if usage_line:
        print(usage_line)


async def _wait_for_session_ready(ws) -> Dict[str, Any]:
    while True:
        raw = await ws.recv()
        event = json.loads(raw)
        etype = str(event.get("type") or "")
        if etype in RAW_INFERENCE_NOISE:
            continue
        if etype == "session_ready":
            return event
        if etype == "error":
            raise RuntimeError(str((event.get("payload") or {}).get("message") or event))


def _print_session_header(session: Dict[str, Any]) -> None:
    title = str(session.get("title") or "").strip() or "(untitled)"
    print(f"\n[session] id={session.get('id')}  title={title}")


async def _connect_chat_ws(base_url: str, token: str, session_id: str):
    ws_url = f"{_ws_base_url(base_url)}/ws/chat/{session_id}?token={token}"
    ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024)
    await _wait_for_session_ready(ws)
    return ws


def _print_history_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no messages)")
        return
    print()
    for msg in rows:
        role = str(msg.get("role") or "").strip() or "?"
        ts = str(msg.get("created_at") or "")
        content = str(msg.get("content") or "").strip()
        print(f"[{ts} {role}] {content}")


async def _send_user_message(ws, text: str) -> None:
    await ws.send(
        json.dumps(
            {
                "type": "user_message",
                "payload": {
                    "role": "user",
                    "text": text,
                    "content_type": "text",
                },
            },
            ensure_ascii=False,
        )
    )


async def _consume_turn(ws) -> Dict[str, Any]:
    printed_delta = False
    while True:
        raw = await ws.recv()
        event = json.loads(raw)
        etype = str(event.get("type") or "")
        if etype in RAW_INFERENCE_NOISE:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        if etype == "message_started":
            print("assistant> ", end="", flush=True)
            continue
        if etype == "message_delta":
            delta = str(payload.get("delta") or "")
            print(delta, end="", flush=True)
            if delta:
                printed_delta = True
            continue
        if etype == "tool_call_started":
            tool_name = str(payload.get("tool_name") or "").strip() or "unknown"
            print(f"\n[tool] start {tool_name}")
            continue
        if etype == "tool_call_completed":
            tool_name = str(payload.get("tool_name") or "").strip() or "unknown"
            print(f"\n[tool] done  {tool_name}")
            continue
        if etype == "tool_call_failed":
            tool_name = str(payload.get("tool_name") or "").strip() or "unknown"
            print(f"\n[tool] fail  {tool_name}: {payload.get('error') or ''}")
            continue
        if etype == "artifact_created":
            url = str(payload.get("http_url") or payload.get("storage_path") or "").strip()
            category = str(payload.get("category") or "").strip()
            print(f"\n[artifact] {category or 'file'} {url}")
            continue
        if etype == "status_changed":
            state = str(payload.get("state") or "").strip()
            task_id = str(payload.get("task_id") or "").strip()
            task_status = str(payload.get("task_status") or "").strip()
            task_kind = str(payload.get("task_kind") or "").strip()
            bits = [f"state={state}"] if state else []
            if task_kind:
                bits.append(f"kind={task_kind}")
            if task_status:
                bits.append(f"task_status={task_status}")
            if task_id:
                bits.append(f"task_id={task_id}")
            if bits:
                print(f"\n[status] {' '.join(bits)}")
            continue
        if etype == "error":
            message = str(payload.get("message") or event)
            print(f"\n[error] {message}")
            return event
        if etype == "message_completed":
            content = str(payload.get("content") or "")
            if not printed_delta and content:
                print(f"assistant> {content}", end="", flush=True)
            if printed_delta or content:
                print()
            usage_line = _format_usage_line(payload.get("usage_metrics"))
            if usage_line:
                print(usage_line)
            return event
        if etype == "session_closed":
            print("\n[info] session closed")
            return event


async def run_chat_once(client: VitoomClient, args: argparse.Namespace, session: Dict[str, Any]) -> int:
    ws = await _connect_chat_ws(client.base_url, client.token or "", str(session["id"]))
    try:
        print(f"user> {args.message}")
        await _send_user_message(ws, str(args.message))
        result = await _consume_turn(ws)
        return 0 if str(result.get("type") or "") == "message_completed" else 1
    finally:
        try:
            await ws.send(json.dumps({"type": "session_close", "payload": {}}, ensure_ascii=False))
        except Exception:
            pass
        await ws.close()


async def run_chat_repl(
    client: VitoomClient,
    args: argparse.Namespace,
    session: Dict[str, Any],
    *,
    force_agent_id: Optional[str],
) -> int:
    current_session = dict(session)
    ws = await _connect_chat_ws(client.base_url, client.token or "", str(current_session["id"]))
    _print_session_header(current_session)
    print("直接输入消息即可发送。用 /help 查看命令，/exit 退出。\n")
    try:
        while True:
            try:
                prompt = f"[chat {str(current_session.get('id') or '')[:8]}]> "
                raw = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            message = raw.strip()
            if not message:
                continue
            if message.startswith("/"):
                parts = message.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""
                if cmd in {"/exit", "/quit", "/q"}:
                    return 0
                if cmd in {"/help", "/?"}:
                    print(
                        "commands:\n"
                        "  /exit                退出\n"
                        "  /new                 新建会话并重连\n"
                        "  /resume <session_id> 切到已有会话\n"
                        "  /messages [N]        打印当前会话最近 N 条历史消息（默认 20）\n"
                        "  /help                显示本帮助"
                    )
                    continue
                if cmd == "/messages":
                    try:
                        limit = int(arg) if arg else 20
                    except ValueError:
                        limit = 20
                    rows = await asyncio.to_thread(
                        client.list_chat_messages,
                        str(current_session["id"]),
                        limit=max(1, min(200, limit)),
                    )
                    _print_history_rows(rows)
                    continue
                if cmd == "/new":
                    new_session = await asyncio.to_thread(
                        client.create_chat_session,
                        title=args.title,
                        agent_id=force_agent_id,
                        load_name=args.load_name,
                        input_mode=args.input_mode,
                        output_mode=args.output_mode,
                    )
                    await ws.close()
                    current_session = dict(new_session)
                    ws = await _connect_chat_ws(client.base_url, client.token or "", str(current_session["id"]))
                    _print_session_header(current_session)
                    continue
                if cmd == "/resume":
                    if not arg:
                        _print_err("/resume requires a session_id")
                        continue
                    resumed = await asyncio.to_thread(client.get_chat_session, arg)
                    await ws.close()
                    current_session = dict(resumed)
                    ws = await _connect_chat_ws(client.base_url, client.token or "", str(current_session["id"]))
                    _print_session_header(current_session)
                    continue
                _print_err(f"unknown command: {cmd}")
                continue

            print(f"user> {message}")
            await _send_user_message(ws, message)
            await _consume_turn(ws)
    finally:
        try:
            await ws.send(json.dumps({"type": "session_close", "payload": {}}, ensure_ascii=False))
        except Exception:
            pass
        await ws.close()


def run_raw_repl(client: VitoomClient, agent: Dict[str, Any]) -> None:
    print(
        f"\n[raw mode] Talking directly to agent: {agent.get('name')} ({agent.get('id')})"
        "\nType your message and press Enter. Use /exit to quit, /help for commands.\n"
    )
    current_agent = agent
    while True:
        try:
            message = input(f"[raw {current_agent.get('id')}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message.startswith("/"):
            parts = message.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in {"/exit", "/quit", "/q"}:
                return
            if cmd in {"/help", "/?"}:
                print(
                    "commands (raw mode):\n"
                    "  /exit                退出\n"
                    "  /agent <id>          切换 agent\n"
                    "  /list                列出 agent\n"
                    "  /run <run_id>        查看某次运行结果\n"
                    "  /help                显示本帮助"
                )
                continue
            if cmd == "/list":
                for idx, item in enumerate(client.list_agents(), start=1):
                    print(_render_agent_row(idx, item))
                continue
            if cmd == "/agent":
                new_agent = _pick_agent(client, arg or None)
                if new_agent:
                    current_agent = new_agent
                    print(f"switched to agent: {current_agent.get('name')} ({current_agent.get('id')})")
                continue
            if cmd == "/run":
                if not arg:
                    _print_err("/run requires a run_id")
                    continue
                try:
                    run = client.get_run(arg)
                    _print_run_result(run)
                except Exception as exc:
                    _print_err(f"failed to fetch run: {exc}")
                continue
            _print_err(f"unknown command: {cmd}")
            continue

        try:
            created = client.create_run(agent_id=str(current_agent["id"]), message=message)
        except Exception as exc:
            _print_err(f"failed to create run: {exc}")
            continue
        run_id = created.get("run_id") or created.get("id")
        if not run_id:
            _print_err(f"unexpected create_run response: {created}")
            continue
        print(f"[info] run_id={run_id} task_id={created.get('task_id')} status={created.get('status')}")
        try:
            final_run = client.wait_for_run(run_id, on_tick=_on_poll_tick)
        except KeyboardInterrupt:
            print("\n[info] interrupted; attempting to cancel run ...")
            try:
                client.cancel_run(str(run_id))
            except Exception as exc:
                _print_err(f"cancel failed: {exc}")
            continue
        except Exception as exc:
            _print_err(f"wait failed: {exc}")
            continue
        _print_run_result(final_run)


def _resolve_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vitoom unified chat CLI")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VITOOM_BASE_URL", "http://127.0.0.1:8888"),
        help="Vitoom backend base URL (default: http://127.0.0.1:8888)",
    )
    parser.add_argument("--email", default=os.environ.get("VITOOM_EMAIL"))
    parser.add_argument("--password", default=os.environ.get("VITOOM_PASSWORD"), help=argparse.SUPPRESS)
    parser.add_argument("--token", default=os.environ.get("VITOOM_TOKEN"))
    parser.add_argument(
        "--force-agent",
        dest="force_agent",
        default=os.environ.get("VITOOM_FORCE_AGENT_ID"),
        help="[debug] 创建新 chat session 时显式绑定 agent_id（id 或别名）",
    )
    parser.add_argument(
        "--agent",
        dest="legacy_agent",
        default=os.environ.get("VITOOM_AGENT_ID"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resume", default=None, help="resume an existing chat session by id")
    parser.add_argument("--load-name", default=None, help="create session with an explicit load_name")
    parser.add_argument("--title", default=None, help="optional title for newly created sessions")
    parser.add_argument("--input-mode", default="text", help="session input_mode (default: text)")
    parser.add_argument("--output-mode", default="text_stream", help="session output_mode (default: text_stream)")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="[debug] 使用 /v1/agents/runs 接口直接调试单次运行",
    )
    parser.add_argument("--no-cache", action="store_true", help="do not read/write the token cache")
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument("--poll-timeout", type=float, default=600.0)
    parser.add_argument("--message", "-m", default=None, help="send a single message and exit")
    return parser.parse_args(argv)


def _ensure_login(args: argparse.Namespace) -> VitoomClient:
    token: Optional[str] = args.token
    source = "arg" if token else None
    if not token and not args.no_cache:
        cached = _load_cached_token(args.base_url)
        if cached:
            token = cached
            source = "cache"

    client = VitoomClient(args.base_url, token=token)
    if token:
        print(f"[auth] using {source} token from {DEFAULT_TOKEN_CACHE}")
        return client

    try:
        token, user_id, email = _auto_issue_token(args.email)
        client.token = token
        if not args.no_cache:
            _save_cached_token(args.base_url, token)
        print(f"[auth] auto-selected user={email or user_id} (id={user_id})；token 已缓存")
        return client
    except Exception as exc:
        _print_err(f"auto-pick user failed: {exc}")

    if args.email and args.password:
        try:
            token = client.login(args.email, args.password)
        except Exception as exc:
            _print_err(f"fallback login failed: {exc}")
            raise SystemExit(2) from exc
        if not args.no_cache:
            _save_cached_token(args.base_url, token)
        print(f"[auth] logged in as {args.email} via /api/auth/login；token 已缓存")
        return client

    _print_err("无法自动拿到 token：请用 `conda activate vitoom` 运行，或显式传 --token。")
    raise SystemExit(2)


def _wrap_wait(client: VitoomClient, poll_interval: float, poll_timeout: float) -> None:
    def wait(run_id: str, **kwargs):
        kwargs.setdefault("poll_interval", poll_interval)
        kwargs.setdefault("max_seconds", poll_timeout)
        return VitoomClient.wait_for_run(client, run_id, **kwargs)

    client.wait_for_run = wait  # type: ignore[assignment]


def main(argv: Optional[List[str]] = None) -> int:
    args = _resolve_args(argv)
    client = _ensure_login(args)
    _wrap_wait(client, args.poll_interval, args.poll_timeout)

    force_agent = args.force_agent or args.legacy_agent
    force_agent_id = _resolve_agent_alias(force_agent) if force_agent else None

    if args.raw:
        if not force_agent_id:
            _print_err("--raw 需要同时提供 --force-agent 或 --agent")
            return 2
        try:
            agent = _pick_agent(client, force_agent_id)
        except Exception as exc:
            _print_err(f"failed to list agents: {exc}")
            return 3
        if not agent:
            return 4
        if args.message:
            try:
                created = client.create_run(agent_id=str(agent["id"]), message=str(args.message))
            except Exception as exc:
                _print_err(f"failed to create run: {exc}")
                return 5
            run_id = created.get("run_id") or created.get("id")
            if not run_id:
                _print_err(f"unexpected create_run response: {created}")
                return 6
            print(f"[info] run_id={run_id} task_id={created.get('task_id')} status={created.get('status')}")
            try:
                final_run = client.wait_for_run(str(run_id), on_tick=_on_poll_tick)
            except Exception as exc:
                _print_err(f"wait failed: {exc}")
                return 7
            _print_run_result(final_run)
            return 0 if str(final_run.get("status") or "").lower() == "completed" else 1
        run_raw_repl(client, agent)
        return 0

    try:
        if args.resume:
            session = client.get_chat_session(str(args.resume))
        else:
            session = client.create_chat_session(
                title=args.title,
                agent_id=force_agent_id,
                load_name=args.load_name,
                input_mode=args.input_mode,
                output_mode=args.output_mode,
            )
    except Exception as exc:
        _print_err(f"failed to prepare chat session: {exc}")
        return 3

    if force_agent_id and not args.resume:
        print(f"[debug] force-agent={force_agent_id}：新会话已显式绑定该 agent_id")

    try:
        if args.message:
            return asyncio.run(run_chat_once(client, args, session))
        return asyncio.run(run_chat_repl(client, args, session, force_agent_id=force_agent_id))
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        _print_err(f"chat session failed: {exc}")
        return 7


if __name__ == "__main__":
    raise SystemExit(main())

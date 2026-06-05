"""统一会话 `/ws/chat/{session_id}` 端到端回归脚本。

对应重构计划 P2/P3 的验收动作：

    1. HTTP ``POST /v1/chat/sessions`` 创建 ChatSession；
    2. WS ``/ws/chat/{session_id}`` 建立连接；
    3. 发送一条 ``user_message``，等待 ``message_started`` →
       ``message_delta`` → ``message_completed`` 流水；
    4. （可选）发送一组 ``audio_chunk`` + ``session_commit`` 模拟
       音频流，验证 turn_buffering 流程；
    5. ``session_close`` 收尾。

前提：

    - ``python -m backend.app.main`` 已运行；
    - 至少有一台 text 推理服务连接到 backend，且它上报过
      ``--model-name`` 对应的 model 绑定（或传 ``--auto-model``
      让脚本从 ``/v1/inference/services`` 自动挑一个）；
    - 想跑音频轮次时：另有一台 audio 推理服务已连线。

示例：

    python test/chat_ws_experiment.py --model-name Qwen/Qwen3-8B
    python test/chat_ws_experiment.py --auto-model --prompt '你好，介绍一下自己' --no-interactive
    python test/chat_ws_experiment.py --auto-model --audio-file samples/hello.wav --no-interactive
    python test/chat_ws_experiment.py --auto-model --prompt '从1写到500，不要调用工具' --interrupt-on first_delta --expect-interrupt --post-interrupt-prompt '你还在吗？只回答：在' --no-interactive
    python test/chat_ws_experiment.py --auto-model --prompt '请帮我生成一张日落海边的图片，直接调用合适工具，不要解释' --interrupt-on first_task --expect-interrupt --expect-task-cancelled --forbid-artifact-after-interrupt --post-interrupt-prompt '你还在吗？只回答：在' --no-interactive
    python test/chat_ws_experiment.py --auto-model --prompt '请生成一段小猫在草地上奔跑的短视频。直接调用 video_generator，不要自行指定 model_name，不要重试，不要解释。' --interrupt-on first_task --expect-tool-name video_generator --expect-task-kind video --expect-interrupt --expect-task-cancelled --forbid-artifact-after-interrupt --post-interrupt-prompt '你还在吗？只回答：在' --no-interactive
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets
from websockets.exceptions import ConnectionClosed


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def _recv_chat_json(ws, *, timeout: Optional[float] = None) -> Dict[str, Any]:
    """接收一条 JSON；若为 ``audio_delta`` 且 ``bytes_len>0``，再收一帧 binary 挂到 ``binary_bytes``。"""
    if timeout is None:
        raw = await ws.recv()
    else:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    if isinstance(raw, (bytes, bytearray)):
        raise RuntimeError("unexpected binary frame while expecting JSON text")
    evt: Any = json.loads(raw)
    if not isinstance(evt, dict):
        return {}
    etype = str(evt.get("type") or "")
    payload = evt.get("payload") if isinstance(evt.get("payload"), dict) else {}
    if etype == "audio_delta":
        try:
            need = int(payload.get("bytes_len") or 0)
        except (TypeError, ValueError):
            need = 0
        if need > 0:
            if timeout is None:
                raw2 = await ws.recv()
            else:
                raw2 = await asyncio.wait_for(ws.recv(), timeout=timeout)
            if not isinstance(raw2, (bytes, bytearray)):
                raise RuntimeError("audio_delta: expected binary PCM frame after JSON meta")
            evt["binary_bytes"] = bytes(raw2)
    return evt


# ---------------------------------------------------------------------------
# Audio dump：把每个 run 的 audio_delta chunks 累积写成 WAV，方便离线听辨 TTS 源头
# ---------------------------------------------------------------------------


_AUDIO_DUMP_DIR: Optional[Path] = None
# run_id -> {"pcm": bytearray, "sample_rate": int|None, "mime": str, "channels": int, "sampwidth": int, "chunk_count": int}
_AUDIO_DUMP_BUFFERS: Dict[str, Dict[str, Any]] = {}


def _parse_pcm_rate(mime: str) -> Optional[int]:
    m = re.search(r"rate\s*=\s*(\d+)", mime or "", flags=re.IGNORECASE)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return val if val > 0 else None
    except ValueError:
        return None


def _decode_wav_chunk(data: bytes) -> Optional[Dict[str, Any]]:
    """单个 chunk 本身是完整 WAV 时，解出 (pcm_bytes, sample_rate, channels, sampwidth)。

    目标 sampwidth=2（Int16 LE）；上游若是 float32 WAV，这里用 soundfile 兜底太重，
    直接返回 None，由调用方按需降级处理或跳过。
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            return {
                "pcm": wf.readframes(wf.getnframes()),
                "sample_rate": wf.getframerate(),
                "channels": wf.getnchannels(),
                "sampwidth": wf.getsampwidth(),
            }
    except wave.Error:
        return None
    except Exception:
        return None


def _audio_dump_accept(
    *,
    run_id: Optional[str],
    pcm_bytes: bytes,
    mime: str,
    sample_rate_hint: Optional[int],
    is_final: bool,
) -> None:
    """累积 chunk，`is_final=True` 时 flush 写盘。"""
    if _AUDIO_DUMP_DIR is None:
        return
    key = (run_id or "").strip() or "_default"

    raw = pcm_bytes or b""
    if raw:
            buf = _AUDIO_DUMP_BUFFERS.setdefault(
                key,
                {
                    "pcm": bytearray(),
                    "sample_rate": None,
                    "mime": mime or "",
                    "channels": 1,
                    "sampwidth": 2,
                    "chunk_count": 0,
                },
            )
            mime_lc = (mime or "").lower()
            if mime_lc.startswith("audio/wav") or mime_lc.startswith("audio/x-wav"):
                parsed = _decode_wav_chunk(raw)
                if parsed is None:
                    print(
                        f"{_C.YELLOW}[audio-dump] WAV chunk not PCM int16 (likely float32). "
                        f"Skip this chunk run={key} bytes={len(raw)}{_C.RESET}"
                    )
                else:
                    buf["pcm"].extend(parsed["pcm"])
                    buf["sample_rate"] = buf["sample_rate"] or int(parsed["sample_rate"])
                    buf["channels"] = int(parsed["channels"])
                    buf["sampwidth"] = int(parsed["sampwidth"])
                    buf["chunk_count"] += 1
            else:
                # 视为裸 PCM16 LE mono
                buf["pcm"].extend(raw)
                if buf["sample_rate"] is None:
                    buf["sample_rate"] = _parse_pcm_rate(mime_lc) or sample_rate_hint
                buf["chunk_count"] += 1

    if is_final:
        _audio_dump_flush(key)


def _audio_dump_flush(key: str) -> None:
    buf = _AUDIO_DUMP_BUFFERS.pop(key, None)
    if not buf or not buf["pcm"]:
        return
    assert _AUDIO_DUMP_DIR is not None
    sr = int(buf["sample_rate"] or 24000)
    channels = int(buf["channels"] or 1)
    sampwidth = int(buf["sampwidth"] or 2)
    safe_key = re.sub(r"[^A-Za-z0-9_\-]", "_", key)[:32] or "run"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _AUDIO_DUMP_DIR / f"tts_{ts}_{safe_key}.wav"
    try:
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sr)
            wf.writeframes(bytes(buf["pcm"]))
    except Exception as exc:
        print(f"{_C.RED}[audio-dump] write wav failed path={out_path} err={exc}{_C.RESET}")
        return
    duration_ms = int(
        len(buf["pcm"]) / max(1, channels * sampwidth) * 1000 / max(1, sr)
    )
    print(
        f"{_C.GREEN}[audio-dump] saved run={key} chunks={buf['chunk_count']} "
        f"bytes={len(buf['pcm'])} sr={sr} ch={channels} sw={sampwidth} "
        f"duration={duration_ms}ms path={out_path}{_C.RESET}"
    )


def _audio_dump_flush_all() -> None:
    for key in list(_AUDIO_DUMP_BUFFERS.keys()):
        print(f"{_C.YELLOW}[audio-dump] force flush run={key} (no is_final received){_C.RESET}")
        _audio_dump_flush(key)

API_BASE_URL = "http://127.0.0.1:8888"
WS_BASE_URL = "ws://127.0.0.1:8888"

DEFAULT_JWT_SECRET = "vitoom-default-secret-key-change-in-production"
DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE = 86400


# ---------------------------------------------------------------------------
# Auth helpers（与既有 text_ws_experiment.py 对齐，简化版）
# ---------------------------------------------------------------------------


def _load_security_settings() -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("需要 PyYAML 自动生成 token，或使用 --token") from e

    merged: Dict[str, Any] = {}
    for candidate in (
        PROJECT_ROOT / "config" / "default.yaml",
        PROJECT_ROOT / "config" / "app.yaml",
    ):
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and isinstance(data.get("security"), dict):
            merged.update(data["security"])
    return merged


def _create_access_token(data: Dict[str, Any]) -> str:
    try:
        from jose import jwt
    except Exception as e:
        raise RuntimeError("需要 python-jose 自动生成 token，或使用 --token") from e

    security = _load_security_settings()
    jwt_cfg = security.get("jwt") if isinstance(security.get("jwt"), dict) else {}
    secret = str(jwt_cfg.get("secret_key") or DEFAULT_JWT_SECRET)
    algorithm = str(jwt_cfg.get("algorithm") or DEFAULT_JWT_ALGORITHM)
    expire = int(jwt_cfg.get("access_token_expire") or DEFAULT_ACCESS_TOKEN_EXPIRE)

    payload = dict(data)
    payload.update(
        {
            "exp": datetime.utcnow() + timedelta(seconds=expire),
            "iat": datetime.utcnow(),
            "type": "access",
        }
    )
    return jwt.encode(payload, secret, algorithm=algorithm)


async def _pick_user_and_token(explicit_token: Optional[str]) -> Tuple[str, str]:
    if explicit_token:
        return "<external-token>", explicit_token
    try:
        from backend.database import User
    except Exception as e:
        raise RuntimeError("无法 import backend.database.User，请用 --token 指定") from e
    users = User.list_all(limit=20)
    if not users:
        raise RuntimeError("users 表为空；请先创建用户或传 --token")
    user = next((u for u in users if str(u.get("status") or "").lower() == "active"), users[0])
    user_id = str(user.get("id") or "")
    token = _create_access_token({"sub": user_id, "email": str(user.get("email") or "")})
    print(f"[auth] user_id={user_id}")
    return user_id, token


# ---------------------------------------------------------------------------
# 模型选择
# ---------------------------------------------------------------------------


async def _auto_pick_load_name(token: str, api_url: str) -> str:
    """``--auto-model`` 的兜底逻辑。

    新架构下 ``load_name`` 只是会话 / 请求参数，不再绑定到某个推理服务。
    如果用户不想手写，本函数只确认存在至少一个 running 的推理服务，然后
    返回空字符串 —— 由后端 ``/api/chat/sessions`` 的默认模型逻辑
    （``agents.default_model``）去兜底。
    """
    del token, api_url
    try:
        from backend.database import InferenceService
    except Exception as exc:
        raise RuntimeError(f"无法导入 ORM，请改传 --load-name：{exc}") from exc

    services = InferenceService.list_all()
    has_running = any(
        str(svc.get("status") or "").lower() == "running" for svc in services
    )
    if not has_running:
        raise RuntimeError("未发现 running 的推理服务，请先启动 inference/text/main.py")
    return ""


# ---------------------------------------------------------------------------
# 创建 ChatSession
# ---------------------------------------------------------------------------


async def _create_chat_session(
    *,
    token: str,
    api_url: str,
    load_name: str,
    input_mode: str,
    output_mode: str,
    audio_output: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "title": title or f"chat_ws_experiment_{int(time.time())}",
        "input_mode": input_mode,
        "output_mode": output_mode,
        "metadata": {"source": "chat_ws_experiment"},
    }
    if load_name:
        payload["load_name"] = load_name
    if audio_output:
        payload["audio_output"] = audio_output
    print("[session.create] request=")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{api_url}/v1/chat/sessions",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code >= 300:
        raise RuntimeError(f"create chat session failed: {resp.status_code} {resp.text}")
    body = resp.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict) or not data.get("id"):
        raise RuntimeError(f"create chat session bad response: {body}")
    return data


# ---------------------------------------------------------------------------
# WS 交互
# ---------------------------------------------------------------------------


def _ws_url(session_id: str, token: str) -> str:
    return f"{WS_BASE_URL}/ws/chat/{session_id}?token={token}"


# 简易 ANSI 着色（终端不支持时也只是多几个字符，影响不大）。
class _C:
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _meta(msg: str) -> str:
    return f"{_C.DIM}{msg}{_C.RESET}"


# 从推理侧经 ``forward_session_message`` 回流过来的"原始"事件。
# 统一会话客户端不需要它们（MasterAgentRuntime 已经聚合成 message_*），
# 全部静默丢弃以避免把流式输出打乱。
_RAW_INFERENCE_NOISE: set = {
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


# 模块级状态：用于区分"握手 session_ready"和后续回流噪音，以及
# 在 streaming 内容前打印一次性前缀。
_PRINT_STATE: Dict[str, Any] = {
    "handshake_ready_seen": False,
    "streaming_prefix_printed": False,
    "in_stream": False,
}


def _fmt_num(v: Any, ndigits: int = 2) -> str:
    """float -> 保留 ndigits 位小数；int 原样；其他转 str。"""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v}"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _normalize_expected_values(values: Optional[List[str]]) -> set[str]:
    normalized: set[str] = set()
    for raw in values or []:
        for part in str(raw or "").split(","):
            item = part.strip().lower()
            if item:
                normalized.add(item)
    return normalized


def _print_usage(usage: Dict[str, Any]) -> None:
    """把推理侧统计以"tokens: ... | tok/s: ... | timing: ..."的紧凑形式打出来。"""
    pt = usage.get("prompt_tokens")
    ot = usage.get("output_tokens")
    tt = usage.get("total_tokens")
    llm_calls = usage.get("successful_requests")
    if tt is None and isinstance(pt, int) and isinstance(ot, int):
        tt = pt + ot

    tok_parts = []
    if pt is not None:
        tok_parts.append(f"prompt={pt}")
    if ot is not None:
        tok_parts.append(f"output={ot}")
    if tt is not None:
        tok_parts.append(f"total={tt}")

    rate_parts = []
    tps_decode = usage.get("tok_s_decode")
    tps_total = usage.get("tok_s_total")
    if tps_decode is not None:
        rate_parts.append(f"decode={_fmt_num(tps_decode)}")
    if tps_total is not None:
        rate_parts.append(f"total={_fmt_num(tps_total)}")

    timing_parts = []
    ttft = usage.get("ttft_seconds")
    total_s = usage.get("total_seconds")
    decode_s = usage.get("decode_seconds")
    if ttft is not None:
        timing_parts.append(f"ttft={_fmt_num(ttft, 3)}s")
    if decode_s is not None:
        timing_parts.append(f"decode={_fmt_num(decode_s, 3)}s")
    if total_s is not None:
        timing_parts.append(f"total={_fmt_num(total_s, 3)}s")

    finish_reason = usage.get("finish_reason")

    segments = []
    if tok_parts:
        segments.append(f"tokens: {', '.join(tok_parts)}")
    if rate_parts:
        segments.append(f"tok/s: {', '.join(rate_parts)}")
    if llm_calls is not None:
        segments.append(f"llm_calls={_fmt_num(llm_calls)}")
    if timing_parts:
        segments.append(f"timing: {', '.join(timing_parts)}")
    if finish_reason:
        segments.append(f"finish={finish_reason}")

    if not segments:
        return
    body = "  |  ".join(segments)
    print(f"{_C.YELLOW}  · usage  {body}{_C.RESET}")


def _print_files(files: Any, *, indent: str = "    ") -> None:
    if not isinstance(files, list) or not files:
        return
    for idx, item in enumerate(files, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("file_name") or item.get("file_id") or f"file-{idx}")
        category = str(item.get("category") or "unknown")
        url = str(item.get("url") or item.get("http_url") or "").strip()
        storage_path = str(item.get("storage_path") or "").strip()
        task_id = str(item.get("derived_task_id") or "").strip()
        source_tool = str(item.get("source_tool") or "").strip()
        extras: List[str] = []
        if task_id:
            extras.append(f"task={task_id}")
        if source_tool:
            extras.append(f"tool={source_tool}")
        extra = f"  ({', '.join(extras)})" if extras else ""
        print(_meta(f"{indent}- [{category}] {name}{extra}"))
        if url:
            print(_meta(f"{indent}  url: {url}"))
        elif storage_path:
            print(_meta(f"{indent}  storage_path: {storage_path}"))


def _file_identity(item: Dict[str, Any]) -> str:
    file_id = str(item.get("file_id") or "").strip()
    if file_id:
        return f"file_id:{file_id}"
    url = str(item.get("url") or item.get("http_url") or "").strip()
    if url:
        return f"url:{url}"
    storage_path = str(item.get("storage_path") or "").strip()
    if storage_path:
        return f"storage:{storage_path}"
    file_name = str(item.get("file_name") or "").strip()
    task_id = str(item.get("derived_task_id") or "").strip()
    return f"name:{task_id}:{file_name}"


def _collect_turn_files(collected: List[Dict[str, Any]], files: Any) -> None:
    if not isinstance(files, list):
        return
    seen = {_file_identity(item) for item in collected if isinstance(item, dict)}
    for item in files:
        if not isinstance(item, dict):
            continue
        identity = _file_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        collected.append(item)


def _print_turn_file_summary(files: List[Dict[str, Any]]) -> None:
    if not files:
        return
    print(_meta(f"  · turn_files  total={len(files)}"))
    _print_files(files, indent="      ")


def _print_event(evt: Dict[str, Any]) -> None:
    etype = str(evt.get("type") or "")
    payload = evt.get("payload") or {}

    # 0) 原始推理事件：静默。推理端的 session_ready 也没有 input/output_mode，
    #    握手用的那条由 runtime 主动下发且 payload 里带 mode 字段——据此区分。
    if etype in _RAW_INFERENCE_NOISE:
        return
    if etype == "session_ready" and _PRINT_STATE["handshake_ready_seen"]:
        return

    # 1) 流式 token：直接内联拼接，不换行、不前缀。
    if etype == "message_delta":
        delta = str(payload.get("delta") or "")
        if not delta:
            return
        if not _PRINT_STATE["streaming_prefix_printed"]:
            sys.stdout.write(f"\n{_C.GREEN}{_C.BOLD}assistant›{_C.RESET} ")
            _PRINT_STATE["streaming_prefix_printed"] = True
        sys.stdout.write(delta)
        sys.stdout.flush()
        return

    # 2) 结构化事件：统一用小标签打印，避免一条事件一大坨 json。
    def _flush_stream_line() -> None:
        if _PRINT_STATE["streaming_prefix_printed"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _PRINT_STATE["streaming_prefix_printed"] = False

    if etype == "status_changed":
        prev = payload.get("prev")
        state = payload.get("state")
        task_id = str(payload.get("task_id") or "").strip()
        task_status = str(payload.get("task_status") or "").strip()
        task_kind = str(payload.get("task_kind") or "").strip()
        progress = payload.get("progress")
        files_count = payload.get("files_count")
        total = payload.get("total")
        error = payload.get("error")
        extras: List[str] = []
        if task_id:
            extras.append(f"task={task_id}")
        if task_status:
            extras.append(f"task_status={task_status}")
        if task_kind:
            extras.append(f"task_kind={task_kind}")
        if progress is not None:
            extras.append(f"progress={progress}")
        if files_count is not None:
            extras.append(f"files={files_count}")
        if total is not None:
            extras.append(f"total={total}")
        if error:
            extras.append(f"error={error}")
        extra_suffix = f"  ({', '.join(extras)})" if extras else ""
        _flush_stream_line()
        print(_meta(f"  · status: {prev} → {state}{extra_suffix}"))
        return

    if etype == "session_ready":
        _PRINT_STATE["handshake_ready_seen"] = True
        in_mode = payload.get("input_mode")
        out_mode = payload.get("output_mode")
        print(
            _meta(f"  · session_ready  input={in_mode}  output={out_mode}")
        )
        return

    if etype == "message_started":
        _PRINT_STATE["streaming_prefix_printed"] = False
        role = payload.get("role") or "assistant"
        print(_meta(f"  · message_started ({role})"))
        return

    if etype == "message_completed":
        _flush_stream_line()
        content_len = len(str(payload.get("content") or ""))
        interrupt = payload.get("interrupt_reason")
        files = payload.get("files") or []
        files_extra = f"  files={len(files)}" if isinstance(files, list) and files else ""
        extra = f"  interrupt={interrupt}" if interrupt else ""
        print(_meta(f"  · message_completed  chars={content_len}{files_extra}{extra}"))
        _print_files(files)

        usage = payload.get("usage_metrics") or {}
        if isinstance(usage, dict) and usage:
            _print_usage(usage)
        return

    if etype == "transcript_delta":
        _flush_stream_line()
        text = str(payload.get("text") or "")
        is_final = bool(payload.get("is_final"))
        state = "final" if is_final else "partial"
        preview = text if len(text) <= 80 else text[:77] + "..."
        print(f"{_C.MAGENTA}  · transcript_delta  {state}  text={preview}{_C.RESET}")
        return

    if etype == "audio_delta":
        _flush_stream_line()
        mime = str(payload.get("mime") or "")
        is_final = bool(payload.get("is_final"))
        sample_rate = payload.get("sample_rate")
        pcm = evt.get("binary_bytes")
        if isinstance(pcm, (bytes, bytearray)):
            pcm_bytes = bytes(pcm)
        else:
            pcm_bytes = b""
        try:
            declared = int(payload.get("bytes_len") or 0)
        except (TypeError, ValueError):
            declared = 0
        byte_len = len(pcm_bytes) if pcm_bytes else declared
        extra = f"  sr={sample_rate}" if sample_rate else ""
        print(
            f"{_C.YELLOW}  · audio_delta  bytes={byte_len}  mime={mime}  final={str(is_final).lower()}{extra}{_C.RESET}"
        )
        run_id = (evt.get("run_id") or "").strip() or None
        sr_hint: Optional[int] = None
        if isinstance(sample_rate, int) and sample_rate > 0:
            sr_hint = sample_rate
        elif isinstance(sample_rate, float) and sample_rate > 0:
            sr_hint = int(sample_rate)
        _audio_dump_accept(
            run_id=run_id,
            pcm_bytes=pcm_bytes,
            mime=mime,
            sample_rate_hint=sr_hint,
            is_final=is_final,
        )
        return

    if etype == "artifact_created":
        _flush_stream_line()
        print(_meta("  · artifact_created"))
        _print_files([payload], indent="      ")
        return

    if etype == "tool_call_started":
        _flush_stream_line()
        tool_name = str(payload.get("tool_name") or "").strip()
        tool_call_id = str(payload.get("tool_call_id") or "").strip()
        arguments_raw = payload.get("arguments")
        model_name = ""
        prompt_preview = ""
        if isinstance(arguments_raw, str) and arguments_raw.strip():
            try:
                parsed_args = json.loads(arguments_raw)
                if isinstance(parsed_args, dict):
                    model_name = str(parsed_args.get("model_name") or "").strip()
                    prompt_preview = str(parsed_args.get("prompt") or "").strip()
            except Exception:
                pass
        extras: List[str] = []
        if tool_call_id:
            extras.append(f"id={tool_call_id}")
        if model_name:
            extras.append(f"model={model_name}")
        if prompt_preview:
            preview = prompt_preview if len(prompt_preview) <= 24 else prompt_preview[:21] + "..."
            extras.append(f"prompt={preview}")
        suffix = f"  ({', '.join(extras)})" if extras else ""
        print(_meta(f"  · tool_call_started  {tool_name}{suffix}"))
        return

    if etype == "tool_call_completed":
        _flush_stream_line()
        tool_name = str(payload.get("tool_name") or "").strip()
        tool_call_id = str(payload.get("tool_call_id") or "").strip()
        output_raw = payload.get("output")
        parsed_output: Dict[str, Any] = {}
        if isinstance(output_raw, str) and output_raw.strip():
            text = output_raw.strip()
            if text.startswith("```json") and text.endswith("```"):
                text = text[len("```json"): -3].strip()
            try:
                maybe = json.loads(text)
                if isinstance(maybe, dict):
                    parsed_output = maybe
            except Exception:
                parsed_output = {}
        extras: List[str] = []
        if tool_call_id:
            extras.append(f"id={tool_call_id}")
        task_id = str(parsed_output.get("task_id") or "").strip()
        status = str(parsed_output.get("status") or "").strip()
        error = str(parsed_output.get("error") or "").strip()
        model_name = str(parsed_output.get("model_name") or "").strip()
        if task_id:
            extras.append(f"task={task_id}")
        if status:
            extras.append(f"status={status}")
        if model_name:
            extras.append(f"model={model_name}")
        if error:
            preview = error if len(error) <= 40 else error[:37] + "..."
            extras.append(f"error={preview}")
        suffix = f"  ({', '.join(extras)})" if extras else ""
        print(_meta(f"  · tool_call_completed  {tool_name}{suffix}"))
        return

    if etype == "error":
        _flush_stream_line()
        code = payload.get("code")
        msg = payload.get("message")
        print(f"{_C.RED}  ✗ error  code={code}  message={msg}{_C.RESET}")
        return

    if etype == "session_closed":
        _flush_stream_line()
        reason = payload.get("reason")
        print(_meta(f"  · session_closed  reason={reason}"))
        return

    # 3) 其他未覆盖事件：用一行精简 JSON 显示 payload，避免把屏幕撑爆。
    _flush_stream_line()
    compact = json.dumps(payload, ensure_ascii=False)
    if len(compact) > 160:
        compact = compact[:157] + "..."
    print(_meta(f"  · {etype}  {compact}"))


async def _drain_until_ready(ws) -> None:
    """等待直到收到 session_ready。"""
    while True:
        try:
            evt = await _recv_chat_json(ws)
        except Exception:
            continue
        _print_event(evt)
        if evt.get("type") == "session_ready":
            return


async def _send_interrupt(ws) -> None:
    await ws.send(json.dumps({"type": "interrupt", "payload": {}}))
    print(_meta("  · client_interrupt sent"))


async def _observe_post_interrupt(
    ws,
    *,
    target_run_id: Optional[str],
    quiet_window_s: float,
    expected_task_ids: Optional[set[str]] = None,
    expected_task_status: str = "",
    expected_task_kind: str = "",
    forbid_artifact_after_interrupt: bool = False,
) -> None:
    """interrupt 后继续观察一段时间，确保 run 正常回 ready 且无脏增量。"""
    deadline = time.monotonic() + max(0.0, float(quiet_window_s or 0.0))
    saw_ready = False
    normalized_expected_status = str(expected_task_status or "").strip().lower()
    normalized_expected_kind = str(expected_task_kind or "").strip().lower()
    normalized_task_ids = {
        str(item).strip() for item in (expected_task_ids or set()) if str(item).strip()
    }
    saw_expected_task_status = not normalized_expected_status

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            evt = await _recv_chat_json(ws, timeout=remaining)
        except asyncio.TimeoutError:
            break
        except Exception:
            continue
        _print_event(evt)

        etype = str(evt.get("type") or "")
        evt_run_id = str(evt.get("run_id") or "").strip() or None
        payload = evt.get("payload") or {}

        if etype == "status_changed":
            if str(payload.get("state") or "") == "ready":
                saw_ready = True
            task_id = str(payload.get("task_id") or "").strip()
            task_status = str(payload.get("task_status") or "").strip().lower()
            task_kind = str(payload.get("task_kind") or "").strip().lower()
            if (
                normalized_expected_status
                and task_id
                and task_id in normalized_task_ids
                and (not normalized_expected_kind or task_kind == normalized_expected_kind)
                and task_status == normalized_expected_status
            ):
                saw_expected_task_status = True

        if target_run_id and evt_run_id == target_run_id and etype in {"message_delta", "message_completed"}:
            raise AssertionError(
                f"interrupt 后静默窗口内仍收到旧 run={target_run_id} 的 {etype}"
            )
        if (
            forbid_artifact_after_interrupt
            and etype == "artifact_created"
            and (not target_run_id or evt_run_id == target_run_id)
        ):
            raise AssertionError("interrupt 后静默窗口内仍收到 artifact_created")

    if not saw_ready:
        raise AssertionError("interrupt 后未在静默观察窗口内看到 status_changed(state=ready)")
    if normalized_expected_status and normalized_task_ids and not saw_expected_task_status:
        raise AssertionError(
            f"interrupt 后未观察到 task_status={normalized_expected_status}，task_ids={sorted(normalized_task_ids)}"
        )


async def _wait_for_turn_done(
    ws,
    *,
    timeout_s: float = 60.0,
    expect_transcript_final: bool = False,
    expect_audio_delta: bool = False,
) -> Dict[str, Any]:
    """等待 message_completed / error，返回该事件。"""
    deadline = time.monotonic() + timeout_s
    last_error: Optional[Dict[str, Any]] = None
    turn_files: List[Dict[str, Any]] = []
    saw_transcript_final = False
    audio_chunk_count = 0
    audio_final_seen = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for message_completed")
        try:
            evt = await _recv_chat_json(ws, timeout=remaining)
        except Exception:
            continue
        _print_event(evt)
        etype = str(evt.get("type") or "")
        payload = evt.get("payload") or {}
        if etype == "artifact_created" and isinstance(payload, dict):
            _collect_turn_files(turn_files, [payload])
        if etype == "transcript_delta" and bool(payload.get("is_final")):
            saw_transcript_final = True
        if etype == "audio_delta":
            try:
                bl = int(payload.get("bytes_len") or 0)
            except (TypeError, ValueError):
                bl = 0
            bin_part = evt.get("binary_bytes")
            bin_len = len(bin_part) if isinstance(bin_part, (bytes, bytearray)) else 0
            if bl > 0 or bin_len > 0:
                audio_chunk_count += 1
            if bool(payload.get("is_final")):
                audio_final_seen = True
        if etype == "message_completed":
            _collect_turn_files(turn_files, payload.get("files") or [])
            _print_turn_file_summary(turn_files)
            if expect_transcript_final and not saw_transcript_final:
                raise AssertionError("expected transcript_delta(is_final=true), but none was observed")
            if expect_audio_delta and audio_chunk_count <= 0:
                raise AssertionError("expected audio_delta chunks, but none were observed")
            if expect_audio_delta and not audio_final_seen:
                raise AssertionError("expected audio_delta(is_final=true), but none was observed")
            return evt
        if etype == "error":
            # 无论是否 recoverable，本轮已经结束（recoverable=True 只代表 session 可继续用）
            last_error = evt
            _print_turn_file_summary(turn_files)
            return evt
        if etype == "session_closed":
            _print_turn_file_summary(turn_files)
            return evt


async def _run_text_turn(
    ws,
    prompt: str,
    *,
    interrupt_on: str = "none",
    interrupt_delay_ms: int = 0,
    interrupt_quiet_window: float = 0.0,
    expect_interrupt: bool = False,
    expect_task_cancelled: bool = False,
    forbid_artifact_after_interrupt: bool = False,
    expect_tool_names: Optional[set[str]] = None,
    expect_task_kind: str = "",
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    msg = {
        "type": "user_message",
        "payload": {"role": "user", "text": prompt, "content_type": "text"},
    }
    print(f"{_C.CYAN}{_C.BOLD}user›{_C.RESET} {prompt}")
    await ws.send(json.dumps(msg))

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    turn_files: List[Dict[str, Any]] = []
    sent_interrupt = False
    target_run_id: Optional[str] = None
    saw_interrupted = False
    saw_ready_after_interrupt = False
    observed_task_ids: set[str] = set()
    observed_task_kinds: set[str] = set()
    saw_task_cancelled = False
    saw_any_tool_call = False
    observed_tool_names: set[str] = set()
    normalized_expected_tool_names = {name for name in (expect_tool_names or set()) if name}
    normalized_expected_task_kind = str(expect_task_kind or "").strip().lower()

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for message_completed")
        try:
            evt = await _recv_chat_json(ws, timeout=remaining)
        except Exception:
            continue
        _print_event(evt)
        etype = str(evt.get("type") or "")
        payload = evt.get("payload") or {}
        evt_run_id = str(evt.get("run_id") or "").strip() or None
        if evt_run_id and not target_run_id:
            target_run_id = evt_run_id

        if etype == "artifact_created" and isinstance(payload, dict):
            _collect_turn_files(turn_files, [payload])
            if sent_interrupt and forbid_artifact_after_interrupt:
                raise AssertionError("interrupt 后本轮仍收到 artifact_created")

        if etype in {"tool_call_started", "tool_call_completed", "tool_call_failed"}:
            saw_any_tool_call = True
            tool_name = str(payload.get("tool_name") or "").strip().lower()
            if tool_name:
                observed_tool_names.add(tool_name)

        if etype == "status_changed":
            state = str(payload.get("state") or "")
            task_id = str(payload.get("task_id") or "").strip()
            task_status = str(payload.get("task_status") or "").strip().lower()
            task_kind = str(payload.get("task_kind") or "").strip().lower()
            if state == "interrupted":
                saw_interrupted = True
            if sent_interrupt and state == "ready":
                saw_ready_after_interrupt = True
            if task_id:
                observed_task_ids.add(task_id)
            if task_kind:
                observed_task_kinds.add(task_kind)
            if task_id and task_status == "cancelled":
                saw_task_cancelled = True

        if (
            interrupt_on == "first_delta"
            and not sent_interrupt
            and etype == "message_delta"
        ):
            if interrupt_delay_ms > 0:
                await asyncio.sleep(interrupt_delay_ms / 1000.0)
            await _send_interrupt(ws)
            sent_interrupt = True
        if (
            interrupt_on == "first_task"
            and not sent_interrupt
            and etype == "status_changed"
            and str(payload.get("task_id") or "").strip()
            and (
                not normalized_expected_task_kind
                or str(payload.get("task_kind") or "").strip().lower() == normalized_expected_task_kind
            )
        ):
            if interrupt_delay_ms > 0:
                await asyncio.sleep(interrupt_delay_ms / 1000.0)
            await _send_interrupt(ws)
            sent_interrupt = True

        if etype == "message_completed":
            _collect_turn_files(turn_files, payload.get("files") or [])
            _print_turn_file_summary(turn_files)

            if expect_interrupt:
                interrupt_reason = str(payload.get("interrupt_reason") or "").strip()
                if not sent_interrupt:
                    if interrupt_on == "first_task" and not observed_task_ids:
                        raise AssertionError(
                            "expect_interrupt=True，但本轮未进入 task 阶段（未观察到任何 task_id），因此 first_task 条件未触发"
                        )
                    if (
                        interrupt_on == "first_task"
                        and normalized_expected_task_kind
                        and normalized_expected_task_kind not in observed_task_kinds
                    ):
                        raise AssertionError(
                            f"expect_interrupt=True，但本轮未观察到匹配的 task_kind={normalized_expected_task_kind}，实际为 {sorted(observed_task_kinds)}"
                        )
                    raise AssertionError("expect_interrupt=True，但本轮未实际发送 interrupt")
                if interrupt_reason != "user_interrupt":
                    raise AssertionError(
                        f"预期 interrupt_reason=user_interrupt，实际为 {interrupt_reason!r}"
                    )
                if not saw_interrupted:
                    raise AssertionError("已发送 interrupt，但未观察到 status_changed(state=interrupted)")
            if expect_task_cancelled and not observed_task_ids:
                raise AssertionError("预期验证 task cancel，但本轮未观察到任何 task_id")
            if normalized_expected_tool_names and not normalized_expected_tool_names.issubset(observed_tool_names):
                raise AssertionError(
                    f"预期命中工具 {sorted(normalized_expected_tool_names)}，实际为 {sorted(observed_tool_names)}"
                )
            if normalized_expected_task_kind and normalized_expected_task_kind not in observed_task_kinds:
                raise AssertionError(
                    f"预期观察到 task_kind={normalized_expected_task_kind}，实际为 {sorted(observed_task_kinds)}"
                )

            if sent_interrupt and interrupt_quiet_window > 0:
                if not saw_ready_after_interrupt:
                    await _observe_post_interrupt(
                        ws,
                        target_run_id=target_run_id,
                        quiet_window_s=interrupt_quiet_window,
                        expected_task_ids=observed_task_ids if expect_task_cancelled else None,
                        expected_task_status="cancelled" if expect_task_cancelled else "",
                        expected_task_kind=normalized_expected_task_kind,
                        forbid_artifact_after_interrupt=forbid_artifact_after_interrupt,
                    )
                else:
                    # 即便已经看到了 ready，也继续观察一段时间，确保旧 run 不再冒泡。
                    await _observe_post_interrupt(
                        ws,
                        target_run_id=target_run_id,
                        quiet_window_s=interrupt_quiet_window,
                        expected_task_ids=observed_task_ids if expect_task_cancelled else None,
                        expected_task_status="cancelled" if expect_task_cancelled else "",
                        expected_task_kind=normalized_expected_task_kind,
                        forbid_artifact_after_interrupt=forbid_artifact_after_interrupt,
                    )
            elif expect_task_cancelled and not saw_task_cancelled:
                raise AssertionError(
                    f"预期 interrupt 后任务进入 cancelled，但未观察到；task_ids={sorted(observed_task_ids)}"
                )
            return {
                "terminal_event": evt,
                "sent_interrupt": sent_interrupt,
                "run_id": target_run_id,
                "task_ids": sorted(observed_task_ids),
                "task_kinds": sorted(observed_task_kinds),
                "tool_names": sorted(observed_tool_names),
                "saw_any_tool_call": saw_any_tool_call,
            }

        if etype == "error":
            _print_turn_file_summary(turn_files)
            return {
                "terminal_event": evt,
                "sent_interrupt": sent_interrupt,
                "run_id": target_run_id,
                "task_ids": sorted(observed_task_ids),
                "task_kinds": sorted(observed_task_kinds),
                "tool_names": sorted(observed_tool_names),
                "saw_any_tool_call": saw_any_tool_call,
            }

        if etype == "session_closed":
            _print_turn_file_summary(turn_files)
            return {
                "terminal_event": evt,
                "sent_interrupt": sent_interrupt,
                "run_id": target_run_id,
                "task_ids": sorted(observed_task_ids),
                "task_kinds": sorted(observed_task_kinds),
                "tool_names": sorted(observed_tool_names),
                "saw_any_tool_call": saw_any_tool_call,
            }


async def _run_audio_turn(
    ws,
    audio_path: Path,
    *,
    chunk_bytes: int = 16000,
    expect_audio_delta: bool = False,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    if not audio_path.exists():
        raise RuntimeError(f"audio file not found: {audio_path}")
    data = audio_path.read_bytes()
    idx = 0
    sent = 0
    while sent < len(data):
        chunk = data[sent : sent + chunk_bytes]
        await ws.send(
            json.dumps(
                {
                    "type": "audio_chunk",
                    "payload": {
                        "seq": idx,
                        "bytes_len": len(chunk),
                        "mime": "audio/pcm;rate=16000",
                    },
                }
            )
        )
        await ws.send(chunk)
        sent += len(chunk)
        idx += 1
    print(f"{_C.CYAN}{_C.BOLD}user›{_C.RESET} sent {idx} audio chunks (total {sent} bytes)")
    await ws.send(json.dumps({"type": "session_commit", "payload": {}}))
    return await _wait_for_turn_done(
        ws,
        timeout_s=timeout_s,
        expect_transcript_final=True,
        expect_audio_delta=expect_audio_delta,
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    global _AUDIO_DUMP_DIR
    if args.save_audio_out:
        out_dir = Path(args.save_audio_out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _AUDIO_DUMP_DIR = out_dir
        print(f"{_C.GREEN}[audio-dump] enabled out_dir={out_dir}{_C.RESET}")

    _, token = await _pick_user_and_token(args.token)

    load_name = args.load_name
    if not load_name and args.auto_model:
        load_name = await _auto_pick_load_name(token, args.api_url)
        if load_name:
            print(f"[auto-model] picked load_name={load_name}")
        elif args.audio_file:
            print("[auto-model] no explicit load_name; backend will use default audio input model")
        else:
            print("[auto-model] no explicit load_name; backend will use agents.default_model")
    # load_name 允许为空：
    # - text 会话回退到 config/default.yaml:agents.default_model
    # - audio 会话回退到后端默认 ASR 模型

    input_mode = "audio_stream" if args.audio_file else "text"
    output_mode = args.output_mode or ("multimodal_result" if args.audio_file and args.expect_audio_delta else "text_stream")
    audio_output: Optional[Dict[str, Any]] = None
    if any(
        [
            args.tts_load_name,
            args.tts_mode,
            args.tts_speaker_name,
            args.tts_voice_preset,
            args.tts_instruct,
            args.tts_language,
            args.tts_sample_rate is not None,
            args.tts_file_type,
        ]
    ):
        audio_output = {}
        if args.tts_load_name:
            audio_output["load_name"] = args.tts_load_name
        if args.tts_mode:
            audio_output["tts_mode"] = args.tts_mode
        if args.tts_speaker_name:
            audio_output["speaker_name"] = args.tts_speaker_name
        if args.tts_voice_preset:
            audio_output["voice_preset"] = args.tts_voice_preset
        if args.tts_instruct:
            audio_output["instruct"] = args.tts_instruct
        if args.tts_language:
            audio_output["language"] = args.tts_language
        if args.tts_sample_rate is not None:
            audio_output["sample_rate"] = args.tts_sample_rate
        if args.tts_file_type:
            audio_output["file_type"] = args.tts_file_type

    session = await _create_chat_session(
        token=token,
        api_url=args.api_url,
        load_name=load_name,
        input_mode=input_mode,
        output_mode=output_mode,
        audio_output=audio_output,
    )
    session_id = str(session["id"])
    session_short_id = session_id[:8] if session_id else "unknown"
    print(f"[session] id={session_id}")

    ws_url = _ws_url(session_id, token)
    print(f"[ws] connecting {ws_url}")

    async with websockets.connect(ws_url, max_size=None, ping_interval=20) as ws:
        # backend 在 WS accept 后会主动 runtime.open() 下发 session_ready，
        # 因此客户端不用再主动发 session_open。
        await _drain_until_ready(ws)

        if args.audio_file:
            await _run_audio_turn(
                ws,
                Path(args.audio_file),
                expect_audio_delta=args.expect_audio_delta,
                timeout_s=max(args.turn_timeout, 180.0),
            )

        if args.prompt:
            expect_tool_names = _normalize_expected_values(args.expect_tool_name)
            result = await _run_text_turn(
                ws,
                args.prompt,
                interrupt_on=args.interrupt_on,
                interrupt_delay_ms=args.interrupt_delay_ms,
                interrupt_quiet_window=args.interrupt_quiet_window,
                expect_interrupt=args.expect_interrupt,
                expect_task_cancelled=args.expect_task_cancelled,
                forbid_artifact_after_interrupt=args.forbid_artifact_after_interrupt,
                expect_tool_names=expect_tool_names,
                expect_task_kind=args.expect_task_kind,
                timeout_s=args.turn_timeout,
            )
            if result.get("sent_interrupt") and args.post_interrupt_prompt:
                print(_meta("  · post_interrupt_probe"))
                await _run_text_turn(ws, args.post_interrupt_prompt, timeout_s=args.turn_timeout)

        if not args.no_interactive:
            loop = asyncio.get_event_loop()
            while True:
                prompt_text = f"\n[{session_short_id}] >>> (quit 退出) "
                user_text = await loop.run_in_executor(
                    None, lambda: input(prompt_text).strip()
                )
                if not user_text:
                    continue
                if user_text.lower() in {"quit", "exit"}:
                    break
                await _run_text_turn(ws, user_text)

        try:
            await ws.send(json.dumps({"type": "session_close", "payload": {}}))
            await _recv_chat_json(ws, timeout=5.0)
        except (ConnectionClosed, asyncio.TimeoutError):
            pass

    _audio_dump_flush_all()
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一会话 /ws/chat 端到端回归")
    p.add_argument("--api-url", default=API_BASE_URL)
    p.add_argument("--token", default=None, help="外部 JWT；不传则自动挑一个 active user 自签")
    p.add_argument("--load-name", default=None, help="会话加载名；audio 模式下表示 ASR load_name")
    p.add_argument("--auto-model", action="store_true", help="从 running 服务里自动挑 load_name")
    p.add_argument("--prompt", default=None, help="非交互模式下发一条文本")
    p.add_argument("--audio-file", default=None, help="非交互模式下发一轮音频（路径）")
    p.add_argument(
        "--output-mode",
        default=None,
        help="会话 output_mode；不传时，文本默认 text_stream，音频+expect-audio-delta 默认 multimodal_result",
    )
    p.add_argument("--tts-load-name", default=None, help="语音回复使用的 TTS load_name")
    p.add_argument("--tts-mode", default=None, help="语音回复 TTS 模式：custom_voice / voice_design / voice_clone")
    p.add_argument("--tts-speaker-name", default=None, help="custom_voice 下的 speaker_name")
    p.add_argument("--tts-voice-preset", default=None, help="speaker_name 的兼容别名")
    p.add_argument("--tts-instruct", default=None, help="语音回复风格描述")
    p.add_argument("--tts-language", default=None, help="语音回复语言")
    p.add_argument("--tts-sample-rate", type=int, default=None, help="语音回复采样率")
    p.add_argument("--tts-file-type", default=None, help="语音回复文件格式，如 wav/mp3")
    p.add_argument(
        "--expect-audio-delta",
        action="store_true",
        help="要求音频 turn 中必须观察到 assistant audio_delta（用于 P2 语音闭环验证）",
    )
    p.add_argument(
        "--save-audio-out",
        default=None,
        help="把收到的 audio_delta chunks 拼成 WAV 存盘的目录；按 run_id 拆文件",
    )
    p.add_argument(
        "--interrupt-on",
        choices=["none", "first_delta", "first_task"],
        default="none",
        help="自动打断策略：none=不打断，first_delta=收到首个 message_delta 后打断，first_task=收到首个匹配 task_id 回流后打断（若传 --expect-task-kind，则仅匹配该 kind）",
    )
    p.add_argument(
        "--interrupt-delay-ms",
        type=int,
        default=0,
        help="自动打断前额外等待的毫秒数（仅与 --interrupt-on 搭配）",
    )
    p.add_argument(
        "--interrupt-quiet-window",
        type=float,
        default=3.0,
        help="打断后静默观察窗口（秒），用于检测旧 run 脏增量是否继续冒泡",
    )
    p.add_argument(
        "--expect-interrupt",
        action="store_true",
        help="要求本轮必须出现 interrupt 收尾语义（message_completed.interrupt_reason=user_interrupt）",
    )
    p.add_argument(
        "--expect-task-cancelled",
        action="store_true",
        help="要求本轮若出现派生 task，则 interrupt 后必须观察到 task_status=cancelled",
    )
    p.add_argument(
        "--forbid-artifact-after-interrupt",
        action="store_true",
        help="要求发送 interrupt 后本轮不得再收到 artifact_created（适合图片/视频任务）",
    )
    p.add_argument(
        "--expect-tool-name",
        action="append",
        default=None,
        help="要求本轮命中指定工具名；可重复传入或用逗号分隔，例如 --expect-tool-name video_generator",
    )
    p.add_argument(
        "--expect-task-kind",
        default=None,
        help="要求本轮观察到指定 task_kind；对 first_task 触发条件也生效，例如 image/video/audio",
    )
    p.add_argument(
        "--turn-timeout",
        type=float,
        default=120.0,
        help="单轮文本回合的超时秒数；视频打断验证可适当调大",
    )
    p.add_argument(
        "--post-interrupt-prompt",
        default=None,
        help="若本轮实际触发了 interrupt，则在其后再发一条探活 prompt",
    )
    p.add_argument("--no-interactive", action="store_true", help="跑完预置 prompt/audio 后直接退出")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"[error] {exc}")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()

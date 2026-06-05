"""
文本推理服务全链路联调脚本

前提：
1. `python -m backend.app.main` 已运行
2. `python inference/text/main.py service_text` 已连接到 backend
3. 数据库中已存在目标 text 模型，且本地模型目录位于 `{models_dir}/{model_name}`

用途：
1. 调用 `/v1/sessions` 创建 `text_chat` session
2. 连接 `/ws/session/{session_id}` 建立会话
3. 默认进入命令行交互式问答，可自由输入多轮问题
4. 输入 `quit` 退出交互，并自动发送 `session_close`
5. 也支持通过 `--prompt` 预置一组非交互问题做回归验证

示例：
  python test/text_ws_experiment.py --model-name "nvidia/Gemma-4-31B-IT-NVFP4"
  python test/text_ws_experiment.py --model-name "Qwen/Qwen3-8B" --model-class "Qwen-text"
  python test/text_ws_experiment.py --model-name "Qwen/Qwen3-8B"
  python test/text_ws_experiment.py --model-name "Qwen/Qwen3-8B" --prompt "请介绍一下你自己"
  python test/text_ws_experiment.py --model-name "Qwen3.5-35B-A3B-GPTQ-Int4" --model-class "Qwen-text" --image-url "https://example.com/demo.jpg" --prompt "请描述这张图片"
  python test/text_ws_experiment.py --model-name "Qwen3.5-35B-A3B-GPTQ-Int4" --model-class "Qwen-text" --video-url "https://example.com/demo.mp4" --prompt "请描述这个视频"
  python test/text_ws_experiment.py --token "<your-token>" --model-name "nvidia/Gemma-4-31B-IT-NVFP4"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE_URL = "http://192.168.0.105:8888"
WS_BASE_URL = "ws://192.168.0.105:8888"

DEFAULT_JWT_SECRET = "vitoom-default-secret-key-change-in-production"
DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE = 86400

def _safe_rate(count: Optional[int], duration_seconds: Optional[float]) -> Optional[float]:
    if count is None or duration_seconds is None or duration_seconds <= 0:
        return None
    return float(count) / float(duration_seconds)


def summarize_turn_stats(
    *,
    started_at: float,
    first_delta_at: Optional[float],
    finished_at: float,
    server_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    server_stats = dict(server_stats or {})
    input_tokens = server_stats.get("prompt_tokens")
    output_tokens = server_stats.get("output_tokens")
    total_seconds = server_stats.get("total_seconds")
    ttft_seconds = server_stats.get("ttft_seconds")
    decode_seconds = server_stats.get("decode_seconds")

    if total_seconds is None:
        total_seconds = max(0.0, finished_at - started_at)
    if ttft_seconds is None and first_delta_at is not None:
        ttft_seconds = max(0.0, first_delta_at - started_at)
    if decode_seconds is None and first_delta_at is not None:
        decode_seconds = max(0.0, finished_at - first_delta_at)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_seconds": total_seconds,
        "ttft_seconds": ttft_seconds,
        "decode_seconds": decode_seconds,
        "output_tps_total": server_stats.get("tok_s_total"),
        "output_tps_decode": server_stats.get("tok_s_decode"),
    }


def print_turn_stats(seq: int, stats: Dict[str, Any]) -> None:
    def _fmt_optional_int(value: Optional[int]) -> str:
        return str(value) if value is not None else "n/a"

    def _fmt_optional_float(value: Optional[float]) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    print(
        f"[turn:{seq}] stats "
        f"input_tokens={_fmt_optional_int(stats.get('input_tokens'))} "
        f"output_tokens={_fmt_optional_int(stats.get('output_tokens'))} "
        f"ttft_s={_fmt_optional_float(stats.get('ttft_seconds'))} "
        f"total_s={_fmt_optional_float(stats.get('total_seconds'))} "
        f"decode_s={_fmt_optional_float(stats.get('decode_seconds'))} "
        f"tok_s_total={_fmt_optional_float(stats.get('output_tps_total'))} "
        f"tok_s_decode={_fmt_optional_float(stats.get('output_tps_decode'))}"
    )


def infer_family(model_name: str, explicit_family: Optional[str]) -> str:
    if explicit_family:
        return str(explicit_family).strip()

    raw = str(model_name or "").strip().lower()
    if "gemma" in raw:
        return "Gemma"
    if "qwen" in raw:
        return "Qwen-text"
    raise ValueError(
        "无法从 model_name 自动推断 family，请显式传 --model-class。"
    )


def load_security_settings() -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("自动生成 token 需要安装 PyYAML，或直接使用 --token") from e

    def _read_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}

    config_dir = PROJECT_ROOT / "config"
    merged: Dict[str, Any] = {}
    for candidate in (config_dir / "default.yaml", config_dir / "app.yaml"):
        data = _read_yaml(candidate)
        security = data.get("security")
        if isinstance(security, dict):
            merged.update(security)
    return merged


def create_access_token_local(data: Dict[str, Any]) -> str:
    try:
        from jose import jwt
    except Exception as e:
        raise RuntimeError("自动生成 token 需要安装 python-jose，或直接使用 --token") from e

    security = load_security_settings()
    jwt_cfg = security.get("jwt") if isinstance(security.get("jwt"), dict) else {}
    secret = str(jwt_cfg.get("secret_key") or DEFAULT_JWT_SECRET)
    algorithm = str(jwt_cfg.get("algorithm") or DEFAULT_JWT_ALGORITHM)
    access_token_expire = int(jwt_cfg.get("access_token_expire") or DEFAULT_ACCESS_TOKEN_EXPIRE)

    payload = dict(data)
    payload.update(
        {
            "exp": datetime.utcnow() + timedelta(seconds=access_token_expire),
            "iat": datetime.utcnow(),
            "type": "access",
        }
    )
    return jwt.encode(payload, secret, algorithm=algorithm)


async def create_test_user_and_token(api_url: str) -> Tuple[str, str]:
    del api_url  # 保持签名一致
    try:
        from backend.database import User
    except Exception as e:
        raise RuntimeError("未提供 --token，且无法导入 backend.database.User 自动选择测试用户") from e

    users = User.list_all(limit=20)
    if not users:
        raise RuntimeError("users 表中没有可用用户，请先创建用户，或直接用 --token")

    user = next((u for u in users if str(u.get("status") or "").lower() == "active"), users[0])
    user_id = str(user.get("id") or "")
    email = str(user.get("email") or "")
    if not user_id:
        raise RuntimeError(f"选中的用户缺少 id: {user}")

    token = create_access_token_local({"sub": user_id, "email": email})
    print(f"user_id: {user_id}, token: {token}")
    return user_id, token


async def resolve_token(api_url: str, explicit_token: Optional[str]) -> Tuple[str, str]:
    if explicit_token:
        return "<external-token>", explicit_token
    return await create_test_user_and_token(api_url)


async def create_text_session(
    *,
    token: str,
    api_url: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "scene": "text_chat",
        "metadata": metadata,
    }
    print("[session.create] request_payload=")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{api_url}/v1/sessions",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

    if resp.status_code != 201:
        raise RuntimeError(f"create session failed: {resp.status_code} {resp.text}")

    body = resp.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        raise RuntimeError(f"session response missing data: {body}")
    if not data.get("id"):
        raise RuntimeError(f"session response missing session id: {body}")
    return data


def build_session_ws_url(ws_url_base: str, ws_path: str, token: str) -> str:
    normalized_base = ws_url_base.rstrip("/")
    normalized_path = ws_path if ws_path.startswith("/") else f"/{ws_path}"
    return f"{normalized_base}{normalized_path}?token={token}"


def print_message(data: Dict[str, Any], *, prefix: str = "[ws]") -> None:
    message_type = str(data.get("type") or "unknown")
    print(f"{prefix} type={message_type}")

    if message_type == "session_ready":
        print(
            f"  session_id={data.get('session_id')} "
            f"scene={data.get('scene')} status={data.get('status')} "
            f"service_id={data.get('service_id')}"
        )
        return

    if message_type == "llm_text_delta":
        delta = str(data.get("delta") or "")
        preview = delta.replace("\n", "\\n")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(
            f"  seq={data.get('seq')} turn={data.get('turn_index')} chunk={data.get('chunk_index')} "
            f"is_final={data.get('is_final')} delta={preview}"
        )
        return

    if message_type == "session_error":
        print(f"  error={data.get('error')}")
        return

    if message_type == "session_closed":
        print(f"  status={data.get('status')}")
        return

    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_multimodal_messages(
    prompt: str,
    image_urls: Optional[List[str]] = None,
    video_urls: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_url in (image_urls or []):
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    for video_url in (video_urls or []):
        content.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url,
                },
            }
        )
    if prompt:
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


async def recv_json(websocket: websockets.WebSocketClientProtocol, timeout_seconds: float) -> Dict[str, Any]:
    try:
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise RuntimeError(f"websocket timeout after {timeout_seconds} seconds")
    except ConnectionClosed as e:
        raise RuntimeError(f"websocket closed: code={e.code}, reason={e.reason}")

    try:
        return json.loads(message)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid json from websocket: {message}") from e


async def wait_initial_session_ready(
    websocket: websockets.WebSocketClientProtocol,
    timeout_seconds: float,
) -> Dict[str, Any]:
    data = await recv_json(websocket, timeout_seconds)
    print_message(data, prefix="[initial]")
    if str(data.get("type") or "") != "session_ready":
        raise RuntimeError(f"expected initial session_ready, got: {data}")
    return data


async def wait_service_ready(
    websocket: websockets.WebSocketClientProtocol,
    timeout_seconds: float,
) -> Dict[str, Any]:
    while True:
        data = await recv_json(websocket, timeout_seconds)
        print_message(data, prefix="[ready.wait]")
        message_type = str(data.get("type") or "")
        if message_type == "session_error":
            raise RuntimeError(str(data.get("error") or "unknown session error"))
        if message_type == "session_ready" and data.get("service_id"):
            return data


async def run_turn(
    websocket: websockets.WebSocketClientProtocol,
    *,
    session_id: str,
    seq: int,
    prompt: str,
    image_urls: Optional[List[str]] = None,
    video_urls: Optional[List[str]] = None,
    timeout_seconds: float,
    disable_token_stats: bool = False,
) -> str:
    normalized_image_urls = [str(item).strip() for item in (image_urls or []) if str(item).strip()]
    normalized_video_urls = [str(item).strip() for item in (video_urls or []) if str(item).strip()]
    payload: Dict[str, Any] = {
        "type": "input_text",
        "session_id": session_id,
        "seq": seq,
    }
    if normalized_image_urls or normalized_video_urls:
        payload["messages"] = build_multimodal_messages(
            prompt,
            image_urls=normalized_image_urls,
            video_urls=normalized_video_urls,
        )
        print(
            f"[turn:{seq}] input={prompt} "
            f"image_urls={normalized_image_urls} video_urls={normalized_video_urls}"
        )
        print(f"[turn:{seq}] request_payload=")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        payload["text"] = prompt
        print(f"[turn:{seq}] input={prompt}")
    started_at = time.perf_counter()
    first_delta_at: Optional[float] = None
    await websocket.send(json.dumps(payload, ensure_ascii=False))

    answer_parts: List[str] = []
    final_server_stats: Dict[str, Any] = {}
    while True:
        data = await recv_json(websocket, timeout_seconds)
        print_message(data, prefix=f"[turn:{seq}]")
        message_type = str(data.get("type") or "")
        if message_type == "session_error":
            raise RuntimeError(str(data.get("error") or "unknown session error"))
        if message_type == "session_closed":
            raise RuntimeError("session closed unexpectedly while waiting for llm_text_delta")
        if message_type != "llm_text_delta":
            continue
        if int(data.get("seq") or -1) != seq:
            continue

        delta = str(data.get("delta") or "")
        if delta:
            if first_delta_at is None:
                first_delta_at = time.perf_counter()
            answer_parts.append(delta)
        if bool(data.get("is_final")):
            answer = "".join(answer_parts)
            finished_at = time.perf_counter()
            final_server_stats = {
                key: data.get(key)
                for key in (
                    "prompt_tokens",
                    "output_tokens",
                    "ttft_seconds",
                    "total_seconds",
                    "decode_seconds",
                    "tok_s_total",
                    "tok_s_decode",
                )
                if data.get(key) is not None
            }
            print(f"[turn:{seq}] answer_complete=")
            print(answer)
            if not disable_token_stats:
                stats = summarize_turn_stats(
                    started_at=started_at,
                    first_delta_at=first_delta_at,
                    finished_at=finished_at,
                    server_stats=final_server_stats,
                )
                print_turn_stats(seq, stats)
            return answer


async def close_session_via_ws(
    websocket: websockets.WebSocketClientProtocol,
    *,
    session_id: str,
    seq: int,
    timeout_seconds: float,
) -> None:
    payload = {
        "type": "session_close",
        "session_id": session_id,
        "seq": seq,
    }
    await websocket.send(json.dumps(payload, ensure_ascii=False))

    try:
        while True:
            data = await recv_json(websocket, timeout_seconds)
            print_message(data, prefix="[close]")
            if str(data.get("type") or "") == "session_closed":
                return
    except Exception as e:
        print(f"[close] ignore close wait error: {e}")


async def run_interactive_loop(
    websocket: websockets.WebSocketClientProtocol,
    *,
    session_id: str,
    start_seq: int,
    image_urls: Optional[List[str]] = None,
    video_urls: Optional[List[str]] = None,
    timeout_seconds: float,
    disable_token_stats: bool = False,
) -> int:
    seq = start_seq
    normalized_image_urls = [str(item).strip() for item in (image_urls or []) if str(item).strip()]
    normalized_video_urls = [str(item).strip() for item in (video_urls or []) if str(item).strip()]
    if normalized_image_urls or normalized_video_urls:
        print(
            "\n[text-chat] 进入多模态交互式问答。"
            "当前会把同一组 image_url / video_url 附加到每轮请求。输入 quit 退出。"
        )
    else:
        print("\n[text-chat] 进入交互式问答。输入 quit 退出。")
    while True:
        prompt = await asyncio.to_thread(input, "\n[text-chat] 输入内容（回车发送，quit 退出）> ")
        prompt = str(prompt or "").strip()
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit", "/quit"}:
            return seq
        await run_turn(
            websocket,
            session_id=session_id,
            seq=seq,
            prompt=prompt,
            image_urls=normalized_image_urls,
            video_urls=normalized_video_urls,
            timeout_seconds=timeout_seconds,
            disable_token_stats=disable_token_stats,
        )
        seq += 1


async def main() -> None:
    parser = argparse.ArgumentParser(description="文本推理服务命令行交互脚本")
    parser.add_argument("--api-url", default=API_BASE_URL, help="后端 API 基地址")
    parser.add_argument("--ws-url", default=WS_BASE_URL, help="后端 WS 基地址")
    parser.add_argument("--token", default=None, help="显式 JWT token；不传则本地自动生成")
    parser.add_argument("--model-name", required=True, help="模型名，例如 nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--model-class", default=None, help="模型分类，例如 Gemma / Qwen-text")
    parser.add_argument("--system-prompt", default="", help="可选 system prompt")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大输出 token")
    parser.add_argument("--enable-thinking", action="store_true", help="启用 thinking")
    parser.add_argument("--prompt", action="append", default=[], help="可重复传入多轮 prompt；不传时默认进入交互式问答")
    parser.add_argument(
        "--image-url",
        action="append",
        default=[],
        help="可重复传入图片 URL；传入后本轮请求会按 OpenAI 风格 messages 发送，启用多模态图片理解",
    )
    parser.add_argument(
        "--video-url",
        action="append",
        default=[],
        help="可重复传入视频 URL；传入后本轮请求会按 OpenAI 风格 messages 发送，启用多模态视频理解",
    )
    parser.add_argument("--no-interactive", action="store_true", help="执行完 `--prompt` 后不进入交互")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次等待超时秒数")
    parser.add_argument("--disable-token-stats", action="store_true", help="关闭 tokens/s 统计")
    args = parser.parse_args()

    user_id, token = await resolve_token(args.api_url, args.token)
    family = infer_family(args.model_name, args.family)
    print(f"[auth] user_id={user_id}")
    print(f"[model] model_name={args.model_name} family={family}")

    prompts = [str(p).strip() for p in (args.prompt or []) if str(p).strip()]
    image_urls = [str(item).strip() for item in (args.image_url or []) if str(item).strip()]
    video_urls = [str(item).strip() for item in (args.video_url or []) if str(item).strip()]

    metadata = {
        "model_name": args.model_name,
        "family": family,
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "enable_thinking": bool(args.enable_thinking),
    }
    session = await create_text_session(
        token=token,
        api_url=args.api_url,
        metadata=metadata,
    )
    session_id = str(session["id"])
    ws_path = str(session["ws_path"])
    ws_url = build_session_ws_url(args.ws_url, ws_path, token)
    print(f"[session] session_id={session_id}")
    print(f"[session] ws_url={ws_url}")
    if image_urls:
        print(f"[session] image_urls={image_urls}")
    if video_urls:
        print(f"[session] video_urls={video_urls}")

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as websocket:
        await wait_initial_session_ready(websocket, args.timeout)

        open_payload = {
            "type": "session_open",
            "session_id": session_id,
            "seq": 1,
        }
        print("[session.open] payload=")
        print(json.dumps(open_payload, ensure_ascii=False, indent=2))
        await websocket.send(json.dumps(open_payload, ensure_ascii=False))
        await wait_service_ready(websocket, args.timeout)

        seq = 2
        for prompt in prompts:
            await run_turn(
                websocket,
                session_id=session_id,
                seq=seq,
                prompt=prompt,
                image_urls=image_urls,
                video_urls=video_urls,
                timeout_seconds=args.timeout,
                disable_token_stats=bool(args.disable_token_stats),
            )
            seq += 1

        if not args.no_interactive:
            seq = await run_interactive_loop(
                websocket,
                session_id=session_id,
                start_seq=seq,
                image_urls=image_urls,
                video_urls=video_urls,
                timeout_seconds=args.timeout,
                disable_token_stats=bool(args.disable_token_stats),
            )

        await close_session_via_ws(
            websocket,
            session_id=session_id,
            seq=seq,
            timeout_seconds=min(args.timeout, 10.0),
        )

    print("[done] text session experiment finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[done] interrupted by user")

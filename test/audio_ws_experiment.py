"""
音频推理服务验证脚本

前提：
1. `python -m backend.app.main` 已运行
2. 已按模型拆分启动所需音频服务，例如：
   - `python inference/audio/main.py qwen_tts`
   - `python inference/audio/main.py qwen_asr`
   - `python inference/audio/main.py voxcpm_tts`

用途：
1. 模拟客户端通过 `/v1/tasks` 提交真实 audio 任务
2. 连接 `/ws/task/{task_id}` 监听流式与最终结果
3. 用现有 backend 链路验证音频推理服务是否正常

示例：
  python test/audio_ws_experiment.py
  python test/audio_ws_experiment.py --only tts,asr
  python test/audio_ws_experiment.py --only tts,tts_stream
  python test/audio_ws_experiment.py --model-name VoxCPM2 --only tts,realtime_tts
  python test/audio_ws_experiment.py --model-name VoxCPM2 --only tts --voxcpm-tts-variant plain
  python test/audio_ws_experiment.py --model-name Qwen3-TTS-12Hz-0.6B-CustomVoice --only tts
  python test/audio_ws_experiment.py --model-name Qwen3-TTS-12Hz-0.6B-CustomVoice --only tts_stream
  python test/audio_ws_experiment.py --model-name Qwen3-ASR-1.7B --only asr

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import websockets
import yaml
from jose import jwt
from websockets.exceptions import ConnectionClosed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.logger import get_app_logger
from backend.database import User

logger = get_app_logger(__name__)

API_BASE_URL = "http://192.168.0.105:8888"
WS_BASE_URL = "ws://192.168.0.105:8888"

VOICE_ROOT = PROJECT_ROOT / "inference" / "third_party" / "vibevoice_demo" / "voices"
# TTS_REFERENCE_WAV = str(VOICE_ROOT / "en-Alice_woman.wav")
TTS_REFERENCE_WAV = "third_party/vibevoice_demo/voices/en-Frank_man.wav"
# TTS_REFERENCE_WAV = "inference/third_party/vibevoice_demo/voices/zh-Bowen_man.wav"

# ASR_INPUT_WAV = str(VOICE_ROOT / "en-Frank_man.wav")
ASR_INPUT_WAV = "inference/third_party/vibevoice_demo/voices/en-Frank_man.wav"

DEFAULT_JWT_SECRET = "vitoom-default-secret-key-change-in-production"
DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE = 86400
REALTIME_TTS_TEST_PROMPT = (
    "Realtime audio synthesis test for websocket chunk streaming. "
    "Please read this as a longer validation sample so the client can observe multiple streamed audio chunks. "
    "We want to confirm that audio arrives incrementally instead of only appearing at the very end of generation. "
    "This sentence adds more duration, and the next sentence keeps the stream active a little longer. "
    "If realtime chunking is working correctly, the websocket listener should receive several audio stream messages "
    "before the final completion event is sent."
)
VOXCPM_PLAIN_TTS_PROMPT = "今天我非常高兴来到这里演讲。请用自然、清晰、稳定的中文朗读这段话。"
VOXCPM_CLONE_TTS_PROMPT = "请沿用参考音频的说话人音色，自然地朗读这段中文内容，保持清晰和稳定。"
QWEN_TTS_PROMPT = "欢迎来到 Vitoom 的 Qwen3-TTS 接入测试。请用自然、亲切、清晰的语气朗读这段中文文本。"
QWEN_TTS_STREAM_PROMPT = (
    "现在开始验证 Qwen3-TTS 的流式输出。"
    "请用自然、稳定、清晰的中文连续朗读这段稍长一些的文本，"
    "这样 websocket 客户端可以观察到多个 audio_stream_chunk 事件逐步到达。"
    "如果流式输出正常，我们应该先收到 audio_stream_start，"
    "随后收到若干个音频分片，最后再收到 audio_stream_end 和最终任务结果。"
)
QWEN_ASR_PROMPT = "Please transcribe the uploaded speech sample and detect its language."


def _is_voxcpm_model(model_name: Optional[str]) -> bool:
    raw = str(model_name or "").strip().lower()
    return "voxcpm" in raw


def _is_qwen_tts_model(model_name: Optional[str]) -> bool:
    raw = str(model_name or "").strip().lower()
    return ("qwen3-tts" in raw) or ("qwen-tts" in raw)


def _is_qwen_asr_model(model_name: Optional[str]) -> bool:
    raw = str(model_name or "").strip().lower()
    return ("qwen3-asr" in raw) or ("qwen-asr" in raw)


def _build_vibevoice_examples() -> Dict[str, Dict[str, Any]]:
    return {
        "tts": {
            "task_type": "audio",
            "audio_mode": "tts",
            "job_type": "TTS",
            # "prompt": (
            #     "Speaker 1: Welcome to the Vitoom audio integration test. "
            #     "This sample verifies the long-form text to speech path."
            # ),
            "prompt": (
                "Speaker 1: Welcome to the Vitoom audio integration test. "
                "This sample verifies the long-form text to speech path."
            ),
            # "prompt_wav_path": TTS_REFERENCE_WAV,
            # "prompt_text": "Welcome to the Vitoom audio integration test.",
            # "prompt_text": "我们的家园：地球是宇宙中的唯一",
            "response_format": "audio_file",
            "stream": False,
            "file_type": "wav",
            "guidance_scale": 1.3,
        },
        "tts_stream": {
            "task_type": "audio",
            "audio_mode": "tts",
            "job_type": "TTS",
            "prompt": (
                "Speaker 1: This is a websocket streaming TTS validation sample. "
                "Please keep speaking long enough so multiple audio chunks can be observed in the client."
            ),
            "response_format": "audio_file",
            "stream": True,
            "file_type": "wav",
            "guidance_scale": 1.3,
        },
        "asr": {
            "task_type": "audio",
            "audio_mode": "asr",
            "job_type": "ASR",
            "prompt": "Please transcribe the uploaded speech sample.",
            "input_audio_url": ASR_INPUT_WAV,
            "response_format": "text_file",
            "stream": True,
            "file_type": "txt",
            "timestamps": True,
            "speaker_diarization": True,
            "language": "en",
        },
        "realtime_tts": {
            "task_type": "audio",
            "audio_mode": "realtime_tts",
            "job_type": "REALTIME_TTS",
            "prompt": REALTIME_TTS_TEST_PROMPT,
            "voice_preset": "Emma",
            "speaker_name": "Emma",
            "response_format": "both",
            "stream": True,
            "file_type": "wav",
            "guidance_scale": 1.5,
        },
    }


def _build_voxcpm_tts_example(variant: str) -> Dict[str, Any]:
    if variant == "plain":
        return {
            "task_type": "audio",
            "audio_mode": "tts",
            "job_type": "TTS",
            "prompt": VOXCPM_PLAIN_TTS_PROMPT,
            "response_format": "audio_file",
            "stream": False,
            "file_type": "wav",
        }
    if variant == "clone":
        return {
            "task_type": "audio",
            "audio_mode": "tts",
            "job_type": "TTS",
            "prompt": VOXCPM_CLONE_TTS_PROMPT,
            "prompt_wav_path": TTS_REFERENCE_WAV,
            "response_format": "audio_file",
            "stream": False,
            "file_type": "wav",
        }
    raise ValueError(f"Unsupported VoxCPM tts variant: {variant}")


def _build_voxcpm_tts_stream_example(variant: str) -> Dict[str, Any]:
    payload = _build_voxcpm_tts_example(variant)
    payload.update(
        {
            "prompt": (
                "欢迎来到 VoxCPM2 流式 TTS 测试。"
                "请连续朗读这段较长的中文文本，以便客户端观察到多个音频分片持续返回。"
                "如果流式链路工作正常，应该先后收到 start、多个 chunk 和 end。"
            ),
            "stream": True,
        }
    )
    return payload


def _build_voxcpm_examples(tts_variant: str) -> Dict[str, Dict[str, Any]]:
    return {
        "tts": _build_voxcpm_tts_example(tts_variant),
        "tts_stream": _build_voxcpm_tts_stream_example(tts_variant),
        "realtime_tts": {
            "task_type": "audio",
            "audio_mode": "realtime_tts",
            "job_type": "REALTIME_TTS",
            "prompt": (
                "欢迎来到 VoxCPM2 实时语音测试。"
                "请连续朗读这段较长的中文文本，以便客户端观察到多个音频分片持续返回。"
                "这段内容会稍微长一些，用来验证 websocket 流式输出是否稳定。"
            ),
            "response_format": "both",
            "stream": True,
            "file_type": "wav",
            "prompt_wav_path": TTS_REFERENCE_WAV,
        },
    }


def _build_qwen_tts_examples() -> Dict[str, Dict[str, Any]]:
    return {
        "tts": {
            "task_type": "audio",
            "audio_mode": "tts",
            "job_type": "TTS",
            "prompt": QWEN_TTS_PROMPT,
            "speaker_name": "Vivian",
            "voice_preset": "Vivian",
            "language": "zh",
            "instruct": "请用温柔、自然、带一点微笑感的语气说。",
            "response_format": "audio_file",
            "stream": False,
            "file_type": "wav",
        },
        "tts_stream": {
            "task_type": "audio",
            "audio_mode": "tts",
            "job_type": "TTS",
            "prompt": QWEN_TTS_STREAM_PROMPT,
            "speaker_name": "Vivian",
            "voice_preset": "Vivian",
            "language": "zh",
            "instruct": "请用自然、亲切、稳定的语气持续朗读，便于观察流式分片输出。",
            "response_format": "audio_file",
            "stream": True,
            "file_type": "wav",
        },
    }


def _build_qwen_asr_examples() -> Dict[str, Dict[str, Any]]:
    return {
        "asr": {
            "task_type": "audio",
            "audio_mode": "asr",
            "job_type": "ASR",
            "prompt": QWEN_ASR_PROMPT,
            "input_audio_url": ASR_INPUT_WAV,
            "prompt_text": "Welcome to the Vitoom audio integration test.",
            "response_format": "text_file",
            "stream": True,
            "file_type": "txt",
            "timestamps": True,
            "speaker_diarization": False,
            "language": "en",
        },
    }


def build_examples(
    model_name: Optional[str] = None,
    *,
    voxcpm_tts_variant: str = "clone",
) -> Dict[str, Dict[str, Any]]:
    if _is_voxcpm_model(model_name):
        return _build_voxcpm_examples(voxcpm_tts_variant)
    if _is_qwen_tts_model(model_name):
        return _build_qwen_tts_examples()
    if _is_qwen_asr_model(model_name):
        return _build_qwen_asr_examples()
    return _build_vibevoice_examples()


async def create_test_user_and_token(api_url: str) -> tuple[str, str]:
    del api_url  # 仅保留函数签名兼容；当前实现直接复用现有用户

    users = User.list_all(limit=20)
    if not users:
        raise RuntimeError("no users found in users table")

    user = next((u for u in users if str(u.get("status") or "").lower() == "active"), users[0])
    user_id = str(user.get("id") or "")
    email = str(user.get("email") or "")
    if not user_id:
        raise RuntimeError(f"selected user missing id: {user}")

    token = create_access_token_local({"sub": user_id, "email": email})
    return user_id, token


def load_security_settings() -> Dict[str, Any]:
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


async def create_audio_task(
    token: str,
    mode: str,
    payload: Dict[str, Any],
    api_url: str,
) -> tuple[str, float, float]:
    """返回 (task_id, t_submit_start, t_submit_end)，均为 time.perf_counter() 单调时钟。"""
    request_data = dict(payload)
    print(f"[{mode}] request_payload=")
    print(json.dumps(request_data, ensure_ascii=False, indent=2))

    t_submit_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{api_url}/v1/tasks",
            json=request_data,
            headers={"Authorization": f"Bearer {token}"},
        )
    t_submit_end = time.perf_counter()
    if resp.status_code != 201:
        raise RuntimeError(f"[{mode}] create task failed: {resp.status_code} {resp.text}")

    body = resp.json()
    data = body.get("data") if isinstance(body, dict) else None
    task_id = (data or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"[{mode}] response missing task_id: {body}")
    return task_id, t_submit_start, t_submit_end


def _print_message(mode: str, data: Dict[str, Any]) -> None:
    msg_type = str(data.get("type") or "unknown")
    print(f"[{mode}] type={msg_type}")

    if msg_type == "task_status":
        print(
            f"  status={data.get('status')} progress={data.get('progress')} "
            f"error={data.get('error')}"
        )
        return

    if msg_type == "result":
        files = data.get("files") or []
        print(f"  status={data.get('status')} progress={data.get('progress')} files={len(files)}")
        for i, file_info in enumerate(files):
            fi = file_info if isinstance(file_info, dict) else {}
            print(
                f"  file={fi.get('file_name')} "
                f"mime={fi.get('mime_type')} "
                f"path={fi.get('storage_path')}"
            )
            # 便于一眼确认推理侧是否带可访问 URL（无键表示 FileInfo.to_dict 未写入）
            print(f"  files[{i}].url={fi.get('url')!r} thumb_url={fi.get('thumb_url')!r}")
        print("  --- raw result message (full JSON) ---")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("  --- end raw result ---")
        return

    if msg_type == "text_stream_delta":
        delta = str(data.get("delta") or "")
        preview = delta.replace("\n", "\\n")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  delta={preview}")
        return

    if msg_type == "transcript_segment":
        seg = data.get("segment") or {}
        print(
            f"  segment speaker={seg.get('speaker_id')} "
            f"[{seg.get('start_time')} - {seg.get('end_time')}] "
            f"text={seg.get('content')}"
        )
        return

    if msg_type == "audio_stream_chunk":
        b64 = str(data.get("data") or "")
        print(f"  audio_chunk_base64_len={len(b64)} sample_rate={data.get('sample_rate')}")
        return

    if msg_type in {"audio_stream_start", "audio_stream_end"}:
        print(
            f"  sequence={data.get('sequence')} "
            f"sample_rate={data.get('sample_rate')} "
            f"is_final={data.get('is_final')}"
        )
        return

    print(json.dumps(data, ensure_ascii=False, indent=2))


async def observe_task(
    task_id: str,
    token: str,
    mode: str,
    ws_url_base: str,
    timeout_seconds: float,
    *,
    t_submit_start: float,
) -> Dict[str, Any]:
    ws_url = f"{ws_url_base}/ws/task/{task_id}?token={token}"
    final_status = "unknown"
    audio_stream_chunk_count = 0
    audio_stream_start_count = 0
    audio_stream_end_count = 0
    result_count = 0
    result_file_count = 0
    t_ws_connected: Optional[float] = None
    t_first_audio_chunk: Optional[float] = None
    t_done: Optional[float] = None

    try:
        async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as websocket:
            t_ws_connected = time.perf_counter()
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    raise RuntimeError(f"[{mode}] websocket timeout after {timeout_seconds} seconds")
                except ConnectionClosed:
                    break

                data = json.loads(message)
                _print_message(mode, data)
                msg_type = str(data.get("type") or "")
                if msg_type == "audio_stream_chunk":
                    if t_first_audio_chunk is None:
                        t_first_audio_chunk = time.perf_counter()
                    audio_stream_chunk_count += 1
                elif msg_type == "audio_stream_start":
                    audio_stream_start_count += 1
                elif msg_type == "audio_stream_end":
                    audio_stream_end_count += 1
                elif msg_type == "result":
                    result_count += 1
                    files = data.get("files") or []
                    try:
                        result_file_count += len(files)
                    except Exception:
                        pass
                status = str(data.get("status") or "")
                if status in {"completed", "failed", "cancelled"}:
                    final_status = status
                    if data.get("type") == "task_status":
                        t_done = time.perf_counter()
                        break
                    if data.get("type") == "result" and status == "completed":
                        t_done = time.perf_counter()
                        break
    finally:
        if t_done is None:
            t_done = time.perf_counter()

    return {
        "final_status": final_status,
        "audio_stream_chunk_count": audio_stream_chunk_count,
        "audio_stream_start_count": audio_stream_start_count,
        "audio_stream_end_count": audio_stream_end_count,
        "result_count": result_count,
        "result_file_count": result_file_count,
        "t_submit_start": t_submit_start,
        "t_ws_connected": t_ws_connected,
        "t_first_audio_chunk": t_first_audio_chunk,
        "t_done": t_done,
    }


async def run_mode(
    *,
    mode: str,
    token: str,
    api_url: str,
    ws_url: str,
    timeout_seconds: float,
    explicit_model_name: Optional[str] = None,
    realtime_min_chunks: int = 2,
    tts_stream_min_chunks: int = 1,
    voxcpm_tts_variant: str = "clone",
) -> str:
    examples = build_examples(explicit_model_name, voxcpm_tts_variant=voxcpm_tts_variant)
    if mode not in examples:
        raise RuntimeError(
            f"[{mode}] model_name={explicit_model_name!r} has no built-in example payload for this mode"
        )
    payload = dict(examples[mode])
    if explicit_model_name:
        payload["model_name"] = explicit_model_name
    if not payload.get("model_name"):
        raise RuntimeError(f"[{mode}] model_name is required")

    task_id, t_submit_start, t_submit_end = await create_audio_task(token, mode, payload, api_url)
    print(f"\n[{mode}] task_id={task_id} model={payload['model_name']}")
    print(
        f"[{mode}] HTTP 创建任务耗时: {(t_submit_end - t_submit_start) * 1000.0:.1f} ms "
        f"({(t_submit_end - t_submit_start):.3f} s)"
    )
    summary = await observe_task(
        task_id,
        token,
        mode,
        ws_url,
        timeout_seconds,
        t_submit_start=t_submit_start,
    )
    status = str(summary["final_status"])
    result_count = int(summary["result_count"])
    result_file_count = int(summary["result_file_count"])
    if mode in {"tts", "tts_stream", "realtime_tts"}:
        print(
            f"[{mode}] result_summary result_messages={result_count} result_files={result_file_count}"
        )
        if result_count < 1:
            raise RuntimeError(f"[{mode}] did not receive final result message")
        if result_file_count < 1:
            raise RuntimeError(f"[{mode}] final result did not contain audio file")
    if mode in {"tts_stream", "realtime_tts"}:
        chunk_count = int(summary["audio_stream_chunk_count"])
        start_count = int(summary["audio_stream_start_count"])
        end_count = int(summary["audio_stream_end_count"])
        print(
            f"[{mode}] stream_summary start={start_count} chunks={chunk_count} end={end_count}"
        )
        if start_count < 1 and chunk_count > 0:
            print(
                f"[{mode}] note: audio_stream_start may have been emitted before websocket subscription; "
                "continuing because chunk stream was observed"
            )
        elif start_count < 1:
            raise RuntimeError(f"[{mode}] did not receive audio_stream_start")
        if end_count < 1:
            raise RuntimeError(f"[{mode}] did not receive audio_stream_end")
        required_chunks = realtime_min_chunks if mode == "realtime_tts" else tts_stream_min_chunks
        if chunk_count < required_chunks:
            raise RuntimeError(
                f"[{mode}] expected at least {required_chunks} audio_stream_chunk messages, got {chunk_count}"
            )
    elif mode == "tts":
        if int(summary["audio_stream_chunk_count"]) > 0:
            raise RuntimeError(f"[{mode}] expected offline file generation, but received audio stream chunks")

    t0 = float(summary["t_submit_start"])
    t_done = float(summary["t_done"] or t0)
    t_first = summary.get("t_first_audio_chunk")
    print(f"[{mode}] final_status={status}")
    if t_first is not None:
        ttft = float(t_first) - t0
        print(f"[{mode}] 发出消息→首条 audio_stream_chunk: {ttft * 1000.0:.1f} ms ({ttft:.3f} s)")
    else:
        print(f"[{mode}] 发出消息→首条 audio_stream_chunk: (未收到，可能未开 stream 或无音频分片)")
    total = t_done - t0
    print(f"[{mode}] 发出消息→完成: {total * 1000.0:.1f} ms ({total:.3f} s)\n")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio websocket experiment runner")
    parser.add_argument(
        "--only",
        type=str,
        default="tts,asr,realtime_tts",
        help="Comma separated modes to run: tts,tts_stream,asr,realtime_tts",
    )
    parser.add_argument("--api-url", type=str, default=API_BASE_URL)
    parser.add_argument("--ws-url", type=str, default=WS_BASE_URL)
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-task websocket timeout in seconds")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="model_name used for all selected modes; if omitted, each mode uses its built-in default",
    )
    parser.add_argument(
        "--realtime-min-chunks",
        type=int,
        default=2,
        help="Minimum number of audio_stream_chunk messages expected for realtime_tts",
    )
    parser.add_argument(
        "--tts-stream-min-chunks",
        type=int,
        default=1,
        help="Minimum number of audio_stream_chunk messages expected for tts_stream",
    )
    parser.add_argument(
        "--voxcpm-tts-variant",
        choices=("plain", "clone"),
        default="clone",
        help="Built-in VoxCPM2 tts example variant: plain=text only, clone=text + prompt_wav_path",
    )
    return parser.parse_args()


async def amain() -> int:
    args = parse_args()
    selected_modes = [
        item.strip() for item in args.only.split(",")
        if item.strip() in {"tts", "tts_stream", "asr", "realtime_tts"}
    ]
    if not selected_modes:
        raise RuntimeError("No valid modes selected. Use: tts,tts_stream,asr,realtime_tts")

    print("=" * 72)
    print("Audio WebSocket Experiment")
    print("=" * 72)
    print(f"api_url={args.api_url}")
    print(f"ws_url={args.ws_url}")
    print(f"modes={selected_modes}")
    print(f"tts_reference={TTS_REFERENCE_WAV}")
    print(f"asr_input={ASR_INPUT_WAV}")
    print(f"voxcpm_tts_variant={args.voxcpm_tts_variant}")

    user_id, token = await create_test_user_and_token(args.api_url)
    print(f"test_user_id={user_id}")
    explicit_names = {
        "tts": args.model_name or "VibeVoice-1.5B",
        "tts_stream": args.model_name or "VibeVoice-1.5B",
        "asr": args.model_name or "VibeVoice-ASR",
        "realtime_tts": args.model_name or "VibeVoice-Realtime-0.5B",
    }

    overall_ok = True
    for mode in selected_modes:
        try:
            status = await run_mode(
                mode=mode,
                token=token,
                api_url=args.api_url,
                ws_url=args.ws_url.replace("http://", "ws://").replace("https://", "wss://"),
                timeout_seconds=args.timeout,
                explicit_model_name=explicit_names.get(mode),
                realtime_min_chunks=args.realtime_min_chunks,
                tts_stream_min_chunks=args.tts_stream_min_chunks,
                voxcpm_tts_variant=args.voxcpm_tts_variant,
            )
            overall_ok = overall_ok and (status == "completed")
        except Exception as exc:
            overall_ok = False
            print(f"[{mode}] ERROR: {exc}")

    print("=" * 72)
    print("DONE" if overall_ok else "DONE WITH ERRORS")
    print("=" * 72)
    return 0 if overall_ok else 1


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())

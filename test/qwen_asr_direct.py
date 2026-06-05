#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.signal import resample_poly

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("soundfile is required to run this script") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = REPO_ROOT / "inference"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from audio.runtime.qwen_asr_bridge import load_asr_bundle, resolve_qwen_asr_bundle_options
from audio.runtime.qwen_asr_vllm_streaming import (  # noqa: E402
    create_vllm_streaming_session,
    load_vllm_asr_bundle,
    resolve_qwen_asr_vllm_options,
)
from audio.runtime.runtime_resolver import (  # noqa: E402
    resolve_audio_model_ref,
    resolve_audio_runtime_policy,
)
from common.config_loader import load_startup_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    default_audio = REPO_ROOT / "inference/third_party/vibevoice_demo/voices/zh-Xinran_woman.wav"
    parser = argparse.ArgumentParser(
        description="本地直测 Qwen-ASR，绕过 WS/联调链路。",
    )
    parser.add_argument("--service-id", default="qwen_asr", help="读取哪份启动配置，默认 qwen_asr")
    parser.add_argument("--audio-file", default=str(default_audio), help="待测试的本地音频文件")
    parser.add_argument("--model-name", default="Qwen3-ASR-0.6B", help="模型名或本地模型路径")
    parser.add_argument("--model-class", default="Qwen-asr", help="默认使用 Qwen-asr")
    parser.add_argument(
        "--mode",
        choices=("streaming-vllm", "offline"),
        default="streaming-vllm",
        help="streaming-vllm 复现实时流式链路；offline 走一次性 transcribe",
    )
    parser.add_argument("--chunk-size-sec", type=float, default=2.0, help="streaming 模式下每个音频块的秒数")
    parser.add_argument("--language", default="", help="可选语言提示，如 zh/en")
    parser.add_argument("--context", default="", help="可选上下文提示")
    parser.add_argument("--timestamps", action="store_true", help="offline 模式输出时间戳")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="覆盖 service 配置里的 vLLM 显存比例")
    parser.add_argument("--max-model-len", type=int, default=None, help="覆盖 service 配置里的 vLLM max_model_len")
    parser.add_argument("--enforce-eager", action="store_true", help="覆盖 service 配置，强制 eager")
    parser.add_argument(
        "--disable-enforce-eager",
        action="store_true",
        help="显式关闭 eager；若同时传了 --enforce-eager，以后者为准",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="覆盖 qwen-asr 的 max_new_tokens")
    parser.add_argument("--show-partials", action="store_true", help="streaming 模式打印每个 chunk 的 partial/delta")
    parser.add_argument("--print-config", action="store_true", help="打印解析后的关键配置")
    return parser.parse_args()


def _read_audio_mono_16k(audio_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)

    mono = np.mean(audio, axis=1, dtype=np.float32)
    if int(sample_rate) != 16000:
        mono = resample_poly(mono, up=16000, down=int(sample_rate)).astype(np.float32, copy=False)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _split_audio(audio: np.ndarray, *, chunk_size_sec: float, sample_rate: int = 16000) -> list[np.ndarray]:
    chunk_size = max(1, int(round(chunk_size_sec * sample_rate)))
    return [audio[i : i + chunk_size].copy() for i in range(0, len(audio), chunk_size)] or [np.zeros((0,), dtype=np.float32)]


def _ensure_runtime_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = model_cfg.get("runtime")
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
        model_cfg["runtime"] = runtime_cfg
    return runtime_cfg


def _build_model_cfg(service_cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    model_cfg = copy.deepcopy(service_cfg if isinstance(service_cfg, dict) else {})
    runtime_cfg = _ensure_runtime_cfg(model_cfg)
    vllm_cfg = runtime_cfg.get("vllm")
    if not isinstance(vllm_cfg, dict):
        vllm_cfg = {}
        runtime_cfg["vllm"] = vllm_cfg

    if args.gpu_memory_utilization is not None:
        vllm_cfg["gpu_memory_utilization"] = float(args.gpu_memory_utilization)
    if args.max_model_len is not None:
        vllm_cfg["max_model_len"] = int(args.max_model_len)
    if args.enforce_eager:
        vllm_cfg["enforce_eager"] = True
    elif args.disable_enforce_eager:
        vllm_cfg["enforce_eager"] = False
    if args.mode == "streaming-vllm":
        runtime_cfg["backend"] = "vllm"
    if args.max_new_tokens is not None:
        runtime_cfg["max_new_tokens"] = int(args.max_new_tokens)
    return model_cfg


def _build_params(model_cfg: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        family=args.family,
        model_cfg=model_cfg,
        timestamps=bool(args.timestamps),
        language=args.language,
        prompt_text=args.context,
    )


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _run_streaming(audio: np.ndarray, bundle: dict[str, Any], *, chunk_size_sec: float, show_partials: bool) -> int:
    session = create_vllm_streaming_session(bundle)
    chunks = _split_audio(audio, chunk_size_sec=chunk_size_sec)
    print(f"\n[streaming] start chunks={len(chunks)} chunk_size_sec={chunk_size_sec:.3f}")
    started_at = time.time()

    for idx, chunk in enumerate(chunks, start=1):
        payload = session.push_chunk(chunk)
        if show_partials:
            text = str(payload.get("text", "") or "")
            delta = str(payload.get("delta", "") or "")
            replaced = str(payload.get("replaced", "") or "")
            language = str(payload.get("language", "") or "")
            print(
                f"[chunk {idx:03d}/{len(chunks):03d}] "
                f"samples={len(chunk):6d} language={language or '-'} "
                f"replaced={replaced!r} delta={delta!r} text={text!r}"
            )

    final_payload = session.finish()
    elapsed = time.time() - started_at
    print(f"\n[streaming] done elapsed={elapsed:.2f}s")
    print(f"[streaming] language={str(final_payload.get('language', '') or '').strip() or '-'}")
    print("[streaming] final_text:")
    print(str(final_payload.get("text", "") or "").strip())
    return 0


def _run_offline(audio_path: Path, bundle: dict[str, Any], args: argparse.Namespace) -> int:
    model = bundle["model"]
    started_at = time.time()
    results = model.transcribe(
        audio=str(audio_path),
        language=(args.language or None),
        context=(args.context or None),
        return_time_stamps=bool(args.timestamps),
    )
    elapsed = time.time() - started_at
    result = results[0] if isinstance(results, (list, tuple)) and results else results
    text = str(getattr(result, "text", "") or "").strip()
    language = str(getattr(result, "language", "") or "").strip()
    print(f"\n[offline] done elapsed={elapsed:.2f}s")
    print(f"[offline] language={language or '-'}")
    print("[offline] text:")
    print(text)
    if args.timestamps:
        stamps = getattr(result, "time_stamps", None)
        print("\n[offline] timestamps:")
        print(json.dumps(stamps, ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    args = _parse_args()
    startup_config = load_startup_config(args.service_id)
    inference_cfg = startup_config.inference_config
    model_cfg = _build_model_cfg(getattr(startup_config, "config", {}) or {}, args)
    params = _build_params(model_cfg, args)

    audio_path = Path(args.audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    audio = _read_audio_mono_16k(audio_path)
    if audio.size == 0:
        raise RuntimeError(f"audio file is empty after decode: {audio_path}")

    model_ref = resolve_audio_model_ref(
        params,
        models_dir=getattr(inference_cfg, "models_dir", None),
        weights_dir=getattr(inference_cfg, "weights_dir", None),
    )
    policy = resolve_audio_runtime_policy(params, audio_mode="asr")
    bundle_options = resolve_qwen_asr_bundle_options(
        params,
        models_dir=getattr(inference_cfg, "models_dir", None),
        weights_dir=getattr(inference_cfg, "weights_dir", None),
    )

    print(f"[input] audio_file={audio_path}")
    print(f"[input] duration_sec={len(audio) / 16000.0:.2f} sample_rate=16000 samples={len(audio)}")
    print(f"[model] model_ref={model_ref}")
    print(f"[model] mode={args.mode}")

    if args.mode == "streaming-vllm":
        vllm_options = resolve_qwen_asr_vllm_options(model_cfg)
        if args.print_config:
            _print_json(
                "Resolved Config",
                {
                    "policy": {
                        "device": policy.device,
                        "dtype": policy.torch_dtype_name,
                        "attn_implementation": policy.attn_implementation,
                        "allow_remote_assets": policy.allow_remote_assets,
                    },
                    "bundle_options": {
                        "forced_aligner_ref": bundle_options.forced_aligner_ref,
                        "max_new_tokens": bundle_options.max_new_tokens,
                        "max_inference_batch_size": bundle_options.max_inference_batch_size,
                        "timestamps_requested": bundle_options.timestamps_requested,
                    },
                    "vllm_options": {
                        "gpu_memory_utilization": vllm_options.gpu_memory_utilization,
                        "max_model_len": vllm_options.max_model_len,
                        "enforce_eager": vllm_options.enforce_eager,
                        "streaming_unfixed_chunk_num": vllm_options.streaming_unfixed_chunk_num,
                        "streaming_unfixed_token_num": vllm_options.streaming_unfixed_token_num,
                        "streaming_chunk_size_sec": vllm_options.streaming_chunk_size_sec,
                    },
                },
            )
        load_started_at = time.time()
        bundle = load_vllm_asr_bundle(model_ref, policy, bundle_options, vllm_options)
        print(f"[model] load_elapsed_sec={time.time() - load_started_at:.2f}")
        return _run_streaming(audio, bundle, chunk_size_sec=args.chunk_size_sec, show_partials=args.show_partials)

    if args.print_config:
        _print_json(
            "Resolved Config",
            {
                "policy": {
                    "device": policy.device,
                    "dtype": policy.torch_dtype_name,
                    "attn_implementation": policy.attn_implementation,
                    "allow_remote_assets": policy.allow_remote_assets,
                },
                "bundle_options": {
                    "forced_aligner_ref": bundle_options.forced_aligner_ref,
                    "max_new_tokens": bundle_options.max_new_tokens,
                    "max_inference_batch_size": bundle_options.max_inference_batch_size,
                    "timestamps_requested": bundle_options.timestamps_requested,
                },
            },
        )
    load_started_at = time.time()
    bundle = load_asr_bundle(model_ref, policy, bundle_options)
    print(f"[model] load_elapsed_sec={time.time() - load_started_at:.2f}")
    return _run_offline(audio_path, bundle, args)


if __name__ == "__main__":
    raise SystemExit(main())

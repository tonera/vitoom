"""Qwen-TTS 输出速度小测：直连 ``QwenTtsEngine.synthesize_stream``，
打印每个 chunk 的 wall time / 累计音频时长 / RTF，最后汇总输出。

诊断用：定位"TTS 生成和朗读几乎同时结束"是模型生成跟不上，还是上层调度卡顿。

用法（贴合产线 nano_vllm 流式后端）::

    python test/bench_qwen_tts_speed.py \
        --model /home/tonera/aimodels/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
        --speaker Vivian --language Chinese \
        --instruct "用特别愤怒的语气说" \
        --output output_custom_voice.wav

切换 transformers 后端（无真流式，整段一次性出）::

    python test/bench_qwen_tts_speed.py --backend transformers ...

关键指标::

    RTF (real-time factor) = wall_seconds / audio_seconds
        < 1.0 → 生成比播放快，可缓冲；越小越好
        ≥ 1.0 → 生成跟不上播放，必然卡顿
    TTFB (time to first audio byte)
        从合成开始到第一段音频可播的延迟，越小越好
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from audio.engines.qwen_tts_engine import QwenTtsEngine  # noqa: E402
from audio.engines.tts_engine import VoiceConfig  # noqa: E402
from audio.runtime.qwen_tts_bridge import load_tts_bundle  # noqa: E402
from audio.runtime.runtime_resolver import (  # noqa: E402
    AudioRuntimePolicy,
    merge_qwen_tts_loader_runtime_cfg,
)


def _build_policy() -> AudioRuntimePolicy:
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        attn = "flash_attention_2"
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        attn = "sdpa"
    else:
        device = "cpu"
        dtype = torch.float32
        attn = "sdpa"
    return AudioRuntimePolicy(
        audio_mode="tts",
        low_vram=False,
        fast_mode=False,
        policy_source="bench-script",
        device=device,
        torch_dtype=dtype,
        torch_dtype_name=str(dtype).replace("torch.", ""),
        attn_implementation=attn,
        device_map=None,
        low_cpu_mem_usage=False,
        allow_remote_assets=False,
    )


def _load_runtime_cfg(yaml_path: Path, backend: str) -> dict:
    if not yaml_path.exists():
        return {}
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    runtime = ((cfg.get("config") or {}).get("runtime")) or {}
    return merge_qwen_tts_loader_runtime_cfg(runtime, backend=backend)


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("bench-qwen-tts")

    policy = _build_policy()
    runtime_cfg = _load_runtime_cfg(Path(args.config), args.backend)
    logger.info("policy=%s", asdict(policy))
    logger.info("backend=%s runtime_cfg=%s", args.backend, runtime_cfg)

    t_load0 = time.perf_counter()
    bundle = await asyncio.to_thread(
        load_tts_bundle, args.model, policy, args.backend, runtime_cfg
    )
    t_load = time.perf_counter() - t_load0
    logger.info(
        "bundle loaded in %.2fs streaming_variant=%s sample_rate=%s caps=%s",
        t_load,
        bundle.get("streaming_variant"),
        bundle.get("sample_rate"),
        bundle.get("capabilities"),
    )

    engine = QwenTtsEngine(bundle_loader=_make_dummy_loader(bundle), logger=logger)
    voice_cfg = VoiceConfig(
        tts_mode=args.tts_mode,
        speaker_name=args.speaker,
        language=args.language,
        instruct=(args.instruct or None),
        model_name=args.model,
    )

    print(
        "\n--- per-chunk timeline (wall vs audio; RTF<1 表示在领先于播放) ---\n"
        f"{'seq':>4} {'final':>5} {'wall_s':>8} {'wall_dt':>8} "
        f"{'audio_s':>8} {'chunk_s':>8} {'rtf_inst':>8} {'lead_s':>8}"
    )

    t_gen0 = time.perf_counter()
    last_wall = 0.0
    cum_audio_seconds = 0.0
    ttfb: float | None = None
    sr_used: int | None = None
    full_audio: np.ndarray | None = None
    chunk_walls: list[float] = []

    async for chunk in engine.synthesize_stream(
        text=args.text, voice_cfg=voice_cfg, stream_mode=args.stream_mode
    ):
        wall = time.perf_counter() - t_gen0
        wall_dt = wall - last_wall
        last_wall = wall
        sr_used = chunk.sample_rate
        chunk_seconds = (chunk.pcm.size / chunk.sample_rate) if chunk.sample_rate else 0.0
        if not chunk.is_final:
            cum_audio_seconds += chunk_seconds
            if ttfb is None and chunk.pcm.size > 0:
                ttfb = wall
            chunk_walls.append(wall_dt)
            rtf_inst = (wall_dt / chunk_seconds) if chunk_seconds > 0 else float("inf")
            lead = cum_audio_seconds - wall
            print(
                f"{chunk.sequence:>4} {'-':>5} {wall:>8.3f} {wall_dt:>8.3f} "
                f"{cum_audio_seconds:>8.3f} {chunk_seconds:>8.3f} "
                f"{rtf_inst:>8.3f} {lead:>+8.3f}"
            )
        else:
            full_audio = chunk.pcm
            print(
                f"{chunk.sequence:>4} {'YES':>5} {wall:>8.3f} {wall_dt:>8.3f} "
                f"{(chunk.pcm.size / chunk.sample_rate):>8.3f} {'-':>8} {'-':>8} {'-':>8}"
            )

    total_wall = time.perf_counter() - t_gen0
    total_audio = (full_audio.size / sr_used) if (full_audio is not None and sr_used) else 0.0
    rtf = (total_wall / total_audio) if total_audio > 0 else float("inf")

    print("\n--- summary ---")
    print(f"  load wall:        {t_load:.2f} s")
    print(f"  total wall:       {total_wall:.3f} s")
    print(f"  total audio:      {total_audio:.3f} s ({sr_used} Hz)")
    print(f"  end-to-end RTF:   {rtf:.3f}  ({'OK ✓' if rtf < 1 else 'BAD ✗ 跟不上播放'})")
    if ttfb is not None:
        print(f"  TTFB (first PCM): {ttfb:.3f} s")
    if chunk_walls:
        print(
            f"  chunks:           n={len(chunk_walls)}  "
            f"wall avg={sum(chunk_walls)/len(chunk_walls)*1000:.1f}ms  "
            f"max={max(chunk_walls)*1000:.1f}ms"
        )

    if args.output and full_audio is not None and full_audio.size > 0:
        try:
            import soundfile as sf

            sf.write(args.output, full_audio, sr_used or 24000)
            print(f"  wrote: {args.output}")
        except Exception as exc:
            print(f"  failed to write {args.output}: {exc}")


def _make_dummy_loader(bundle: dict):
    """``QwenTtsEngine`` 期望一个 ``bundle_loader`` 协议；本脚本只跑单权重，
    直接把已加载的 bundle 还回去即可。"""

    async def _loader(_mode: str, *, model_name: str | None = None) -> dict:
        return bundle

    return _loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="本地权重绝对路径")
    parser.add_argument(
        "--text",
        default="王勃的成名作是《滕王阁序》（全称《秋日登洪府滕王阁饯别序》）。这篇骈文是唐高宗上元二年（675年）王勃在南昌滕王阁宴会上即兴创作的，以其华丽的辞藻、精巧的构思和深邃的意境著称，尤其是其中的名句“落霞与孤鹜齐飞，秋水共长天一色”流传千古。《滕王阁序》不仅奠定了王勃在文学史上的地位，也被视为中国骈文的巅峰之作之一。",
    )
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="")
    parser.add_argument(
        "--tts-mode",
        default="custom_voice",
        choices=["custom_voice", "voice_design"],
    )
    parser.add_argument("--backend", default="nano_vllm", choices=["nano_vllm", "transformers"])
    parser.add_argument("--stream-mode", action=argparse.BooleanOptionalAction, default=True,
                        help="底层 generate 的 non_streaming_mode 反向，仅 nano_vllm 后端有效")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "inference" / "config" / "qwen_tts.yaml"))
    parser.add_argument("--output", default="bench_qwen_tts.wav")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))

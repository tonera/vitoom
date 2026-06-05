"""VoxCPM 纯合成引擎。

从 `audio.handlers.voxcpm_tts_handler` 抽出 tts / realtime_tts 两条合成路径
（两者合并为同一 `synthesize_stream`，非 realtime 是 realtime 的退化）。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Mapping, Optional

import numpy as np

from common.io_utils import download_url_to_tempfile
from audio.engines.tts_engine import (
    AudioChunk,
    CancelCheck,
    VoiceConfig,
    normalize_audio_array,
)

# inference/ 根目录：inference/audio/engines/voxcpm_tts_engine.py → parents[2]
# 用于把 yaml 里相对路径（如 "third_party/vibevoice_demo/voices/zh-linda.wav"）
# 解析成绝对路径，避免受进程 CWD 影响。
_INFERENCE_ROOT = Path(__file__).resolve().parents[2]


BundleLoader = Callable[[str], Awaitable[Dict[str, Any]]]


class VoxCPMTtsEngine:
    """VoxCPM 纯合成引擎。

    `audio_mode` 在构造时固定（"tts" / "realtime_tts"），决定加载哪份 bundle。
    两条路径共用同一个 `synthesize_stream`；非流式仅是流式的一次性退化。
    """

    def __init__(
        self,
        *,
        audio_mode: str,
        bundle_loader: BundleLoader,
        logger: logging.Logger,
        speaker_presets: Optional[Mapping[str, str]] = None,
        default_speaker: Optional[str] = None,
    ):
        mode = str(audio_mode or "tts").strip().lower()
        if mode not in ("tts", "realtime_tts"):
            raise ValueError(f"VoxCPMTtsEngine expects audio_mode tts|realtime_tts, got {mode!r}")
        self._audio_mode = mode
        self._bundle_loader = bundle_loader
        self._logger = logger
        # 名字 → 参考音频路径（来自共享 config/tts_speakers.json）
        # key 统一 lower/strip，匹配时对入参做同样处理。
        self._speaker_presets: Dict[str, str] = {
            str(k).strip().lower(): str(v)
            for k, v in (speaker_presets or {}).items()
            if str(k).strip() and str(v).strip()
        }
        default_key = str(default_speaker or "").strip().lower()
        self._default_speaker: Optional[str] = default_key or None

    @property
    def audio_mode(self) -> str:
        return self._audio_mode

    async def synthesize_stream(
        self,
        *,
        text: str,
        voice_cfg: VoiceConfig,
        cancel_check: Optional[CancelCheck] = None,
        stream_mode: bool = True,
    ) -> AsyncIterator[AudioChunk]:
        if self._audio_mode == "realtime_tts" and not stream_mode:
            raise ValueError("VoxCPM realtime_tts requires stream_mode=True")

        bundle = await self._bundle_loader(self._audio_mode)
        model = bundle["model"]
        sample_rate = int(bundle.get("sample_rate") or 48000)
        runtime_policy = bundle.get("runtime_policy")
        reference_wav_path = await self._resolve_reference_wav_path(voice_cfg)
        generation_kwargs = self._build_generation_kwargs(
            text=text, voice_cfg=voice_cfg, reference_wav_path=reference_wav_path
        )

        self._logger.info(
            "[%s][voxcpm] device=%s sample_rate=%s reference=%s policy=%s stream_mode=%s",
            self._audio_mode,
            bundle.get("device"),
            sample_rate,
            reference_wav_path,
            getattr(runtime_policy, "cache_key", ""),
            stream_mode,
        )

        if stream_mode:
            async for item in self._run_streaming(
                model, sample_rate=sample_rate, generation_kwargs=generation_kwargs,
                cancel_check=cancel_check,
            ):
                yield item
            return

        final_audio = await asyncio.to_thread(model.generate, **generation_kwargs)
        final_audio = normalize_audio_array(final_audio)
        if cancel_check and await cancel_check():
            yield AudioChunk(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=sample_rate,
                is_final=True,
            )
            return
        yield AudioChunk(
            pcm=final_audio,
            sample_rate=sample_rate,
            is_final=True,
        )

    # --------------- streaming ---------------

    async def _run_streaming(
        self,
        model: Any,
        *,
        sample_rate: int,
        generation_kwargs: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
    ) -> AsyncIterator[AudioChunk]:
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        result_box: Dict[str, Any] = {"error": None}
        chunks: list[np.ndarray] = []

        def _enqueue(item: np.ndarray | None) -> None:
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            fut.result()

        def _generate() -> None:
            try:
                if not hasattr(model, "generate_streaming"):
                    raise RuntimeError("Current voxcpm package does not provide generate_streaming()")
                for chunk in model.generate_streaming(**generation_kwargs):
                    if stop_event.is_set():
                        break
                    normalized = normalize_audio_array(chunk)
                    if normalized.size == 0:
                        continue
                    chunks.append(normalized)
                    _enqueue(normalized)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc
            finally:
                _enqueue(None)

        worker = threading.Thread(target=_generate, name="audio-voxcpm-engine", daemon=True)
        worker.start()

        sequence = 0
        cancelled = False
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if cancel_check and await cancel_check():
                    stop_event.set()
                    cancelled = True
                    continue
                yield AudioChunk(
                    pcm=chunk,
                    sample_rate=sample_rate,
                    sequence=sequence,
                    is_final=False,
                )
                sequence += 1
        finally:
            stop_event.set()
            await asyncio.to_thread(worker.join)

        if result_box["error"] is not None:
            raise result_box["error"]

        if cancelled or not chunks:
            final_pcm = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
        else:
            final_pcm = np.concatenate(chunks, axis=0)
        yield AudioChunk(
            pcm=final_pcm,
            sample_rate=sample_rate,
            sequence=sequence,
            is_final=True,
        )

    # --------------- helpers ---------------

    async def _resolve_reference_wav_path(self, voice_cfg: VoiceConfig) -> Optional[str]:
        # 1) 显式 prompt_wav_path 优先（task 通道用户自带音频）；
        # 2) 按 speaker_name 命中 preset；
        # 3) custom_voice 才退回 default_speaker preset；
        # 4) 都没有就 zero-shot（返回 None）。
        # 注：当 continuation_prompt=True（chat 实时 chained prompt 注入），
        # voice_cfg.prompt_wav_path 表示"上一段合成结果，仅用作韵律 prompt"，
        # 不应抢占 reference 解析（避免连续多段后音色累积漂移）。
        chain_only = bool(getattr(voice_cfg, "continuation_prompt", False))
        if chain_only:
            # chain 模式下优先级：continuation_reference_wav_path（drama 显式 seed）
            # > speaker_name preset（chat 实时）> default_speaker。
            # 严禁读 voice_cfg.prompt_wav_path——它在 chain 模式下表示"上一段合成结果"，
            # 用作韵律 prompt，不能反向当成下一段的音色锚点。
            chain_ref = str(getattr(voice_cfg, "continuation_reference_wav_path", None) or "").strip()
            if chain_ref:
                reference = chain_ref
                source = "continuation_reference"
            else:
                reference = ""
                source = "speaker_preset"
        else:
            reference = str(voice_cfg.prompt_wav_path or "").strip()
            source = "prompt_wav_path"
        if not reference:
            if str(voice_cfg.tts_mode or "").strip().lower() == "voice_design":
                return None
            reference = self._lookup_preset_by_speaker(voice_cfg.speaker_name)
            if reference:
                source = f"speaker_preset:{str(voice_cfg.speaker_name).strip().lower()}"
            else:
                reference = self._lookup_preset_by_speaker(self._default_speaker)
                if reference:
                    source = f"default_speaker:{self._default_speaker}"

        if not reference:
            if str(voice_cfg.prompt_text or "").strip():
                raise ValueError("VoxCPM prompt_text requires prompt_wav_path")
            return None

        resolved = self._resolve_path_like(reference)
        self._logger.debug(
            "[voxcpm] resolved reference_wav: source=%s raw=%s resolved=%s",
            source,
            reference,
            resolved,
        )
        path = await download_url_to_tempfile(
            resolved,
            default_suffix=".wav",
            timeout_seconds=300.0,
            max_bytes=200 * 1024 * 1024,
        )
        return str(path)

    def _lookup_preset_by_speaker(self, speaker_name: Optional[str]) -> str:
        key = str(speaker_name or "").strip().lower()
        if not key:
            return ""
        return self._speaker_presets.get(key, "")

    @staticmethod
    def _resolve_path_like(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return text
        if text.startswith(("http://", "https://", "file://")):
            return text
        p = Path(text)
        if p.is_absolute():
            return str(p)
        return str((_INFERENCE_ROOT / p).resolve())

    def _build_generation_kwargs(
        self,
        *,
        text: str,
        voice_cfg: VoiceConfig,
        reference_wav_path: Optional[str],
    ) -> Dict[str, Any]:
        effective_text = text or ""
        if not reference_wav_path and str(voice_cfg.tts_mode or "").strip().lower() == "voice_design":
            design_prompt = str(voice_cfg.design_instruct or voice_cfg.instruct or "").strip()
            if design_prompt:
                effective_text = f"({design_prompt}){effective_text}"

        kwargs: Dict[str, Any] = {
            "text": effective_text,
            "cfg_value": self._resolve_cfg_value(voice_cfg),
            "inference_timesteps": self._resolve_inference_timesteps(voice_cfg),
        }

        if not reference_wav_path:
            return kwargs

        prompt_text = str(voice_cfg.prompt_text or "").strip()
        if prompt_text:
            chain_only = bool(getattr(voice_cfg, "continuation_prompt", False))
            chain_wav = str(voice_cfg.prompt_wav_path or "").strip() if chain_only else ""
            if chain_only and chain_wav:
                # Chained prompt（方案 D）：reference_wav 锁住音色身份（speaker preset），
                # prompt_wav 用上一段合成结果续韵律。两者**分开**，避免音色逐步漂移。
                kwargs["reference_wav_path"] = reference_wav_path
                kwargs["prompt_wav_path"] = self._resolve_path_like(chain_wav)
                kwargs["prompt_text"] = prompt_text
                return kwargs
            # VoxCPM2 "Ultimate Cloning"（task 通道经典语义）：
            # prompt_wav 与 reference_wav 复用同一份用户自带音频。
            kwargs["reference_wav_path"] = reference_wav_path
            kwargs["prompt_wav_path"] = reference_wav_path
            kwargs["prompt_text"] = prompt_text
            return kwargs

        # VoxCPM2 "Controllable Voice Cloning"：仅 reference_wav
        kwargs["reference_wav_path"] = reference_wav_path
        return kwargs

    def _resolve_cfg_value(self, voice_cfg: VoiceConfig) -> float:
        gen_cfg = voice_cfg.generation_cfg or {}
        if gen_cfg.get("cfg_value") is not None:
            return max(1.0, float(gen_cfg["cfg_value"]))

        raw = voice_cfg.guidance_scale
        if raw is None:
            return 2.0
        raw_value = float(raw)
        if raw_value <= 0 or abs(raw_value - 7.5) < 1e-6:
            return 2.0
        return max(1.0, raw_value)

    def _resolve_inference_timesteps(self, voice_cfg: VoiceConfig) -> int:
        gen_cfg = voice_cfg.generation_cfg or {}
        if gen_cfg.get("inference_timesteps") is not None:
            return max(1, int(gen_cfg["inference_timesteps"]))

        raw_steps = int(voice_cfg.num_inference_steps or 0)
        if raw_steps <= 0 or raw_steps == 30:
            return 10
        return max(1, raw_steps)


__all__ = ["VoxCPMTtsEngine"]

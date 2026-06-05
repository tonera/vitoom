"""Qwen3-TTS 纯合成引擎。

从 `audio.handlers.qwen_tts_handler` 抽出 4 种 tts_mode 的合成逻辑，
不再依赖 task / file / result_handler / service_id。
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import urllib.parse
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional
from urllib.request import urlopen

import numpy as np

from audio.engines.tts_engine import (
    AudioChunk,
    CancelCheck,
    VoiceConfig,
    normalize_audio_array,
)
from audio.runtime.qwen_tts_bridge import (
    load_tts_bundle,
    normalize_qwen_language,
    resolve_qwen_custom_speaker,
)


# 单一规则：``voice_cfg.load_name`` 显式 → 用之；否则按 ``tts_mode`` / ``instruct``
# 选默认权重并对齐 mode（task / session 两条路径共用，handler 不再做二次默认）。
DEFAULT_QWEN_VOICE_DESIGN_MODEL = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_QWEN_CUSTOM_VOICE_MODEL = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_QWEN_BASE_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"


# ``load_name`` kwarg：让 bundle_loader 在一次合成里串行加载多份权重，
# 触发 LRU=1 evict 释放上一份的 VRAM。
BundleLoader = Callable[..., Awaitable[Dict[str, Any]]]


class QwenTtsEngine:
    """Qwen3-TTS 纯合成引擎。

    职责：
      - 根据 `VoiceConfig.tts_mode` 在 custom_voice / voice_design / voice_clone /
        voice_design_then_clone 中分派
      - 以 async generator 的方式流式产出 `AudioChunk`
      - 终态 chunk（is_final=True）承载**全量高质量音频**，供上层落盘；中间
        chunk 仅供实时播放/流式转发
      - 尊重 `cancel_check`：每个 chunk 产出前轮询一次，命中则尽快退出
    """

    def __init__(
        self,
        *,
        bundle_loader: BundleLoader,
        logger: logging.Logger,
    ):
        self._bundle_loader = bundle_loader
        self._logger = logger
        self._base_bundle_cache: Dict[tuple, Dict[str, Any]] = {}

    # --------------- entry point ---------------

    async def synthesize_stream(
        self,
        *,
        text: str,
        voice_cfg: VoiceConfig,
        cancel_check: Optional[CancelCheck] = None,
        stream_mode: bool = True,
    ) -> AsyncIterator[AudioChunk]:
        """产出 `AudioChunk` 序列，最后一条 `is_final=True` 为全量音频。

        `stream_mode` 仅用于驱动底层 `generate_*_async(non_streaming_mode=...)`
        的性能提示；输出协议（chunk + final）与 `stream_mode` 无关。
        """
        weight_name, tts_mode = self._resolve_weight_and_mode(voice_cfg)
        bundle = await self._bundle_loader("tts", load_name=weight_name)
        caps = bundle.get("capabilities") or {}

        if tts_mode == "custom_voice":
            self._assert_capability(caps, "custom_voice", tts_mode, bundle)
            iterator = self._stream_custom_voice(text, voice_cfg, bundle, cancel_check, stream_mode)
        elif tts_mode == "voice_design":
            self._assert_capability(caps, "voice_design", tts_mode, bundle)
            iterator = self._stream_voice_design(text, voice_cfg, bundle, cancel_check, stream_mode)
        elif tts_mode == "voice_clone":
            iterator = self._stream_voice_clone(text, voice_cfg, bundle, cancel_check, stream_mode)
        elif tts_mode == "voice_design_then_clone":
            self._assert_capability(caps, "voice_design", tts_mode, bundle)
            iterator = self._stream_voice_design_then_clone(
                text, voice_cfg, bundle, cancel_check, stream_mode
            )
        else:
            raise ValueError(
                f"Unsupported tts_mode={tts_mode}; expected one of "
                "custom_voice/voice_design/voice_clone/voice_design_then_clone"
            )

        async for chunk in iterator:
            yield chunk

    # --------------- per-mode streams ---------------

    async def _stream_custom_voice(
        self,
        text: str,
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
        stream_mode: bool,
    ) -> AsyncIterator[AudioChunk]:
        model = bundle["model"]
        dialogue_lines = self._normalize_custom_voice_dialogues(voice_cfg, bundle)
        if dialogue_lines:
            async for item in self._stream_custom_voice_dialogues(
                dialogue_lines, voice_cfg, bundle, cancel_check, stream_mode
            ):
                yield item
            return

        speaker = resolve_qwen_custom_speaker(
            voice_cfg.speaker_name or voice_cfg.voice_preset,
            supported_speakers=bundle.get("supported_speakers"),
        )
        language = normalize_qwen_language(
            voice_cfg.language, supported_languages=bundle.get("supported_languages")
        )
        instruct = self._resolve_instruct(voice_cfg)
        self._logger.info(
            "[qwen-tts][custom_voice] speaker=%s language=%s instruct=%s stream_mode=%s",
            speaker,
            language,
            bool(instruct),
            stream_mode,
        )

        if self._is_streaming_variant(bundle):
            async def _raw() -> AsyncIterator[Any]:
                await self._ensure_streaming_runtime(bundle)
                async for chunk in model.generate_custom_voice_async(
                    text=text or "",
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    non_streaming_mode=(not stream_mode),
                ):
                    yield chunk

            async for item in self._consume_streaming_generator(
                voice_cfg, bundle, _raw(), cancel_check
            ):
                yield item
            return

        kwargs = self._build_common_generation_kwargs(voice_cfg)
        kwargs.update(
            text=text or "",
            language=language,
            speaker=speaker,
            non_streaming_mode=(not stream_mode),
        )
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = await asyncio.to_thread(model.generate_custom_voice, **kwargs)
        audio = self._extract_first_audio(wavs)
        sr = int(sr or bundle.get("sample_rate") or 24000)
        async for item in self._chunk_full_audio(audio, sr, voice_cfg, cancel_check):
            yield item

    async def _stream_custom_voice_dialogues(
        self,
        dialogue_lines: list[Dict[str, str]],
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
        stream_mode: bool,
    ) -> AsyncIterator[AudioChunk]:
        model = bundle["model"]
        sample_rate = int(bundle.get("sample_rate") or 24000)
        self._logger.info(
            "[qwen-tts][custom_voice][dialogue] turns=%s stream_mode=%s",
            len(dialogue_lines),
            stream_mode,
        )

        if self._is_streaming_variant(bundle):
            audios: list[np.ndarray] = []
            for line in dialogue_lines:
                async def _raw(line=line) -> AsyncIterator[Any]:
                    await self._ensure_streaming_runtime(bundle)
                    async for chunk in model.generate_custom_voice_async(
                        text=line["text"],
                        language=line["language"],
                        speaker=line["speaker"],
                        instruct=line["instruct"],
                        non_streaming_mode=True,
                    ):
                        yield chunk

                audio, sr = await self._drain_streaming_for_final_audio(
                    bundle, _raw(), cancel_check
                )
                sample_rate = int(sr or sample_rate)
                if cancel_check and await cancel_check():
                    break
                audios.append(audio)
            final_audio = self._concat_audio_outputs(audios, sample_rate)
            async for item in self._chunk_full_audio(final_audio, sample_rate, voice_cfg, cancel_check):
                yield item
            return

        kwargs = self._build_common_generation_kwargs(voice_cfg)
        kwargs.update(
            text=[line["text"] for line in dialogue_lines],
            language=[line["language"] for line in dialogue_lines],
            speaker=[line["speaker"] for line in dialogue_lines],
            instruct=[line["instruct"] for line in dialogue_lines],
            non_streaming_mode=(not stream_mode),
        )
        wavs, sr = await asyncio.to_thread(model.generate_custom_voice, **kwargs)
        sample_rate = int(sr or sample_rate)
        audio = self._concat_audio_outputs(wavs, sample_rate)
        async for item in self._chunk_full_audio(audio, sample_rate, voice_cfg, cancel_check):
            yield item

    async def _stream_voice_design(
        self,
        text: str,
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
        stream_mode: bool,
    ) -> AsyncIterator[AudioChunk]:
        model = bundle["model"]
        dialogue_lines = self._normalize_voice_design_dialogues(voice_cfg, bundle)
        if dialogue_lines:
            async for item in self._stream_voice_design_dialogues(
                dialogue_lines, voice_cfg, bundle, cancel_check, stream_mode
            ):
                yield item
            return

        language = normalize_qwen_language(
            voice_cfg.language, supported_languages=bundle.get("supported_languages")
        )
        instruct = self._resolve_instruct(voice_cfg)
        if not instruct:
            raise ValueError("voice_design mode requires 'instruct' (声线/风格的自然语言描述)")
        self._logger.info(
            "[qwen-tts][voice_design] language=%s instruct_len=%s stream_mode=%s",
            language,
            len(instruct),
            stream_mode,
        )

        if self._is_streaming_variant(bundle):
            async def _raw() -> AsyncIterator[Any]:
                await self._ensure_streaming_runtime(bundle)
                async for chunk in model.generate_voice_design_async(
                    text=text or "",
                    language=language,
                    instruct=instruct,
                    non_streaming_mode=(not stream_mode),
                ):
                    yield chunk

            async for item in self._consume_streaming_generator(
                voice_cfg, bundle, _raw(), cancel_check
            ):
                yield item
            return

        kwargs = self._build_common_generation_kwargs(voice_cfg)
        kwargs.update(text=text or "", language=language, instruct=instruct)
        wavs, sr = await asyncio.to_thread(model.generate_voice_design, **kwargs)
        sr = int(sr or bundle.get("sample_rate") or 24000)
        audio = self._concat_audio_outputs(wavs, sr)
        async for item in self._chunk_full_audio(audio, sr, voice_cfg, cancel_check):
            yield item

    async def _stream_voice_design_dialogues(
        self,
        dialogue_lines: list[Dict[str, str]],
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
        stream_mode: bool,
    ) -> AsyncIterator[AudioChunk]:
        model = bundle["model"]
        sample_rate = int(bundle.get("sample_rate") or 24000)
        self._logger.info(
            "[qwen-tts][voice_design][dialogue] turns=%s stream_mode=%s",
            len(dialogue_lines),
            stream_mode,
        )

        if self._is_streaming_variant(bundle):
            audios: list[np.ndarray] = []
            for line in dialogue_lines:
                async def _raw(line=line) -> AsyncIterator[Any]:
                    await self._ensure_streaming_runtime(bundle)
                    async for chunk in model.generate_voice_design_async(
                        text=line["text"],
                        language=line["language"],
                        instruct=line["instruct"],
                        non_streaming_mode=True,
                    ):
                        yield chunk

                audio, sr = await self._drain_streaming_for_final_audio(
                    bundle, _raw(), cancel_check
                )
                sample_rate = int(sr or sample_rate)
                if cancel_check and await cancel_check():
                    break
                audios.append(audio)
            final_audio = self._concat_audio_outputs(audios, sample_rate)
            async for item in self._chunk_full_audio(final_audio, sample_rate, voice_cfg, cancel_check):
                yield item
            return

        kwargs = self._build_common_generation_kwargs(voice_cfg)
        kwargs.update(
            text=[line["text"] for line in dialogue_lines],
            language=[line["language"] for line in dialogue_lines],
            instruct=[line["instruct"] for line in dialogue_lines],
        )
        wavs, sr = await asyncio.to_thread(model.generate_voice_design, **kwargs)
        sample_rate = int(sr or sample_rate)
        audio = self._concat_audio_outputs(wavs, sample_rate)
        async for item in self._chunk_full_audio(audio, sample_rate, voice_cfg, cancel_check):
            yield item

    async def _stream_voice_clone(
        self,
        text: str,
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
        stream_mode: bool,
    ) -> AsyncIterator[AudioChunk]:
        base_bundle = await self._ensure_base_bundle(bundle, voice_cfg)
        model = base_bundle["model"]
        language = normalize_qwen_language(
            voice_cfg.language, supported_languages=base_bundle.get("supported_languages")
        )

        ref_audio_raw = str(voice_cfg.ref_audio or "").strip()
        if not ref_audio_raw:
            raise ValueError("voice_clone mode requires 'ref_audio' (URL or local path)")
        ref_text_raw = str(voice_cfg.ref_text or "").strip()
        x_vector_only = bool(voice_cfg.x_vector_only)
        if not ref_text_raw and not x_vector_only:
            raise ValueError(
                "voice_clone mode requires 'ref_text' unless x_vector_only=True"
            )

        ref_audio_path, cleanup_path = await self._materialize_ref_audio(ref_audio_raw)
        try:
            self._logger.info(
                "[qwen-tts][voice_clone] language=%s ref_audio=%s ref_text_len=%s "
                "x_vector_only=%s stream_mode=%s",
                language,
                ref_audio_path,
                len(ref_text_raw),
                x_vector_only,
                stream_mode,
            )
            if self._is_streaming_variant(base_bundle):
                async def _raw() -> AsyncIterator[Any]:
                    await self._ensure_streaming_runtime(base_bundle)
                    async for chunk in model.generate_voice_clone_async(
                        text=text or "",
                        language=language,
                        ref_audio=ref_audio_path,
                        ref_text=(ref_text_raw or None),
                        x_vector_only_mode=x_vector_only,
                        non_streaming_mode=(not stream_mode),
                    ):
                        yield chunk

                async for item in self._consume_streaming_generator(
                    voice_cfg, base_bundle, _raw(), cancel_check
                ):
                    yield item
                return

            kwargs = self._build_common_generation_kwargs(voice_cfg)
            kwargs.update(
                text=text or "",
                language=language,
                ref_audio=ref_audio_path,
            )
            if ref_text_raw:
                kwargs["ref_text"] = ref_text_raw
            if x_vector_only:
                kwargs["x_vector_only_mode"] = True
            wavs, sr = await asyncio.to_thread(model.generate_voice_clone, **kwargs)
            audio = self._extract_first_audio(wavs)
            sr = int(sr or base_bundle.get("sample_rate") or 24000)
            async for item in self._chunk_full_audio(audio, sr, voice_cfg, cancel_check):
                yield item
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                try:
                    os.unlink(cleanup_path)
                except Exception:
                    pass

    async def _stream_voice_design_then_clone(
        self,
        text: str,
        voice_cfg: VoiceConfig,
        design_bundle: Dict[str, Any],
        cancel_check: Optional[CancelCheck],
        stream_mode: bool,
    ) -> AsyncIterator[AudioChunk]:
        design_model = design_bundle["model"]
        design_language = normalize_qwen_language(
            voice_cfg.language, supported_languages=design_bundle.get("supported_languages")
        )
        design_seed_text = str(voice_cfg.design_seed_text or "").strip()
        if not design_seed_text:
            raise ValueError(
                "voice_design_then_clone mode requires 'design_seed_text' "
                "(VoiceDesign 阶段要念的那句原文，会成为 Base 克隆的 ref_text)"
            )
        design_instruct = str(
            voice_cfg.design_instruct or self._resolve_instruct(voice_cfg) or ""
        ).strip()
        if not design_instruct:
            raise ValueError(
                "voice_design_then_clone mode requires 'design_instruct' (也可回退到 instruct)"
            )

        self._logger.info(
            "[qwen-tts][dtc][1/2:design] language=%s seed_text_len=%s instruct_len=%s",
            design_language,
            len(design_seed_text),
            len(design_instruct),
        )

        if self._is_streaming_variant(design_bundle):
            async def _design_raw() -> AsyncIterator[Any]:
                await self._ensure_streaming_runtime(design_bundle)
                async for chunk in design_model.generate_voice_design_async(
                    text=design_seed_text,
                    language=design_language,
                    instruct=design_instruct,
                    non_streaming_mode=True,
                ):
                    yield chunk

            ref_audio_array, design_sr = await self._drain_streaming_for_final_audio(
                design_bundle, _design_raw(), cancel_check
            )
        else:
            design_wavs, design_sr = await asyncio.to_thread(
                design_model.generate_voice_design,
                text=design_seed_text,
                language=design_language,
                instruct=design_instruct,
            )
            ref_audio_array = self._extract_first_audio(design_wavs)
            design_sr = int(design_sr or design_bundle.get("sample_rate") or 24000)

        if cancel_check and await cancel_check():
            sr = int(design_sr or design_bundle.get("sample_rate") or 24000)
            yield AudioChunk(pcm=np.zeros(0, dtype=np.float32), sample_rate=sr, is_final=True)
            return

        base_bundle = await self._ensure_base_bundle(design_bundle, voice_cfg)
        base_model = base_bundle["model"]
        base_language = normalize_qwen_language(
            voice_cfg.language, supported_languages=base_bundle.get("supported_languages")
        )
        x_vector_only = bool(voice_cfg.x_vector_only)

        create_prompt = getattr(base_model, "create_voice_clone_prompt", None)
        if create_prompt is None:
            raise ValueError(
                "voice_design_then_clone requires Base weight exposing create_voice_clone_prompt()"
            )

        prompt_items = await asyncio.to_thread(
            create_prompt,
            ref_audio=(ref_audio_array, design_sr),
            ref_text=design_seed_text,
            x_vector_only_mode=x_vector_only,
        )

        self._logger.info(
            "[qwen-tts][dtc][2/2:clone] language=%s x_vector_only=%s stream_mode=%s",
            base_language,
            x_vector_only,
            stream_mode,
        )

        if self._is_streaming_variant(base_bundle):
            async def _raw() -> AsyncIterator[Any]:
                await self._ensure_streaming_runtime(base_bundle)
                async for chunk in base_model.generate_voice_clone_async(
                    text=text or "",
                    language=base_language,
                    voice_clone_prompt=prompt_items,
                    x_vector_only_mode=x_vector_only,
                    non_streaming_mode=(not stream_mode),
                ):
                    yield chunk

            async for item in self._consume_streaming_generator(
                voice_cfg, base_bundle, _raw(), cancel_check
            ):
                yield item
            return

        kwargs = self._build_common_generation_kwargs(voice_cfg)
        kwargs.update(
            text=text or "",
            language=base_language,
            voice_clone_prompt=prompt_items,
        )
        wavs, sr = await asyncio.to_thread(base_model.generate_voice_clone, **kwargs)
        audio = self._extract_first_audio(wavs)
        sr = int(sr or base_bundle.get("sample_rate") or 24000)
        async for item in self._chunk_full_audio(audio, sr, voice_cfg, cancel_check):
            yield item

    # --------------- streaming helpers ---------------

    async def _consume_streaming_generator(
        self,
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
        generator: AsyncIterator[Any],
        cancel_check: Optional[CancelCheck],
    ) -> AsyncIterator[AudioChunk]:
        """消费 nano-vllm 风格的 codec chunk 流：等齐 codec → 一次性 decode → 切片喂出。

        关键约束（与 voxcpm engine 行为一致）：**所有中间 chunks 拼接 == final chunk**。
        Qwen3-TTS 的 ``speech_tokenizer.decode`` 走 flow-matching 类 vocoder：
          - 不是 state-preserving 流式（无法增量喂 codec、续算 PCM）；
          - 含 sampling 噪声（同一段 codec 多次 decode 输出不会逐样本相等）。
        任何"边收 codec 边独立 decode 出小段"或"累积 redecode + emit growth tail"的
        方案都会在段边界产生不连续，前端按 PCM 时间线无缝拼接就会听到周期性"破音"。

        因此放弃"边算边解"的伪流式：
          1. 收齐全部 codec frame；
          2. 一次性 ``decode(collected_codes)`` 得到 **唯一稳定** PCM；
          3. 走 ``_chunk_full_audio`` 按 ``stream_chunk_seconds`` 切片喂出中间 chunks
             和 final chunk（与 transformers 后端的非流式分支完全等价）。

        代价：失去"边算边播"的真流式，TTFB ≈ 整段 codec 生成时间 + 一次 decode。
        优势：所有 chunks 都是同一份 PCM 的字面切片，**前端按时间线拼接绝对连续**。
        若 vendor 后续暴露 state-preserving streaming vocoder，可以再引入真流式路径。
        """
        sample_rate = int(bundle.get("sample_rate") or 24000)
        collected_codes: list[list[int]] = []
        cancelled = False

        try:
            async for raw_chunk in generator:
                if cancel_check and await cancel_check():
                    cancelled = True
                    break
                codec_chunks = self._normalize_audio_codes_chunk(raw_chunk)
                if codec_chunks:
                    collected_codes.extend(codec_chunks)
        except BaseException:
            cancelled = True
            raise

        if cancelled or not collected_codes:
            yield AudioChunk(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=sample_rate,
                sequence=0,
                is_final=True,
            )
            return

        final_audio, final_sr = await asyncio.to_thread(
            self._decode_audio_codes, bundle, collected_codes
        )
        sample_rate = int(final_sr or sample_rate)
        async for item in self._chunk_full_audio(
            final_audio, sample_rate, voice_cfg, cancel_check
        ):
            yield item

    async def _drain_streaming_for_final_audio(
        self,
        bundle: Dict[str, Any],
        generator: AsyncIterator[Any],
        cancel_check: Optional[CancelCheck],
    ) -> tuple[np.ndarray, int]:
        """仅保留整段全量音频，忽略中间 chunk；用于 DtC 的第 1 阶段。"""
        sample_rate = int(bundle.get("sample_rate") or 24000)
        collected_codes: list[list[int]] = []
        async for raw_chunk in generator:
            if cancel_check and await cancel_check():
                break
            codec_chunks = self._normalize_audio_codes_chunk(raw_chunk)
            if codec_chunks:
                collected_codes.extend(codec_chunks)
        if not collected_codes:
            return np.zeros((0,), dtype=np.float32), sample_rate
        audio, sr = await asyncio.to_thread(self._decode_audio_codes, bundle, collected_codes)
        return audio, int(sr or sample_rate)

    async def _chunk_full_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        voice_cfg: VoiceConfig,
        cancel_check: Optional[CancelCheck],
    ) -> AsyncIterator[AudioChunk]:
        """把整段非流式合成结果按 `stream_chunk_seconds` 切片，尾部再给一条全量 final。"""
        sequence = 0
        if audio.size > 0:
            chunk_samples = self._resolve_stream_chunk_samples(voice_cfg, sample_rate)
            for start in range(0, audio.shape[0], chunk_samples):
                if cancel_check and await cancel_check():
                    break
                piece = audio[start:start + chunk_samples]
                if piece.size == 0:
                    continue
                yield AudioChunk(
                    pcm=piece,
                    sample_rate=sample_rate,
                    sequence=sequence,
                    is_final=False,
                )
                sequence += 1
        yield AudioChunk(
            pcm=audio if audio.size > 0 else np.zeros(0, dtype=np.float32),
            sample_rate=sample_rate,
            sequence=sequence,
            is_final=True,
        )

    # --------------- bundle / ref helpers ---------------

    def _assert_capability(
        self,
        caps: Dict[str, Any],
        capability: str,
        tts_mode: str,
        bundle: Dict[str, Any],
    ) -> None:
        if caps.get(capability):
            return
        raise ValueError(
            f"Qwen3-TTS weight at {bundle.get('model_ref')!r} does not expose "
            f"'generate_{capability}' and cannot serve tts_mode={tts_mode}. "
            "Please point load_name to the correct weight "
            "(CustomVoice / VoiceDesign / Base)."
        )

    async def _ensure_base_bundle(
        self,
        current_bundle: Dict[str, Any],
        voice_cfg: VoiceConfig,
    ) -> Dict[str, Any]:
        caps = current_bundle.get("capabilities") or {}
        if caps.get("voice_clone") and caps.get("create_voice_clone_prompt"):
            return current_bundle

        explicit_model = str(voice_cfg.clone_base_load_name or "").strip()
        policy = current_bundle.get("runtime_policy")
        if policy is None:
            raise RuntimeError(
                "current qwen-tts bundle has no runtime_policy; cannot load Base weight"
            )
        if not explicit_model:
            raise ValueError(
                "voice_clone / voice_design_then_clone requires 'clone_base_load_name' "
                "when load_name is not a Base weight."
            )

        base_ref = await self._resolve_sibling_weight_ref(current_bundle, explicit_model)
        runtime_backend = str(current_bundle.get("runtime_backend") or "transformers").strip().lower()
        cache_key = (base_ref, policy.cache_key if policy else "", runtime_backend)
        cached = self._base_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        self._logger.info(
            "[qwen-tts] loading Base clone weight model_ref=%s (companion to %s)",
            base_ref,
            current_bundle.get("model_ref"),
        )
        runtime_cfg = current_bundle.get("runtime_config")
        base_bundle = await asyncio.to_thread(
            load_tts_bundle,
            base_ref,
            policy,
            runtime_backend,
            dict(runtime_cfg or {}),
        )
        caps = base_bundle.get("capabilities") or {}
        if not (caps["voice_clone"] and caps["create_voice_clone_prompt"]):
            raise ValueError(
                f"Companion weight {base_ref!r} does not expose voice-clone API; "
                "please ensure clone_base_load_name points to Qwen3-TTS-12Hz-*-Base."
            )
        self._base_bundle_cache[cache_key] = base_bundle
        return base_bundle

    async def _resolve_sibling_weight_ref(
        self,
        current_bundle: Dict[str, Any],
        sibling_name: str,
    ) -> str:
        name = sibling_name.strip()
        if os.path.isabs(name):
            return name
        if "/" in name:
            return name
        current_ref = str(current_bundle.get("model_ref") or "").strip()
        if current_ref and os.path.isabs(current_ref):
            parent = os.path.dirname(current_ref.rstrip("/\\"))
            if parent:
                candidate = os.path.join(parent, name)
                return candidate
        return name

    async def _materialize_ref_audio(self, ref_audio: str) -> tuple[str, Optional[str]]:
        if os.path.exists(ref_audio):
            return ref_audio, None
        parsed = urllib.parse.urlparse(ref_audio)
        if parsed.scheme in ("http", "https"):
            suffix = os.path.splitext(parsed.path)[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(prefix="qwen_tts_ref_", suffix=suffix, delete=False)
            tmp_path = tmp.name
            tmp.close()
            self._logger.info("[qwen-tts] downloading ref_audio=%s -> %s", ref_audio, tmp_path)
            try:
                await asyncio.to_thread(_download_to_file, ref_audio, tmp_path)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                raise
            return tmp_path, tmp_path
        raise FileNotFoundError(f"ref_audio not found and not an http(s) URL: {ref_audio!r}")

    # --------------- generation / codec helpers ---------------

    def _build_common_generation_kwargs(self, voice_cfg: VoiceConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        gen_cfg = voice_cfg.generation_cfg or {}
        for key in (
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "do_sample",
            "repetition_penalty",
        ):
            if gen_cfg.get(key) is not None:
                kwargs[key] = gen_cfg[key]
        return kwargs

    def _normalize_custom_voice_dialogues(
        self,
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
    ) -> list[Dict[str, str]]:
        lines: list[Dict[str, str]] = []
        for raw in self._normalize_drama_dialogues(voice_cfg):
            text = str(raw.get("text") or "").strip()
            speaker = resolve_qwen_custom_speaker(
                raw.get("speaker_name") or voice_cfg.speaker_name or voice_cfg.voice_preset,
                supported_speakers=bundle.get("supported_speakers"),
            )
            language = normalize_qwen_language(
                raw.get("language") or voice_cfg.language,
                supported_languages=bundle.get("supported_languages"),
            )
            lines.append({
                "text": text,
                "speaker": speaker,
                "language": language,
                "instruct": str(raw.get("instruct") or "").strip(),
            })
        return lines

    def _normalize_voice_design_dialogues(
        self,
        voice_cfg: VoiceConfig,
        bundle: Dict[str, Any],
    ) -> list[Dict[str, str]]:
        lines: list[Dict[str, str]] = []
        for raw in self._normalize_drama_dialogues(voice_cfg):
            text = str(raw.get("text") or "").strip()
            instruct = str(raw.get("instruct") or "").strip()
            if not instruct:
                raise ValueError("voice_design dialogue requires 'instruct' for every non-empty line")
            language = normalize_qwen_language(
                raw.get("language") or voice_cfg.language,
                supported_languages=bundle.get("supported_languages"),
            )
            lines.append({
                "text": text,
                "language": language,
                "instruct": instruct,
            })
        return lines

    def _normalize_drama_dialogues(self, voice_cfg: VoiceConfig) -> list[Dict[str, str]]:
        drama = voice_cfg.drama or {}
        if not isinstance(drama, dict):
            return []

        raw_characters = drama.get("characters")
        raw_dialogues = drama.get("dialogues")
        if not isinstance(raw_characters, list) or not isinstance(raw_dialogues, list):
            return []

        characters: Dict[str, Dict[str, Any]] = {}
        for raw in raw_characters:
            if not isinstance(raw, dict):
                continue
            character_id = str(raw.get("id") or "").strip()
            if character_id:
                characters[character_id] = raw

        lines: list[Dict[str, str]] = []
        for raw in raw_dialogues:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            speaker_id = str(raw.get("speaker_id") or "").strip()
            character = characters.get(speaker_id, {})
            lines.append({
                "text": text,
                "speaker": str(character.get("name") or speaker_id or "").strip(),
                "speaker_name": str(character.get("speaker_name") or "").strip(),
                "language": str(raw.get("language") or character.get("language") or voice_cfg.language or "").strip(),
                "instruct": str(raw.get("instruct") or character.get("instruct") or "").strip(),
            })
        return lines

    def _concat_audio_outputs(
        self,
        wavs: Any,
        sample_rate: int,
        *,
        pause_seconds: float = 0.2,
    ) -> np.ndarray:
        if isinstance(wavs, (list, tuple)):
            audios = [normalize_audio_array(item) for item in wavs]
        else:
            raw = wavs.detach().cpu().float().numpy() if hasattr(wavs, "detach") else np.asarray(wavs)
            if raw.ndim == 2 and raw.shape[0] > 1:
                audios = [normalize_audio_array(item) for item in raw]
            else:
                audios = [normalize_audio_array(raw)]

        audios = [audio for audio in audios if audio.size > 0]
        if not audios:
            raise RuntimeError("qwen-tts generation returned no audio output")
        if len(audios) == 1:
            return audios[0]

        pause_samples = max(int(sample_rate * pause_seconds), 0)
        pause = np.zeros((pause_samples,), dtype=np.float32) if pause_samples else None
        segments: list[np.ndarray] = []
        for audio in audios:
            if segments and pause is not None:
                segments.append(pause)
            segments.append(audio)
        concatenated = np.concatenate(segments, axis=0).astype(np.float32, copy=False)
        return np.ascontiguousarray(concatenated)

    def _resolve_stream_chunk_samples(self, voice_cfg: VoiceConfig, sample_rate: int) -> int:
        gen_cfg = voice_cfg.generation_cfg or {}
        raw_seconds = gen_cfg.get("stream_chunk_seconds")
        if raw_seconds is None:
            return max(sample_rate // 2, 1)
        try:
            seconds = float(raw_seconds)
        except Exception:
            return max(sample_rate // 2, 1)
        return max(int(sample_rate * seconds), 1)

    def _resolve_instruct(self, voice_cfg: VoiceConfig) -> Optional[str]:
        for candidate in (voice_cfg.instruct, voice_cfg.prompt_text):
            text = str(candidate or "").strip()
            if text:
                return text
        return None

    def _resolve_weight_and_mode(self, voice_cfg: VoiceConfig) -> tuple[str, str]:
        """决定本次合成的 (weight_name, effective_tts_mode)。

        ``voice_cfg.load_name`` 显式 → 原样用，``tts_mode`` 也尊重原值；空 → 按 mode
        选默认权重；mode 也是 default('custom_voice') + 有 ``instruct`` 时整体升档为
        voice_design——避免 instruct 被降格成 CustomVoice 的 tone tweak。
        """
        explicit = str(voice_cfg.load_name or "").strip()
        mode = (voice_cfg.tts_mode or "custom_voice").strip().lower()
        if explicit:
            return explicit, mode
        if mode in ("voice_design", "voice_design_then_clone"):
            return DEFAULT_QWEN_VOICE_DESIGN_MODEL, mode
        if mode == "voice_clone":
            return DEFAULT_QWEN_BASE_MODEL, mode
        if str(voice_cfg.instruct or "").strip():
            return DEFAULT_QWEN_VOICE_DESIGN_MODEL, "voice_design"
        return DEFAULT_QWEN_CUSTOM_VOICE_MODEL, "custom_voice"

    def _extract_first_audio(self, wavs: Any) -> np.ndarray:
        if isinstance(wavs, np.ndarray):
            return normalize_audio_array(wavs)
        if isinstance(wavs, (list, tuple)):
            for item in wavs:
                audio = normalize_audio_array(item)
                if audio.size > 0:
                    return audio
        raise RuntimeError("qwen-tts generation returned no audio output")

    def _is_streaming_variant(self, bundle: Dict[str, Any]) -> bool:
        return bool(bundle.get("streaming_variant"))

    async def _ensure_streaming_runtime(self, bundle: Dict[str, Any]) -> None:
        if not self._is_streaming_variant(bundle):
            return
        model = bundle["model"]
        start_fn = getattr(model, "start_zmq_tasks", None)
        if not callable(start_fn):
            raise RuntimeError("nano qwen3tts bundle is missing start_zmq_tasks()")
        await start_fn()

    def _normalize_audio_codes_chunk(self, chunk: Any) -> list[list[int]]:
        if chunk is None:
            return []
        if hasattr(chunk, "tolist"):
            chunk = chunk.tolist()
        if not isinstance(chunk, (list, tuple)):
            raise TypeError(f"Unsupported audio code chunk type: {type(chunk)!r}")
        if not chunk:
            return []
        first = chunk[0]
        if isinstance(first, (list, tuple)):
            return [list(map(int, item)) for item in chunk]
        return [list(map(int, chunk))]

    def _decode_audio_codes(
        self, bundle: Dict[str, Any], codes: list[list[int]]
    ) -> tuple[np.ndarray, int]:
        if not codes:
            return np.zeros((0,), dtype=np.float32), int(bundle.get("sample_rate") or 24000)
        speech_tokenizer = getattr(bundle.get("model"), "speech_tokenizer", None)
        if speech_tokenizer is None:
            raise RuntimeError("nano qwen3tts runtime has no speech_tokenizer; cannot decode audio codes")
        wavs, sr = speech_tokenizer.decode([{"audio_codes": codes}])
        return self._extract_first_audio(wavs), int(sr or bundle.get("sample_rate") or 24000)


def _download_to_file(url: str, dest_path: str) -> None:
    with urlopen(url, timeout=60) as resp, open(dest_path, "wb") as fp:
        fp.write(resp.read())


__all__ = ["QwenTtsEngine"]

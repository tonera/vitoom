"""VoxCPM task 通道薄壳 handler。

职责：
  - 承接 task 通道的 `InferenceRequestParams`
  - 委托 [`VoxCPMTtsEngine`](../engines/voxcpm_tts_engine.py) 做纯合成
  - 维持 task 通道的外部行为：`audio_stream_*` 流事件 + 可选 `result_handler.process_single_result` 落盘
"""

from __future__ import annotations

import asyncio
import os
import base64
import io
import tempfile
import time
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from common.logger import get_logger
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

from audio.engines.tts_engine import VoiceConfig, voice_config_from_params
from audio.engines.voxcpm_tts_engine import VoxCPMTtsEngine
from audio.handlers.progress_estimator import (
    estimate_tts_generation_seconds,
    tick_estimated_progress,
)
from audio.runtime.audio_wav_utils import audio_tensor_to_pcm16_bytes

logger = get_logger(__name__)


class VoxCPMTtsHandler:
    """VoxCPM task 通道 handler（薄壳）。"""

    def __init__(
        self,
        *,
        audio_mode: str,
        result_handler: ResultHandler,
        service_id: str,
        logger=logger,
        bundle_loader: Callable[[str], Awaitable[Dict[str, Any]]],
        stream_sender: Callable[[Dict[str, Any]], Awaitable[bool]],
        check_cancelled: Optional[Callable[[str], Awaitable[bool]]] = None,
        status_sender: Optional[Callable[..., Awaitable[None]]] = None,
        speaker_presets: Optional[Mapping[str, str]] = None,
        default_speaker: Optional[str] = None,
    ):
        self.audio_mode = audio_mode
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.stream_sender = stream_sender
        self.check_cancelled = check_cancelled
        self.status_sender = status_sender
        self._engine = VoxCPMTtsEngine(
            audio_mode=audio_mode,
            bundle_loader=bundle_loader,
            logger=logger,
            speaker_presets=speaker_presets,
            default_speaker=default_speaker,
        )

    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        if params.type != "audio":
            raise ValueError(f"VoxCPMTtsHandler expected req.type='audio', got '{params.type}'")

        stream_enabled = bool(params.stream) or self.audio_mode == "realtime_tts"
        if self.audio_mode == "realtime_tts" and not bool(params.stream):
            raise ValueError("VoxCPM realtime_tts requires stream=true")

        voice_cfg = voice_config_from_params(params)
        started_at = time.time()
        await self._send_progress(task_id, 5, "准备音频生成")

        drama = getattr(params, "drama", None)
        if isinstance(drama, dict) and drama.get("characters") and drama.get("dialogues"):
            await self._run_drama(params=params, task_id=task_id, voice_cfg=voice_cfg, started_at=started_at)
            return

        collected: list[np.ndarray] = []
        final_audio: np.ndarray | None = None
        final_sr: int | None = None
        sequence = 0
        started = False

        self.logger.info(
            "[%s][voxcpm] task_id=%s stream=%s",
            self.audio_mode,
            task_id,
            stream_enabled,
        )

        progress_ticker: Optional[asyncio.Task[None]] = None
        progress_stop_event: Optional[asyncio.Event] = None
        try:
            progress_cursor = 15
            await self._send_progress(task_id, progress_cursor, "开始合成音频")
            estimated_seconds: Optional[float] = None
            if not stream_enabled:
                estimated_seconds = await estimate_tts_generation_seconds(
                    engine=self._engine,
                    task_id=task_id,
                    text=params.prompt or "",
                    voice_cfg=voice_cfg,
                    send_progress=self._send_progress,
                    cancel_check=self._cancel_check,
                    logger=self.logger,
                    log_prefix="voxcpm",
                )
                if estimated_seconds is not None:
                    progress_cursor = 20
                    await self._send_progress(task_id, progress_cursor, "测速完成，开始合成音频")
                    progress_stop_event = asyncio.Event()
                    progress_ticker = asyncio.create_task(
                        tick_estimated_progress(
                            task_id=task_id,
                            send_progress=self._send_progress,
                            start=progress_cursor,
                            end=88,
                            estimated_seconds=estimated_seconds,
                            message="正在合成音频",
                            stop_event=progress_stop_event,
                        )
                    )
            async for chunk in self._engine.synthesize_stream(
                text=params.prompt or "",
                voice_cfg=voice_cfg,
                cancel_check=self._cancel_check,
                stream_mode=stream_enabled,
            ):
                if stream_enabled and not started:
                    await self._emit_event(
                        task_id=task_id,
                        event_type="audio_stream_start",
                        sequence=sequence,
                        sample_rate=chunk.sample_rate,
                        is_final=False,
                    )
                    sequence += 1
                    started = True

                if chunk.is_final:
                    final_audio = chunk.pcm
                    final_sr = chunk.sample_rate
                    continue

                collected.append(chunk.pcm)
                if stream_enabled:
                    next_progress = min(90, progress_cursor + 3)
                    if next_progress > progress_cursor:
                        progress_cursor = next_progress
                        await self._send_progress(task_id, progress_cursor, "正在合成音频")
                if stream_enabled and chunk.pcm.size > 0:
                    await self._emit_event(
                        task_id=task_id,
                        event_type="audio_stream_chunk",
                        sequence=sequence,
                        sample_rate=chunk.sample_rate,
                        is_final=False,
                        data=base64.b64encode(audio_tensor_to_pcm16_bytes(chunk.pcm)).decode("ascii"),
                    )
                    sequence += 1
        finally:
            if progress_ticker:
                if progress_stop_event:
                    progress_stop_event.set()
                try:
                    await progress_ticker
                except asyncio.CancelledError:
                    pass
            if stream_enabled and started:
                await self._emit_event(
                    task_id=task_id,
                    event_type="audio_stream_end",
                    sequence=sequence,
                    sample_rate=final_sr or 48000,
                    is_final=True,
                )

        if self.check_cancelled and await self.check_cancelled(f"after {self.audio_mode} generation"):
            return

        if final_audio is None or final_audio.size == 0:
            if collected:
                final_audio = np.concatenate(collected, axis=0)
                final_sr = final_sr or 48000
        if final_audio is None or final_audio.size == 0:
            raise RuntimeError("VoxCPM streaming generation returned no audio chunk")

        should_persist_file = (
            self.audio_mode == "tts" or params.response_format in {"audio_file", "both"}
        )
        if should_persist_file:
            params.file_type = "wav"
            await self._send_progress(task_id, 95, "正在保存音频")
            await self.result_handler.process_single_result(
                file_data=_audio_bytes(final_audio, final_sr or 48000),
                request_params=params,
                generate_time=time.time() - started_at,
                service_id=self.service_id,
                index=0,
                total=1,
            )

    async def _run_drama(
        self,
        *,
        params: InferenceRequestParams,
        task_id: str,
        voice_cfg: VoiceConfig,
        started_at: float,
    ) -> None:
        drama = getattr(params, "drama", None)
        if not isinstance(drama, dict):
            raise ValueError("audio drama requires params.drama")
        characters = self._normalize_drama_characters(drama.get("characters"))
        dialogues = self._normalize_drama_dialogues(drama.get("dialogues"))
        if not characters:
            raise ValueError("audio drama requires at least one character")
        if not dialogues:
            raise ValueError("audio drama requires at least one dialogue line")

        temp_seed_paths: List[str] = []
        # 方案 D（drama 扩展）：每个 character 独立维护 chain prompt 缓存——
        # ``character_id → (上一段合成 wav 路径, 上一段文本)``。同 character 第 N 段
        # 合成时把上一段 wav + 文本注入到 voice_cfg 作为 VoxCPM2 Ultimate Cloning 的
        # 韵律 prompt，reference 仍锚定在该 character 的 seed_path（音色身份不漂移）。
        # 不同 character 间是隔离的，A 的 chain 不会污染 B。
        chain_state: Dict[str, Tuple[str, str]] = {}
        chain_temp_paths: List[str] = []
        try:
            await self._send_progress(task_id, 8, "准备角色对白")
            seed_paths: Dict[str, str] = {}
            seed_texts: Dict[str, str] = {}
            character_total = max(1, len(characters))
            for character_index, (character_id, character) in enumerate(characters.items(), start=1):
                if self.check_cancelled and await self.check_cancelled("before drama voice seed"):
                    return
                seed_text = self._seed_text_for_character(character)
                seed_cfg = self._voice_cfg_for_seed(voice_cfg, character)
                seed_audio, seed_sr = await self._synthesize_final_audio(
                    text=seed_text,
                    voice_cfg=seed_cfg,
                    stream_mode=False,
                )
                seed_path = self._write_temp_wav(seed_audio, seed_sr)
                temp_seed_paths.append(seed_path)
                seed_paths[character_id] = seed_path
                seed_texts[character_id] = seed_text
                self.logger.info(
                    "[voxcpm][drama] task_id=%s character=%s seed_path=%s seed_chars=%d",
                    task_id,
                    character_id,
                    seed_path,
                    len(seed_text),
                )
                seed_progress = 10 + int(round((character_index / character_total) * 35))
                await self._send_progress(task_id, seed_progress, "正在生成角色音色")

            dialogue_blocks = self._build_dialogue_blocks(dialogues)
            self.logger.info(
                "[voxcpm][drama] task_id=%s dialogue_lines=%d blocks=%d",
                task_id,
                len(dialogues),
                len(dialogue_blocks),
            )

            pieces: List[np.ndarray] = []
            output_sr: Optional[int] = None
            block_total = max(1, len(dialogue_blocks))
            for block_index, block in enumerate(dialogue_blocks, start=1):
                if self.check_cancelled and await self.check_cancelled("before drama line synthesis"):
                    return
                character_id = block["speaker_id"]
                character = characters.get(character_id)
                seed_path = seed_paths.get(character_id)
                if character is None or not seed_path:
                    raise ValueError(f"unknown drama speaker_id: {character_id}")
                block_text = self._block_text_with_style(block)
                cached = chain_state.get(character_id)
                if cached is not None:
                    cached_wav, cached_text = cached
                    # chain 模式：reference 锚 seed（音色身份），prompt 续上一段（韵律）。
                    line_cfg = replace(
                        voice_cfg,
                        tts_mode="custom_voice",
                        speaker_name=None,
                        voice_preset=None,
                        instruct=character.get("instruct"),
                        prompt_wav_path=cached_wav,
                        prompt_text=cached_text,
                        continuation_prompt=True,
                        continuation_reference_wav_path=seed_path,
                        design_seed_text=None,
                        design_instruct=None,
                    )
                else:
                    # 该 character 首次出现：保持原 Controllable Voice Cloning 语义。
                    line_cfg = replace(
                        voice_cfg,
                        tts_mode="custom_voice",
                        speaker_name=None,
                        voice_preset=None,
                        instruct=character.get("instruct"),
                        prompt_wav_path=seed_path,
                        prompt_text=None,
                        continuation_prompt=False,
                        continuation_reference_wav_path=None,
                        design_seed_text=None,
                        design_instruct=None,
                    )
                audio, sr = await self._synthesize_final_audio(
                    text=block_text,
                    voice_cfg=line_cfg,
                    stream_mode=False,
                )
                if output_sr is None:
                    output_sr = sr
                elif sr != output_sr:
                    raise RuntimeError(f"drama line sample_rate mismatch: {sr} != {output_sr}")

                # 更新 chain cache：用 trim/loudness 之前的"原始"audio 作韵律 prompt
                # （trim/loudness 是为最终拼接做的后处理，不应改变模型 prompt 语义）。
                # 长段保护：> _CHAIN_MAX_SECONDS 跳过更新，避免把超长 prompt 喂下去拖慢推理。
                self._maybe_update_drama_chain(
                    chain_state=chain_state,
                    chain_temp_paths=chain_temp_paths,
                    character_id=character_id,
                    audio=audio,
                    sample_rate=sr,
                    block_text=block_text,
                    task_id=task_id,
                )

                pieces.append(self._match_loudness(self._trim_silence(audio)))
                pause_ms = self._block_pause_after_ms(block)
                if pause_ms > 0:
                    pieces.append(np.zeros(int(output_sr * pause_ms / 1000), dtype=np.float32))
                block_progress = 45 + int(round((block_index / block_total) * 45))
                await self._send_progress(task_id, block_progress, "正在合成对白")

            final_audio = np.concatenate(pieces, axis=0) if pieces else np.zeros(0, dtype=np.float32)
            final_sr = int(output_sr or 48000)
            if final_audio.size == 0:
                raise RuntimeError("audio drama generation returned no audio")

            params.file_type = "wav"
            await self._send_progress(task_id, 95, "正在保存对白音频")
            await self.result_handler.process_single_result(
                file_data=_audio_bytes(final_audio, final_sr),
                request_params=params,
                generate_time=time.time() - started_at,
                service_id=self.service_id,
                index=0,
                total=1,
            )
        finally:
            for path in temp_seed_paths + chain_temp_paths:
                try:
                    if path and os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass

    async def _synthesize_final_audio(
        self,
        *,
        text: str,
        voice_cfg: VoiceConfig,
        stream_mode: bool,
    ) -> Tuple[np.ndarray, int]:
        collected: List[np.ndarray] = []
        final_audio: Optional[np.ndarray] = None
        final_sr: Optional[int] = None
        async for chunk in self._engine.synthesize_stream(
            text=text,
            voice_cfg=voice_cfg,
            cancel_check=self._cancel_check,
            stream_mode=stream_mode,
        ):
            if chunk.is_final:
                final_audio = chunk.pcm
                final_sr = chunk.sample_rate
            elif chunk.pcm.size > 0:
                collected.append(chunk.pcm)
                final_sr = chunk.sample_rate
        if final_audio is None or final_audio.size == 0:
            final_audio = np.concatenate(collected, axis=0) if collected else np.zeros(0, dtype=np.float32)
        if final_audio.size == 0:
            raise RuntimeError("VoxCPM generation returned no audio")
        return final_audio.astype(np.float32, copy=False), int(final_sr or 48000)

    @staticmethod
    def _normalize_drama_characters(raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, list):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip()
            if not cid:
                continue
            out[cid] = dict(item)
        return out

    @staticmethod
    def _normalize_drama_dialogues(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            speaker_id = str(item.get("speaker_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if speaker_id and text:
                next_item = dict(item)
                next_item["speaker_id"] = speaker_id
                next_item["text"] = text
                out.append(next_item)
        return out

    @staticmethod
    def _seed_text_for_character(character: Dict[str, Any]) -> str:
        seed_text = str(character.get("seed_text") or "").strip()
        name = str(character.get("name") or character.get("id") or "这个角色").strip()
        instruct = str(character.get("instruct") or "").strip()
        if instruct:
            base = f"我是{name}。我的声音设定是：{instruct}。这段声音将作为我的角色音色基准，请保持稳定、清晰、自然。"
        else:
            base = f"我是{name}。这段声音将作为我的角色音色基准，请保持稳定、清晰、自然。"
        return f"{base}{seed_text}" if seed_text else base

    @staticmethod
    def _voice_cfg_for_seed(base: VoiceConfig, character: Dict[str, Any]) -> VoiceConfig:
        mode = str(character.get("voice_mode") or "").strip().lower()
        speaker_name = str(character.get("speaker_name") or "").strip() or None
        instruct = str(character.get("instruct") or "").strip() or None
        if mode == "custom_voice" and speaker_name:
            return replace(
                base,
                tts_mode="custom_voice",
                speaker_name=speaker_name,
                voice_preset=None,
                instruct=instruct,
                prompt_wav_path=None,
                prompt_text=None,
                design_instruct=None,
            )
        return replace(
            base,
            tts_mode="voice_design",
            speaker_name=None,
            voice_preset=None,
            instruct=instruct or "自然、清晰、有角色辨识度的声音",
            design_instruct=instruct or "自然、清晰、有角色辨识度的声音",
            prompt_wav_path=None,
            prompt_text=None,
        )

    @staticmethod
    def _line_text_with_style(line: Dict[str, Any]) -> str:
        text = str(line.get("text") or "").strip()
        return text
        style = str(line.get("instruct") or line.get("emotion") or "").strip()
        return f"({style}){text}" if style else text

    @classmethod
    def _build_dialogue_blocks(
        cls,
        dialogues: List[Dict[str, Any]],
        *,
        max_lines_per_block: int = 4,
        merge_pause_threshold_ms: int = 800,
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for line in dialogues:
            if not isinstance(line, dict):
                continue
            speaker_id = str(line.get("speaker_id") or "").strip()
            text = str(line.get("text") or "").strip()
            if not speaker_id or not text:
                continue
            if blocks and cls._can_merge_into_block(
                blocks[-1],
                line,
                max_lines_per_block=max_lines_per_block,
                merge_pause_threshold_ms=merge_pause_threshold_ms,
            ):
                blocks[-1]["lines"].append(line)
            else:
                blocks.append({"speaker_id": speaker_id, "lines": [line]})
        return blocks

    @classmethod
    def _can_merge_into_block(
        cls,
        block: Dict[str, Any],
        line: Dict[str, Any],
        *,
        max_lines_per_block: int,
        merge_pause_threshold_ms: int,
    ) -> bool:
        lines = block.get("lines")
        if not isinstance(lines, list) or len(lines) >= max_lines_per_block:
            return False
        if str(block.get("speaker_id") or "").strip() != str(line.get("speaker_id") or "").strip():
            return False
        previous = lines[-1] if lines else {}
        return cls._pause_after_ms(previous) < merge_pause_threshold_ms

    @classmethod
    def _block_text_with_style(cls, block: Dict[str, Any]) -> str:
        lines = block.get("lines")
        if not isinstance(lines, list):
            return ""
        return "\n".join(
            item for item in (cls._line_text_with_style(line) for line in lines)
            if item
        )

    @classmethod
    def _block_pause_after_ms(cls, block: Dict[str, Any]) -> int:
        lines = block.get("lines")
        if not isinstance(lines, list) or not lines:
            return 450
        return cls._pause_after_ms(lines[-1])

    @staticmethod
    def _pause_after_ms(line: Dict[str, Any]) -> int:
        raw = line.get("pause_after_ms")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 450
        return max(0, min(3000, value))

    @staticmethod
    def _trim_silence(audio: np.ndarray, *, threshold: float = 0.008, pad: int = 960) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        idx = np.flatnonzero(np.abs(arr) > threshold)
        if idx.size == 0:
            return arr
        start = max(0, int(idx[0]) - pad)
        end = min(arr.size, int(idx[-1]) + pad)
        return arr[start:end]

    @staticmethod
    def _match_loudness(audio: np.ndarray, *, target_rms: float = 0.08) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        rms = float(np.sqrt(np.mean(np.square(arr))) + 1e-8)
        if rms <= 1e-6:
            return arr
        gain = min(4.0, max(0.25, target_rms / rms))
        return np.clip(arr * gain, -0.98, 0.98).astype(np.float32, copy=False)

    @staticmethod
    def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
        import soundfile as sf

        fd, path = tempfile.mkstemp(prefix="voxcpm-drama-seed-", suffix=".wav")
        os.close(fd)
        sf.write(path, audio.astype(np.float32), int(sample_rate), format="WAV")
        return path

    # drama chain prompt 的单段时长上限（秒）。超阈值跳过 cache 更新，
    # 维持上一段缓存不变——这样下一段仍能续韵律，但不会被超长 prompt 拖累首响。
    _CHAIN_MAX_SECONDS = 18.0

    def _maybe_update_drama_chain(
        self,
        *,
        chain_state: Dict[str, Tuple[str, str]],
        chain_temp_paths: List[str],
        character_id: str,
        audio: np.ndarray,
        sample_rate: int,
        block_text: str,
        task_id: str,
    ) -> None:
        """把刚合成完的 dialogue block 写为该 character 的下一段 chain prompt。

        - 写盘失败 / 长段越界 / 空 audio / 空 text → 静默跳过，保留上一段 cache；
        - 旧文件不会立刻删（统一交给 ``_run_drama`` 的 finally 清理），
          目的是：即使本次 cache 更新失败，下一段仍能用上一段健康的 cache。
        """
        if sample_rate <= 0 or audio is None:
            return
        try:
            arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            return
        if arr.size == 0:
            return
        text_clean = block_text.strip()
        if not text_clean:
            return
        duration = arr.size / float(sample_rate)
        if duration > self._CHAIN_MAX_SECONDS:
            self.logger.debug(
                "[voxcpm][drama][chain] task_id=%s character=%s skip update duration=%.2fs > %.2fs",
                task_id,
                character_id,
                duration,
                self._CHAIN_MAX_SECONDS,
            )
            return
        try:
            new_path = self._write_chain_temp_wav(arr, sample_rate)
        except Exception:
            self.logger.exception(
                "[voxcpm][drama][chain] task_id=%s character=%s write tempfile failed",
                task_id,
                character_id,
            )
            return
        chain_temp_paths.append(new_path)
        chain_state[character_id] = (new_path, text_clean)
        self.logger.debug(
            "[voxcpm][drama][chain] task_id=%s character=%s wav=%s duration=%.2fs",
            task_id,
            character_id,
            new_path,
            duration,
        )

    async def _send_progress(self, task_id: str, progress: int, message: str) -> None:
        if not self.status_sender:
            return
        try:
            await self.status_sender(
                task_id,
                "processing",
                progress=progress,
                message=message,
            )
        except Exception:
            self.logger.debug("[voxcpm] progress update failed task_id=%s", task_id, exc_info=True)

    @staticmethod
    def _write_chain_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
        import soundfile as sf

        fd, path = tempfile.mkstemp(prefix="voxcpm-drama-chain-", suffix=".wav")
        os.close(fd)
        sf.write(path, np.clip(audio, -1.0, 1.0).astype(np.float32), int(sample_rate), format="WAV")
        return path

    async def _cancel_check(self) -> bool:
        if not self.check_cancelled:
            return False
        return await self.check_cancelled(f"streaming {self.audio_mode}")

    async def _emit_event(
        self,
        *,
        task_id: str,
        event_type: str,
        sequence: int,
        sample_rate: int,
        is_final: bool,
        data: Optional[str] = None,
    ) -> None:
        # 流式 chunk 协议：mime 为 ``audio/pcm;rate=N``，``data`` 是 base64(裸 Int16 LE PCM)。
        # 与 chat_ws_protocol.md 中 audio_delta 的 BINARY=裸 PCM16 LE 对齐。
        payload: Dict[str, Any] = {
            "type": event_type,
            "task_id": task_id,
            "service_id": self.service_id,
            "audio_mode": self.audio_mode,
            "mime_type": f"audio/pcm;rate={int(sample_rate)}",
            "sequence": sequence,
            "is_final": is_final,
            "sample_rate": sample_rate,
        }
        if data is not None:
            payload["data"] = data
        sent = await self.stream_sender(payload)
        if not sent:
            raise RuntimeError("stream transport disconnected, audio task aborted")


def _audio_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), sample_rate, format="WAV")
    return buffer.getvalue()

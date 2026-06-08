from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from common.io_utils import download_url_to_tempfile
from common.logger import get_logger
from common.result_handler import ResultHandler
from schemas import InferenceRequestParams

from audio.runtime.qwen_asr_bridge import normalize_qwen_asr_language
from audio.runtime.audio_wav_utils import build_transcript_text

logger = get_logger(__name__)


class QwenAsrHandler:
    def __init__(
        self,
        *,
        result_handler: ResultHandler,
        service_id: str,
        logger=logger,
        bundle_loader: Callable[[str], Awaitable[Dict[str, Any]]],
        stream_sender: Callable[[Dict[str, Any]], Awaitable[bool]],
        check_cancelled: Optional[Callable[[str], Awaitable[bool]]] = None,
        status_sender: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger
        self.bundle_loader = bundle_loader
        self.stream_sender = stream_sender
        self.check_cancelled = check_cancelled
        self.status_sender = status_sender

    async def run(self, params: InferenceRequestParams, *, task_id: str) -> None:
        if params.type != "audio":
            raise ValueError(f"QwenAsrHandler expected req.type='audio', got '{params.type}'")

        audio_url = params.input_audio_url or params.prompt_wav_path
        if not audio_url:
            raise ValueError("ASR requires input_audio_url or prompt_wav_path")

        await self._send_progress(task_id, 5, "准备语音识别")
        bundle = await self.bundle_loader("asr")
        await self._send_progress(task_id, 15, "语音识别模型已就绪")
        model = bundle["model"]
        runtime_policy = bundle.get("runtime_policy")
        forced_aligner_ref = bundle.get("forced_aligner_ref")
        normalized_language = normalize_qwen_asr_language(params.language)
        context = str(params.prompt_text or "").strip() or None
        if params.speaker_diarization:
            self.logger.info("[qwen-asr] task_id=%s speaker diarization is not supported and will be ignored", task_id)

        self.logger.info(
            "[qwen-asr] task_id=%s device=%s audio_url=%s language=%s timestamps=%s forced_aligner=%s policy=%s",
            task_id,
            bundle.get("device"),
            audio_url,
            normalized_language,
            bool(params.timestamps),
            forced_aligner_ref,
            getattr(runtime_policy, "cache_key", ""),
        )

        audio_path = await download_url_to_tempfile(
            audio_url,
            default_suffix=".wav",
            timeout_seconds=300.0,
            max_bytes=500 * 1024 * 1024,
        )
        self.logger.info("[qwen-asr] task_id=%s input audio resolved to %s", task_id, audio_path)
        await self._send_progress(task_id, 25, "音频下载完成，开始识别")

        started_at = time.time()
        results = await asyncio.to_thread(
            model.transcribe,
            audio=str(audio_path),
            language=normalized_language,
            context=context,
            return_time_stamps=bool(params.timestamps),
        )
        if self.check_cancelled and await self.check_cancelled("after qwen-asr transcription"):
            return
        await self._send_progress(task_id, 85, "语音识别完成，正在整理结果")

        result = self._extract_first_result(results)
        raw_text = str(getattr(result, "text", "") or "").strip()
        detected_language = str(getattr(result, "language", "") or "").strip()
        segments = self._convert_time_stamps(getattr(result, "time_stamps", None))

        self.logger.info(
            "[qwen-asr] task_id=%s transcription finished text_length=%s detected_language=%s segments=%s",
            task_id,
            len(raw_text),
            detected_language,
            len(segments),
        )

        if params.stream:
            next_sequence = await self._emit_text_stream(task_id, raw_text)
            if segments:
                await self._emit_segments(task_id, segments, start_sequence=next_sequence)
            if self.check_cancelled and await self.check_cancelled("after qwen-asr stream emit"):
                return
            await self._send_progress(task_id, 92, "识别文本已推送")

        params.file_type = "txt"
        text_payload = build_transcript_text(raw_text, segments)
        if detected_language:
            text_payload = f"# Detected Language\n{detected_language}\n\n{text_payload}"
        await self._send_progress(task_id, 95, "正在保存识别文本")
        await self.result_handler.process_single_result(
            file_data=text_payload,
            request_params=params,
            generate_time=time.time() - started_at,
            service_id=self.service_id,
            index=0,
            total=1,
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
            self.logger.debug("[qwen-asr] progress update failed task_id=%s", task_id, exc_info=True)

    async def _emit_text_stream(self, task_id: str, raw_text: str) -> int:
        sequence = 0
        started = await self.stream_sender(
            {
                "type": "text_stream_delta",
                "task_id": task_id,
                "service_id": self.service_id,
                "audio_mode": "asr",
                "sequence": sequence,
                "delta": "",
                "is_final": False,
            }
        )
        if not started:
            raise RuntimeError("stream transport disconnected, audio task aborted")
        sequence += 1

        if raw_text:
            sent = await self.stream_sender(
                {
                    "type": "text_stream_delta",
                    "task_id": task_id,
                    "service_id": self.service_id,
                    "audio_mode": "asr",
                    "sequence": sequence,
                    "delta": raw_text,
                    "is_final": False,
                }
            )
            if not sent:
                raise RuntimeError("stream transport disconnected, audio task aborted")
            sequence += 1
        return sequence

    async def _emit_segments(
        self,
        task_id: str,
        segments: list[dict[str, Any]],
        *,
        start_sequence: int,
    ) -> None:
        sequence = start_sequence
        for segment in segments:
            if self.check_cancelled and await self.check_cancelled("qwen-asr transcript segment emit"):
                return
            sent = await self.stream_sender(
                {
                    "type": "transcript_segment",
                    "task_id": task_id,
                    "service_id": self.service_id,
                    "audio_mode": "asr",
                    "sequence": sequence,
                    "segment": segment,
                    "is_final": False,
                }
            )
            if not sent:
                raise RuntimeError("stream transport disconnected, audio task aborted")
            sequence += 1

    def _extract_first_result(self, results: Any) -> Any:
        if isinstance(results, (list, tuple)) and results:
            return results[0]
        raise RuntimeError("qwen-asr transcription returned no result")

    def _convert_time_stamps(self, time_stamps: Any) -> list[dict[str, Any]]:
        if not time_stamps:
            return []

        segments: list[dict[str, Any]] = []
        for item in time_stamps:
            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
                start_time = item.get("start_time")
                end_time = item.get("end_time")
            else:
                text = str(getattr(item, "text", "") or "").strip()
                start_time = getattr(item, "start_time", None)
                end_time = getattr(item, "end_time", None)
            if not text:
                continue
            segments.append(
                {
                    "speaker_id": "",
                    "content": text,
                    "start_time": self._format_timestamp_value(start_time),
                    "end_time": self._format_timestamp_value(end_time),
                }
            )
        return segments

    @staticmethod
    def _format_timestamp_value(value: Any) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        return f"{numeric:.3f}"

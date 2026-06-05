from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterator

import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)


class VoxCPMNanoVllmFacade:
    """Wrap ``nano-vllm-voxcpm`` Sync pool API to match ``voxcpm`` handler expectations.

    The upstream ``voxcpm`` package uses ``generate`` / ``generate_streaming(text=...)``.
    Nano-vLLM-VoxCPM uses ``generate(target_text=..., ref_audio_latents=..., ...)``.
    """

    def __init__(self, pool: Any, *, sample_rate: int) -> None:
        self._pool = pool
        self.tts_model = SimpleNamespace(sample_rate=int(sample_rate))

    def shutdown(self) -> None:
        stop = getattr(self._pool, "stop", None)
        if callable(stop):
            stop()

    def _encode_wav_file(self, path: str) -> bytes:
        with open(path, "rb") as handle:
            raw = handle.read()
        if not raw:
            raise ValueError(f"Empty reference audio file: {path}")
        encode = getattr(self._pool, "encode_latents", None)
        if not callable(encode):
            raise RuntimeError("Nano-vLLM-VoxCPM pool is missing encode_latents()")
        return encode(raw, "wav")

    def _generation_kwargs_to_nano(self, generation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        text = str(generation_kwargs.get("text") or "")
        if not text.strip():
            raise ValueError("VoxCPM generation requires non-empty text")

        nano_kw: Dict[str, Any] = {
            "target_text": text,
            "cfg_value": float(generation_kwargs.get("cfg_value", 2.0)),
            "prompt_latents": None,
            "prompt_text": "",
            "ref_audio_latents": None,
            "max_generate_length": int(generation_kwargs.get("max_generate_length", 2000)),
            "temperature": float(generation_kwargs.get("temperature", 1.0)),
            "lora_name": generation_kwargs.get("lora_name"),
        }

        prompt_text = str(generation_kwargs.get("prompt_text") or "").strip()
        ref_path = generation_kwargs.get("reference_wav_path")
        prompt_wav = generation_kwargs.get("prompt_wav_path")

        if prompt_wav and prompt_text:
            nano_kw["prompt_latents"] = self._encode_wav_file(str(prompt_wav))
            nano_kw["prompt_text"] = prompt_text
        elif prompt_text and not prompt_wav:
            raise ValueError("prompt_text without prompt_wav_path is not supported for Nano-vLLM-VoxCPM")

        if ref_path:
            ref_s = str(ref_path)
            if nano_kw["prompt_latents"] is not None and prompt_wav and ref_s == str(prompt_wav):
                nano_kw["ref_audio_latents"] = nano_kw["prompt_latents"]
            else:
                nano_kw["ref_audio_latents"] = self._encode_wav_file(ref_s)

        if nano_kw.get("lora_name") in ("", None):
            nano_kw.pop("lora_name", None)

        return nano_kw

    @staticmethod
    def _nano_generate_call(nano_kw: Dict[str, Any]) -> Dict[str, Any]:
        call: Dict[str, Any] = {
            "target_text": nano_kw["target_text"],
            "prompt_text": str(nano_kw.get("prompt_text") or ""),
            "max_generate_length": int(nano_kw["max_generate_length"]),
            "temperature": float(nano_kw["temperature"]),
            "cfg_value": float(nano_kw["cfg_value"]),
        }
        if nano_kw.get("prompt_latents") is not None:
            call["prompt_latents"] = nano_kw["prompt_latents"]
        if nano_kw.get("ref_audio_latents") is not None:
            call["ref_audio_latents"] = nano_kw["ref_audio_latents"]
        if nano_kw.get("lora_name"):
            call["lora_name"] = nano_kw["lora_name"]
        return call

    def generate_streaming(self, **generation_kwargs: Any) -> Iterator[np.ndarray]:
        nano_kw = self._generation_kwargs_to_nano(generation_kwargs)
        gen = self._pool.generate(**self._nano_generate_call(nano_kw))
        for chunk in gen:
            if chunk is None:
                continue
            yield np.asarray(chunk, dtype=np.float32)

    def generate(self, **generation_kwargs: Any) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for part in self.generate_streaming(**generation_kwargs):
            if part.size:
                chunks.append(part)
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

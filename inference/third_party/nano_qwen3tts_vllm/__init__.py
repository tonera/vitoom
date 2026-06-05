"""Qwen3-TTS with vLLM-style optimizations."""

from nano_qwen3tts_vllm.sampling_params import SamplingParams
from nano_qwen3tts_vllm.config import Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig

__version__ = "0.1.0"

__all__ = [
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "SamplingParams",
]

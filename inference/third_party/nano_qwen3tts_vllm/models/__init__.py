"""TTS model implementations adapted for nano-vllm style."""

from nano_qwen3tts_vllm.models.qwen3_tts_talker import Qwen3TTSTalkerForCausalLM
from nano_qwen3tts_vllm.models.qwen3_tts_predictor import Qwen3TTSCodePredictorForCausalLM

__all__ = ["Qwen3TTSTalkerForCausalLM", "Qwen3TTSCodePredictorForCausalLM"]

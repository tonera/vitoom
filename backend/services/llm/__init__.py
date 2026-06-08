"""LLM-oriented helpers shared across the backend."""

from .multimodal import MultimodalCompletionError, run_multimodal_completion

__all__ = ["MultimodalCompletionError", "run_multimodal_completion"]

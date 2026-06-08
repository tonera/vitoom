"""TTS 引擎层：纯合成逻辑，不依赖 task / file / result_handler。

供 task 通道（handlers）与 session 通道（session_runtime）共用。
"""

from audio.engines.tts_engine import (
    AudioChunk,
    TtsEngine,
    VoiceConfig,
    normalize_audio_array,
    voice_config_from_params,
)


def __getattr__(name: str):
    if name == "QwenTtsEngine":
        from audio.engines.qwen_tts_engine import QwenTtsEngine

        return QwenTtsEngine
    if name == "VoxCPMTtsEngine":
        from audio.engines.voxcpm_tts_engine import VoxCPMTtsEngine

        return VoxCPMTtsEngine
    raise AttributeError(name)

__all__ = [
    "AudioChunk",
    "TtsEngine",
    "VoiceConfig",
    "QwenTtsEngine",
    "VoxCPMTtsEngine",
    "normalize_audio_array",
    "voice_config_from_params",
]

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from audio.engines.qwen_tts_engine import QwenTtsEngine
from audio.engines.tts_engine import VoiceConfig


class _FakeCustomVoiceModel:
    def __init__(self):
        self.kwargs = None

    def generate_custom_voice(self, **kwargs):
        self.kwargs = kwargs
        return [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4, 0.5], dtype=np.float32),
        ], 10


class _FakeVoiceDesignModel:
    def __init__(self):
        self.kwargs = None

    def generate_voice_design(self, **kwargs):
        self.kwargs = kwargs
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
        ], 10


async def _collect_final_audio(engine, *, text, voice_cfg):
    chunks = [
        chunk
        async for chunk in engine.synthesize_stream(
            text=text,
            voice_cfg=voice_cfg,
            stream_mode=False,
        )
    ]
    return chunks[-1].pcm


def test_qwen_custom_voice_dialogues_use_batch_inputs_and_concat_outputs():
    model = _FakeCustomVoiceModel()

    async def bundle_loader(_kind, **_kwargs):
        return {
            "model": model,
            "capabilities": {"custom_voice": True},
            "sample_rate": 10,
        }

    engine = QwenTtsEngine(bundle_loader=bundle_loader, logger=logging.getLogger(__name__))
    audio = asyncio.run(_collect_final_audio(
        engine,
        text="ignored",
        voice_cfg=VoiceConfig(
            tts_mode="custom_voice",
            drama={
                "characters": [
                    {
                        "id": "vivian",
                        "name": "Vivian",
                        "voice_mode": "custom_voice",
                        "speaker_name": "Vivian",
                        "language": "Chinese",
                    },
                    {
                        "id": "ryan",
                        "name": "Ryan",
                        "voice_mode": "custom_voice",
                        "speaker_name": "Ryan",
                        "language": "English",
                    },
                ],
                "dialogues": [
                    {"speaker_id": "vivian", "text": "你好"},
                    {"speaker_id": "ryan", "text": "Hello", "instruct": "Very happy."},
                    {"speaker_id": "vivian", "text": "再见", "instruct": "轻声说"},
                ],
            },
        ),
    ))

    assert model.kwargs["text"] == ["你好", "Hello", "再见"]
    assert model.kwargs["speaker"] == ["Vivian", "Ryan", "Vivian"]
    assert model.kwargs["language"] == ["Chinese", "English", "Chinese"]
    assert model.kwargs["instruct"] == ["", "Very happy.", "轻声说"]
    np.testing.assert_allclose(
        audio,
        np.array([0.1, 0.2, 0.0, 0.0, 0.3, 0.4, 0.5], dtype=np.float32),
    )


def test_qwen_voice_design_batch_outputs_are_concatenated():
    model = _FakeVoiceDesignModel()

    async def bundle_loader(_kind, **_kwargs):
        return {
            "model": model,
            "capabilities": {"voice_design": True},
            "sample_rate": 10,
        }

    engine = QwenTtsEngine(bundle_loader=bundle_loader, logger=logging.getLogger(__name__))
    audio = asyncio.run(_collect_final_audio(
        engine,
        text="hello",
        voice_cfg=VoiceConfig(tts_mode="voice_design", instruct="Warm voice."),
    ))

    np.testing.assert_allclose(
        audio,
        np.array([1.0, 2.0, 0.0, 0.0, 3.0], dtype=np.float32),
    )


def test_qwen_voice_design_dialogues_use_batch_inputs_without_speaker():
    model = _FakeVoiceDesignModel()

    async def bundle_loader(_kind, **_kwargs):
        return {
            "model": model,
            "capabilities": {"voice_design": True},
            "sample_rate": 10,
        }

    engine = QwenTtsEngine(bundle_loader=bundle_loader, logger=logging.getLogger(__name__))
    audio = asyncio.run(_collect_final_audio(
        engine,
        text="ignored",
        voice_cfg=VoiceConfig(
            tts_mode="voice_design",
            drama={
                "characters": [
                    {
                        "id": "girl",
                        "name": "小妹",
                        "voice_mode": "voice_design",
                        "instruct": "撒娇稚嫩的萝莉女声",
                        "language": "Chinese",
                    },
                    {
                        "id": "guest",
                        "name": "Guest",
                        "voice_mode": "voice_design",
                        "instruct": "Speak in an incredulous tone.",
                        "language": "English",
                    },
                ],
                "dialogues": [
                    {"speaker_id": "girl", "text": "哥哥，你回来啦"},
                    {"speaker_id": "guest", "text": "No way!"},
                ],
            },
        ),
    ))

    assert model.kwargs["text"] == ["哥哥，你回来啦", "No way!"]
    assert model.kwargs["language"] == ["Chinese", "English"]
    assert model.kwargs["instruct"] == ["撒娇稚嫩的萝莉女声", "Speak in an incredulous tone."]
    assert "speaker" not in model.kwargs
    np.testing.assert_allclose(
        audio,
        np.array([1.0, 2.0, 0.0, 0.0, 3.0], dtype=np.float32),
    )

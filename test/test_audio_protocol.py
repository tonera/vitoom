import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from backend.api.tasks.routes import TaskCreateRequest, _extract_params_from_request
from backend.services.agent.tools.builtin.audio_drama_tts import synthesize_drama_audio_sync
from inference.audio.engines.tts_engine import voice_config_from_params
from inference.audio.runtime.qwen_asr_bridge import resolve_qwen_asr_bundle_options
from inference.audio.runtime.runtime_resolver import resolve_audio_model_ref, resolve_audio_runtime
from inference.common.result_handler import ResultHandler
from inference.schemas import InferenceRequestParams


def _sample_drama():
    return {
        "characters": [
            {"id": "vivian", "name": "Vivian", "voice_mode": "custom_voice", "speaker_name": "Vivian", "language": "Chinese"},
            {"id": "ryan", "name": "Ryan", "voice_mode": "custom_voice", "speaker_name": "Ryan", "language": "English"},
        ],
        "dialogues": [
            {"speaker_id": "vivian", "text": "你好", "instruct": ""},
            {"speaker_id": "ryan", "text": "Hello", "instruct": "Very happy."},
        ],
    }


def test_extract_audio_params_contains_qwen_protocol_fields():
    drama = _sample_drama()
    request = TaskCreateRequest(
        task_type="audio",
        prompt="hello world",
        audio_mode="tts",
        speaker_name="Ryan",
        voice_preset="Ryan",
        instruct="Speak warmly and naturally.",
        response_format="audio_file",
        stream=True,
        sample_rate=24000,
        language="en",
        drama=drama,
        model_id="model-1",
    )

    params = _extract_params_from_request(request, "audio")

    assert params["audio_mode"] == "tts"
    assert params["speaker_name"] == "Ryan"
    assert params["voice_preset"] == "Ryan"
    assert params["instruct"] == "Speak warmly and naturally."
    assert params["response_format"] == "audio_file"
    assert params["stream"] is True
    assert params["sample_rate"] == 24000
    assert params["language"] == "en"
    assert params["drama"] == drama
    assert "dialogues" not in params


def test_inference_request_params_from_task_dict_supports_qwen_audio_fields():
    drama = _sample_drama()
    params = InferenceRequestParams.from_task_dict(
        {
            "id": "task-1",
            "type": "audio",
            "user_id": "user-1",
            "prompt": "Please read this sentence.",
            "storage": "server",
            "params": {
                "job_type": "TTS",
                "audio_mode": "tts",
                "speaker_name": "Ryan",
                "voice_preset": "Ryan",
                "instruct": "Friendly podcast host style.",
                "response_format": "audio_file",
                "stream": True,
                "sample_rate": 24000,
                "language": "en",
                "drama": drama,
                "file_type": "wav",
            },
            "model": {
                "name": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "family": "Qwen-tts",
            },
        }
    )

    assert params.audio_mode == "tts"
    assert params.speaker_name == "Ryan"
    assert params.voice_preset == "Ryan"
    assert params.instruct == "Friendly podcast host style."
    assert params.response_format == "audio_file"
    assert params.stream is True
    assert params.sample_rate == 24000
    assert params.language == "en"
    assert params.drama == drama
    assert params.file_type == "wav"
    assert params.model_name == "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert params.family == "Qwen-tts"

    voice_cfg = voice_config_from_params(params)
    assert voice_cfg.drama == params.drama


def test_inference_request_params_from_task_dict_supports_qwen_asr_fields():
    params = InferenceRequestParams.from_task_dict(
        {
            "id": "task-2",
            "type": "audio",
            "user_id": "user-1",
            "prompt": "Please transcribe this sample.",
            "storage": "server",
            "params": {
                "job_type": "ASR",
                "audio_mode": "asr",
                "input_audio_url": "https://example.com/input.wav",
                "prompt_text": "Welcome to the Vitoom audio integration test.",
                "response_format": "text_file",
                "stream": True,
                "timestamps": True,
                "speaker_diarization": False,
                "sample_rate": 16000,
                "language": "en",
                "file_type": "txt",
            },
            "model": {
                "name": "Qwen3-ASR-1.7B",
                "family": "Qwen-asr",
            },
        }
    )

    assert params.audio_mode == "asr"
    assert params.input_audio_url == "https://example.com/input.wav"
    assert params.prompt_text == "Welcome to the Vitoom audio integration test."
    assert params.response_format == "text_file"
    assert params.stream is True
    assert params.timestamps is True
    assert params.speaker_diarization is False
    assert params.sample_rate == 16000
    assert params.language == "en"
    assert params.file_type == "txt"
    assert params.model_name == "Qwen3-ASR-1.7B"
    assert params.family == "Qwen-asr"


def test_extract_audio_params_allows_voice_design_drama_without_top_level_instruct():
    drama = {
        "characters": [
            {"id": "a", "name": "A", "voice_mode": "voice_design", "instruct": "急切的，怒吼", "language": "Chinese"},
            {"id": "b", "name": "B", "voice_mode": "voice_design", "instruct": "轻蔑地笑着说"},
        ],
        "dialogues": [
            {"speaker_id": "a", "text": "放开那个女孩"},
            {"speaker_id": "b", "text": "怎么，你想英雄救美吗？"},
        ],
    }
    request = TaskCreateRequest(
        task_type="audio",
        prompt="放开那个女孩\n怎么，你想英雄救美吗？",
        audio_mode="tts",
        tts_mode="voice_design",
        model_name="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        drama=drama,
    )

    params = _extract_params_from_request(request, "audio")

    assert params["tts_mode"] == "voice_design"
    assert params["instruct"] is None
    assert params["drama"] == drama
    assert "dialogues" not in params


def test_audio_drama_tts_requires_voice_source_for_each_character():
    result = synthesize_drama_audio_sync(
        user_id="user-1",
        characters=[
            {
                "id": "juliet",
                "name": "朱丽叶",
                "voice_mode": "voice_design",
                "instruct": "",
            }
        ],
        dialogues=[
            {
                "speaker_id": "juliet",
                "text": "罗密欧，命运为何如此残酷。",
            }
        ],
    )

    assert result["status"] == "failed"
    assert result["error"] == "character juliet requires instruct when voice_mode=voice_design"


def test_result_handler_supports_text_file_for_audio_task(tmp_path):
    async def _run():
        cfg = SimpleNamespace(
            outputs_dir=str(tmp_path),
            storage_default="local",
            api_base_url="http://127.0.0.1:8888",
        )
        handler = ResultHandler(
            ws_client=None,
            storage_base_path=str(tmp_path),
            inference_config=cfg,
        )
        req = InferenceRequestParams(
            type="audio",
            job_type="ASR",
            storage="local",
            reference_id="",
            id="msg-1",
            user_id="user-1",
            task_id="task-1",
            prompt="",
            audio_mode="asr",
            input_audio_url="https://example.com/audio.wav",
            response_format="text_file",
            stream=False,
            file_type="txt",
        )

        resp = await handler.process_single_result(
            file_data="hello\nworld\n",
            request_params=req,
            generate_time=0.01,
            service_id="audio-service",
            index=0,
            total=1,
        )

        assert resp.files[0].file_name.endswith(".txt")
        assert resp.files[0].mime_type == "text/plain; charset=utf-8"

    asyncio.run(_run())


def test_audio_runtime_is_selected_by_family():
    assert resolve_audio_runtime(SimpleNamespace(family="Voxcpm")) == "voxcpm"
    assert resolve_audio_runtime(SimpleNamespace(family="Qwen-tts")) == "qwen_tts"
    assert resolve_audio_runtime(SimpleNamespace(family="Qwen-asr")) == "qwen_asr"


def test_audio_runtime_requires_supported_family():
    params = SimpleNamespace(family="Qwen-Audio")

    try:
        resolve_audio_runtime(params)
    except ValueError as exc:
        assert "Qwen-asr" in str(exc)
    else:
        raise AssertionError("resolve_audio_runtime should reject unsupported family")


def test_audio_model_ref_resolves_from_models_dir_then_weights_dir(tmp_path):
    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    models_dir.mkdir()
    weights_dir.mkdir()

    model_name = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    local_model = weights_dir / model_name
    local_model.mkdir()

    params = SimpleNamespace(model_name=model_name)
    resolved = resolve_audio_model_ref(
        params,
        models_dir=str(models_dir),
        weights_dir=str(weights_dir),
    )

    assert resolved == str(local_model.resolve())


def test_qwen_asr_timestamps_default_to_local_forced_aligner_only(tmp_path):
    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    models_dir.mkdir()
    weights_dir.mkdir()

    options = resolve_qwen_asr_bundle_options(
        SimpleNamespace(timestamps=True, model_cfg=None),
        models_dir=str(models_dir),
        weights_dir=str(weights_dir),
    )

    assert options.timestamps_requested is True
    assert options.forced_aligner_ref is None
    assert options.forced_aligner_source == "missing_local_default"


def test_qwen_asr_timestamps_pick_local_forced_aligner_when_present(tmp_path):
    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    models_dir.mkdir()
    weights_dir.mkdir()
    aligner_dir = weights_dir / "Qwen3-ForcedAligner-0.6B"
    aligner_dir.mkdir()

    options = resolve_qwen_asr_bundle_options(
        SimpleNamespace(timestamps=True, model_cfg=None),
        models_dir=str(models_dir),
        weights_dir=str(weights_dir),
    )

    assert options.timestamps_requested is True
    assert options.forced_aligner_source == "auto_local"
    assert options.forced_aligner_ref == str(aligner_dir.resolve())

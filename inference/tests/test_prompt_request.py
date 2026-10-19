from pathlib import Path

from prompt.runtime.request import PromptRequestError, resolve_max_new_tokens, resolve_upsample_input
from prompt.runtime.upsampler import BUNDLED_SYSTEM_PROMPTS, _load_system_prompts, resolve_model_paths
from schemas import InferenceRequestParams


def test_text_only_request() -> None:
    text, sources = resolve_upsample_input(
        prompt="A man walking with his dog",
        image_sources=[],
        messages=None,
    )
    assert text == "A man walking with his dog"
    assert sources == []


def test_messages_take_text_and_images() -> None:
    text, sources = resolve_upsample_input(
        prompt="",
        image_sources=["ignored.jpg"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}},
                    {"type": "text", "text": "put a red bandana around the dog's neck"},
                ],
            }
        ],
    )
    assert text == "put a red bandana around the dog's neck"
    assert sources == ["https://example.com/dog.jpg"]


def test_too_many_images_rejected() -> None:
    try:
        resolve_upsample_input(
            prompt="edit",
            image_sources=[f"{index}.jpg" for index in range(3)],
            messages=None,
            max_images=2,
        )
    except PromptRequestError as exc:
        assert "At most 2" in str(exc)
    else:
        raise AssertionError("expected PromptRequestError")


def test_max_new_tokens_is_capped() -> None:
    assert resolve_max_new_tokens(None, default=128, cap=64) == 64
    assert resolve_max_new_tokens(32, default=128, cap=64) == 32
    assert resolve_max_new_tokens(999, default=128, cap=64) == 64


def test_prompt_task_dict_keeps_messages_and_txt(tmp_path: Path) -> None:
    del tmp_path
    params = InferenceRequestParams.from_task_dict(
        {
            "id": "task-1",
            "user_id": "user-1",
            "type": "prompt",
            "prompt": "",
            "params": {
                "messages": [{"role": "user", "content": "afternoon sun"}],
                "max_tokens": 128,
                "tpl_list": ["dog.jpg"],
            },
        }
    )
    assert params.file_type == "txt"
    assert params.extract["messages"][0]["content"] == "afternoon sun"
    assert params.extract["max_tokens"] == 128
    assert params.tpl_list == ["dog.jpg"]


def test_bundled_system_prompts_are_used() -> None:
    prompts = _load_system_prompts(BUNDLED_SYSTEM_PROMPTS)
    assert "t2i" in prompts and "i2i" in prompts
    assert BUNDLED_SYSTEM_PROMPTS.name == "system_prompts.json"


def test_resolve_model_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "flux-3-image-klein-pup-nunchaku"
    model_dir.mkdir(parents=True)
    quant = model_dir / "svdq-int4-flux-3-image-klein-pup-nunchaku.safetensors"
    quant.write_bytes(b"weights")
    pup, found = resolve_model_paths(str(tmp_path / "models"), str(tmp_path / "weights"), "flux-3-image-klein-pup-nunchaku")
    assert pup == model_dir
    assert found == quant

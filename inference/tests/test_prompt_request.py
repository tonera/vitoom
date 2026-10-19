from pathlib import Path

from prompt.runtime.request import (
    MAX_SYSTEM_PROMPT_CHARS,
    PromptRequestError,
    resolve_max_new_tokens,
    resolve_system_prompt,
    resolve_upsample_input,
)
from prompt.runtime.upsampler import (
    BUNDLED_SYSTEM_PROMPTS,
    PromptUpsamplerBundle,
    _build_messages,
    _load_system_prompts,
    resolve_model_paths,
)
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


def test_system_prompt_override() -> None:
    assert resolve_system_prompt(None) is None
    assert resolve_system_prompt("  ") is None
    assert resolve_system_prompt("只描述人物外貌") == "只描述人物外貌"
    try:
        resolve_system_prompt("x" * (MAX_SYSTEM_PROMPT_CHARS + 1))
    except PromptRequestError as exc:
        assert "at most" in str(exc)
    else:
        raise AssertionError("expected PromptRequestError")
    try:
        resolve_system_prompt({"text": "nope"})
    except PromptRequestError as exc:
        assert "string" in str(exc)
    else:
        raise AssertionError("expected PromptRequestError")


def test_custom_system_prompt_replaces_builtin_wrapper() -> None:
    bundle = PromptUpsamplerBundle(
        model_ref="flux-3-image-klein-pup-nunchaku",
        pup_dir="",
        quant_path="",
        device="cuda",
        processor=None,
        model=None,
        system_prompts={"t2i": "T2I", "i2i": "I2I"},
    )
    plain = _build_messages(bundle, prompt="a dog", images=[])
    assert plain == [
        {"role": "system", "content": "T2I"},
        {"role": "user", "content": "a dog"},
    ]
    builtin = _build_messages(bundle, prompt="a dog", images=["img"])
    assert builtin[0]["content"] == "I2I"
    assert "Output the caption." in builtin[1]["content"][1]["text"]

    custom = _build_messages(bundle, prompt="站在窗边的女人", images=["img"], system_prompt="只描述人物外貌")
    assert custom[0] == {"role": "system", "content": "只描述人物外貌"}
    assert custom[1]["content"][0] == {"type": "image", "image": "img"}
    assert custom[1]["content"][1] == {"type": "text", "text": "站在窗边的女人"}

    text_only = _build_messages(bundle, prompt="雨夜街道", images=[], system_prompt="只描述场景")
    assert text_only == [
        {"role": "system", "content": "只描述场景"},
        {"role": "user", "content": "雨夜街道"},
    ]


def test_prompt_task_dict_keeps_system_prompt_only_from_payload(tmp_path: Path) -> None:
    del tmp_path
    params = InferenceRequestParams.from_task_dict(
        {
            "id": "task-2",
            "user_id": "user-1",
            "type": "prompt",
            "prompt": "站在窗边的女人",
            "params": {
                "system_prompt": "  只描述人物外貌  ",
                "extract": {"system_prompt": "不要用这段"},
                "messages": [
                    {"role": "system", "content": "不要用这段"},
                    {"role": "user", "content": "站在窗边的女人"},
                ],
            },
        }
    )
    assert params.system_prompt == "  只描述人物外貌  "
    assert params.extract["system_prompt"] == "不要用这段"
    assert resolve_system_prompt(params.system_prompt) == "只描述人物外貌"


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

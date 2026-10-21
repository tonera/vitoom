from types import SimpleNamespace
from unittest.mock import patch

import torch

import image.inference_param_specs as specs
from image.inference_param_specs import SdxlMainInferenceParamSpec
from schemas import InferenceRequestParams


def _params(**overrides) -> InferenceRequestParams:
    data = {
        "type": "image",
        "job_type": "MK",
        "id": "1",
        "user_id": "1",
        "task_id": "1",
        "prompt": "a baby racoon",
        "negative_prompt": "",
    }
    data.update(overrides)
    return InferenceRequestParams(**data)


def _base(params: InferenceRequestParams) -> dict:
    return {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt or "",
        "guidance_scale": 4.5,
    }


def _fake_embeds(*_args, **_kwargs):
    prompt = torch.ones(1, 77, 4)
    pooled = torch.ones(1, 8)
    negative = torch.full((1, 77, 4), 2.0)
    negative_pooled = torch.full((1, 8), 3.0)
    return prompt, negative, pooled, negative_pooled


def _build(params: InferenceRequestParams) -> dict:
    return SdxlMainInferenceParamSpec().build(
        pipeline_class_name="StableDiffusionXLPipeline",
        request_params=params,
        base=_base(params),
        width=1024,
        height=1024,
        pipeline_instance=SimpleNamespace(),
    )


def test_blank_negative_prompt_is_left_to_pipeline() -> None:
    with patch.object(specs, "get_weighted_text_embeddings_sdxl", _fake_embeds):
        for raw in ("", "   "):
            built = _build(_params(negative_prompt=raw))
            assert "negative_prompt" not in built
            assert "negative_prompt_embeds" not in built
            assert "negative_pooled_prompt_embeds" not in built
            assert "prompt" not in built
            assert built["prompt_embeds"].shape == (1, 77, 4)
            assert built["pooled_prompt_embeds"].shape == (1, 8)


def test_nonempty_negative_prompt_keeps_embeds() -> None:
    with patch.object(specs, "get_weighted_text_embeddings_sdxl", _fake_embeds):
        built = _build(_params(negative_prompt="low quality"))
    assert "negative_prompt" not in built
    assert torch.equal(built["negative_prompt_embeds"], torch.full((1, 77, 4), 2.0))
    assert torch.equal(built["negative_pooled_prompt_embeds"], torch.full((1, 8), 3.0))


def test_blank_negative_prompt_dropped_when_embed_fails() -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("encode failed")

    with patch.object(specs, "get_weighted_text_embeddings_sdxl", _boom):
        built = _build(_params(negative_prompt=""))
    assert "negative_prompt" not in built
    assert built["prompt"] == "a baby racoon"

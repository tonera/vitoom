from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from image.inference_params_builder import build_inference_params
import image.inference_param_specs as inference_param_specs
from schemas import InferenceRequestParams


def test_qwen_editplus_does_not_rewrite_user_steps_or_negative_prompt(monkeypatch):
    monkeypatch.setattr(inference_param_specs, "load_images_from_list", lambda images: ["mock-image"])

    params = InferenceRequestParams(
        type="image",
        action="MK",
        job_type="ED",
        storage="server",
        id="task-1",
        user_id="user-1",
        task_id="task-1",
        prompt="remake this in Makoto Shinkai style",
        negative_prompt="keep this",
        width=1024,
        height=1024,
        guidance_scale=3.5,
        num_inference_steps=8,
        fast_mode=True,
        model_name="Qwen-Image-Edit-2511-Lightning-Nunchaku",
        family="qwen.edit",
        tpl_list=["mock://image-1"],
    )

    pipeline_class = type("QwenImageEditPlusPipeline", (), {})
    result = build_inference_params(pipeline_class, params)

    assert result["num_inference_steps"] == 8
    assert result["negative_prompt"] == "keep this"
    assert "true_cfg_scale" not in result
    assert "guidance_scale" not in result
    assert result["image"] == ["mock-image"]

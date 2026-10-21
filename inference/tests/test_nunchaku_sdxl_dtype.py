import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from common.pipeline_component_injector import build_component_overrides


def _plan():
    return SimpleNamespace(torch_dtype=torch.float16, device="cuda")


def test_nunchaku_sdxl_pipeline_uses_bfloat16(tmp_path: Path) -> None:
    model_dir = tmp_path / "oneObsession_v24-nunchaku"
    model_dir.mkdir()
    quant = model_dir / "svdq-fp4_r32-oneObsession_v24-nunchaku.safetensors"
    quant.write_bytes(b"x")
    plan = _plan()

    loaded = build_component_overrides(
        family="sdxl",
        model_config=None,
        inference_config=SimpleNamespace(models_dir=str(tmp_path), weights_dir=str(tmp_path)),
        device_plan=plan,
        model_info=SimpleNamespace(
            repo_id=str(model_dir),
            method="from_pretrained",
            load_name="oneObsession_v24-nunchaku",
        ),
        logger=logging.getLogger("nunchaku-sdxl-dtype"),
        num_inference_steps=30,
        fast_mode=True,
        low_vram=False,
    )

    assert loaded["torch_dtype"] is torch.bfloat16
    assert plan.torch_dtype is torch.bfloat16
    assert str(quant) in loaded["__component_sig"]["unet"]

    # build_pipeline_params 先写入 device plan 的 fp16，再用 overrides 覆盖。
    args = {"torch_dtype": torch.float16}
    args.update(loaded)
    assert args["torch_dtype"] is torch.bfloat16


def test_plain_sdxl_stays_float16(tmp_path: Path) -> None:
    model_dir = tmp_path / "sdxl-base"
    model_dir.mkdir()
    plan = _plan()

    loaded = build_component_overrides(
        family="sdxl",
        model_config=None,
        inference_config=SimpleNamespace(models_dir=str(tmp_path), weights_dir=str(tmp_path)),
        device_plan=plan,
        model_info=SimpleNamespace(
            repo_id=str(model_dir),
            method="from_pretrained",
            load_name="sdxl-base",
        ),
        logger=logging.getLogger("nunchaku-sdxl-dtype"),
        num_inference_steps=30,
        fast_mode=True,
        low_vram=False,
    )

    assert "torch_dtype" not in loaded
    assert plan.torch_dtype is torch.float16

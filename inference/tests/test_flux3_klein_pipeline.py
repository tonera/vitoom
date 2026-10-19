import logging
from pathlib import Path
from types import SimpleNamespace

from common.model_registry import MODEL_REGISTRY
from common.pipeline_component_injector import build_component_overrides
from image.runtime.flux3_klein_pipeline import Flux3KleinPipeline


class _Scheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return args


class _Inner:
    def __init__(self) -> None:
        self.scheduler = _Scheduler()
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.scheduler.step(None, 1, None)
        return ["image"]


def test_generic_model_index_does_not_hide_blocks(tmp_path: Path) -> None:
    from common.model_metadata import read_family_name
    from common.pipeline_detector import PipelineDetector

    model_dir = tmp_path / "flux-3-image-klein"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text('{"_class_name": "ModularPipeline"}', encoding="utf-8")
    (model_dir / "modular_model_index.json").write_text(
        '{"_class_name": "ModularPipeline", "_blocks_class_name": "Flux3AutoBlocks"}',
        encoding="utf-8",
    )
    assert read_family_name(model_dir) == "Flux3AutoBlocks"
    detector = PipelineDetector()
    params = SimpleNamespace(load_name=str(model_dir), family=None, url=None, model_cfg=None)
    assert detector.get_pipeline(params).__name__ == "Flux3KleinPipeline"
    assert detector.family == "flux3_klein"


def test_modular_index_uses_catalog(tmp_path: Path) -> None:
    assert MODEL_REGISTRY.to_family("Flux.3 Klein") == "flux3_klein"
    model_dir = tmp_path / "flux-3-image-klein-nunchaku"
    model_dir.mkdir()
    (model_dir / "modular_model_index.json").write_text(
        '{"_class_name": "Flux3DistilledModularPipeline"}',
        encoding="utf-8",
    )
    from common.pipeline_detector import PipelineDetector

    detector = PipelineDetector()
    params = SimpleNamespace(load_name=str(model_dir), family=None, url=None, model_cfg=None)
    assert detector.get_pipeline(params).__name__ == "Flux3KleinPipeline"
    assert detector.family == "flux3_klein"


def test_fast_mode_finds_weights_in_nunchaku_dir(tmp_path: Path) -> None:
    models = tmp_path / "models"
    weights = tmp_path / "weights"
    model_dir = models / "flux-3-image-klein-nunchaku"
    model_dir.mkdir(parents=True)
    quant = model_dir / "svdq-fp4_r32-flux-3-image-klein-nunchaku.safetensors"
    quant.write_bytes(b"x")
    te_dir = weights / "Qwen3-VL-4B-Instruct-Nunchaku"
    te_dir.mkdir(parents=True)
    te = te_dir / "svdq-int4-Qwen3-VL-4B-Instruct-Nunchaku.safetensors"
    te.write_bytes(b"x")

    loaded = build_component_overrides(
        family="flux3_klein",
        model_config=None,
        inference_config=SimpleNamespace(models_dir=str(models), weights_dir=str(weights)),
        device_plan=SimpleNamespace(torch_dtype=None, device="cpu"),
        model_info=SimpleNamespace(
            repo_id=str(model_dir),
            method="from_pretrained",
            load_name="flux-3-image-klein-nunchaku",
        ),
        logger=logging.getLogger("flux3-test"),
        num_inference_steps=16,
        fast_mode=True,
        low_vram=False,
    )
    assert str(quant) in loaded["__component_sig"]["transformer"]
    assert str(te) in loaded["__component_sig"]["text_encoder"]


def test_from_pretrained_skips_upsampler_and_call_returns_images(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "flux-3-image-klein-nunchaku"
    model_dir.mkdir()
    (model_dir / "modular_model_index.json").write_text("{}", encoding="utf-8")

    class _Blocks:
        def __init__(self) -> None:
            self.sub_blocks = {"prompt_upsampler": object(), "denoise": object()}

        def init_pipeline(self, model: str):
            assert model == str(model_dir)
            pipe = _Pipe()
            pipe.loaded_from = model
            return pipe

    class _Spec:
        def __init__(self, repo: str) -> None:
            self.pretrained_model_name_or_path = repo
            self.revision = "main"

    class _Pipe(_Inner):
        def __init__(self) -> None:
            super().__init__()
            self._component_specs = {
                "transformer": _Spec("black-forest-labs/flux-3-image-klein"),
                "text_encoder": _Spec("Qwen/Qwen3-VL-4B-Instruct"),
                "vae": _Spec("black-forest-labs/flux-3-image-klein"),
                "prompt_upsampler": _Spec("black-forest-labs/flux-3-image-klein-pup"),
            }
            self.updated: dict = {}
            self.loaded = None
            self.order: list[str] = []

        def update_components(self, **kwargs):
            self.order.append("update")
            self.updated.update(kwargs)

        def load_components(self, names=None, **kwargs):
            if "torch_dtype" in kwargs:
                raise TypeError("use dtype")
            self.order.append("load")
            self.loaded = (list(names or []), kwargs)

    blocks = _Blocks()
    monkeypatch.setattr("image.runtime.flux3_klein_pipeline._load_modular_blocks", lambda pipeline_dir: blocks)
    transformer = object()
    wrapper = Flux3KleinPipeline.from_pretrained(str(model_dir), transformer=transformer, torch_dtype="bf16")
    assert "prompt_upsampler" not in blocks.sub_blocks
    assert wrapper._pipe.order == ["load", "update"]
    assert wrapper._pipe.updated["transformer"] is transformer
    assert wrapper._pipe.loaded[0] == ["text_encoder", "vae"]
    loaded_kwargs = wrapper._pipe.loaded[1]
    assert loaded_kwargs["pretrained_model_name_or_path"] == str(model_dir)
    assert loaded_kwargs["local_files_only"] is True
    assert loaded_kwargs["dtype"] == "bf16"
    assert wrapper._pipe._component_specs["text_encoder"].pretrained_model_name_or_path == "Qwen/Qwen3-VL-4B-Instruct"
    assert wrapper._pipe._component_specs["prompt_upsampler"].pretrained_model_name_or_path.endswith("klein-pup")

    result = wrapper("caption", guidance_scale=4, negative_prompt="no", height=512, width=512, image=["ref"])
    assert result.images == ["image"]
    assert wrapper._pipe.calls[0]["output"] == "images"
    assert "guidance_scale" not in wrapper._pipe.calls[0]
    assert wrapper._pipe.calls[0]["image"] == ["ref"]

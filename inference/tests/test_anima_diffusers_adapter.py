from pathlib import Path
from types import SimpleNamespace

from common.model_catalog.types import PipelineRef
from common.pipeline_component_injector import _nunchaku_diffusers_sibling_dir, _nunchaku_file_precision
from image.runtime.anima_diffusers_pipeline import AnimaDiffusersPipeline, resolve_anima_scheduler


class _Scheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            shift=3.0,
            stochastic_sampling=False,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
        )
        self.steps = 0

    def register_to_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def step(self, *args, **kwargs):
        self.steps += 1
        return args


class _Inner:
    def __init__(self) -> None:
        self.scheduler = _Scheduler()
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.scheduler.step(None, 7, None)
        return SimpleNamespace(images=["ok"])


def test_callback_flow_shift_and_max_sequence_length() -> None:
    inner = _Inner()
    wrapper = AnimaDiffusersPipeline(inner)
    seen_steps: list[tuple] = []

    def callback(pipe, step_index, timestep, callback_kwargs):
        seen_steps.append((step_index, timestep))
        return callback_kwargs

    result = wrapper(
        "a prompt",
        max_sequence_length=256,
        flow_shift=5,
        callback_on_step_end=callback,
        latents="kept",
    )
    assert result.images == ["ok"]
    assert inner.calls[0]["max_sequence_length"] == 256
    assert inner.calls[0]["latents"] == "kept"
    assert seen_steps == [(0, 7)]
    assert inner.scheduler.config.shift == 5.0

    wrapper("again")
    assert inner.scheduler.config.shift == 3.0


def test_callback_restores_scheduler_step_when_cancelled() -> None:
    inner = _Inner()
    original = inner.scheduler.step
    wrapper = AnimaDiffusersPipeline(inner)

    def callback(pipe, step_index, timestep, callback_kwargs):
        raise RuntimeError("cancelled")

    try:
        wrapper("a prompt", callback_on_step_end=callback)
    except RuntimeError as exc:
        assert str(exc) == "cancelled"
    else:
        raise AssertionError("callback did not cancel")

    assert inner.scheduler.step.__func__ is original.__func__
    inner.scheduler.step(None, 1, None)
    assert inner.scheduler.steps == 1


def test_resolve_anima_scheduler_names() -> None:
    assert resolve_anima_scheduler(None) is None
    assert resolve_anima_scheduler("Euler") is None
    assert resolve_anima_scheduler("DDIM") is None
    assert resolve_anima_scheduler("DPM++ 2M") is None
    assert resolve_anima_scheduler("UniPC") is None
    assert resolve_anima_scheduler("qwen_scheduler_lightning") is None

    euler_a = resolve_anima_scheduler("Euler a")
    assert euler_a is not None
    assert euler_a["stochastic_sampling"] is True
    assert euler_a["use_karras_sigmas"] is False

    karras = resolve_anima_scheduler("DPM++ 2M Karras")
    assert karras is not None
    assert karras["stochastic_sampling"] is False
    assert karras["use_karras_sigmas"] is True

    both = resolve_anima_scheduler("DPM++ 2M SDE Karras")
    assert both is not None
    assert both["stochastic_sampling"] is True
    assert both["use_karras_sigmas"] is True
    assert both["use_exponential_sigmas"] is False
    assert both["use_beta_sigmas"] is False

    ancestral = resolve_anima_scheduler("DPM2 a")
    assert ancestral is not None
    assert ancestral["stochastic_sampling"] is True
    assert ancestral["use_karras_sigmas"] is False


def test_scheduler_name_applies_per_call_and_is_not_forwarded() -> None:
    inner = _Inner()
    wrapper = AnimaDiffusersPipeline(inner)

    wrapper("prompt", schedulerName="Euler a", flow_shift=4)
    assert inner.scheduler.config.stochastic_sampling is True
    assert inner.scheduler.config.use_karras_sigmas is False
    assert inner.scheduler.config.shift == 4
    assert "schedulerName" not in inner.calls[0]

    wrapper("prompt", schedulerName="LMS Karras")
    assert inner.scheduler.config.stochastic_sampling is False
    assert inner.scheduler.config.use_karras_sigmas is True
    assert inner.scheduler.config.shift == 3.0

    wrapper("prompt", schedulerName="DDIM")
    assert inner.scheduler.config.stochastic_sampling is False
    assert inner.scheduler.config.use_karras_sigmas is False
    assert "schedulerName" not in inner.calls[-1]


def test_anima_nunchaku_sibling_and_precision(tmp_path: Path) -> None:
    export = tmp_path / "Anima-Nunchaku"
    diffusers_dir = export / "diffusers"
    diffusers_dir.mkdir(parents=True)
    int4 = export / "svdq-int4_r32-anima-int8note.safetensors"
    int8 = export / "svdq-int8_r128-anima.safetensors"
    assert _nunchaku_diffusers_sibling_dir(SimpleNamespace(family="anima"), diffusers_dir) == export
    assert _nunchaku_file_precision(int4) == "int4"
    assert _nunchaku_file_precision(int8) == "int8"


def test_runtime_backend_wins_for_single_file(tmp_path: Path, monkeypatch) -> None:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(PipelineRef, "resolve", lambda self: self.attr)
    from common.pipeline_detector import PipelineDetector

    detector = PipelineDetector()
    runtime = SimpleNamespace(
        load_name=str(ckpt),
        family="anima",
        model_cfg={"anima": {"backend": "runtime"}},
        url=None,
    )
    diffusers = SimpleNamespace(
        load_name=str(ckpt),
        family="anima",
        model_cfg={},
        url=None,
    )
    assert detector.get_pipeline(runtime) == "AnimaPipeline"
    assert detector.get_pipeline(diffusers) == "AnimaDiffusersPipeline"

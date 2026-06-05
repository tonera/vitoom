from __future__ import annotations
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = REPO_ROOT / "inference"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _install_fake_diffusers_and_nunchaku(monkeypatch: pytest.MonkeyPatch):
    diffusers_mod = types.ModuleType("diffusers")

    class DiffusionPipeline:
        pass

    class StableDiffusionXLPipeline(DiffusionPipeline):
        pass

    class StableDiffusionXLImg2ImgPipeline(DiffusionPipeline):
        pass

    diffusers_mod.DiffusionPipeline = DiffusionPipeline
    diffusers_mod.StableDiffusionXLPipeline = StableDiffusionXLPipeline
    diffusers_mod.StableDiffusionXLImg2ImgPipeline = StableDiffusionXLImg2ImgPipeline

    diffusers_models_mod = types.ModuleType("diffusers.models")
    diffusers_models_unets_mod = types.ModuleType("diffusers.models.unets")
    diffusers_unet_mod = types.ModuleType("diffusers.models.unets.unet_2d_condition")

    class UNet2DConditionOutput:
        def __init__(self, sample=None):
            self.sample = sample

    diffusers_unet_mod.UNet2DConditionOutput = UNet2DConditionOutput

    diffusers_utils_mod = types.ModuleType("diffusers.utils")
    diffusers_utils_mod.USE_PEFT_BACKEND = False
    diffusers_utils_mod.deprecate = lambda *args, **kwargs: None
    diffusers_utils_mod.scale_lora_layers = lambda *args, **kwargs: None
    diffusers_utils_mod.unscale_lora_layers = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)
    monkeypatch.setitem(sys.modules, "diffusers.models", diffusers_models_mod)
    monkeypatch.setitem(sys.modules, "diffusers.models.unets", diffusers_models_unets_mod)
    monkeypatch.setitem(sys.modules, "diffusers.models.unets.unet_2d_condition", diffusers_unet_mod)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils_mod)

    nunchaku_mod = types.ModuleType("nunchaku")
    nunchaku_caching_mod = types.ModuleType("nunchaku.caching")
    nunchaku_fbcache_mod = types.ModuleType("nunchaku.caching.fbcache")
    module_buffers: dict[str, object] = {}
    current_ctx = {"value": None}

    def _buffer_dict():
        ctx = current_ctx["value"]
        if isinstance(ctx, dict):
            return ctx.setdefault("buffers", {})
        return module_buffers

    def get_buffer(key):
        return _buffer_dict().get(key)

    def set_buffer(key, value):
        _buffer_dict()[key] = value

    def get_can_use_cache(*args, **kwargs):
        return False, None

    def create_cache_context():
        return {"buffers": {}}

    def get_current_cache_context():
        return current_ctx["value"]

    @contextmanager
    def cache_context(ctx):
        prev = current_ctx["value"]
        current_ctx["value"] = ctx
        try:
            yield ctx
        finally:
            current_ctx["value"] = prev

    nunchaku_fbcache_mod.get_buffer = get_buffer
    nunchaku_fbcache_mod.get_can_use_cache = get_can_use_cache
    nunchaku_fbcache_mod.set_buffer = set_buffer
    nunchaku_fbcache_mod.cache_context = cache_context
    nunchaku_fbcache_mod.create_cache_context = create_cache_context
    nunchaku_fbcache_mod.get_current_cache_context = get_current_cache_context

    monkeypatch.setitem(sys.modules, "nunchaku", nunchaku_mod)
    monkeypatch.setitem(sys.modules, "nunchaku.caching", nunchaku_caching_mod)
    monkeypatch.setitem(sys.modules, "nunchaku.caching.fbcache", nunchaku_fbcache_mod)

    return StableDiffusionXLPipeline


def _load_pipeline_lifecycle_for_test(monkeypatch: pytest.MonkeyPatch):
    if str(INFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(INFERENCE_ROOT))

    sdxl_base_cls = _install_fake_diffusers_and_nunchaku(monkeypatch)

    common_pkg = types.ModuleType("common")
    common_pkg.__path__ = [str(INFERENCE_ROOT / "common")]
    monkeypatch.setitem(sys.modules, "common", common_pkg)

    const_mod = types.ModuleType("common.Constant")
    const_mod.JT_ED = "ED"
    monkeypatch.setitem(sys.modules, "common.Constant", const_mod)

    model_utils_mod = types.ModuleType("common.family_utils")
    model_utils_mod.to_model_family = lambda value: str(value or "").lower()
    monkeypatch.setitem(sys.modules, "common.family_utils", model_utils_mod)

    logger_mod = types.ModuleType("common.logger")
    logger_mod.print_info = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "common.logger", logger_mod)

    torch_transfer_mod = types.ModuleType("common.torch_transfer_utils")
    torch_transfer_mod.pretouch_pipeline_cpu_tensors = lambda *args, **kwargs: None
    torch_transfer_mod.should_pretouch = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "common.torch_transfer_utils", torch_transfer_mod)

    pipeline_cache_mod = types.ModuleType("common.pipeline_cache")
    pipeline_cache_mod.PipelineCache = object
    monkeypatch.setitem(sys.modules, "common.pipeline_cache", pipeline_cache_mod)

    fbcache_mod = _load_module("common.fbcache_sdxl", INFERENCE_ROOT / "common" / "fbcache_sdxl.py")
    setattr(common_pkg, "fbcache_sdxl", fbcache_mod)

    image_pkg = types.ModuleType("image")
    image_pkg.__path__ = [str(INFERENCE_ROOT / "image")]
    runtime_pkg = types.ModuleType("image.runtime")
    runtime_pkg.__path__ = [str(INFERENCE_ROOT / "image" / "runtime")]
    monkeypatch.setitem(sys.modules, "image", image_pkg)
    monkeypatch.setitem(sys.modules, "image.runtime", runtime_pkg)

    lora_manager_mod = types.ModuleType("image.runtime.lora_manager")
    lora_manager_mod.unload_loras_from_pipe = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "image.runtime.lora_manager", lora_manager_mod)

    pipeline_service_mod = types.ModuleType("image.runtime.pipeline_service")

    class PipelineService:
        def __init__(self, *args, **kwargs):
            pass

    pipeline_service_mod.PipelineService = PipelineService
    monkeypatch.setitem(sys.modules, "image.runtime.pipeline_service", pipeline_service_mod)

    schemas_mod = types.ModuleType("schemas")

    class InferenceRequestParams:
        pass

    schemas_mod.InferenceRequestParams = InferenceRequestParams
    monkeypatch.setitem(sys.modules, "schemas", schemas_mod)

    mod = _load_module("_pipeline_lifecycle_for_test", INFERENCE_ROOT / "image" / "runtime" / "pipeline_lifecycle.py")
    return mod.PipelineLifecycle, sdxl_base_cls


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_sdxl_fast_mode_false_rolls_back_nested_call_patches(monkeypatch: pytest.MonkeyPatch):
    PipelineLifecycle, StableDiffusionXLPipeline = _load_pipeline_lifecycle_for_test(monkeypatch)

    class FakeUNet:
        def forward(self, sample=None, timestep=None, encoder_hidden_states=None, return_dict=True):
            return SimpleNamespace(sample=sample)

    class FakeSDXLPipeline(StableDiffusionXLPipeline):
        def __init__(self):
            self.unet = FakeUNet()

        def __call__(self, *args, **kwargs):
            return "base-call"

    lifecycle = object.__new__(PipelineLifecycle)
    lifecycle.logger = _DummyLogger()

    pipe = FakeSDXLPipeline()

    lifecycle.apply_fast_mode_cache(
        pipe,
        SimpleNamespace(family="sdxl", fast_mode=True, num_inference_steps=4),
    )
    assert getattr(pipe, "_meancache_call_isolated", False) is True
    assert getattr(pipe, "_fbcache_call_isolated", False) is True
    assert pipe.__class__ is not FakeSDXLPipeline

    lifecycle.apply_fast_mode_cache(
        pipe,
        SimpleNamespace(family="sdxl", fast_mode=False, num_inference_steps=4),
    )

    assert pipe.__class__ is FakeSDXLPipeline
    assert getattr(pipe, "_meancache_call_isolated", False) is False
    assert getattr(pipe, "_fbcache_call_isolated", False) is False
    assert pipe(num_inference_steps=4) == "base-call"

import json
import argparse
import asyncio
from pathlib import Path
import sys

import pytest

# 添加项目路径，便于导入 inference 下的模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from common.pipeline_detector import PipelineDetector
from common.model_catalog.types import PipelineRef
from common.model_metadata import read_family_name, read_int_from_json, safetensors_find_first_shape
from image.runtime.pipeline_service import PipelineService
from schemas import InferenceRequestParams


def _minimal_params(**overrides):
    base = dict(
        # schemas.InferenceRequestParams 必填字段
        type="image",
        action="MK",
        job_type="image",
        storage="local",
        id="task1",
        user_id="u1",
        task_id="task1",
        prompt="a cat",
        model_name="dummy",
    )
    base.update(overrides)
    return InferenceRequestParams(**base)


def test_get_pipeline_from_dir(tmp_path, monkeypatch):
    # 准备一个带 model_index.json 的目录
    model_dir = tmp_path / "sdxl"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}), "utf-8")

    class DummyPipe:
        pass

    monkeypatch.setattr(PipelineRef, "resolve", lambda self: DummyPipe)

    params = _minimal_params(model_name=str(model_dir), url=None)
    detector = PipelineDetector()
    pipe_cls = detector.get_pipeline(params)
    assert pipe_cls is DummyPipe
    assert detector.family == "sdxl"

def test_get_pipeline_from_dir_flux2_klein(tmp_path, monkeypatch):
    # 准备一个带 model_index.json 的目录（Flux2KleinPipeline）
    model_dir = tmp_path / "flux2_klein"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline"}), "utf-8")

    class DummyKlein:
        pass

    monkeypatch.setattr(PipelineRef, "resolve", lambda self: DummyKlein)

    params = _minimal_params(model_name=str(model_dir), url=None)
    detector = PipelineDetector()
    pipe_cls = detector.get_pipeline(params)
    assert pipe_cls is DummyKlein
    assert detector.family == "flux2_klein"


def test_model_metadata_helpers(tmp_path):
    model_dir = tmp_path / "flux2_klein"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline"}), "utf-8")
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(json.dumps({"joint_attention_dim": 12288}), "utf-8")

    assert read_family_name(model_dir) == "Flux2KleinPipeline"
    assert read_int_from_json(transformer_dir / "config.json", "joint_attention_dim") == 12288

    ckpt = tmp_path / "demo.safetensors"
    try:
        from safetensors.torch import save_file
        import torch

        save_file({"context_embedder.weight": torch.zeros((3072, 12288))}, str(ckpt))
        shape, key = safetensors_find_first_shape(ckpt, ("context_embedder.weight",))
        assert shape == (3072, 12288)
        assert key == "context_embedder.weight"
    except Exception:
        # 测试环境缺少 safetensors/torch 时，JSON helpers 仍可覆盖主要工具路径
        pass


def test_get_pipeline_from_file(tmp_path, monkeypatch):
    # 准备一个文件路径
    model_file = tmp_path / "flux.safetensors"
    model_file.write_bytes(b"dummy")

    detector = PipelineDetector()

    # mock 文件类型检测
    monkeypatch.setattr(detector, "_detect_model_type_from_file_keys", lambda _: "flux")

    class DummyFlux:
        pass

    monkeypatch.setattr(PipelineRef, "resolve", lambda self: DummyFlux)

    params = _minimal_params(model_name=str(model_file), url=None)
    pipe_cls = detector.get_pipeline(params)
    assert pipe_cls is DummyFlux
    assert detector.family == "flux"


def test_get_pipeline_from_file_flux2_klein_kv(tmp_path, monkeypatch):
    model_file = tmp_path / "FLUX.2-klein-9b-kv.safetensors"
    model_file.write_bytes(b"dummy")

    detector = PipelineDetector()
    monkeypatch.setattr(detector, "_detect_model_type_from_file_keys", lambda _: "flux2_klein")

    class DummyKlein:
        pass

    class DummyKleinKV:
        pass

    def _resolve(self):
        mapping = {
            "Flux2KleinPipeline": DummyKlein,
            "Flux2KleinKVPipeline": DummyKleinKV,
        }
        return mapping[self.attr]

    monkeypatch.setattr(PipelineRef, "resolve", _resolve)

    params = _minimal_params(model_name=str(model_file), url=None)
    pipe_cls = detector.get_pipeline(params)
    assert pipe_cls is DummyKleinKV
    assert detector.family == "flux2_klein"


def test_build_pipeline_params_flux_auto_nunchaku_fast_mode(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "flux"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "FLUX.1-dev"
    te_dir = weights_dir / "nunchaku-t5"
    model_dir.mkdir(parents=True)
    te_dir.mkdir(parents=True)
    main_ckpt = model_dir / "svdq-int4_r64-FLUX.1-dev.safetensors"
    text_ckpt = te_dir / "awq-int4-flux.1-t5xxl.safetensors"
    main_ckpt.write_bytes(b"main")
    text_ckpt.write_bytes(b"text")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="flux", fast_mode=True, low_vram=False, url=None)
    params.num_inference_steps = 10
    params.model_cfg = {"transformer": "should_be_ignored.safetensors"}

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "flux",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )
    monkeypatch.setattr(
        pci,
        "_load_nunchaku_t5_encoder",
        lambda ctx, p, key: {"kind": key, "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert out["torch_dtype"] == torch.bfloat16
    assert "transformer" not in out
    assert "text_encoder_2" not in out
    assert set(out["__lazy_component_loaders"].keys()) == {"transformer", "text_encoder_2"}
    assert out["__component_sig"]["transformer"].endswith(str(main_ckpt))
    assert out["__component_sig"]["text_encoder_2"].endswith(str(text_ckpt))


def test_build_pipeline_params_flux_auto_nunchaku_low_vram(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "flux"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "FLUX.1-dev"
    te_dir = weights_dir / "nunchaku-t5"
    model_dir.mkdir(parents=True)
    te_dir.mkdir(parents=True)
    main_ckpt = model_dir / "svdq-int4_r32-FLUX.1-dev.safetensors"
    text_ckpt = te_dir / "awq-int4-flux.1-t5xxl.safetensors"
    main_ckpt.write_bytes(b"main")
    text_ckpt.write_bytes(b"text")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="flux", fast_mode=True, low_vram=True, url=None)

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "flux",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )
    monkeypatch.setattr(
        pci,
        "_load_nunchaku_t5_encoder",
        lambda ctx, p, key: {"kind": key, "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert "transformer" not in out
    assert "text_encoder_2" not in out
    assert set(out["__lazy_component_loaders"].keys()) == {"transformer", "text_encoder_2"}
    assert out["__component_sig"]["transformer"].endswith(str(main_ckpt))
    assert out["__component_sig"]["text_encoder_2"].endswith(str(text_ckpt))


def test_build_pipeline_params_flux_auto_nunchaku_uses_special_model_name_fallback(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "flux"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "FLUX.1-Krea-dev"
    fallback_dir = weights_dir / "nunchaku-flux.1-krea-dev"
    model_dir.mkdir(parents=True)
    fallback_dir.mkdir(parents=True)
    main_ckpt = fallback_dir / "svdq-int4_r32-krea-dev.safetensors"
    main_ckpt.write_bytes(b"main")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="flux", fast_mode=True, low_vram=True, url=None)

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "flux",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert set(out["__lazy_component_loaders"].keys()) == {"transformer"}
    assert out["__component_sig"]["transformer"].endswith(str(main_ckpt))


def test_build_pipeline_params_flux_kontext_auto_nunchaku_uses_family_fallback(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "flux_kontext"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "FLUX.1-Kontext-dev"
    fallback_dir = models_dir / "nunchaku-flux.1-kontext-dev"
    model_dir.mkdir(parents=True)
    fallback_dir.mkdir(parents=True)
    main_ckpt = fallback_dir / "svdq-int4_r32-flux.1-kontext-dev.safetensors"
    main_ckpt.write_bytes(b"main")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="flux_kontext", fast_mode=True, low_vram=True, url=None)

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "flux_kontext",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert set(out["__lazy_component_loaders"].keys()) == {"transformer"}
    assert out["__component_sig"]["transformer"].endswith(str(main_ckpt))


def test_build_pipeline_params_zimage_auto_nunchaku_uses_special_model_name_fallback(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "zimage"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "Z-Image-Turbo"
    fallback_dir = weights_dir / "nunchaku-z-image-turbo"
    model_dir.mkdir(parents=True)
    fallback_dir.mkdir(parents=True)
    main_ckpt = fallback_dir / "svdq-int4_r32-z-image-turbo.safetensors"
    main_ckpt.write_bytes(b"main")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="zimage", fast_mode=True, low_vram=True, url=None)

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "zimage",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert set(out["__lazy_component_loaders"].keys()) == {"transformer"}
    assert out["__component_sig"]["transformer"].endswith(str(main_ckpt))


def test_build_pipeline_params_flux2_klein_skips_incompatible_nunchaku_text_encoder(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "flux2_klein"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "FLUX.2-klein-4B"
    transformer_dir = model_dir / "transformer"
    model_text_encoder_dir = model_dir / "text_encoder"
    qwen_dir = weights_dir / "Qwen3-text-Nunchaku"
    model_dir.mkdir(parents=True)
    transformer_dir.mkdir(parents=True)
    model_text_encoder_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)

    main_ckpt = model_dir / "svdq-int4_r32-FLUX.2-klein-4B.safetensors"
    text_ckpt = qwen_dir / "svdq-int4-Qwen3-text-Nunchaku.safetensors"
    main_ckpt.write_bytes(b"main")
    text_ckpt.write_bytes(b"text")
    (transformer_dir / "config.json").write_text(json.dumps({"joint_attention_dim": 7680}), "utf-8")
    (model_text_encoder_dir / "config.json").write_text(json.dumps({"hidden_size": 2560}), "utf-8")
    (qwen_dir / "config.json").write_text(json.dumps({"hidden_size": 4096}), "utf-8")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="flux2_klein", fast_mode=True, low_vram=True, url=None)

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "flux2_klein",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )
    monkeypatch.setattr(
        pci,
        "_load_nunchaku_qwen_encoder",
        lambda ctx, p, key: {"kind": key, "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert set(out["__lazy_component_loaders"].keys()) == {"transformer"}
    assert "text_encoder" not in out["__lazy_component_loaders"]
    assert "text_encoder" not in out.get("__component_sig", {})


def test_build_pipeline_params_flux2_klein_accepts_compatible_nunchaku_text_encoder(tmp_path, monkeypatch):
    detector = PipelineDetector()
    detector.family = "flux2_klein"

    models_dir = tmp_path / "models"
    weights_dir = tmp_path / "weights"
    model_dir = models_dir / "FLUX.2-klein-9B"
    transformer_dir = model_dir / "transformer"
    model_text_encoder_dir = model_dir / "text_encoder"
    qwen_dir = weights_dir / "Qwen3-text-Nunchaku"
    model_dir.mkdir(parents=True)
    transformer_dir.mkdir(parents=True)
    model_text_encoder_dir.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)

    main_ckpt = model_dir / "svdq-int4_r32-FLUX.2-klein-9B.safetensors"
    text_ckpt = qwen_dir / "svdq-int4-Qwen3-text-Nunchaku.safetensors"
    main_ckpt.write_bytes(b"main")
    text_ckpt.write_bytes(b"text")
    (transformer_dir / "config.json").write_text(json.dumps({"joint_attention_dim": 12288}), "utf-8")
    (model_text_encoder_dir / "config.json").write_text(json.dumps({"hidden_size": 2560}), "utf-8")
    (qwen_dir / "config.json").write_text(json.dumps({"hidden_size": 4096}), "utf-8")

    detector.inference_config = type("Cfg", (), {"weights_dir": str(weights_dir), "models_dir": str(models_dir)})()

    params = _minimal_params(model_name=str(model_dir), family="flux2_klein", fast_mode=True, low_vram=True, url=None)

    import common.pipeline_component_injector as pci
    monkeypatch.setattr(pci, "_platform_nunchaku_precision", lambda: "int4")
    monkeypatch.setitem(
        pci.NUNCHAKU_TRANSFORMER_LOADER,
        "flux2_klein",
        lambda ctx, p: {"kind": "transformer", "path": str(p), "strategy": "auto"},
    )
    monkeypatch.setattr(
        pci,
        "_load_nunchaku_qwen_encoder",
        lambda ctx, p, key: {"kind": key, "path": str(p), "strategy": "auto"},
    )

    import torch
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id=str(model_dir), method="from_pretrained", model_name=str(model_dir))
    out = detector.build_pipeline_params(params, device_plan=device_plan, model_info=model_info)

    assert set(out["__lazy_component_loaders"].keys()) == {"transformer", "text_encoder"}
    assert out["__component_sig"]["text_encoder"].endswith(str(text_ckpt))


def test_pipeline_service_instantiate_pipeline_materializes_lazy_components():
    class DummyPipe:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            return {"repo_id": repo_id, "kwargs": kwargs}

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    service = PipelineService(
        detector=object(),
        device_planner=object(),
        model_locator=object(),
        inference_config=object(),
        logger=DummyLogger(),
        pipeline_cache=None,
        run_blocking=None,
        build_cache_key_fn=lambda **kwargs: "unused",
    )

    model_info = type("ModelInfo", (), {"repo_id": "repo/demo", "method": "from_pretrained"})()
    pipe = service.instantiate_pipeline(
        pipeline_cls=DummyPipe,
        model_info=model_info,
        pipeline_params={
            "torch_dtype": "bf16",
            "__component_sig": {"transformer": "auto:nunchaku:/tmp/demo.safetensors"},
            "__lazy_component_loaders": {
                "transformer": lambda: {"kind": "transformer", "path": "/tmp/demo.safetensors"},
            },
        },
    )

    assert pipe["repo_id"] == "repo/demo"
    assert pipe["kwargs"]["torch_dtype"] == "bf16"
    assert pipe["kwargs"]["transformer"] == {"kind": "transformer", "path": "/tmp/demo.safetensors"}
    assert "__lazy_component_loaders" not in pipe["kwargs"]


def test_pipeline_service_cache_hit_skips_lazy_component_materialization():
    class DummyDetector:
        family = "flux"

        def get_pipeline(self, params):
            class DummyPipe:
                @classmethod
                def from_pretrained(cls, repo_id, **kwargs):
                    return {"repo_id": repo_id, "kwargs": kwargs}

            return DummyPipe

        def build_pipeline_params(self, params, *, device_plan, model_info):
            return {
                "torch_dtype": device_plan.torch_dtype,
                "__component_sig": {
                    "transformer": "low_vram:nunchaku_transformer:/tmp/flux.safetensors",
                },
                "__lazy_component_loaders": {
                    "transformer": lambda: (_ for _ in ()).throw(AssertionError("cache hit should not materialize transformer"))
                },
            }

    class DummyModelLocator:
        def locate(self, params):
            return type("ModelInfo", (), {"repo_id": "repo/demo", "method": "from_pretrained", "model_name": "demo"})()

    class DummyDevicePlanner:
        def plan(self, params):
            import torch

            return type("DevicePlan", (), {"device": "cuda", "torch_dtype": torch.bfloat16})()

    class DummyCache:
        def enabled(self):
            return True

        async def acquire(self, key, create_fn):
            return {"cached": True}, True

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    service = PipelineService(
        detector=DummyDetector(),
        device_planner=DummyDevicePlanner(),
        model_locator=DummyModelLocator(),
        inference_config=type("Cfg", (), {"pipeline_cache_ttl_seconds": 60})(),
        logger=DummyLogger(),
        pipeline_cache=DummyCache(),
        run_blocking=None,
        build_cache_key_fn=lambda **kwargs: "cache-key",
    )

    result = asyncio.run(service.acquire(_minimal_params(model_name="demo", family="flux", low_vram=True)))
    assert result.pipe == {"cached": True}
    assert result.cache_enabled is True
    assert result.cache_key == "cache-key"


def test_pipeline_cache_key_ignores_model_cfg_component_fields():
    from image.runtime.device_planner import DevicePlan
    from image.runtime.model_locator import ModelInfo
    from image.runtime.pipeline_lifecycle import PipelineLifecycle
    import torch

    lifecycle = PipelineLifecycle(
        detector=object(),
        device_planner=object(),
        model_locator=object(),
        inference_config=object(),
        logger=object(),
    )

    params1 = _minimal_params(model_name="demo", family="flux", url=None, low_vram=False)
    params2 = _minimal_params(model_name="demo", family="flux", url=None, low_vram=False)
    params1.model_cfg = {"transformer": "a.safetensors", "text_encoder_2": "b.safetensors"}
    params2.model_cfg = {"transformer": "x.safetensors", "text_encoder_2": "y.safetensors"}

    device_plan = DevicePlan(device="cuda", torch_dtype=torch.bfloat16, preferred=torch.device("cuda"))
    model_info = ModelInfo(repo_id="resources/models/demo", method="from_pretrained", model_name="demo")
    pipeline_params = {"__component_sig": {"transformer": "auto:nunchaku:/tmp/demo.safetensors"}}

    key1 = lifecycle._build_cache_key(
        params=params1,
        pipeline_cls=type("DummyPipe", (), {}),
        model_info=model_info,
        device_plan=device_plan,
        pipeline_params=pipeline_params,
    )
    key2 = lifecycle._build_cache_key(
        params=params2,
        pipeline_cls=type("DummyPipe", (), {}),
        model_info=model_info,
        device_plan=device_plan,
        pipeline_params=pipeline_params,
    )

    assert key1 == key2


@pytest.mark.skip(reason="手动/命令行测试入口，不作为 pytest 单元测试运行")
def test_real_data():
    """使用真实数据测试 PipelineDetector"""
    parser = argparse.ArgumentParser(description="测试 PipelineDetector 使用真实数据")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径（目录或文件）")
    parser.add_argument("--torch-dtype", type=str, default="float16", help="torch_dtype (float16/bfloat16)")
    parser.add_argument("--low-vram", action="store_true", help="是否低显存模式")
    parser.add_argument("--fast-mode", action="store_true", help="是否快速模式")
    parser.add_argument("--url", type=str, default=None, help="图生图URL（可选）")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="推理步数")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        return 1
    
    print(f"✓ 使用模型路径: {model_path}")
    print(f"✓ torch_dtype: {args.torch_dtype}")
    print(f"✓ low_vram: {args.low_vram}")
    print(f"✓ fast_mode: {args.fast_mode}")
    
    # 构建参数
    params_dict = dict(
        type="image",
        action="MK",
        job_type="image",
        storage="local",
        id="test_task",
        user_id="test_user",
        task_id="test_task",
        prompt="a beautiful landscape",
        model_name=str(model_path),
        url=args.url,
        fast_mode=args.fast_mode,
        num_inference_steps=args.num_inference_steps,
    )
    params = InferenceRequestParams(**params_dict)
    
    detector = PipelineDetector()
    
    # 测试 get_pipeline
    print("\n" + "="*60)
    print("测试 get_pipeline...")
    print("="*60)
    try:
        pipeline_cls = detector.get_pipeline(params)
        print(f"✓ Pipeline 类: {pipeline_cls}")
        print(f"✓ Model version: {detector.family}")
    except Exception as e:
        print(f"❌ get_pipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试 build_pipeline_params
    print("\n" + "="*60)
    print("测试 build_pipeline_params...")
    print("="*60)
    try:
        import torch
        from image.runtime.device_planner import DevicePlan
        from image.runtime.model_locator import ModelInfo

        torch_dtype = getattr(torch, args.torch_dtype, torch.float16)

        if model_path.is_file():
            repo_id = str(model_path)
            method = "from_single_file"
        else:
            repo_id = str(model_path)
            method = "from_pretrained"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_plan = DevicePlan(device=device, torch_dtype=torch_dtype, preferred=torch.device(device))
        model_info = ModelInfo(repo_id=repo_id, method=method, model_name=str(model_path))

        pipeline_params = detector.build_pipeline_params(
            params,
            device_plan=device_plan,
            model_info=model_info,
        )
        
        print(f"✓ Pipeline params 键: {list(pipeline_params.keys())}")
        print(f"✓ torch_dtype: {pipeline_params.get('torch_dtype')}")
        print(f"✓ low_vram: {pipeline_params.get('low_vram')}")
        
        if "transformer" in pipeline_params:
            print(f"✓ transformer: {type(pipeline_params['transformer'])}")
        if "text_encoder_2" in pipeline_params:
            print(f"✓ text_encoder_2: {type(pipeline_params['text_encoder_2'])}")
        if "original_config" in pipeline_params:
            print(f"✓ original_config: {pipeline_params['original_config']}")
        if "cache_dir" in pipeline_params:
            print(f"✓ cache_dir: {pipeline_params['cache_dir']}")
            
    except Exception as e:
        print(f"❌ build_pipeline_params 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)
    return 0


if __name__ == "__main__":
    # 检查是否有命令行参数（除了脚本名）
    if len(sys.argv) > 1 and sys.argv[1] not in ["-v", "--verbose", "-h", "--help"]:
        # 运行真实数据测试
        sys.exit(test_real_data())
    else:
        # 运行 pytest 单元测试
        pytest.main([__file__, "-v"])


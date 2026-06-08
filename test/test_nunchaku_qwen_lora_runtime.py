from __future__ import annotations

from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

import common.pipeline_component_injector as pci
from image.runtime.lora_manager import unload_loras_from_pipe


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_load_nunchaku_transformer_qwen_defaults_offload_false(monkeypatch):
    captured: dict[str, object] = {}

    class DummyTransformer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = dict(kwargs)
            return {"path": path, **kwargs}

    pkg_nunchaku = types.ModuleType("nunchaku")
    pkg_models = types.ModuleType("nunchaku.models")
    pkg_transformers = types.ModuleType("nunchaku.models.transformers")
    mod_qwen = types.ModuleType("nunchaku.models.transformers.transformer_qwenimage")
    mod_qwen.NunchakuQwenImageTransformer2DModel = DummyTransformer

    monkeypatch.setitem(sys.modules, "nunchaku", pkg_nunchaku)
    monkeypatch.setitem(sys.modules, "nunchaku.models", pkg_models)
    monkeypatch.setitem(sys.modules, "nunchaku.models.transformers", pkg_transformers)
    monkeypatch.setitem(sys.modules, "nunchaku.models.transformers.transformer_qwenimage", mod_qwen)

    ctx = types.SimpleNamespace(
        logger=_DummyLogger(),
        device_plan=types.SimpleNamespace(torch_dtype="bf16"),
    )

    out = pci._load_nunchaku_transformer_qwen(ctx, Path("/tmp/qwen-edit.safetensors"))

    assert out["path"] == "/tmp/qwen-edit.safetensors"
    assert captured["kwargs"]["pin_memory"] is False
    assert captured["kwargs"]["offload"] is False
    assert captured["kwargs"]["torch_dtype"] == "bf16"


def test_unload_loras_from_pipe_resets_nunchaku_transformer():
    class DummyTransformer:
        __module__ = "nunchaku.models.transformers.transformer_qwenimage"

        def __init__(self):
            self.reset_calls = 0

        def reset_lora(self):
            self.reset_calls += 1

    pipe = types.SimpleNamespace(transformer=DummyTransformer())

    unload_loras_from_pipe(pipe, "qwen.edit", logger=_DummyLogger())

    assert pipe.transformer.reset_calls == 1

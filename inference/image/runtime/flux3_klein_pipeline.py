"""FLUX.3 Image Klein 走 diffusers ModularPipeline。

量化 transformer / 文本编码器仍由组件注入层传入，这里只补 ModularPipeline 和普通
pipeline 的差别：去掉 prompt upsampler，并把 output="images" 收成 result.images。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_modular_blocks(pipeline_dir: Path) -> Any:
    from diffusers import ModularPipeline

    return ModularPipeline.from_pretrained(str(pipeline_dir), local_files_only=True).blocks


def _load_components(pipe: Any, names: list[str], torch_dtype: Any, pipeline_dir: Path) -> None:
    # 索引里的仓库名可能仍是 Hub id（含 Qwen3-VL）。本地目录必须作为 load_components 参数传入。
    load_kwargs = {
        "local_files_only": True,
        "pretrained_model_name_or_path": str(pipeline_dir),
        "revision": None,
        "torch_dtype": torch_dtype,
    }
    try:
        pipe.load_components(names, **load_kwargs)
    except TypeError:
        load_kwargs.pop("torch_dtype")
        pipe.load_components(names, dtype=torch_dtype, **load_kwargs)


class Flux3KleinPipeline:
    _pipeline_base_class_name = "Flux3KleinPipeline"

    def __init__(self, pipe: Any):
        self._pipe = pipe

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        transformer: Any = None,
        text_encoder: Any = None,
        torch_dtype: Any = None,
        **kwargs: Any,
    ):
        del kwargs
        pipeline_dir = Path(str(repo_id)).expanduser()
        if not (pipeline_dir / "modular_model_index.json").is_file():
            raise FileNotFoundError(f"flux3_klein 缺少 modular_model_index.json: {pipeline_dir}")

        blocks = _load_modular_blocks(pipeline_dir)
        sub = getattr(blocks, "sub_blocks", None)
        if sub is not None and "prompt_upsampler" in sub:
            sub.pop("prompt_upsampler")
        pipe = blocks.init_pipeline(str(pipeline_dir))
        if torch_dtype is None:
            import torch

            torch_dtype = torch.bfloat16
        skip = {name for name, value in (("transformer", transformer), ("text_encoder", text_encoder)) if value is not None}
        names = [
            str(name)
            for name in (getattr(pipe, "_component_specs", None) or {})
            if "prompt_upsampler" not in str(name) and str(name) not in skip
        ]
        if names:
            _load_components(pipe, names, torch_dtype, pipeline_dir)
        if transformer is not None:
            pipe.update_components(transformer=transformer)
        if text_encoder is not None:
            pipe.update_components(text_encoder=text_encoder)
        return cls(pipe)

    def to(self, device: Any, *args: Any, **kwargs: Any):
        self._pipe.to(device, *args, **kwargs)
        return self

    def __call__(
        self,
        prompt: str,
        image: Any = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        generator: Any = None,
        callback_on_step_end: Any = None,
        **kwargs: Any,
    ):
        del kwargs
        call_kwargs: dict[str, Any] = {"prompt": prompt, "output": "images"}
        if image is not None:
            call_kwargs["image"] = image
        if height is not None:
            call_kwargs["height"] = int(height)
        if width is not None:
            call_kwargs["width"] = int(width)
        if num_inference_steps is not None:
            call_kwargs["num_inference_steps"] = int(num_inference_steps)
        if generator is not None:
            call_kwargs["generator"] = generator
        restore = _hook_cancel(self._pipe, callback_on_step_end)
        try:
            raw = self._pipe(**call_kwargs)
        finally:
            if restore is not None:
                restore()
        images = getattr(raw, "images", None)
        if images:
            return raw
        if isinstance(raw, dict) and raw.get("images") is not None:
            images = raw["images"]
            return SimpleNamespace(images=images if isinstance(images, list) else [images])
        if isinstance(raw, (list, tuple)):
            return SimpleNamespace(images=list(raw))
        return SimpleNamespace(images=[raw])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipe, name)


def _hook_cancel(pipe: Any, callback: Any):
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None or callback is None or not callable(getattr(scheduler, "step", None)):
        return None
    original = scheduler.step
    index = {"i": 0}

    def step(*args: Any, **kwargs: Any):
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        callback(pipe, index["i"], timestep, {})
        index["i"] += 1
        return original(*args, **kwargs)

    scheduler.step = step
    return lambda: setattr(scheduler, "step", original)

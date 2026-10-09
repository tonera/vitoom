"""diffusers Anima 的加载适配器。

目录：AnimaModularPipeline.from_pretrained，再 load_components。
单文件：只替换 transformer，其余组件来自 models/Anima-Base-v1.0-Diffusers。
快速模式的量化 transformer 由组件注入层传入，这里不再加载目录里的 bf16 transformer。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from common.anima_checkpoint import (
    is_anima_diffusers_dir,
    load_anima_transformer_from_single_file,
    resolve_anima_diffusers_base,
)

_COMPONENT_NAMES = (
    "text_encoder",
    "tokenizer",
    "t5_tokenizer",
    "text_conditioner",
    "scheduler",
    "vae",
    "transformer",
)
_MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json", "spiece.model")


def _component_has_local_files(component_dir: Path, name: str) -> bool:
    if not component_dir.is_dir():
        return False
    filenames = [path.name for path in component_dir.iterdir()]
    if name in {"tokenizer", "t5_tokenizer"}:
        return any(marker in filenames for marker in _TOKENIZER_MARKERS)
    if name == "scheduler":
        return "scheduler_config.json" in filenames
    return any(filename.endswith(_MODEL_WEIGHT_SUFFIXES) or filename.endswith(".index.json") for filename in filenames)


def _pin_specs_to_local(pipe: Any, pipeline_dir: Path, names: tuple[str, ...] | list[str]) -> None:
    missing: list[str] = []
    for name in names:
        spec = pipe._component_specs.get(name)
        if spec is None:
            missing.append(f"{name}: pipeline 里没有这个组件")
            continue
        subfolder = getattr(spec, "subfolder", None) or name
        component_dir = pipeline_dir / subfolder
        if not _component_has_local_files(component_dir, name):
            missing.append(f"{name}: {component_dir}")
            continue
        spec.pretrained_model_name_or_path = str(pipeline_dir)
        if hasattr(spec, "revision"):
            spec.revision = None
    if missing:
        raise FileNotFoundError(
            "Anima diffusers 目录缺少组件，不会访问 Hugging Face：\n" + "\n".join(missing)
        )


def _configured_model_roots() -> list[Path]:
    try:
        from common.config_loader import load_inference_config

        cfg = load_inference_config()
    except Exception:
        return []
    roots: list[Path] = []
    for attr in ("models_dir", "weights_dir"):
        raw = str(getattr(cfg, attr, "") or "").strip()
        if raw:
            roots.append(Path(raw).expanduser())
    return roots


def _load_components(pipe: Any, names: list[str], torch_dtype: Any, pipeline_dir: Path) -> None:
    # modular_model_index.json 里常仍写着 Hub 仓库名。和 test_anima_diffusers.py 一样，
    # 组件路径必须作为 load_components 参数传入，不能只改 spec。
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


def _apply_guidance(pipe: Any, guidance_scale: float) -> None:
    guider = getattr(pipe, "guider", None)
    if guider is not None and hasattr(guider, "guidance_scale"):
        guider.guidance_scale = float(guidance_scale)
        return
    from diffusers.guiders import ClassifierFreeGuidance

    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(guidance_scale)))


def _install_step_callback(pipe: Any, callback: Any):
    """Anima 的去噪循环没有 callback_on_step_end。挂在 scheduler.step 上，每步检查一次取消。"""
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

    def restore() -> None:
        scheduler.step = original

    return restore


class AnimaDiffusersPipeline:
    """包一层，让现有 PipelineService 的 from_pretrained / from_single_file 能跑 Anima。"""

    _pipeline_base_class_name = "AnimaModularPipeline"

    def __init__(self, pipe: Any):
        self._pipe = pipe

    @classmethod
    def from_pretrained(cls, repo_id: str, transformer: Any = None, torch_dtype: Any = None, **kwargs: Any):
        from diffusers import AnimaModularPipeline

        pipeline_dir = Path(str(repo_id)).expanduser()
        if not is_anima_diffusers_dir(pipeline_dir):
            raise FileNotFoundError(f"不是 Anima diffusers 目录（缺少 modular_model_index.json）: {pipeline_dir}")

        pipe = AnimaModularPipeline.from_pretrained(str(pipeline_dir), local_files_only=True)
        names = list(_COMPONENT_NAMES)
        if transformer is not None:
            names = [name for name in names if name != "transformer"]
        _pin_specs_to_local(pipe, pipeline_dir, names)
        if torch_dtype is None:
            import torch

            torch_dtype = torch.bfloat16
        _load_components(pipe, names, torch_dtype, pipeline_dir)
        if transformer is not None:
            if hasattr(pipe, "update_components"):
                pipe.update_components(transformer=transformer)
            else:
                pipe.transformer = transformer
        return cls(pipe)

    @classmethod
    def from_single_file(cls, repo_id: str, transformer: Any = None, torch_dtype: Any = None, **kwargs: Any):
        ckpt = Path(str(repo_id)).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Anima 单文件不存在: {ckpt}")
        extra_roots = _configured_model_roots()
        for key in ("models_dir", "weights_dir"):
            raw = kwargs.get(key)
            if raw:
                extra_roots.append(Path(str(raw)).expanduser())
        base_dir = resolve_anima_diffusers_base(ckpt, extra_roots=extra_roots)
        if torch_dtype is None:
            import torch

            torch_dtype = torch.bfloat16
        if transformer is None:
            transformer = load_anima_transformer_from_single_file(ckpt, torch_dtype=torch_dtype)
        return cls.from_pretrained(str(base_dir), transformer=transformer, torch_dtype=torch_dtype)

    def to(self, device: Any, *args: Any, **kwargs: Any):
        self._pipe.to(device, *args, **kwargs)
        return self

    def _apply_flow_shift(self, flow_shift: float | None) -> None:
        scheduler = getattr(self._pipe, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "register_to_config"):
            return
        if not hasattr(self, "_anima_base_scheduler_shift"):
            self._anima_base_scheduler_shift = getattr(getattr(scheduler, "config", None), "shift", None)
        target = self._anima_base_scheduler_shift if flow_shift is None else float(flow_shift)
        if target is None:
            return
        scheduler.register_to_config(shift=float(target))

    def __call__(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        max_sequence_length: int | None = None,
        generator: Any = None,
        output_type: str = "pil",
        flow_shift: float | None = None,
        callback_on_step_end: Any = None,
        callback_on_step_end_tensor_inputs: Any = None,
        **kwargs: Any,
    ):
        del callback_on_step_end_tensor_inputs
        if guidance_scale is not None:
            _apply_guidance(self._pipe, float(guidance_scale))
        self._apply_flow_shift(flow_shift)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "generator": generator,
            "output_type": output_type,
        }
        call_kwargs.update(kwargs)
        if height is not None:
            call_kwargs["height"] = int(height)
        if width is not None:
            call_kwargs["width"] = int(width)
        if num_inference_steps is not None:
            call_kwargs["num_inference_steps"] = int(num_inference_steps)
        if max_sequence_length is not None:
            call_kwargs["max_sequence_length"] = int(max_sequence_length)

        restore_callback = _install_step_callback(self._pipe, callback_on_step_end)
        try:
            return self._pipe(**call_kwargs)
        finally:
            if restore_callback is not None:
                restore_callback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipe, name)

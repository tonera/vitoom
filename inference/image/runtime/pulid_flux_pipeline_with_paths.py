"""
Project-local wrapper for nunchaku's PuLIDFluxPipeline.

Why this exists:
- Upstream `nunchaku.pipeline.pipeline_flux_pulid.PuLIDFluxPipeline` constructs `PuLIDPipeline`
  inside `__init__` without forwarding `facexlib_dirpath/insightface_dirpath/pulid_path/eva_clip_path`,
  causing unwanted downloads when local weights are available elsewhere.

This subclass allows passing those paths through `from_pretrained(...)` without modifying the
third-party package.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Optional, Union

import torch
from diffusers import FluxPipeline

from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline, PuLIDPipeline  # type: ignore


class PuLIDFluxPipelineWithPaths(PuLIDFluxPipeline):
    """
    `PuLIDFluxPipeline` variant that forwards model directory paths into `PuLIDPipeline`.

    Important: We intentionally DO NOT call `super().__init__()` here because the upstream
    `PuLIDFluxPipeline.__init__` would create `PuLIDPipeline` with default paths (and may download).
    """

    @classmethod
    def from_pretrained(cls, pretrained_load_name_or_path: str | PathLike, *args: Any, **kwargs: Any):
        """
        Custom `from_pretrained` that avoids diffusers' kwarg type-validation bug for custom init kwargs.

        Strategy:
        - Pop PuLID-specific kwargs first (so diffusers/FluxPipeline won't see them).
        - Load a vanilla `FluxPipeline` via `FluxPipeline.from_pretrained(...)`.
        - Re-wrap the loaded components into `cls(...)`, passing the PuLID-specific kwargs.
        """
        pulid_device = kwargs.pop("pulid_device", "cuda")
        onnx_provider = kwargs.pop("onnx_provider", "gpu")

        # If caller didn't specify weight_dtype, default to `torch_dtype` (if provided) else bf16.
        torch_dtype = kwargs.get("torch_dtype", None)
        weight_dtype = kwargs.pop("weight_dtype", torch_dtype if torch_dtype is not None else torch.bfloat16)

        pulid_path = kwargs.pop("pulid_path", "guozinan/PuLID/pulid_flux_v0.9.1.safetensors")
        eva_clip_path = kwargs.pop("eva_clip_path", "QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt")
        insightface_dirpath = kwargs.pop("insightface_dirpath", None)
        facexlib_dirpath = kwargs.pop("facexlib_dirpath", None)

        base = FluxPipeline.from_pretrained(pretrained_load_name_or_path, *args, **kwargs)

        pipe = cls(
            scheduler=base.scheduler,
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            text_encoder_2=getattr(base, "text_encoder_2", None),
            tokenizer_2=getattr(base, "tokenizer_2", None),
            transformer=base.transformer,
            image_encoder=getattr(base, "image_encoder", None),
            feature_extractor=getattr(base, "feature_extractor", None),
            pulid_device=pulid_device,
            weight_dtype=weight_dtype,
            onnx_provider=onnx_provider,
            pulid_path=pulid_path,
            eva_clip_path=eva_clip_path,
            insightface_dirpath=insightface_dirpath,
            facexlib_dirpath=facexlib_dirpath,
        )

        # Help GC: we only need the modules we re-used above.
        del base
        return pipe

    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        image_encoder=None,
        feature_extractor=None,
        pulid_device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        onnx_provider: str = "gpu",
        # NOTE: Keep annotations conservative for diffusers' `from_pretrained` kwarg validation.
        # Some diffusers versions choke on PEP604 unions with `os.PathLike[str]` (generic) and can raise KeyError.
        pulid_path: Union[str, PathLike] = "guozinan/PuLID/pulid_flux_v0.9.1.safetensors",
        eva_clip_path: Union[str, PathLike] = "QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt",
        insightface_dirpath: Optional[Union[str, PathLike]] = None,
        facexlib_dirpath: Optional[Union[str, PathLike]] = None,
    ):
        # Match upstream init behavior: call FluxPipeline init to register modules.
        FluxPipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        self.pulid_device = torch.device(pulid_device)
        self.weight_dtype = weight_dtype
        self.onnx_provider = onnx_provider

        # Create PuLIDPipeline with user-provided paths.
        self.pulid_model = PuLIDPipeline(
            dit=self.transformer,  # directly mutates transformer with pulid_ca
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
            pulid_path=pulid_path,
            eva_clip_path=eva_clip_path,
            insightface_dirpath=insightface_dirpath,
            facexlib_dirpath=facexlib_dirpath,
        )

    def release(self) -> None:
        """
        best-effort 释放显存/句柄：
        - 尝试把 pipeline/pulid_model 搬回 CPU（降低仍被引用的 CUDA allocations）
        - 断开 pulid_model 引用，帮助 GC 回收引用环
        """
        # diffusers/accelerate hooks：尽量先解除
        try:
            if hasattr(self, "maybe_free_model_hooks"):
                self.maybe_free_model_hooks()
        except Exception:
            pass
        try:
            if hasattr(self, "_remove_all_hooks"):
                self._remove_all_hooks()  # type: ignore[attr-defined]
        except Exception:
            pass

        # pulid_model 可能包含 insightface/onnxruntime 资源：优先调用其 release（若存在）
        try:
            pm = getattr(self, "pulid_model", None)
            if pm is not None and hasattr(pm, "release") and callable(getattr(pm, "release")):
                pm.release()  # type: ignore[call-arg]
        except Exception:
            pass

        # teacache 等 forward 缓存可能把大 tensor 留在 transformer 上：best-effort 清理
        try:
            tr = getattr(self, "transformer", None)
            if tr is not None:
                for attr in (
                    "previous_modulated_input",
                    "previous_residual",
                    "accumulated_rel_l1_distance",
                    "cnt",
                ):
                    try:
                        if hasattr(tr, attr):
                            setattr(tr, attr, None)
                    except Exception:
                        pass
        except Exception:
            pass

        # 尽量把 torch 模型搬回 CPU
        try:
            if hasattr(self, "to"):
                self.to("cpu")
        except Exception:
            pass
        try:
            pm = getattr(self, "pulid_model", None)
            if pm is not None and hasattr(pm, "to"):
                pm.to("cpu")  # type: ignore[call-arg]
        except Exception:
            pass

        # 断开引用（让 PipelineLifecycle.release_pipeline 的 gc/empty_cache 更稳定）
        try:
            setattr(self, "pulid_model", None)
        except Exception:
            pass



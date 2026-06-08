from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import torch

from .inferencer import AnimaInferencer, AnimaPaths, AnimaRunConfig
from .tokenizers import Qwen3LocalPaths


class AnimaPipelineOutput:
    """
    兼容 diffusers 输出形态：DiffusionHandler 只依赖 `.images`。
    """

    def __init__(self, images: list[Any]):
        self.images = images


def _resolve_path(root: Optional[Path], value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("empty path")
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if root is not None:
        cand = root / p
        if cand.exists():
            return str(cand)
    return str(p)


def _parse_anima_paths(raw: Union[AnimaPaths, dict], *, root: Optional[Path]) -> AnimaPaths:
    if isinstance(raw, AnimaPaths):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"anima_paths must be AnimaPaths or dict, got {type(raw)}")

    qwen3_raw = raw.get("qwen3")
    if not isinstance(qwen3_raw, dict):
        raise ValueError("anima_paths.qwen3 must be a dict")

    qwen3 = Qwen3LocalPaths(
        model_or_weights_path=_resolve_path(root, qwen3_raw.get("model_or_weights_path")),
        config_dir=_resolve_path(root, qwen3_raw.get("config_dir")),
        tokenizer_dir=_resolve_path(root, qwen3_raw.get("tokenizer_dir")),
    )
    return AnimaPaths(
        dit_path=_resolve_path(root, raw.get("dit_path")),
        vae_path=_resolve_path(root, raw.get("vae_path")),
        qwen3=qwen3,
        t5_tokenizer_dir=_resolve_path(root, raw.get("t5_tokenizer_dir")),
    )


class AnimaPipeline:
    """
    Anima 的“diffusers 适配器”：
    - 支持 `from_pretrained` / `from_single_file`，便于接入现有 PipelineService。
    - 支持 `__call__` 返回 `.images`，便于复用 DiffusionHandler 的迭代/回传逻辑。

    约定（推荐）：
    - `repo_id` 指向一个模型目录（model bundle）。
    - 目录下可放一个 `anima_paths.json`（或 `anima.json`）作为组件清单；
      若未提供，则要求通过 kwargs 传入 `anima_paths`。
    """

    # 便于 inference_params_builder 识别“原始类名”
    _pipeline_base_class_name = "AnimaPipeline"

    def __init__(self, inferencer: AnimaInferencer, *, model_root: Optional[str] = None):
        self._inf = inferencer
        self._model_root = model_root

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> "AnimaPipeline":
        root = Path(str(repo_id)).expanduser()
        root_dir = root if root.is_dir() else root.parent

        # 1) paths
        anima_paths = kwargs.get("anima_paths")
        if anima_paths is None:
            # 目录内 manifest（可选）
            manifest = None
            for name in ("anima_paths.json", "anime_paths.json", "anima.json"):
                p = root_dir / name
                if p.is_file():
                    manifest = p
                    break
            if manifest is not None:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                anima_paths = data.get("anima_paths") if isinstance(data, dict) else data

        if anima_paths is None:
            raise ValueError(
                "AnimaPipeline requires `anima_paths`.\n"
                "- 推荐：在模型目录放置 `anima_paths.json`（或 `anime_paths.json` / `anima.json`）\n"
                "- 或者：通过 request.model_config 传入 anima_paths（由组件注入层转换并透传）。"
            )
        paths = _parse_anima_paths(anima_paths, root=root_dir)

        # 2) runtime options（均为可选）
        device = str(kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch_dtype = kwargs.get("torch_dtype")
        dtype = torch_dtype if torch_dtype is not None else kwargs.get("dtype", "bf16")

        text_device = kwargs.get("text_device")
        text_dtype = kwargs.get("text_dtype")
        attn_mode = str(kwargs.get("attn_mode") or "torch")
        split_attn = bool(kwargs.get("split_attn", False))
        vae_spatial_chunk_size = kwargs.get("vae_spatial_chunk_size")
        vae_disable_cache = bool(kwargs.get("vae_disable_cache", False))
        enable_block_swap = kwargs.get("enable_block_swap")
        dit_loading_device = kwargs.get("dit_loading_device")
        qwen3_loading_device = kwargs.get("qwen3_loading_device")
        pretouch_cpu_tensors_before_to_cuda = bool(kwargs.get("pretouch_cpu_tensors_before_to_cuda", False))

        inf = AnimaInferencer(
            paths,
            device=device,
            dtype=dtype,
            text_device=str(text_device) if text_device else None,
            text_dtype=text_dtype,
            attn_mode=attn_mode,
            split_attn=split_attn,
            vae_spatial_chunk_size=vae_spatial_chunk_size,
            vae_disable_cache=vae_disable_cache,
            enable_block_swap=enable_block_swap,
            dit_loading_device=str(dit_loading_device) if dit_loading_device else None,
            qwen3_loading_device=str(qwen3_loading_device) if qwen3_loading_device else None,
            pretouch_cpu_tensors_before_to_cuda=pretouch_cpu_tensors_before_to_cuda,
        )
        return cls(inf, model_root=str(root_dir))

    @classmethod
    def from_single_file(cls, repo_id: str, **kwargs: Any) -> "AnimaPipeline":
        # 兼容：把 repo_id 当成 dit_path（仍要求其它组件来自 anima_paths / manifest / kwargs）
        if "anima_paths" not in kwargs and "dit_path" not in kwargs:
            kwargs["dit_path"] = repo_id
        return cls.from_pretrained(repo_id, **kwargs)

    def to(self, device: str) -> "AnimaPipeline":
        """
        兼容 PipelineLifecycle.move_to_device：best-effort 迁移到目标 device。
        注：若在 from_pretrained 阶段已按最终 device 加载，这里基本是 no-op。
        """
        try:
            dev = torch.device(str(device))
        except Exception:
            return self

        inf = getattr(self, "_inf", None)
        if inf is None:
            return self

        # 统一迁移：把 text/dit/vae 都迁到同一 device（最少 surprise）
        try:
            inf.device = dev
            inf.text_device = dev
        except Exception:
            pass

        for attr in ("dit", "vae", "qwen3_text_encoder"):
            m = getattr(inf, attr, None)
            if m is None:
                continue
            try:
                if hasattr(m, "to"):
                    m.to(dev)
            except Exception:
                pass
        return self

    def enable_model_cpu_offload(self) -> bool:
        """
        兼容 PipelineLifecycle.enable_cpu_offload：
        Anima 没有 accelerate hooks，这里用“全模型搬回 CPU”作为保守回退。
        """
        try:
            self.to("cpu")
            return True
        except Exception:
            return False

    def enable_sequential_cpu_offload(self) -> bool:
        return self.enable_model_cpu_offload()

    def __call__(self, prompt: str, **kwargs: Any) -> AnimaPipelineOutput:
        """
        接受 diffusers 风格 kwargs（由 inference_params_builder 产出）并映射到 AnimaRunConfig。
        """
        negative_prompt = str(kwargs.get("negative_prompt") or "")
        height = int(kwargs.get("height", 1024) or 1024)
        width = int(kwargs.get("width", 1024) or 1024)
        steps = int(kwargs.get("num_inference_steps", 40) or 40)
        cfg = float(kwargs.get("guidance_scale", 4.5) or 4.5)
        flow_shift = float(kwargs.get("flow_shift", 3.0) or 3.0)
        qwen3_max_len = int(kwargs.get("qwen3_max_len", 512) or 512)
        t5_max_len = int(kwargs.get("t5_max_len", 512) or 512)

        seed: Optional[int] = None
        gen = kwargs.get("generator")
        try:
            if isinstance(gen, torch.Generator):
                seed = int(gen.initial_seed())
        except Exception:
            seed = None

        config = AnimaRunConfig(
            height=height,
            width=width,
            steps=steps,
            cfg=cfg,
            flow_shift=flow_shift,
            seed=seed,
            qwen3_max_len=qwen3_max_len,
            t5_max_len=t5_max_len,
        )

        img = self._inf.generate(prompt, negative_prompt=negative_prompt, config=config)
        return AnimaPipelineOutput([img])

    def release(self) -> None:
        """
        供 PipelineLifecycle.release_pipeline 调用：尽量释放显存/引用。
        """
        try:
            self.to("cpu")
        except Exception:
            pass
        try:
            inf = getattr(self, "_inf", None)
            if inf is not None:
                for attr in ("dit", "vae", "qwen3_text_encoder", "qwen3_tokenizer", "t5_tokenizer"):
                    try:
                        setattr(inf, attr, None)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self._inf = None  # type: ignore[assignment]
        except Exception:
            pass

    def export_anima_paths(self) -> Optional[dict]:
        """
        便于排障/落盘：把当前 AnimaPaths 导出为 dict（尽量不丢信息）。
        """
        inf = getattr(self, "_inf", None)
        paths = getattr(inf, "paths", None) if inf is not None else None
        if isinstance(paths, AnimaPaths):
            try:
                d = asdict(paths)
                # Qwen3LocalPaths 也是 dataclass；asdict 会递归
                return d
            except Exception:
                return None
        return None


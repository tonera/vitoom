"""把 Anima 单文件 DiT 转成 diffusers 的 CosmosTransformer3DModel 权重。

Civitai / ComfyUI 用 model.diffusion_model.*，sd-scripts 用 net.*。
llm_adapter 留在基座 text_conditioner 里，这里丢掉，只保留 transformer。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANIMA_DIFFUSERS_BASE_DIRNAME = "Anima-Base-v1.0-Diffusers"

_COSMOS2_RENAME = (
    ("t_embedder.1", "time_embed.t_embedder"),
    ("t_embedding_norm", "time_embed.norm"),
    ("blocks", "transformer_blocks"),
    ("adaln_modulation_self_attn.1", "norm1.linear_1"),
    ("adaln_modulation_self_attn.2", "norm1.linear_2"),
    ("adaln_modulation_cross_attn.1", "norm2.linear_1"),
    ("adaln_modulation_cross_attn.2", "norm2.linear_2"),
    ("adaln_modulation_mlp.1", "norm3.linear_1"),
    ("adaln_modulation_mlp.2", "norm3.linear_2"),
    ("self_attn", "attn1"),
    ("cross_attn", "attn2"),
    ("q_proj", "to_q"),
    ("k_proj", "to_k"),
    ("v_proj", "to_v"),
    ("output_proj", "to_out.0"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("mlp.layer1", "ff.net.0.proj"),
    ("mlp.layer2", "ff.net.2"),
    ("x_embedder.proj.1", "patch_embed.proj"),
    ("final_layer.adaln_modulation.1", "norm_out.linear_1"),
    ("final_layer.adaln_modulation.2", "norm_out.linear_2"),
    ("final_layer.linear", "proj_out"),
)
_DROP_SUBSTRINGS = (
    "accum_video_sample_counter",
    "accum_image_sample_counter",
    "accum_iteration",
    "accum_train_in_hours",
    "pos_embedder.seq",
    "pos_embedder.dim_spatial_range",
    "pos_embedder.dim_temporal_range",
    "_extra_state",
)


def is_anima_diffusers_dir(path: Path) -> bool:
    index_path = path / "modular_model_index.json"
    if not index_path.is_file():
        return False
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    class_name = str(data.get("_class_name") or "")
    blocks_name = str(data.get("_blocks_class_name") or "")
    return class_name == "AnimaModularPipeline" or blocks_name == "AnimaAutoBlocks"


def resolve_anima_diffusers_base(ckpt_path: Path, *, extra_roots: list[Path] | None = None) -> Path:
    """从单文件上级目录，以及 models_dir / weights_dir，找 Anima-Base-v1.0-Diffusers。"""
    start = Path(ckpt_path).expanduser()
    parents = [start.parent, *start.parents]
    candidates = [parent / ANIMA_DIFFUSERS_BASE_DIRNAME for parent in parents[:6]]
    for root in extra_roots or []:
        candidates.append(Path(root).expanduser() / ANIMA_DIFFUSERS_BASE_DIRNAME)

    seen: set[str] = set()
    searched: list[str] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        searched.append(key)
        if is_anima_diffusers_dir(candidate):
            return candidate
    raise FileNotFoundError(
        "Anima 单文件需要基座目录 "
        f"{ANIMA_DIFFUSERS_BASE_DIRNAME}（text encoder / VAE / tokenizer），没有找到。已查找: {', '.join(searched)}"
    )


def normalize_anima_dit_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    keys = list(state_dict)
    if any(key.startswith("model.diffusion_model.") for key in keys):
        prefix = "model.diffusion_model."
    elif any(key.startswith("diffusion_model.") for key in keys):
        prefix = "diffusion_model."
    elif any(key.startswith("net.") for key in keys):
        prefix = "net."
    else:
        prefix = ""
    if not prefix:
        return dict(state_dict)
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


def anima_transformer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """丢掉 llm_adapter，把剩余 DiT 键名改成 CosmosTransformer3DModel。"""
    normalized = normalize_anima_dit_keys(state_dict)
    renamed: dict[str, Any] = {}
    for key, value in normalized.items():
        if key.startswith("llm_adapter.") or key == "llm_adapter":
            continue
        if any(part in key for part in _DROP_SUBSTRINGS):
            continue
        new_key = key
        for old, new in _COSMOS2_RENAME:
            new_key = new_key.replace(old, new)
        if new_key in renamed:
            raise ValueError(f"Anima transformer 键名冲突: {key} -> {new_key}")
        renamed[new_key] = value
    if not any(key.startswith("transformer_blocks.") for key in renamed):
        raise ValueError("Anima 权重里没有 transformer blocks，无法当作 DiT 单文件")
    return renamed


def infer_num_layers(state_dict: dict[str, Any]) -> int:
    indices = []
    for key in state_dict:
        if not key.startswith("transformer_blocks."):
            continue
        index = key.split(".", 2)[1]
        if index.isdigit():
            indices.append(int(index))
    if not indices:
        raise ValueError("转换后的权重里没有 transformer_blocks.*")
    count = max(indices) + 1
    missing = [i for i in range(count) if i not in set(indices)]
    if missing:
        raise ValueError(f"Anima transformer blocks 不连续，缺少: {missing[:10]}")
    return count


def load_anima_transformer_from_single_file(ckpt_path: str | Path, *, torch_dtype: Any):
    from accelerate import init_empty_weights
    from diffusers import CosmosTransformer3DModel
    from safetensors.torch import load_file

    raw = load_file(str(ckpt_path), device="cpu")
    state_dict = anima_transformer_state_dict(raw)
    del raw
    num_layers = infer_num_layers(state_dict)
    config = {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": num_layers,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": (1.0, 4.0, 4.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    }
    with init_empty_weights():
        transformer = CosmosTransformer3DModel(**config)
    expected = set(transformer.state_dict().keys())
    got = set(state_dict.keys())
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing={missing[:20]}")
        if unexpected:
            detail.append(f"unexpected={unexpected[:20]}")
        raise RuntimeError("Anima 单文件无法对齐 CosmosTransformer3DModel: " + "; ".join(detail))
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    return transformer.to(dtype=torch_dtype)

import json
from pathlib import Path

from common.anima_checkpoint import (
    anima_transformer_state_dict,
    infer_num_layers,
    normalize_anima_dit_keys,
    resolve_anima_diffusers_base,
)


def test_civitai_prefix_becomes_cosmos_block_names() -> None:
    raw = {
        "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight": object(),
        "model.diffusion_model.blocks.1.self_attn.q_proj.weight": object(),
        "model.diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight": object(),
        "model.diffusion_model.x_embedder.proj.1.weight": object(),
    }
    converted = anima_transformer_state_dict(raw)
    assert "transformer_blocks.0.norm1.linear_1.weight" in converted
    assert "transformer_blocks.1.attn1.to_q.weight" in converted
    assert "patch_embed.proj.weight" in converted
    assert not any("llm_adapter" in key for key in converted)
    assert infer_num_layers(converted) == 2


def test_resolve_base_from_models_dir(tmp_path: Path) -> None:
    weights = tmp_path / "weights" / "ckpt"
    weights.mkdir(parents=True)
    ckpt = weights / "model.safetensors"
    ckpt.write_bytes(b"")
    base = tmp_path / "models" / "Anima-Base-v1.0-Diffusers"
    base.mkdir(parents=True)
    (base / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "AnimaModularPipeline"}),
        encoding="utf-8",
    )
    found = resolve_anima_diffusers_base(ckpt, extra_roots=[tmp_path / "models"])
    assert found == base


def test_net_prefix_is_stripped() -> None:
    raw = {"net.blocks.0.cross_attn.k_norm.weight": object()}
    normalized = normalize_anima_dit_keys(raw)
    assert "blocks.0.cross_attn.k_norm.weight" in normalized
    converted = anima_transformer_state_dict(raw)
    assert "transformer_blocks.0.attn2.norm_k.weight" in converted

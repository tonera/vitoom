"""Qwen3-VL 文本编码器：文件里只有 50 层，最终 norm 换成 Identity；带关键帧时再装上未量化的视觉塔。"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG
from safetensors import safe_open
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel

from video.handlers.minimax_h3.quant import QuantLinear, build_qtensor, group_quant_tensors, parse_quant, set_module

VISION_CONFIG = dict(
    depth=27,
    hidden_size=1152,
    intermediate_size=4304,
    num_heads=16,
    out_hidden_size=5120,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    num_position_embeddings=2304,
    deepstack_visual_indexes=[8, 16, 24],
    hidden_act="gelu_pytorch_tanh",
    in_channels=3,
)


def text_config(num_layers: int) -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig(
        hidden_size=5120,
        intermediate_size=25600,
        num_hidden_layers=num_layers,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000,
        vocab_size=151936,
        max_position_embeddings=262144,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
    )


class H3TextEncoder(nn.Module):
    """model 为 Qwen3VLTextModel（纯文本）或 Qwen3VLModel（带视觉塔）。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, **vision_kwargs) -> torch.Tensor:
        # norm 已换成 Identity，last_hidden_state 就是完整模型的 hidden_states[50]。
        return self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            **vision_kwargs,
        ).last_hidden_state


def load_processor(tokenizer_dir: str):
    processor = AutoProcessor.from_pretrained(tokenizer_dir)
    # 参考视频逐帧加像素上限（qwen-vl-utils 的做法，transformers 5.22 起的默认）。
    # 不加时 5 秒参考视频在文本编码器里约 6000 个 token。
    processor.video_processor.cap_pixels_per_frame = True
    return processor


def load_text_encoder(path: str, *, with_vision: bool = False, device: str = "cuda", log=print) -> H3TextEncoder:
    prefixes = ("model.", "visual.") if with_vision else ("model.",)
    with safe_open(path, "pt", device=device) as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefixes)}
    language = _strip(tensors, "model.")
    config = text_config(1 + max(int(key.split(".")[1]) for key in language if key.startswith("layers.")))
    text_model = _load_language_model(config, language, device, log)
    if not with_vision:
        return H3TextEncoder(text_model)

    vision_config = Qwen3VLVisionConfig(**VISION_CONFIG)
    visual = Qwen3VLVisionModel._from_config(vision_config, attn_implementation="sdpa", dtype=torch.bfloat16).to(device)
    visual.load_state_dict(_strip(tensors, "visual."), strict=True)
    with torch.device("meta"):
        model = Qwen3VLModel(Qwen3VLConfig(text_config=config.to_dict(), vision_config=vision_config.to_dict()))
    model.visual = visual
    model.language_model = text_model
    log(f"[te] vision tower depth={vision_config.depth} out={vision_config.out_hidden_size}")
    return H3TextEncoder(model.eval().requires_grad_(False))


def build_presentation(processor, prompt: str, keyframes: list) -> tuple[list[int], list[int], dict]:
    """返回 (token_ids, 每个 token 的模态 tag, 视觉输入)。

    必须与 diffusers MiniMaxH3FL2VATextEncoderStep 逐 token 一致：每个关键帧一个 "<Picture i>: " 标签加视觉块，
    然后是原样的 prompt；不套 chat template，不加特殊 token。没有关键帧时就是 t2va 的纯文本。
    """
    tokenizer = processor.tokenizer
    token_ids: list[int] = []
    token_tags: list[int] = []
    vision_inputs: dict = {}
    if keyframes:
        vision = processor.image_processor(images=keyframes, return_tensors="pt")
        vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]}
        merge = processor.image_processor.merge_size**2
        for index, grid in enumerate(vision["image_grid_thw"]):
            label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            block = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label + block
            token_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [MINIMAX_H3_VIDEO_TAG] * len(block)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags, vision_inputs


def build_reference_presentation(processor, prompt: str, references: list) -> tuple[list[int], list[int], dict]:
    """ref2va 的拼装：每个参考一个按模态编号的标签（图片、视频再加视觉块），然后是 prompt。

    直接调用 diffusers MiniMaxH3Ref2VATextEncoderStep 的两个内部方法，保证与官方逐 token 一致；
    它们不是公开接口，升级 diffusers 时要回归 ref2va。
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_FPS

    step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_counts, video_counts, video_timestamps = step._gather_vision_features(
        processor, references, MINIMAX_H3_FPS
    )
    token_ids, token_tags = step._build_presentation(
        processor.tokenizer,
        prompt,
        references,
        image_counts,
        video_counts,
        video_timestamps,
        text_tag=MINIMAX_H3_TEXT_TAG,
        video_tag=MINIMAX_H3_VIDEO_TAG,
    )
    return token_ids, token_tags, vision_inputs


@torch.inference_mode()
def encode_prompt(
    encoder: H3TextEncoder,
    token_ids: list[int],
    *,
    processor=None,
    vision_inputs: dict | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """vision_inputs 用 Qwen3-VL 的参数名：pixel_values / image_grid_thw，pixel_values_videos / video_grid_thw。"""
    if not token_ids:
        raise ValueError("文本编码器收到空 token 序列")
    tokens = torch.tensor([token_ids], device=device)
    kwargs = {}
    if vision_inputs:
        kwargs["mm_token_type_ids"] = torch.tensor(processor.create_mm_token_type_ids([token_ids]), device=device)
        for name, value in vision_inputs.items():
            kwargs[name] = value.to(device, torch.bfloat16) if name.startswith("pixel_") else value.to(device)
    return encoder(tokens, **kwargs).to(torch.bfloat16)


def _strip(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)}


def _load_language_model(
    config: Qwen3VLTextConfig, tensors: dict[str, torch.Tensor], device: str, log
) -> Qwen3VLTextModel:
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    model.norm = nn.Identity()

    groups, plain = group_quant_tensors(tensors)
    embedding = groups.pop("embed_tokens.")
    weight = embedding["weight"].float() * embedding["weight_scale"].float()
    model.embed_tokens = nn.Embedding.from_pretrained(weight.to(torch.bfloat16), freeze=True)

    for prefix, layer_tensors in groups.items():
        cfg = parse_quant(layer_tensors["comfy_quant"])
        pre_quant_scale = layer_tensors.pop("pre_quant_scale", None)
        if pre_quant_scale is not None:
            # AWQ 约定是乘在输入上。用除法会让数值发散。
            pre_quant_scale = pre_quant_scale.to(torch.bfloat16)
        set_module(
            model,
            prefix.rstrip("."),
            QuantLinear(build_qtensor(cfg, layer_tensors), None, pre_quant_scale=pre_quant_scale),
        )

    missing, unexpected = model.load_state_dict(plain, strict=False, assign=True)
    held = {prefix.rstrip(".") for prefix in groups} | {"embed_tokens"}
    missing = [name for name in missing if name.rsplit(".", 1)[0] not in held]
    log(
        f"[te] layers={config.num_hidden_layers} quant={len(groups)} plain={len(plain)} "
        f"missing={missing[:6]} unexpected={unexpected[:6]}"
    )
    if missing or unexpected:
        raise RuntimeError(f"文本编码器键未对齐 missing={missing[:8]} unexpected={unexpected[:8]}")
    model.rotary_emb = type(model.rotary_emb)(config=model.config, device=device)
    leftovers = [name for name, buf in list(model.named_parameters()) + list(model.named_buffers()) if buf.device.type == "meta"]
    if leftovers:
        raise RuntimeError(f"文本编码器仍有 meta 张量: {leftovers[:8]}")
    return model.eval().requires_grad_(False)

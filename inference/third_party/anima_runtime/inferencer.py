from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import Optional, Tuple, Union

import torch
from PIL import Image

from .logging_utils import setup_logging
from .models import Anima, AutoencoderKLQwenImage, load_vae
from .tokenizers import Qwen3LocalPaths, load_qwen3_text_encoder, load_t5_tokenizer
from .torch_transfer_utils import pretouch_module_cpu_tensors
from .weights import load_state_dict_any

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnimaPaths:
    dit_path: str
    vae_path: str
    qwen3: Qwen3LocalPaths
    t5_tokenizer_dir: str


@dataclass(frozen=True)
class AnimaRunConfig:
    height: int = 1024
    width: int = 1024
    steps: int = 40
    cfg: float = 4.5
    flow_shift: float = 3.0
    seed: Optional[int] = None
    qwen3_max_len: int = 512
    t5_max_len: int = 512


def _ensure_size_multiple_of_16(h: int, w: int) -> Tuple[int, int]:
    h = max(64, h - (h % 16))
    w = max(64, w - (w % 16))
    return h, w


def _parse_dtype(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"不支持的 dtype: {name}")


def _load_anima_dit(
    dit_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_mode: str = "torch",
    split_attn: bool = False,
    enable_block_swap: Optional[int] = None,
    loading_device: Optional[torch.device] = None,
    pretouch_cpu_before_to_cuda: bool = False,
) -> Anima:
    # Fixed config from sd-scripts `anima_utils.load_anima_model`
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": 28,
        "num_heads": 16,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
    }

    if loading_device is None:
        loading_device = torch.device("cpu")

    # 为了避免“随机初始化大模型参数”造成的构建耗时，推理默认不初始化权重（反正立刻加载 checkpoint）。
    build_device = loading_device if loading_device.type != "cpu" else torch.device("cpu")
    model = Anima(**dit_config, initialize_weights=False).to(device=build_device, dtype=dtype).eval()

    logger.info(f"Loading DiT weights: {dit_path} (loading_device={loading_device})")
    t0 = time.perf_counter()
    sd = load_state_dict_any(dit_path, device=loading_device, dtype=None, strip_net_prefix=True)
    t1 = time.perf_counter()

    load_sig = inspect.signature(model.load_state_dict)
    if "assign" in load_sig.parameters:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    t2 = time.perf_counter()

    # sd-scripts 允许这些 buffer 缺失（由 __init__ 创建）
    allow_missing_substrings = ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq")
    unexpected_missing = [k for k in missing if not any(s in k for s in allow_missing_substrings)]
    if unexpected_missing:
        raise RuntimeError(f"DiT missing keys（示例前 10 个）: {unexpected_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"DiT unexpected keys（示例前 10 个）: {list(unexpected)[:10]}")

    # 如果 loading_device 不是最终 device，这里才需要一次显式迁移
    if model.device != device:
        if (
            pretouch_cpu_before_to_cuda
            and model.device.type == "cpu"
            and device.type == "cuda"
        ):
            # 参考 torch_transfer_utils：避免 mmap/懒加载导致 `.to("cuda")` 缺页风暴
            t_mat0 = time.perf_counter()
            pretouch_module_cpu_tensors(model)
            t_mat1 = time.perf_counter()
            logger.info(f"DiT pretouch CPU tensors: {t_mat1 - t_mat0:.2f}s")
        model.to(device=device, dtype=dtype)
    t3 = time.perf_counter()
    logger.info(
        f"DiT load timing: read_state_dict={t1 - t0:.2f}s, load_state_dict={t2 - t1:.2f}s, move_device={t3 - t2:.2f}s"
    )

    if enable_block_swap and int(enable_block_swap) > 0:
        model.enable_block_swap(int(enable_block_swap), device=device)
        model.switch_block_swap_for_inference()
    return model


@torch.inference_mode()
def _do_sample(
    *,
    height: int,
    width: int,
    seed: Optional[int],
    dit: Anima,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    flow_shift: float = 3.0,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    latent_h = height // 8
    latent_w = width // 8
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
    else:
        generator = None
    noise = torch.randn((1, 16, 1, latent_h, latent_w), dtype=torch.float32, generator=generator, device="cpu").to(dtype).to(device)

    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    flow_shift = float(flow_shift)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    x = noise.clone()
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None
    for i in range(int(steps)):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0)  # (1,)
        if use_cfg:
            pos_out = dit(x, t, crossattn_emb, padding_mask=padding_mask).float()
            neg_out = dit(x, t, neg_crossattn_emb, padding_mask=padding_mask).float()
            model_output = neg_out + guidance_scale * (pos_out - neg_out)
        else:
            model_output = dit(x, t, crossattn_emb, padding_mask=padding_mask).float()

        dt = sigmas[i + 1] - sigma
        x = (x + model_output * dt).to(dtype)

    return x


class AnimaInferencer:
    """
    可复制推理器：你把 `anima_runtime/` 目录拷到新项目，并安装依赖后即可集成。

    关键点：
    - 所有组件都支持手动指定本地路径（DiT/Qwen3/T5 tokenizer/VAE）
    - 默认只支持 attn_mode='torch'（最少依赖、可移植）
    """

    def __init__(
        self,
        paths: AnimaPaths,
        *,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "bf16",
        text_device: Optional[str] = None,
        text_dtype: Optional[Union[str, torch.dtype]] = None,
        attn_mode: str = "torch",
        split_attn: bool = False,
        vae_spatial_chunk_size: Optional[int] = None,
        vae_disable_cache: bool = False,
        enable_block_swap: Optional[int] = None,
        dit_loading_device: Optional[str] = None,
        qwen3_loading_device: Optional[str] = None,
        pretouch_cpu_tensors_before_to_cuda: bool = False,
    ) -> None:
        self.paths = paths
        self.device = torch.device(device)
        self.dtype = _parse_dtype(dtype)
        self.text_device = torch.device(text_device) if text_device else self.device
        self.text_dtype = _parse_dtype(text_dtype) if text_dtype is not None else (self.dtype if self.text_device.type != "cpu" else torch.float32)

        t_init0 = time.perf_counter()

        # 1) Qwen3 + tokenizer
        t0 = time.perf_counter()
        self.qwen3_text_encoder, self.qwen3_tokenizer = load_qwen3_text_encoder(
            paths.qwen3,
            device=str(self.text_device),
            dtype=self.text_dtype,
            loading_device=qwen3_loading_device,
            pretouch_cpu_before_to_cuda=pretouch_cpu_tensors_before_to_cuda,
        )
        self.qwen3_text_encoder.eval()
        t1 = time.perf_counter()
        logger.info(f"Qwen3 load time: {t1 - t0:.2f}s")

        # 2) T5 tokenizer (only tokenizer)
        t0 = time.perf_counter()
        self.t5_tokenizer = load_t5_tokenizer(paths.t5_tokenizer_dir)
        t1 = time.perf_counter()
        logger.info(f"T5 tokenizer load time: {t1 - t0:.2f}s")

        # 3) DiT
        t0 = time.perf_counter()
        loading_dev = torch.device(dit_loading_device) if dit_loading_device else torch.device("cpu")
        self.dit = _load_anima_dit(
            paths.dit_path,
            device=self.device,
            dtype=self.dtype,
            attn_mode=attn_mode,
            split_attn=split_attn,
            enable_block_swap=enable_block_swap,
            loading_device=loading_dev,
            pretouch_cpu_before_to_cuda=pretouch_cpu_tensors_before_to_cuda,
        )
        t1 = time.perf_counter()
        logger.info(f"DiT total load time: {t1 - t0:.2f}s")

        # 4) VAE
        t0 = time.perf_counter()
        self.vae: AutoencoderKLQwenImage = load_vae(
            paths.vae_path,
            device=self.device,
            spatial_chunk_size=vae_spatial_chunk_size,
            disable_cache=bool(vae_disable_cache),
        )
        self.vae.to(device=self.device, dtype=self.dtype).eval()
        t1 = time.perf_counter()
        logger.info(f"VAE total load time: {t1 - t0:.2f}s")

        t_init1 = time.perf_counter()
        logger.info(f"AnimaInferencer init total: {t_init1 - t_init0:.2f}s")

    def encode_prompt(self, text: str, *, qwen3_max_len: int, t5_max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Qwen3 tokens
        qwen3_enc = self.qwen3_tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=int(qwen3_max_len),
        )
        qwen3_input_ids = qwen3_enc["input_ids"].to(self.text_device)
        qwen3_attn_mask = qwen3_enc["attention_mask"].to(self.text_device)

        out = self.qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = out.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        # T5 tokens (for LLM adapter target ids)
        t5_enc = self.t5_tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=int(t5_max_len),
        )
        t5_input_ids = t5_enc["input_ids"]
        t5_attn_mask = t5_enc["attention_mask"]
        return prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask

    def _to_crossattn(self, prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask) -> torch.Tensor:
        pe = prompt_embeds.to(device=self.device, dtype=self.dit.dtype)
        am = qwen3_attn_mask.to(device=self.device)
        t5_ids = t5_input_ids.to(device=self.device, dtype=torch.long)
        t5_am = t5_attn_mask.to(device=self.device)

        if getattr(self.dit, "use_llm_adapter", False) and hasattr(self.dit, "llm_adapter"):
            cross = self.dit.llm_adapter(
                source_hidden_states=pe,
                target_input_ids=t5_ids,
                target_attention_mask=t5_am,
                source_attention_mask=am,
            )
            cross[~t5_am.bool()] = 0
            return cross

        pe[~am.bool()] = 0
        return pe

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        config: Optional[AnimaRunConfig] = None,
    ) -> Image.Image:
        config = config or AnimaRunConfig()
        height, width = _ensure_size_multiple_of_16(int(config.height), int(config.width))

        prompt_embeds, attn_mask, t5_ids, t5_am = self.encode_prompt(prompt, qwen3_max_len=config.qwen3_max_len, t5_max_len=config.t5_max_len)
        neg_embeds, neg_mask, neg_t5_ids, neg_t5_am = self.encode_prompt(
            negative_prompt or "", qwen3_max_len=config.qwen3_max_len, t5_max_len=config.t5_max_len
        )

        cross = self._to_crossattn(prompt_embeds, attn_mask, t5_ids, t5_am)
        neg_cross = self._to_crossattn(neg_embeds, neg_mask, neg_t5_ids, neg_t5_am)

        latents = _do_sample(
            height=height,
            width=width,
            seed=config.seed,
            dit=self.dit,
            crossattn_emb=cross,
            steps=int(config.steps),
            dtype=self.dit.dtype,
            device=self.device,
            guidance_scale=float(config.cfg),
            flow_shift=float(config.flow_shift),
            neg_crossattn_emb=neg_cross if float(config.cfg) > 1.0 else None,
        )

        pixels = self.vae.decode_to_pixels(latents)
        if pixels.dim() == 5:
            pixels = pixels[:, :, 0, :, :]
        pixels = ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)

        try:
            import numpy as np  # optional dependency; used only for saving/conversion
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "当前环境未安装 numpy。anima_runtime 仅在把 tensor 转成 PIL 图片时需要 numpy。\n"
                "请先安装：pip install numpy"
            ) from e

        img = (pixels[0].permute(1, 2, 0).float().detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(img)


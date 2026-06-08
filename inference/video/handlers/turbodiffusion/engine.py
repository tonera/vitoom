import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from .checkpoints import infer_quant_linear, resolve_tokenizer_dir
from .imports import import_turbodiffusion_core


@dataclass(frozen=True)
class TurboModelPaths:
    mode: str  # "t2v" | "i2v"
    model_root: Path
    # common
    vae_path: str
    text_encoder_path: str
    tokenizer_dir: Optional[str]
    # t2v
    dit_path: Optional[str] = None
    model_arch: Optional[str] = None
    # i2v
    high_noise_model_path: Optional[str] = None
    low_noise_model_path: Optional[str] = None


@dataclass
class TurboModels:
    tokenizer: object
    t5_encoder: object
    net: Optional[object] = None
    high_noise_model: Optional[object] = None
    low_noise_model: Optional[object] = None


@dataclass(frozen=True)
class TurboArgs:
    # sampling
    num_steps: int
    num_frames: int
    num_samples: int
    sigma_max: float
    seed: int
    # model opts
    attention_type: str
    sla_topk: float
    quant_linear: bool
    default_norm: bool
    # vram / device
    # If true, we try to reduce GPU peak by keeping aux models (umt5/vae) on CPU when possible.
    force_offload: bool = False
    # Offload behavior when force_offload is enabled:
    # - "balanced": use GPU for umt5/vae compute stages, but offload them back to CPU between stages to lower peak VRAM.
    # - "max": keep umt5 on CPU and decode VAE on CPU (slowest, lowest VRAM).
    offload_level: str = "balanced"
    # i2v opts
    boundary: float = 0.9
    adaptive_resolution: bool = False
    ode: bool = False


def _cache_key(paths: TurboModelPaths, args: TurboArgs) -> str:
    return "|".join(
        [
            paths.mode,
            str(paths.model_root),
            str(paths.dit_path or ""),
            str(paths.high_noise_model_path or ""),
            str(paths.low_noise_model_path or ""),
            paths.vae_path,
            paths.text_encoder_path,
            str(paths.tokenizer_dir or ""),
            args.attention_type,
            str(args.sla_topk),
            str(int(args.quant_linear)),
            str(int(args.default_norm)),
            str(int(bool(args.force_offload))),
            str(getattr(args, "offload_level", "balanced")),
            str(args.boundary),
            str(int(args.adaptive_resolution)),
            str(int(args.ode)),
        ]
    )


def _move_t5_encoder(t5_encoder: object, *, device: str) -> None:
    # rcm.utils.umt5.UMT5EncoderModel is a thin wrapper with a `.model` (nn.Module) and `.device`
    m = getattr(t5_encoder, "model", None)
    if m is not None and hasattr(m, "to"):
        m.to(device)  # type: ignore[call-arg]
    if hasattr(t5_encoder, "device"):
        setattr(t5_encoder, "device", device)


def _move_vae_interface(tokenizer: object, *, device: str) -> None:
    # rcm.tokenizers.wan2pt1.Wan2pt1VAEInterface stores WanVAE at `.model`
    vae = getattr(tokenizer, "model", None)
    if vae is None:
        return
    inner = getattr(vae, "model", None)
    if inner is not None and hasattr(inner, "to"):
        inner.to(device)  # type: ignore[call-arg]
    for name in ("mean", "std"):
        t = getattr(vae, name, None)
        if t is not None and hasattr(t, "to"):
            setattr(vae, name, t.to(device))
    mean = getattr(vae, "mean", None)
    std = getattr(vae, "std", None)
    if mean is not None and std is not None:
        try:
            setattr(vae, "scale", [mean, 1.0 / std])
        except Exception:
            pass
    if hasattr(vae, "device"):
        setattr(vae, "device", device)


def build_models_cache_key(*, paths: TurboModelPaths, args: TurboArgs) -> str:
    """
    返回“模型权重兼容性签名”（不包含 seed/steps/frames 这类 request 级变量）。
    用于上层 PipelineCache 的 key。
    """
    return _cache_key(paths, args)


def load_models(*, paths: TurboModelPaths, args: TurboArgs, logger) -> TurboModels:
    key = build_models_cache_key(paths=paths, args=args)
    td = import_turbodiffusion_core()
    t0 = time.time()

    tokenizer = td.Wan2pt1VAEInterface(vae_pth=paths.vae_path)
    tok_dir = paths.tokenizer_dir or "google/umt5-xxl"
    level = str(getattr(args, "offload_level", "balanced")).strip().lower()
    if level not in ("balanced", "max", "none"):
        level = "balanced" if bool(args.force_offload) else "none"
    t5_encoder = td.UMT5EncoderModel(
        text_len=512,
        # Upstream loads ckpt with map_location="cuda" internally; however setting device="cpu"
        # still ends with the module on CPU, which helps reduce peak VRAM during inference.
        device=("cpu" if (bool(args.force_offload) and level == "max") else "cuda"),
        checkpoint_path=paths.text_encoder_path,
        tokenizer_path=tok_dir,
    )

    models = TurboModels(tokenizer=tokenizer, t5_encoder=t5_encoder)

    if bool(args.force_offload):
        # Keep aux models on CPU by default; generation can move them as needed.
        try:
            _move_vae_interface(models.tokenizer, device="cpu")
        except Exception:
            pass
        try:
            _move_t5_encoder(models.t5_encoder, device="cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()

    # The upstream factory expects an argparse-like args. We'll give it a lightweight object.
    class _A:
        pass

    a = _A()
    a.mode = paths.mode
    a.model = paths.model_arch or ("Wan2.1-14B" if paths.mode == "t2v" else "Wan2.2-A14B")
    a.attention_type = args.attention_type
    a.sla_topk = args.sla_topk
    a.quant_linear = bool(args.quant_linear)
    a.default_norm = bool(args.default_norm)
    # i2v
    a.boundary = args.boundary
    a.adaptive_resolution = bool(args.adaptive_resolution)
    a.ode = bool(args.ode)

    if paths.mode == "t2v":
        assert paths.dit_path
        if bool(args.force_offload):
            # Cache on CPU to reduce idle VRAM; generation will move to CUDA for denoise.
            net = td.create_model(dit_path=paths.dit_path, args=a).cpu().eval()
        else:
            # Fast path: keep on GPU.
            net = td.create_model(dit_path=paths.dit_path, args=a).cuda().eval()
        torch.cuda.empty_cache()
        models.net = net
    else:
        assert paths.high_noise_model_path and paths.low_noise_model_path
        high_noise_model = td.create_model(dit_path=paths.high_noise_model_path, args=a).cpu().eval()
        torch.cuda.empty_cache()
        low_noise_model = td.create_model(dit_path=paths.low_noise_model_path, args=a).cpu().eval()
        torch.cuda.empty_cache()
        models.high_noise_model = high_noise_model
        models.low_noise_model = low_noise_model

    logger.info(f"[TurboDiffusion] models loaded in {time.time() - t0:.2f}s cache_key={key}")
    return models


def _raise_if_cancelled(
    cancel_checker: Optional[Callable[[str], None]],
    stage: str,
) -> None:
    if cancel_checker is None:
        return
    cancel_checker(stage)


def generate_t2v(
    *,
    models: TurboModels,
    args: TurboArgs,
    prompt: str,
    width: int,
    height: int,
    logger,
    cancel_checker: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """
    Returns video tensor: (T,H,W,C) uint8 on CPU.
    """
    import numpy as np
    from einops import repeat

    td = import_turbodiffusion_core()
    tokenizer = models.tokenizer
    t5_encoder = models.t5_encoder
    net = models.net
    assert net is not None

    level = str(getattr(args, "offload_level", "balanced")).strip().lower()
    if level not in ("balanced", "max", "none"):
        level = "balanced" if bool(getattr(args, "force_offload", False)) else "none"

    _raise_if_cancelled(cancel_checker, "turbo t2v before text encode")
    logger.info(f"[TurboDiffusion][t2v] stage=text-encode start")
    with torch.no_grad():
        if bool(getattr(args, "force_offload", False)) and level == "max":
            # Max VRAM saving: keep umt5 on CPU; move only embeddings to CUDA.
            text_emb = t5_encoder(str(prompt), device=torch.device("cpu")).to(**td.tensor_kwargs)
        else:
            # Fast: use GPU for embedding.
            try:
                _move_t5_encoder(t5_encoder, device="cuda")
            except Exception:
                pass
            text_emb = t5_encoder(str(prompt), device=td.tensor_kwargs["device"]).to(**td.tensor_kwargs)
    _raise_if_cancelled(cancel_checker, "turbo t2v after text encode")
    logger.info(f"[TurboDiffusion][t2v] stage=text-encode done")

    # If we are in force_offload mode, drop umt5 from GPU after embedding to lower peak VRAM during denoise.
    if bool(getattr(args, "force_offload", False)) and level in ("balanced", "max"):
        try:
            _move_t5_encoder(t5_encoder, device="cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()

    condition = {"crossattn_emb": repeat(text_emb.to(**td.tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    generator = torch.Generator(device=td.tensor_kwargs["device"])
    generator.manual_seed(int(args.seed))

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        height // tokenizer.spatial_compression_factor,
        width // tokenizer.spatial_compression_factor,
    ]
    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=td.tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(float(args.sigma_max)), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    net.cuda()
    total_steps = max(0, len(t_steps) - 1)
    logger.info(f"[TurboDiffusion][t2v] stage=denoise start steps={total_steps}")
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # num_steps is tiny (1~4), so per-step logs are acceptable and helpful for diagnosing stalls.
        _raise_if_cancelled(cancel_checker, f"turbo t2v denoise step={i + 1}/{total_steps}")
        logger.info(f"[TurboDiffusion][t2v] stage=denoise step={i+1}/{total_steps}")
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**td.tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**td.tensor_kwargs),
                **condition,
            ).to(torch.float64)
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=td.tensor_kwargs["device"],
                generator=generator,
            )
    _raise_if_cancelled(cancel_checker, "turbo t2v after denoise")
    samples = x.float()
    net.cpu()
    torch.cuda.empty_cache()
    logger.info(f"[TurboDiffusion][t2v] stage=denoise done")

    _raise_if_cancelled(cancel_checker, "turbo t2v before decode")
    logger.info(f"[TurboDiffusion][t2v] stage=decode start")
    with torch.no_grad():
        if bool(getattr(args, "force_offload", False)) and level == "max":
            # Max VRAM saving: decode on CPU (slow).
            try:
                _move_vae_interface(tokenizer, device="cpu")
            except Exception:
                pass
            video = tokenizer.decode(samples.cpu())  # B,C,T,H,W in [-1,1]
        else:
            # Balanced/fast: decode on GPU (DiT 已经 cpu()，峰值仍可控）
            try:
                _move_vae_interface(tokenizer, device="cuda")
            except Exception:
                pass
            video = tokenizer.decode(samples.to(device=td.tensor_kwargs["device"]))  # B,C,T,H,W in [-1,1]
            if bool(getattr(args, "force_offload", False)) and level == "balanced":
                try:
                    _move_vae_interface(tokenizer, device="cpu")
                except Exception:
                    pass
                torch.cuda.empty_cache()
    _raise_if_cancelled(cancel_checker, "turbo t2v after decode")
    logger.info(f"[TurboDiffusion][t2v] stage=decode done")
    vid = (video[0].clamp(-1, 1) + 1.0) / 2.0  # C,T,H,W in [0,1]
    frames = (vid.permute(1, 2, 3, 0).contiguous().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return torch.from_numpy(np.asarray(frames))


def generate_i2v(
    *,
    models: TurboModels,
    args: TurboArgs,
    prompt: str,
    init_image_path: str,
    width: int,
    height: int,
    logger,
    cancel_checker: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    import numpy as np
    from einops import repeat
    from PIL import Image
    import torchvision.transforms.v2 as T

    td = import_turbodiffusion_core()
    tokenizer = models.tokenizer
    t5_encoder = models.t5_encoder
    high_noise_model = models.high_noise_model
    low_noise_model = models.low_noise_model
    assert high_noise_model is not None and low_noise_model is not None

    level = str(getattr(args, "offload_level", "balanced")).strip().lower()
    if level not in ("balanced", "max", "none"):
        level = "balanced" if bool(getattr(args, "force_offload", False)) else "none"

    _raise_if_cancelled(cancel_checker, "turbo i2v before text encode")
    logger.info(f"[TurboDiffusion][i2v] stage=text-encode start")
    with torch.no_grad():
        if bool(getattr(args, "force_offload", False)) and level == "max":
            # Max VRAM saving: keep umt5 on CPU; move only embeddings to CUDA.
            text_emb = t5_encoder(str(prompt), device=torch.device("cpu")).to(**td.tensor_kwargs)
        else:
            # Fast: use GPU for embedding.
            try:
                _move_t5_encoder(t5_encoder, device="cuda")
            except Exception:
                pass
            text_emb = t5_encoder(str(prompt), device=td.tensor_kwargs["device"]).to(**td.tensor_kwargs)
    _raise_if_cancelled(cancel_checker, "turbo i2v after text encode")
    logger.info(f"[TurboDiffusion][i2v] stage=text-encode done")

    if bool(getattr(args, "force_offload", False)) and level in ("balanced", "max"):
        try:
            _move_t5_encoder(t5_encoder, device="cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()

    condition = {"crossattn_emb": repeat(text_emb.to(**td.tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    input_image = Image.open(str(init_image_path)).convert("RGB")

    # NOTE: we honor requested width/height; adaptive_resolution can override for best quality like upstream.
    if bool(args.adaptive_resolution):
        max_resolution_area = width * height
        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w
        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)
        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        height = lat_h * stride
        width = lat_w * stride

    F = args.num_frames
    lat_h = height // tokenizer.spatial_compression_factor
    lat_w = width // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    image_transforms = T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(height, width), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # VAE encode stage uses tokenizer; ensure tokenizer is on CUDA for this stage.
    try:
        _move_vae_interface(tokenizer, device="cuda")
    except Exception:
        pass
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=td.tensor_kwargs["device"], dtype=torch.float32)

    _raise_if_cancelled(cancel_checker, "turbo i2v before vae encode")
    logger.info(f"[TurboDiffusion][i2v] stage=vae-encode start")
    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, height, width, device=image_tensor.device)], dim=2
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()
    _raise_if_cancelled(cancel_checker, "turbo i2v after vae encode")
    logger.info(f"[TurboDiffusion][i2v] stage=vae-encode done")

    if bool(getattr(args, "force_offload", False)) and level in ("balanced", "max"):
        # Free VAE VRAM before denoise; we'll decode on CPU at the end.
        try:
            _move_vae_interface(tokenizer, device="cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=td.tensor_kwargs["device"], dtype=td.tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0
    y = torch.cat([msk, encoded_latents.to(**td.tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)
    condition["y_B_C_T_H_W"] = y

    generator = torch.Generator(device=td.tensor_kwargs["device"])
    generator.manual_seed(int(args.seed))

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=td.tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(float(args.sigma_max)), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

    high_noise_model.cuda()
    net = high_noise_model
    switched = False
    boundary = float(args.boundary)
    total_steps = max(0, len(t_steps) - 1)
    logger.info(f"[TurboDiffusion][i2v] stage=denoise start steps={total_steps} boundary={boundary}")
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        _raise_if_cancelled(cancel_checker, f"turbo i2v denoise step={i + 1}/{total_steps}")
        logger.info(f"[TurboDiffusion][i2v] stage=denoise step={i+1}/{total_steps}")
        if float(t_cur.item()) < boundary and not switched:
            high_noise_model.cpu()
            torch.cuda.empty_cache()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            logger.info(f"[TurboDiffusion][i2v] stage=denoise switched_to=low_noise_model at_t={float(t_cur.item()):.4f}")
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**td.tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**td.tensor_kwargs),
                **condition,
            ).to(torch.float64)
            if bool(args.ode):
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=td.tensor_kwargs["device"],
                    generator=generator,
                )

    _raise_if_cancelled(cancel_checker, "turbo i2v after denoise")
    samples = x.float()
    if switched:
        low_noise_model.cpu()
    else:
        high_noise_model.cpu()
    torch.cuda.empty_cache()
    logger.info(f"[TurboDiffusion][i2v] stage=denoise done switched={int(switched)}")

    _raise_if_cancelled(cancel_checker, "turbo i2v before decode")
    logger.info(f"[TurboDiffusion][i2v] stage=decode start")
    with torch.no_grad():
        if bool(getattr(args, "force_offload", False)) and level == "max":
            # Max VRAM saving: decode on CPU (slow).
            try:
                _move_vae_interface(tokenizer, device="cpu")
            except Exception:
                pass
            video = tokenizer.decode(samples.cpu())  # B,C,T,H,W in [-1,1]
        else:
            # Balanced/fast: decode on GPU (DiT 已经 cpu()，峰值仍可控）
            try:
                _move_vae_interface(tokenizer, device="cuda")
            except Exception:
                pass
            video = tokenizer.decode(samples.to(device=td.tensor_kwargs["device"]))  # B,C,T,H,W in [-1,1]
            if bool(getattr(args, "force_offload", False)) and level == "balanced":
                try:
                    _move_vae_interface(tokenizer, device="cpu")
                except Exception:
                    pass
                torch.cuda.empty_cache()
    _raise_if_cancelled(cancel_checker, "turbo i2v after decode")
    logger.info(f"[TurboDiffusion][i2v] stage=decode done")
    vid = (video[0].clamp(-1, 1) + 1.0) / 2.0
    frames = (vid.permute(1, 2, 3, 0).contiguous().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return torch.from_numpy(np.asarray(frames))


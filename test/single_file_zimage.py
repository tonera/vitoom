import torch
from diffusers import ZImagePipeline
import time
from datetime import datetime
start_time = time.time()
import inspect
from contextlib import nullcontext
import argparse
import os
from typing import Dict, Iterable, Optional, Tuple
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

"""
说明：
- 你遇到的报错是 `ZImagePipeline.from_single_file()` 在加载 `text_encoder=Qwen3Model` 时失败。
- 原因：diffusers 的 single_file loader **不会**从 checkpoint 自动构建/灌权重到 `Qwen3Model`（它只内置支持 CLIP/T5 的 transformers 单文件加载）。
- 解决：不改 diffusers 源码的前提下，**先从同一个 .safetensors 里把 Qwen3 权重提取出来，手工构建 text_encoder，再传给 from_single_file**。

进一步：
- 你现在遇到 `AutoencoderKL` 报 “Weights ... missing in the checkpoint”，是因为你安装的 diffusers 版本在 pipeline 的 single_file loader
  路径上并不会自动从 checkpoint 构建/灌权重到 VAE（会退回到从 config 目录找权重文件）。
- 解决同样是：脚本里先构建 `vae` 并灌权重，再把 `vae=...` 传给 `from_single_file`。

- 同理：为了确保 transformer 也确实来自该单文件，我们也先构建 `transformer`，再传给 `from_single_file`。
"""


def _load_safetensors_subset(path: str, prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    """
    只加载指定前缀的权重，避免把整个 AIO checkpoint 都读进内存。
    注意：`ZImagePipeline.from_single_file()` 内部仍会再读取一次 checkpoint（无法避免）。
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能从单文件读取权重：pip install safetensors") from e

    prefixes = tuple(prefixes)
    out: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefixes):
                out[k] = f.get_tensor(k)
    return out


def _load_safetensors_excluding_prefixes(path: str, excluded_prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    """
    加载除指定前缀之外的所有权重（用于 transformer：避免把 text_encoder/vae 的 key 一起喂给 transformer loader 造成超长“unused keys”提示）。
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能从单文件读取权重：pip install safetensors") from e

    excluded_prefixes = tuple(excluded_prefixes)
    out: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(excluded_prefixes):
                continue
            out[k] = f.get_tensor(k)
    return out


def _remap_to_target_keys(
    tensors: Dict[str, torch.Tensor],
    target_keys: Iterable[str],
    prefix_to_strip: str,
    also_strip_model_prefix: bool,
) -> Dict[str, torch.Tensor]:
    target_keys = set(target_keys)
    remapped: Dict[str, torch.Tensor] = {}
    plen = len(prefix_to_strip)
    for k, v in tensors.items():
        if not k.startswith(prefix_to_strip):
            continue
        kk = k[plen:]
        if also_strip_model_prefix and kk.startswith("model."):
            kk = kk[len("model.") :]
        if kk in target_keys:
            remapped[kk] = v
    return remapped


def _build_qwen3_text_encoder_from_ckpt(
    ckpt_path: str,
    config: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
) -> "torch.nn.Module":
    t_start = time.time()
    # 1) 读出 checkpoint 里 text_encoder 相关权重
    # 你提供的 key 形如：text_encoders.qwen3_4b.transformer.model.layers.0....
    prefixes = [
        "text_encoders.qwen3_4b.transformer.",
        "text_encoders.qwen3_4b.",
        "text_encoder.",
    ]
    tensors = _load_safetensors_subset(ckpt_path, prefixes=prefixes)
    if not tensors:
        raise RuntimeError(
            "在 checkpoint 里没找到 text encoder 权重前缀；请确认是否存在 `text_encoders.qwen3_4b.*` 或 `text_encoder.*`"
        )
    t_after_tensors = time.time()
    print(f"[text_encoder] tensors_loaded={len(tensors)} time={t_after_tensors - t_start:.3f}s")

    # 2) 用 diffusers repo 里的 text_encoder/config.json 创建 Qwen3Model（只读配置，不读权重）
    try:
        from transformers import AutoConfig, Qwen3Model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装支持 Qwen3 的 transformers 才能构建 text_encoder") from e

    text_cfg = AutoConfig.from_pretrained(
        config,
        subfolder="text_encoder",
        local_files_only=local_files_only,
    )
    t_after_cfg = time.time()
    print(f"[text_encoder] config_loaded time={t_after_cfg - t_after_tensors:.3f}s")
    # 关键优化：避免 Qwen3Model(...) 触发大规模参数随机初始化（你日志里 28s 就耗在这里）。
    # 使用 accelerate 的 init_empty_weights + PyTorch 的 load_state_dict(assign=True)，直接把权重“贴”到参数上。
    assign_supported = "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters
    ctx = nullcontext
    use_assign = False
    try:
        from accelerate import init_empty_weights  # type: ignore

        if assign_supported:
            ctx = init_empty_weights
            use_assign = True
    except Exception:
        # 没有 accelerate 或者 torch 不支持 assign，就退回普通构建（会慢）
        ctx = nullcontext
        use_assign = False

    with ctx():
        text_encoder = Qwen3Model(text_cfg)
    t_after_init = time.time()
    print(f"[text_encoder] model_init time={t_after_init - t_after_cfg:.3f}s")

    # 3) 尝试把 checkpoint key 对齐到 Qwen3Model 的 state_dict
    target_keys = text_encoder.state_dict().keys()
    candidates: list[tuple[str, bool]] = [
        ("text_encoders.qwen3_4b.transformer.", True),  # strip ".transformer." and optional "model."
        ("text_encoders.qwen3_4b.", True),
        ("text_encoder.", True),
        ("text_encoders.qwen3_4b.transformer.", False),
        ("text_encoders.qwen3_4b.", False),
        ("text_encoder.", False),
    ]

    best = {}
    best_desc: Optional[tuple[str, bool]] = None
    for p, strip_model in candidates:
        remapped = _remap_to_target_keys(tensors, target_keys, prefix_to_strip=p, also_strip_model_prefix=strip_model)
        if len(remapped) > len(best):
            best = remapped
            best_desc = (p, strip_model)

    if not best:
        raise RuntimeError(
            "无法将 checkpoint 的 text encoder 权重映射到 Qwen3Model（可能是前缀/结构不一致，或该文件并非 Qwen3Model 权重）"
        )

    load_kwargs = {"assign": True} if use_assign else {}
    incompat = text_encoder.load_state_dict(best, strict=False, **load_kwargs)
    missing = len(getattr(incompat, "missing_keys", []))
    unexpected = len(getattr(incompat, "unexpected_keys", []))
    print(f"[text_encoder] mapped={len(best)} missing={missing} unexpected={unexpected} via={best_desc}")
    t_after_load = time.time()
    print(f"[text_encoder] load_state_dict time={t_after_load - t_after_init:.3f}s total={t_after_load - t_start:.3f}s")

    # 4) dtype + eval（device 交给 pipe.to(...)）
    text_encoder = text_encoder.to(dtype=torch_dtype)
    text_encoder.eval()
    return text_encoder


def _build_vae_from_ckpt(
    ckpt_path: str,
    config: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
) -> "torch.nn.Module":
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("无法导入 diffusers.AutoencoderKL") from e

    # 1) 基于 diffusers repo 的 vae/config.json 构建结构（不依赖权重）
    vae_cfg = AutoencoderKL.load_config(config, subfolder="vae", local_files_only=local_files_only)
    vae = AutoencoderKL.from_config(vae_cfg)

    # 2) 从 checkpoint 只读 VAE 相关权重
    vae_tensors = _load_safetensors_subset(ckpt_path, prefixes=["vae.", "first_stage_model."])
    if not vae_tensors:
        raise RuntimeError("在 checkpoint 里没找到 VAE 权重前缀：`vae.*` 或 `first_stage_model.*`")

    # 3) 先尝试直接前缀剥离匹配（若该 checkpoint 已经是 diffusers 风格）
    target_keys = vae.state_dict().keys()
    direct_best = {}
    direct_best_desc: Optional[str] = None
    for p in ["vae.", "first_stage_model."]:
        remapped = _remap_to_target_keys(vae_tensors, target_keys, prefix_to_strip=p, also_strip_model_prefix=False)
        if len(remapped) > len(direct_best):
            direct_best = remapped
            direct_best_desc = f"direct_strip({p})"

    loaded = False
    if direct_best:
        incompat = vae.load_state_dict(direct_best, strict=False)
        missing = len(getattr(incompat, "missing_keys", []))
        unexpected = len(getattr(incompat, "unexpected_keys", []))
        print(f"[vae] direct_mapped={len(direct_best)} missing={missing} unexpected={unexpected} via={direct_best_desc}")
        # 经验阈值：匹配太少基本说明需要转换
        loaded = len(direct_best) > 1000

    # 4) 若直接匹配不足，尝试使用 diffusers 自带的 VAE ckpt->diffusers 转换器
    if not loaded:
        try:
            from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint  # type: ignore

            converted = convert_ldm_vae_checkpoint(vae_tensors, vae_cfg)
            incompat = vae.load_state_dict(converted, strict=False)
            missing = len(getattr(incompat, "missing_keys", []))
            unexpected = len(getattr(incompat, "unexpected_keys", []))
            print(f"[vae] converted_mapped={len(converted)} missing={missing} unexpected={unexpected} via=convert_ldm_vae_checkpoint")
        except Exception as e:
            raise RuntimeError(
                "VAE 权重未能直接匹配，且 convert_ldm_vae_checkpoint 转换也失败。"
                "这通常意味着该单文件的 VAE 结构/命名与 `--config` 的 vae/config.json 不一致。"
            ) from e

    vae = vae.to(dtype=torch_dtype)
    vae.eval()
    return vae


def _build_zimage_transformer_from_ckpt(
    ckpt_path: str,
    config: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
) -> "torch.nn.Module":
    try:
        from diffusers import ZImageTransformer2DModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("无法导入 diffusers.ZImageTransformer2DModel") from e

    # 只喂给 transformer 相关权重，避免把 text_encoder/vae 的 key 一起带进去导致超长 unused keys 提示
    transformer_ckpt = _load_safetensors_excluding_prefixes(
        ckpt_path,
        excluded_prefixes=[
            "text_encoders.",
            "text_encoder.",
            "vae.",
            "first_stage_model.",
        ],
    )

    transformer = ZImageTransformer2DModel.from_single_file(
        transformer_ckpt,
        config=config,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    transformer.eval()
    return transformer


def main(ckpt: str, config: str, out: str, local_files_only: bool):
    if not ckpt or not os.path.isfile(ckpt):
        raise ValueError("--ckpt 必须是本地单文件权重路径，例如 *.safetensors")
    if not config:
        raise ValueError("--config 必须提供：diffusers 模型目录或 repo id（用于读取 text_encoder/config.json 等配置文件）")
    print(f"Loading ZImagePipeline from single file: {ckpt}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    torch_dtype = torch.bfloat16
    text_encoder = _build_qwen3_text_encoder_from_ckpt(
        ckpt_path=ckpt,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    print(f"载入text_encoder完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    vae = _build_vae_from_ckpt(
        ckpt_path=ckpt,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    print(f"载入vae完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    transformer = _build_zimage_transformer_from_ckpt(
        ckpt_path=ckpt,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    print(f"载入transformer完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 关键：仍然用 ZImagePipeline.from_single_file()，但把不支持自动加载的组件提前传入
    pipe = ZImagePipeline.from_single_file(
        ckpt,
        config=config,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    print(f"载入pipeline完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pretouch_pipeline_cpu_tensors(pipe, on_component=lambda n: print(f"{n} pretouch完成"))
    pipe.to("cuda")
    print(f"To cuda完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    negative_promp = "bad quality,worst quality,worst detail,censor"

    # 2. Generate Image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_promp,
        height=1024,
        width=1024,
        num_inference_steps=4,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    print(f"生成图片完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    image.save(out)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="单文件 AIO 权重（.safetensors/.ckpt）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="resources/models/Z-Image-Turbo",
        help="diffusers 模型目录或 repo id（用于读取各组件 config.json；建议本地目录以便离线）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/test_z_image_singlefile2.png",
        help="输出图片路径",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="仅使用本地文件/缓存（无网络）。如果 config 传的是 repo id 且本地无缓存，会失败。",
    )
    args = parser.parse_args()
    main(args.ckpt, args.config, args.out, args.local_files_only)

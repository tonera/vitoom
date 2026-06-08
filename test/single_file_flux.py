import argparse
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, Iterable, Optional

import torch
from diffusers import FluxPipeline


"""
Flux 单文件（.safetensors/.ckpt）加载实验脚本（不修改 diffusers 源码）。

目标：
- 验证 `FluxPipeline.from_single_file()` 是否能正确加载第三方“单文件打包”的 Flux 模型（例如 realDream_flux1V1.safetensors）。
- 在某些 diffusers 版本里，pipeline 的 single_file loader 可能无法自动把 VAE/Transformer 从 checkpoint 灌进去；
  因此这里提供“脚本先构建组件再传给 from_single_file”的兜底方式，确保使用单文件里的组件权重，避免影响出图效果。

用法示例：
python test1.py \
  --ckpt /path/to/realDream_flux1V1.safetensors \
  --config black-forest-labs/FLUX.1-schnell \
  --prompt "A cat holding a sign that says hello world" \
  --steps 4 --guidance 0.0 \
  --out outputs/flux_single_file.png
"""

def _probe_ckpt_component_prefixes(ckpt_path: str) -> dict[str, int]:
    """
    只扫描 safetensors key（不读取 tensor 数据），统计一些常见前缀/特征 key 的出现次数，用于判断 ckpt 是否“包含某组件权重”。
    注意：第三方模型打包方式很多，这里是启发式判断（尽量覆盖主流 Flux 单文件命名）。
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能读取单文件权重：pip install safetensors") from e

    # 前缀：越通用的越靠前；Flux 原始权重有时带 model.diffusion_model. 前缀
    prefixes: dict[str, tuple[str, ...]] = {
        # VAE
        "vae": ("vae.", "first_stage_model.", "autoencoder."),
        # Flux transformer（原始实现常见：double_blocks/single_blocks/time_in/img_in/txt_in/...）
        "transformer": (
            "double_blocks.",
            "single_blocks.",
            "time_in.",
            "vector_in.",
            "img_in.",
            "txt_in.",
            "model.diffusion_model.double_blocks.",
            "model.diffusion_model.single_blocks.",
            "model.diffusion_model.time_in.",
            "model.diffusion_model.vector_in.",
            "model.diffusion_model.img_in.",
            "model.diffusion_model.txt_in.",
        ),
        # TE1（CLIP/OpenCLIP）
        "text_encoder": (
            "text_encoder.",
            "cond_stage_model.transformer.",
            "conditioner.embedders.0.transformer.",
            "text_encoders.clip_l.transformer.",
            "text_encoders.clip_g.transformer.",
        ),
        # TE2（T5）
        "text_encoder_2": (
            "text_encoder_2.",
            "text_encoders.t5xxl.transformer.",
        ),
        # tokenizers（可选）
        "tokenizer": ("tokenizer.",),
        "tokenizer_2": ("tokenizer_2.",),
    }

    counts = {k: 0 for k in prefixes.keys()}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            for name, pfxs in prefixes.items():
                if k.startswith(pfxs):
                    counts[name] += 1
    return counts


def _print_ckpt_component_probe(ckpt_path: str) -> None:
    counts = _probe_ckpt_component_prefixes(ckpt_path)
    def yn(n: int) -> str:
        return "有" if n > 0 else "无/未识别"
    print(
        "[ckpt_probe] "
        f"vae={yn(counts['vae'])}(keys={counts['vae']}) "
        f"transformer={yn(counts['transformer'])}(keys={counts['transformer']}) "
        f"text_encoder={yn(counts['text_encoder'])}(keys={counts['text_encoder']}) "
        f"text_encoder_2={yn(counts['text_encoder_2'])}(keys={counts['text_encoder_2']})"
    )


def _count_key_matches_only(
    safetensors_path: str, prefix: str, target_keys: set[str], also_strip_model_prefix: bool = False
) -> tuple[int, int]:
    """
    只扫描 safetensors 的 key 名字（不读取 tensor 数据），统计：
    - found: 以 prefix 开头的 key 数
    - matched: 去掉 prefix（以及可选的 model.）后，能命中 target_keys 的 key 数
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能读取单文件权重：pip install safetensors") from e

    found = matched = 0
    plen = len(prefix)
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            found += 1
            kk = k[plen:]
            if also_strip_model_prefix and kk.startswith("model."):
                kk = kk[len("model.") :]
            if kk in target_keys:
                matched += 1
    return found, matched


def _diff_key_matches_only(
    safetensors_path: str,
    prefix: str,
    target_keys: set[str],
    also_strip_model_prefix: bool = False,
    limit: int = 10,
    alias_equivalents: Optional[dict[str, str]] = None,
) -> tuple[int, int, list[str], list[str]]:
    """
    只扫描 safetensors 的 key（不读取 tensor 数据），并返回：
    - found: prefix 命中的 key 数
    - matched: 剥前缀后命中 target_keys 的 key 数
    - missing_keys_sample: target_keys 中缺失的 key（最多 limit 个）
    - unexpected_keys_sample: ckpt 前缀下多余的 key（剥前缀后不在 target_keys，最多 limit 个）
    """
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 `safetensors` 才能读取单文件权重：pip install safetensors") from e

    found_keys_stripped: set[str] = set()
    matched_keys: set[str] = set()

    plen = len(prefix)
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            kk = k[plen:]
            if also_strip_model_prefix and kk.startswith("model."):
                kk = kk[len("model.") :]
            found_keys_stripped.add(kk)
            if kk in target_keys:
                matched_keys.add(kk)

    missing_set = set(target_keys - matched_keys)
    # 一些模型存在“绑权重/别名权重”，例如 T5EncoderModel 的 encoder.embed_tokens.weight 通常与 shared.weight 绑定。
    # 单文件可能只保存 shared.weight，导致按 key 名称统计时出现 1 个 missing，但实际推理不受影响。
    if alias_equivalents:
        for missing_k, alt_k in alias_equivalents.items():
            if missing_k in missing_set and alt_k in found_keys_stripped:
                missing_set.remove(missing_k)

    missing = sorted(list(missing_set))
    unexpected = sorted(list(found_keys_stripped - target_keys))
    return len(found_keys_stripped), len(matched_keys), missing[:limit], unexpected[:limit]


def _print_text_encoder_match_stats(pipe: FluxPipeline, ckpt_path: str) -> None:
    """
    打印 TE/TE2 的“单文件 key -> 组件 state_dict key”匹配统计。
    注意：这里只做 key 名字级的匹配统计，不涉及 diffusers 的复杂转换逻辑（因此对某些非常规命名可能低估）。
    """
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        target = set(pipe.text_encoder.state_dict().keys())
        candidates = [
            ("text_encoder.", False),
            # LDM / SDXL 风格（如果第三方单文件沿用这类命名）
            ("cond_stage_model.transformer.", False),
            ("conditioner.embedders.0.transformer.", False),
            ("text_encoders.clip_l.transformer.", False),
            ("text_encoders.clip_g.transformer.", False),
        ]
        best = None
        for pfx, strip_model in candidates:
            found, matched = _count_key_matches_only(ckpt_path, pfx, target, also_strip_model_prefix=strip_model)
            if found == 0:
                continue
            if best is None or matched > best[2]:
                best = (pfx, found, matched)
        if best is None:
            print("[text_encoder] 单文件中未发现可匹配的 key（可能 TE 权重不在 ckpt，实际从 --config 目录加载）")
        else:
            pfx, found, matched = best
            found2, matched2, missing_s, unexpected_s = _diff_key_matches_only(ckpt_path, pfx, target)
            missing = len(target) - matched
            unexpected = found - matched
            print(
                f"[text_encoder] target_keys={len(target)} ckpt_prefixed_keys={found} matched={matched} missing≈{missing} unexpected≈{unexpected} via={pfx}"
            )
            if missing > 0 or unexpected > 0:
                print(f"[text_encoder] missing_sample={missing_s} unexpected_sample={unexpected_s}")

    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        target = set(pipe.text_encoder_2.state_dict().keys())
        candidates = [
            ("text_encoder_2.", False),
            # SD3 风格 T5 前缀
            ("text_encoders.t5xxl.transformer.", False),
        ]
        best = None
        for pfx, strip_model in candidates:
            found, matched = _count_key_matches_only(ckpt_path, pfx, target, also_strip_model_prefix=strip_model)
            if found == 0:
                continue
            if best is None or matched > best[2]:
                best = (pfx, found, matched)
        if best is None:
            print("[text_encoder_2] 单文件中未发现可匹配的 key（可能 TE2 权重不在 ckpt，实际从 --config 目录加载）")
        else:
            pfx, found, matched = best
            found2, matched2, missing_s, unexpected_s = _diff_key_matches_only(
                ckpt_path,
                pfx,
                target,
                alias_equivalents={
                    # T5EncoderModel 常见绑权重：encoder.embed_tokens.weight == shared.weight
                    "encoder.embed_tokens.weight": "shared.weight",
                },
            )
            missing = len(target) - matched
            unexpected = found - matched
            print(
                f"[text_encoder_2] target_keys={len(target)} ckpt_prefixed_keys={found} matched={matched} missing≈{missing} unexpected≈{unexpected} via={pfx}"
            )
            if missing > 0 or unexpected > 0:
                print(f"[text_encoder_2] missing_sample={missing_s} unexpected_sample={unexpected_s}")


def _load_safetensors_subset(path: str, prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    """
    只加载指定前缀的权重，避免把整个单文件 checkpoint 都读进内存。
    注意：`FluxPipeline.from_single_file()` 内部仍会再读取一次 checkpoint（无法避免）。
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
    加载除指定前缀之外的所有权重（用于 transformer：避免把 text_encoder/vae 的 key 一起喂给 transformer loader）。
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


def _best_prefix_strip_map(
    tensors: Dict[str, torch.Tensor], target_keys: Iterable[str], candidates: Iterable[tuple[str, bool]]
) -> Dict[str, torch.Tensor]:
    """
    对一组候选前缀做“剥前缀 +（可选）剥 model. 前缀”的匹配，选命中 key 数量最多的一种。
    """
    target_keys = set(target_keys)
    best: Dict[str, torch.Tensor] = {}

    for p, strip_model in candidates:
        plen = len(p)
        remapped: Dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            if not k.startswith(p):
                continue
            kk = k[plen:]
            if strip_model and kk.startswith("model."):
                kk = kk[len("model.") :]
            if kk in target_keys:
                remapped[kk] = v
        if len(remapped) > len(best):
            best = remapped
    return best


def _build_vae_from_ckpt(ckpt_path: str, config: str, torch_dtype: torch.dtype, local_files_only: bool):
    """
    构建 AutoencoderKL 并从单文件灌权重（直接匹配优先；不匹配则尝试 convert_ldm_vae_checkpoint）。
    """
    try:
        from diffusers import AutoencoderKL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("无法导入 diffusers.AutoencoderKL") from e

    vae_cfg = AutoencoderKL.load_config(config, subfolder="vae", local_files_only=local_files_only)
    vae = AutoencoderKL.from_config(vae_cfg)

    vae_tensors = _load_safetensors_subset(
        ckpt_path,
        prefixes=[
            "vae.",
            "first_stage_model.",
            "autoencoder.",
        ],
    )
    if not vae_tensors:
        raise RuntimeError("在 checkpoint 里没找到 VAE 权重前缀：`vae.* / first_stage_model.* / autoencoder.*`")

    # 先尝试直接剥前缀匹配
    direct = _best_prefix_strip_map(
        vae_tensors,
        vae.state_dict().keys(),
        candidates=[
            ("vae.", False),
            ("first_stage_model.", False),
            ("autoencoder.", False),
        ],
    )

    if direct and len(direct) > 1000:
        incompat = vae.load_state_dict(direct, strict=False)
        missing = len(getattr(incompat, "missing_keys", []))
        unexpected = len(getattr(incompat, "unexpected_keys", []))
        print(
            f"[vae] tensors_loaded={len(vae_tensors)} target_keys={len(vae.state_dict())} direct_mapped={len(direct)} missing={missing} unexpected={unexpected}"
        )
    else:
        # 再尝试 LDM->diffusers 的转换（对很多第三方 checkpoint 很有效）
        try:
            from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint  # type: ignore

            converted = convert_ldm_vae_checkpoint(vae_tensors, vae_cfg)
            incompat = vae.load_state_dict(converted, strict=False)
            missing = len(getattr(incompat, "missing_keys", []))
            unexpected = len(getattr(incompat, "unexpected_keys", []))
            print(
                f"[vae] tensors_loaded={len(vae_tensors)} target_keys={len(vae.state_dict())} converted_mapped={len(converted)} missing={missing} unexpected={unexpected}"
            )
        except Exception as e:
            raise RuntimeError(
                "VAE 权重未能直接匹配，且 convert_ldm_vae_checkpoint 转换失败。"
                "这通常意味着该单文件的 VAE 结构/命名与 `--config` 的 vae/config.json 不一致。"
            ) from e

    vae = vae.to(dtype=torch_dtype)
    vae.eval()
    return vae


def _build_flux_transformer_from_ckpt(
    ckpt_path: str,
    config: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    disable_mmap: bool,
):
    """
    构建 FluxTransformer2DModel 并从单文件灌权重。

    说明：
    - Flux 的 transformer 原始权重命名往往是 `double_blocks.* / single_blocks.* / time_in.* ...`，
      diffusers 内置了 convert_flux_transformer_checkpoint_to_diffusers 来完成映射；
    - 因此这里直接调用 `FluxTransformer2DModel.from_single_file(...)`（它会走映射函数）。
    """
    try:
        from diffusers import FluxTransformer2DModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("无法导入 diffusers.FluxTransformer2DModel（你的 diffusers 版本可能过旧）") from e

    # 尽量只喂给 transformer 相关权重：排除 VAE / TextEncoders，避免无用 key 造成额外处理/日志刷屏
    transformer_ckpt = _load_safetensors_excluding_prefixes(
        ckpt_path,
        excluded_prefixes=[
            "vae.",
            "first_stage_model.",
            "autoencoder.",
            "text_encoder.",
            "text_encoder_2.",
            "text_encoders.",
            "tokenizer.",
            "tokenizer_2.",
        ],
    )

    transformer = FluxTransformer2DModel.from_single_file(
        transformer_ckpt,
        config=config,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        disable_mmap=disable_mmap,
    )
    # `from_single_file` 内部会做一层 key 映射，这里至少把“读入的 key 数 / 模型期望 key 数”打印出来，方便观察
    print(f"[transformer] tensors_loaded={len(transformer_ckpt)} target_keys={len(transformer.state_dict())}")
    transformer.eval()
    return transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="单文件 AIO 权重（.safetensors/.ckpt）")
    parser.add_argument(
        "--config",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="diffusers 模型目录或 repo id（用于读取各组件 config.json；建议本地目录以便离线）",
    )
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--negative", type=str, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/flux.png")
    parser.add_argument("--local-files-only", action="store_true", help="仅使用本地文件/缓存（无网络）")
    parser.add_argument("--disable-mmap", action="store_true", help="禁用 safetensors mmap（某些环境更快）")
    parser.add_argument(
        "--no-override-components",
        action="store_true",
        help="不手工构建 vae/transformer，完全依赖 FluxPipeline.from_single_file 自动推断加载（用于对比）",
    )
    parser.add_argument(
        "--pretouch",
        action="store_true",
        help="在 pipe.to('cuda') 前对 CPU 参数做 pretouch（可显著减少 GB10 上的 to cuda 长尾）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise ValueError(f"--ckpt 不是本地文件：{args.ckpt}")

    print(f"Loading FluxPipeline from single file: {args.ckpt}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _print_ckpt_component_probe(args.ckpt)

    torch_dtype = torch.bfloat16

    vae = transformer = None
    if not args.no_override_components:
        t0 = time.time()
        vae = _build_vae_from_ckpt(args.ckpt, args.config, torch_dtype, args.local_files_only)
        print(f"载入vae完成,耗时={time.time()-t0:.3f}s,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        t0 = time.time()
        transformer = _build_flux_transformer_from_ckpt(
            args.ckpt, args.config, torch_dtype, args.local_files_only, args.disable_mmap
        )
        print(
            f"载入transformer完成,耗时={time.time()-t0:.3f}s,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # 关键：仍然用 FluxPipeline.from_single_file()，但可把组件提前传入，确保使用单文件里的组件权重
    pipe = FluxPipeline.from_single_file(
        args.ckpt,
        config=args.config,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
        disable_mmap=args.disable_mmap,
        vae=vae,
        transformer=transformer,
    )
    print(f"载入pipeline完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _print_text_encoder_match_stats(pipe, args.ckpt)

    if args.pretouch:
        try:
            from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

            pretouch_pipeline_cpu_tensors(pipe, on_component=lambda n: print(f"{n} pretouch完成"))
        except Exception as e:
            raise RuntimeError("启用 --pretouch 需要本仓库的 inference/common/torch_transfer_utils.py") from e

    pipe.to("cuda")
    print(f"To cuda完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    generator = torch.Generator("cuda").manual_seed(args.seed)

    call_kwargs = dict(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    if args.negative is not None:
        call_kwargs["negative_prompt"] = args.negative
    if args.height is not None:
        call_kwargs["height"] = args.height
    if args.width is not None:
        call_kwargs["width"] = args.width
    if args.max_seq_len is not None:
        call_kwargs["max_sequence_length"] = args.max_seq_len

    t0 = time.time()
    image = pipe(**call_kwargs).images[0]
    print(f"生成图片完成,耗时={time.time()-t0:.3f}s,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    image.save(args.out)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测单文件模型里是否包含常见组件权重：vae / text_encoder / text_encoder_2 / transformer / unet

用法：
  python scripts/detect_singlefile_components.py /path/to/model.safetensors --max-samples 2 --prefix-depth 3
  python scripts/detect_singlefile_components.py /path/to/model.ckpt --json

说明：
- 对 .safetensors：仅读取 header 的 key（很快，不会把全部 tensor 读入内存）
- 对 .ckpt/.pt/.pth：需要 torch.load 才能拿到 key（可能较慢/占内存）
- transformer vs unet 在很多单文件里会用通用前缀 `model.diffusion_model.*`，无法严格区分。
  对 SD/SDXL 这类 checkpoint（出现 input_blocks/middle_block/output_blocks）会判定为 UNet=yes。
  其它只有 diffusion_model 但缺乏结构信号时，脚本会给 transformer/unet 标记为 "maybe"（可能）避免误判。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def _load_keys_from_safetensors(path: str) -> List[str]:
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖：safetensors。请先安装：pip install safetensors") from e

    with safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys())


def _unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    """
    兼容常见 checkpoint 结构：
    - 直接就是 state_dict
    - {"state_dict": ...}
    - {"model": ...}
    - {"module": ...}
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "module", "net", "ema", "params"):
            v = obj.get(k)
            if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                return v
        if all(isinstance(kk, str) for kk in obj.keys()):
            return obj
    raise RuntimeError("无法从该文件解析出 state_dict（不支持的格式或损坏文件）")


def _load_keys_from_torch_checkpoint(path: str) -> List[str]:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖：torch。请先安装 PyTorch") from e

    obj = torch.load(path, map_location="cpu")
    sd = _unwrap_state_dict(obj)
    return list(sd.keys())


def load_keys(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return _load_keys_from_safetensors(path)
    if ext in (".ckpt", ".pt", ".pth", ".bin"):
        return _load_keys_from_torch_checkpoint(path)
    raise ValueError(f"不支持的文件后缀：{ext}（支持：.safetensors/.ckpt/.pt/.pth/.bin）")


def prefix_counts(keys: Iterable[str], depth: int = 2, prefix_filter: str = "") -> List[Tuple[str, int]]:
    """
    按 key 的前缀进行计数，便于快速看“有哪些大类/子类”。

    示例：
    - depth=1: model / vae / text_encoders / conditioner ...
    - depth=2: model.diffusion_model / text_encoders.qwen3_4b / conditioner.embedders ...
    """
    depth = max(1, int(depth))
    pf = (prefix_filter or "").strip()
    c = Counter()
    for k in keys:
        if pf and not k.startswith(pf):
            continue
        parts = k.split(".")
        p = ".".join(parts[:depth]) if len(parts) >= depth else ".".join(parts)
        c[p] += 1
    return c.most_common()


def detect_components(keys: List[str], max_samples: int = 30) -> Dict[str, Any]:
    # 显式信号（强指示）
    sig_transformer = ("transformer.",)
    sig_unet = ("unet.",)

    # 通用 diffusion backbone（很多单文件只有这个，无法严格区分 UNet vs transformer）
    sig_diffusion_generic = ("model.diffusion_model.", "diffusion_model.")

    out: Dict[str, Any] = {
        "total_keys": len(keys),
        "components": {},
        "signals": {},
    }

    def scan(prefixes: Tuple[str, ...]) -> Tuple[int, List[str]]:
        cnt = 0
        samples: List[str] = []
        for k in keys:
            if k.startswith(prefixes):
                cnt += 1
                if len(samples) < max_samples:
                    samples.append(k)
        return cnt, samples

    def scan_contains(tokens: Tuple[str, ...]) -> Tuple[int, List[str]]:
        cnt = 0
        samples: List[str] = []
        tokens_l = tuple(t.lower() for t in tokens)
        for k in keys:
            kl = k.lower()
            if any(t in kl for t in tokens_l):
                cnt += 1
                if len(samples) < max_samples:
                    samples.append(k)
        return cnt, samples

    # VAE / TE 用“contains”更稳一点（前缀命名差异较大）
    vae_cnt, vae_samples = scan_contains(("vae.", "first_stage_model.", "autoencoder"))
    te1_cnt, te1_samples = scan_contains(("text_encoder.", "cond_stage_model.", "clip.", "t5.", "text_model."))
    te2_cnt, te2_samples = scan_contains(("text_encoder_2.", "cond_stage_model_2.", "text_encoder2."))

    # 一些项目会用 text_encoders.<name>.*（例如 text_encoders.qwen3_4b.*）
    te_any_cnt, te_any_samples = scan(("text_encoders.",))
    out["signals"]["text_encoders_prefix"] = {"text_encoders": te_any_cnt}

    # SDXL checkpoint 常见命名：conditioner.embedders.0/1（0/1 分别对应两个文本编码器）
    emb0_cnt, emb0_samples = scan(("conditioner.embedders.0.",))
    emb1_cnt, emb1_samples = scan(("conditioner.embedders.1.",))
    out["signals"]["conditioner_embedders"] = {
        "embedders_0": emb0_cnt,
        "embedders_1": emb1_cnt,
    }

    # transformer/unet：优先看显式前缀；否则只看 diffusion_model 作为“可能”
    tr_exp_cnt, tr_exp_samples = scan(sig_transformer)
    un_exp_cnt, un_exp_samples = scan(sig_unet)
    diff_cnt, diff_samples = scan(sig_diffusion_generic)

    # 结构化判别：
    # - UNet checkpoint 强信号：input_blocks/middle_block/output_blocks（SD/SDXL 常见）
    unet_struct_cnt, unet_struct_samples = scan(
        (
            "model.diffusion_model.input_blocks.",
            "model.diffusion_model.middle_block.",
            "model.diffusion_model.output_blocks.",
        )
    )
    # - DiT/Transformer 强信号：time_text_embed / img_in / txt_in / pos_embed 等（避免被 UNet 内部 transformer_blocks 误导）
    tr_struct_cnt, tr_struct_samples = scan(
        (
            # Qwen/DiT 系
            "model.diffusion_model.time_text_embed.",
            "model.diffusion_model.pos_embed.",
            "model.diffusion_model.img_in.",
            "model.diffusion_model.txt_in.",
            "model.diffusion_model.transformer_blocks.",
            # 一些 DiT/Transformer 模型会有 caption/token embedder
            "model.diffusion_model.cap_embedder.",
            "model.diffusion_model.caption_embedder.",
            "model.diffusion_model.token_embedder.",
            "model.diffusion_model.patch_embed.",
            "model.diffusion_model.blocks.",
        )
    )
    out["signals"]["diffusion_structure"] = {
        "unet_like_blocks": unet_struct_cnt,
        "dit_like_blocks": tr_struct_cnt,
    }

    def yn(x: int) -> str:
        return "yes" if x > 0 else "no"

    out["components"]["vae"] = {"status": yn(vae_cnt), "matches": vae_cnt, "samples": vae_samples}
    # text_encoder / text_encoder_2：同时兼容 diffusers 命名与 SDXL ckpt 的 conditioner.embedders.{0,1}
    te1_has = (te1_cnt > 0) or (emb0_cnt > 0) or (te_any_cnt > 0)
    te2_has = (te2_cnt > 0) or (emb1_cnt > 0)
    out["components"]["text_encoder"] = {
        "status": "yes" if te1_has else "no",
        "matches": te1_cnt,
        "matches_conditioner_embedders_0": emb0_cnt,
        "matches_text_encoders_prefix": te_any_cnt,
        "samples": te1_samples or emb0_samples or te_any_samples,
    }
    out["components"]["text_encoder_2"] = {
        "status": "yes" if te2_has else "no",
        "matches": te2_cnt,
        "matches_conditioner_embedders_1": emb1_cnt,
        "samples": te2_samples or emb1_samples,
    }

    # UNet 判别优先级更高：
    # - 只要出现 UNet 结构块（input/middle/output_blocks）或显式 unet.*，就判定 unet=yes
    # - 如果出现明显 DiT/Transformer 结构信号且没有任何 UNet 结构信号，则判定 unet=no（避免误报）
    if un_exp_cnt > 0 or unet_struct_cnt > 0:
        un_status = "yes"
    elif tr_struct_cnt > 0 and unet_struct_cnt == 0 and un_exp_cnt == 0:
        un_status = "no"
    elif diff_cnt > 0 and tr_exp_cnt == 0:
        # 只有 diffusion_model 前缀但缺乏结构信号：保守给 maybe
        un_status = "maybe"
    else:
        un_status = "no"

    if tr_exp_cnt > 0:
        tr_status = "yes"
    elif tr_struct_cnt > 0 and unet_struct_cnt == 0 and un_exp_cnt == 0:
        tr_status = "yes"
    elif unet_struct_cnt > 0 and tr_exp_cnt == 0:
        tr_status = "no"
    elif diff_cnt > 0 and un_exp_cnt == 0:
        tr_status = "maybe"
    else:
        tr_status = "no"

    out["components"]["transformer"] = {
        "status": tr_status,
        "matches_explicit": tr_exp_cnt,
        "matches_generic_diffusion_model": diff_cnt,
        "matches_struct_dit_like": tr_struct_cnt,
        "samples_explicit": tr_exp_samples,
        "samples_generic": diff_samples[:max_samples],
        "samples_struct": tr_struct_samples,
    }
    out["components"]["unet"] = {
        "status": un_status,
        "matches_explicit": un_exp_cnt,
        "matches_generic_diffusion_model": diff_cnt,
        "matches_struct_unet_like": unet_struct_cnt,
        "samples_explicit": un_exp_samples,
        "samples_generic": diff_samples[:max_samples],
        "samples_struct": unet_struct_samples,
    }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="单文件模型路径（.safetensors/.ckpt/.pt/.pth/.bin）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--max-samples", type=int, default=30, help="每个组件最多打印多少条 key 样例")
    ap.add_argument("--prefix-depth", type=int, default=2, help="key 前缀统计深度（1/2/3...）。例如 depth=2 可看到 text_encoders.qwen3_4b")
    ap.add_argument("--prefix-filter", type=str, default="", help="仅统计以该前缀开头的 key（例如 conditioner.embedders. 或 text_encoders.）")
    ap.add_argument("--prefix-top", type=int, default=50, help="前缀统计输出前 N 条（按计数降序）。设为 0 表示不限制")
    ap.add_argument("--prefix-all", action="store_true", help="输出全部前缀统计（等价于 --prefix-top 0）")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print(f"[err] 文件不存在：{args.path}", file=sys.stderr)
        return 2

    try:
        keys = load_keys(args.path)
        result = detect_components(keys, max_samples=max(0, int(args.max_samples)))
        result["file"] = os.path.abspath(args.path)
        result["ext"] = os.path.splitext(args.path)[1].lower()
        # 额外：输出 key 分类统计，便于后续迭代规则而无需“猜”
        depth = int(args.prefix_depth)
        pf = args.prefix_filter or ""
        pc = prefix_counts(keys, depth=depth, prefix_filter=pf)
        top_n = 0 if args.prefix_all else int(args.prefix_top)
        if top_n > 0:
            pc = pc[:top_n]
        result["prefix_counts"] = {"depth": depth, "filter": pf, "counts": pc}
    except Exception as e:
        print(f"[err] {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"file: {result['file']}")
    print(f"total_keys: {result['total_keys']}")
    pc = result.get("prefix_counts", {})
    depth = pc.get("depth", 2)
    pf = pc.get("filter", "")
    counts = pc.get("counts", [])
    print(f"\n== prefix counts (depth={depth}, filter={pf or 'NONE'}) ==")
    for p, cnt in counts:
        print(f"{cnt:8d}  {p}")

    print("\n== components ==")
    comps = result["components"]
    for name in ("vae", "text_encoder", "text_encoder_2", "transformer", "unet"):
        info = comps[name]
        status = info["status"]
        if name in ("transformer", "unet"):
            extra = []
            extra.append(f"explicit={info['matches_explicit']}")
            extra.append(f"generic_diffusion_model={info['matches_generic_diffusion_model']}")
            if name == "transformer" and "matches_struct_dit_like" in info:
                extra.append(f"struct_dit_like={info['matches_struct_dit_like']}")
            if name == "unet" and "matches_struct_unet_like" in info:
                extra.append(f"struct_unet_like={info['matches_struct_unet_like']}")
            print(f"- {name}: {status} ({', '.join(extra)})")
        else:
            extra = [f"matches={info.get('matches', 0)}"]
            if name == "text_encoder" and "matches_conditioner_embedders_0" in info:
                extra.append(f"embedders_0={info['matches_conditioner_embedders_0']}")
            if name == "text_encoder" and "matches_text_encoders_prefix" in info:
                extra.append(f"text_encoders={info['matches_text_encoders_prefix']}")
            if name == "text_encoder_2" and "matches_conditioner_embedders_1" in info:
                extra.append(f"embedders_1={info['matches_conditioner_embedders_1']}")
            print(f"- {name}: {status} ({', '.join(extra)})")

    print("\n== samples (first N) ==")
    for name in ("vae", "text_encoder", "text_encoder_2"):
        info = comps[name]
        if info["samples"]:
            print(f"\n[{name}]")
            for k in info["samples"]:
                print(k)

    # transformer/unet 的样例优先显示 explicit，否则显示 generic
    for name in ("transformer", "unet"):
        info = comps[name]
        samples = info["samples_explicit"] or info["samples_generic"]
        if samples:
            print(f"\n[{name}] ({'explicit' if info['samples_explicit'] else 'generic_diffusion_model'})")
            for k in samples:
                print(k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


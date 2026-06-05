import argparse
import inspect
import os
import time
from datetime import datetime

import torch
from diffusers import QwenImagePipeline

"""
Qwen 单文件权重试验脚本（简洁版）

### 背景/结论
- 你本地的 diffusers `QwenImagePipeline` **没有** `from_single_file`。
- 很多第三方“单文件 .safetensors”实际只包含 `model.diffusion_model.*`（也就是 diffusion transformer 权重）。
- 因此这里采用稳定路线：
  1) 用 `--config` 下的各组件 `config.json` 构建 **空壳**（scheduler/vae/text_encoder/tokenizer/transformer）
  2) 把 `--ckpt` 里的 `model.diffusion_model.*` 剥前缀后灌入 `pipe.transformer`
  3) 可选 `--pretouch` 加速 GB10 等平台的 H2D
  4) `pipe.to(cuda)` 后跑一次推理并保存图片

### 已尝试/评估过的加速方案（简要）
- **跳过超慢随机初始化（已保留，默认启用）**
  - 用 `accelerate.init_empty_weights()` + `to_empty()` 构建 `text_encoder/transformer` 空壳，再 `load_state_dict(assign=True)` 直接挂权重；
  - 这能把“build text_encoder/transformer”从 ~60s 级别降到秒级。
- **CPU 端 pretouch（已保留，可选）**
  - `pretouch_pipeline_cpu_tensors(pipe, ...)`：在 `to(cuda)` 前预触达参数页，减少 GB10 这类平台的 H2D 长尾；
  - 代价：会增加一些 CPU 预处理时间。
- **强制 SDPA/Flash/Mem‑Efficient / 禁用 attn_mask（已移除，不建议）**
  - QwenImage/SD3 系模型常见“非空 mask + GQA/跨流 attention”组合；
  - 强制 flash/mem_efficient 或禁用 mask 容易导致 **结果异常** 或直接报 “No available kernel”，且对主循环加速不稳定。
- **dtype 覆盖（已保留，默认 bf16）**
  - 注入权重时强制 cast 到 `--dtype`，避免 `assign=True` 把参数 dtype 意外覆盖到 FP32 导致极慢。

### 用法
python test/single_file_qwen.py \
  --ckpt /home/tonera/models/copaxTimeless_qwenUltraRealistic.safetensors \
  --config inference/config/qwen \
  --out outputs/qwen_singlefile.png \
  --pretouch

注意：有些 模型只有transformer权重，没有text_encoder/vae/tokenizer权重，
"""


def _resolve_dtype(dtype: str) -> torch.dtype:
    d = (dtype or "").lower().strip()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"不支持的 dtype：{dtype}（可选：bf16/fp16/fp32）")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="单文件权重（.safetensors/.ckpt）")
    parser.add_argument("--config", type=str, required=True, help="本地 diffusers 配置目录（含各子目录 config.json）")
    parser.add_argument("--prompt", type=str, default="A cute cat, high quality")
    parser.add_argument("--negative", type=str, default=" ")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--move",
        type=str,
        default="all",
        choices=["all", "transformer-only", "none"],
        help="all=pipe.to(device), transformer-only=只搬 transformer, none=不搬（仅验证注入链路）",
    )
    parser.add_argument(
        "--pretouch",
        action="store_true",
        help="在搬运到 CUDA 前对 CPU 参数做 pretouch（可显著减少 GB10 上的 to cuda 长尾）",
    )
    parser.add_argument("--out", type=str, default="outputs/qwen_singlefile.png")
    parser.add_argument("--local-files-only", action="store_true", help="仅使用本地文件/缓存（无网络）")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise ValueError(f"--ckpt 不是本地文件：{args.ckpt}")

    start_time = time.time()
    torch_dtype = _resolve_dtype(args.dtype)

    print(f"[qwen_singlefile] start={_now()}")
    print(f"[qwen_singlefile] ckpt={args.ckpt}")
    print(f"[qwen_singlefile] config={args.config}")

    # accelerate（可选）：用于跳过超慢随机初始化
    try:
        from accelerate import init_empty_weights  # type: ignore

        has_accelerate = True
    except Exception:
        init_empty_weights = None
        has_accelerate = False

    # 1) 从 config 构建空壳组件
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    t0 = time.time()
    scheduler_cfg = FlowMatchEulerDiscreteScheduler.load_config(args.config, subfolder="scheduler", local_files_only=args.local_files_only)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)
    print(f"[scheduler] built,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    t0 = time.time()
    vae_cfg = AutoencoderKLQwenImage.load_config(args.config, subfolder="vae", local_files_only=args.local_files_only)
    vae = AutoencoderKLQwenImage.from_config(vae_cfg).to(dtype=torch_dtype)
    print(f"[vae] built,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    t0 = time.time()
    te_cfg = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(
        args.config, subfolder="text_encoder", local_files_only=args.local_files_only
    )
    if has_accelerate and init_empty_weights is not None:
        with init_empty_weights():
            text_encoder = Qwen2_5_VLForConditionalGeneration(te_cfg)
        if hasattr(text_encoder, "to_empty"):
            text_encoder.to_empty(device="cpu")
    else:
        text_encoder = Qwen2_5_VLForConditionalGeneration(te_cfg)
    text_encoder = text_encoder.to(dtype=torch_dtype)
    print(f"[text_encoder] built,耗时={time.time()-t0:.3f}s,当前时间: {_now()}{' (empty_init)' if has_accelerate else ''}")

    t0 = time.time()
    tokenizer = Qwen2Tokenizer.from_pretrained(args.config, subfolder="tokenizer", local_files_only=args.local_files_only)
    print(f"[tokenizer] loaded,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    # 2) 读取单文件并抽取 transformer 权重
    try:
        from safetensors.torch import load_file as safetensors_load_file  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 safetensors 才能读取 .safetensors") from e

    t0 = time.time()
    sd = safetensors_load_file(args.ckpt, device="cpu")
    print(f"[ckpt] loaded,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    prefix = "model.diffusion_model."
    t0 = time.time()
    remapped = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if not remapped:
        raise RuntimeError("未在 ckpt 中发现 `model.diffusion_model.*` 前缀权重，无法注入 transformer。")
    # 保持 dtype 一致（避免 assign=True 把参数 dtype 覆盖成 FP32）
    for k in list(remapped.keys()):
        v = remapped[k]
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch_dtype:
            remapped[k] = v.to(dtype=torch_dtype)
    print(f"[ckpt] remap_done keys={len(remapped)},耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    # 3) 构建 transformer（空壳）并注入权重
    t0 = time.time()
    transformer_cfg = QwenImageTransformer2DModel.load_config(
        args.config, subfolder="transformer", local_files_only=args.local_files_only
    )
    if has_accelerate and init_empty_weights is not None:
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel.from_config(transformer_cfg)
    else:
        transformer = QwenImageTransformer2DModel.from_config(transformer_cfg)
    transformer = transformer.to(dtype=torch_dtype)
    print(f"[transformer] built,耗时={time.time()-t0:.3f}s,当前时间: {_now()}{' (empty_init)' if has_accelerate else ''}")

    t0 = time.time()
    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    print(f"[pipeline] constructed,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    t0 = time.time()
    load_kwargs = {}
    assign_supported = "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters
    if has_accelerate and assign_supported:
        load_kwargs["assign"] = True
    incompat = pipe.transformer.load_state_dict(remapped, strict=False, **load_kwargs)
    missing = len(getattr(incompat, "missing_keys", []))
    unexpected = len(getattr(incompat, "unexpected_keys", []))
    print(f"[inject] transformer loaded keys={len(remapped)} missing={missing} unexpected={unexpected},耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    # 4) pretouch + to(device)
    if args.move != "none":
        if args.pretouch:
            from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

            t0 = time.time()
            pretouch_pipeline_cpu_tensors(pipe, on_component=lambda n: print(f"[pretouch] {n} done"))
            print(f"[pretouch] all_done,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

        t0 = time.time()
        if args.move == "all":
            pipe.to(args.device)
            print(f"[to_device] pipe.to({args.device}) done,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")
        elif args.move == "transformer-only":
            pipe.transformer.to(args.device)
            print(f"[to_device] transformer.to({args.device}) done,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    # 5) 推理 + 保存
    gen = torch.Generator("cpu").manual_seed(args.seed)
    with torch.inference_mode():
        t0 = time.time()
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            generator=gen,
        ).images[0]
        print(f"[infer] done,耗时={time.time()-t0:.3f}s,当前时间: {_now()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    image.save(args.out)
    print(f"[save] saved: {args.out},耗时={time.time()-t0:.3f}s,当前时间: {_now()}")
    print(f"[total] total_time={time.time()-start_time:.3f}s")


if __name__ == "__main__":
    main()


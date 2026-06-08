"""
SDXL + torch.compile 单项加速测试脚本（尽量“零额外安装”）

目标：
- 在不引入第三方编译扩展（如 sfast）的前提下，测试 torch.compile 对 SDXL(base) 的提速效果
- 支持 baseline / compile 两种模式（你可按需单独跑）
- 固定 seed + 相同推理参数，便于对比耗时与图片质量
- compile 失败自动降级到 baseline（适配复杂环境/多样硬件）

使用示例（项目根目录）：
    # baseline
    python test/test_sdxl_compile.py --mode baseline

    # compile（只编译 unet，最常见的收益点）
    python test/test_sdxl_compile.py --mode compile
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline

from inference.common.sdpa_utils import sdpa_ctx


@dataclass
class RunResult:
    elapsed_s: float
    image: Any


def _torch_dtype(name: str) -> torch.dtype:
    name = (name or "").lower().strip()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _print_env() -> None:
    print("=== ENV ===")
    print("python:", platform.python_version())
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print("gpu:", torch.cuda.get_device_name(idx))
        print("cc:", torch.cuda.get_device_capability(idx))
        if hasattr(torch.cuda, "get_arch_list"):
            try:
                print("arch_list:", torch.cuda.get_arch_list())
            except Exception:
                pass
    print(
        "SDP backends:",
        "flash=", torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else "N/A",
        "mem_efficient=", torch.backends.cuda.mem_efficient_sdp_enabled() if torch.cuda.is_available() else "N/A",
        "math=", torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else "N/A",
    )
    print("============")


def _load_pipe(model: str, *, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _try_compile_unet(
    pipe: StableDiffusionXLPipeline,
    *,
    compile_backend: str,
    compile_mode: str,
    fullgraph: bool,
    dynamic: bool,
) -> bool:
    """
    只 compile UNet（SDXL 主要耗时在这里），并保持其它模块不动。
    返回是否成功启用 compile。
    """
    if not hasattr(torch, "compile"):
        print("[compile] torch.compile 不存在，跳过。")
        return False

    # inductor 后端通常依赖 “可用的 triton”，仅 import triton 不够（有时能 import 但不可用）。
    if compile_backend == "inductor":
        triton_ok = False
        triton_err = None
        try:
            from torch._inductor import utils as inductor_utils  # type: ignore

            if hasattr(inductor_utils, "has_triton"):
                triton_ok = bool(inductor_utils.has_triton())
            else:
                # 兜底：仅能做 import 检查
                import triton  # type: ignore  # noqa: F401
                triton_ok = True
        except Exception as e:
            triton_err = e
            triton_ok = False

        if not triton_ok:
            msg = f"{triton_err}" if triton_err is not None else "unknown"
            print(f"[compile] backend=inductor 需要可用的 triton，但当前不可用：{msg}")
            print("[compile] 将跳过 compile（保持 baseline）。你也可以尝试：--compile-backend aot_eager")
            return False

    try:
        # 经验：把 compile 放在 .to(cuda) 之后更稳
        pipe.unet = torch.compile(  # type: ignore[attr-defined]
            pipe.unet,
            backend=compile_backend,
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        print(
            f"[compile] enabled: backend={compile_backend}, mode={compile_mode}, "
            f"fullgraph={fullgraph}, dynamic={dynamic}"
        )
        return True
    except Exception as e:
        print(f"[compile] failed, fallback to baseline: {e}")
        return False


def _run_once(
    pipe: StableDiffusionXLPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    seed: int,
) -> RunResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    _sync_if_cuda()
    t0 = time.perf_counter()
    with torch.inference_mode():
        with sdpa_ctx():
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator,
            )
    _sync_if_cuda()
    t1 = time.perf_counter()
    return RunResult(elapsed_s=t1 - t0, image=out.images[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "compile"], default="compile")
    parser.add_argument("--model", default="resources/models/xl-base")
    parser.add_argument("--dtype", default=os.environ.get("SDXL_DTYPE", "bf16"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default="outputs/_tmp_sdxl_compile.png")

    # compile 相关参数
    parser.add_argument("--compile-backend", default=os.environ.get("SDXL_COMPILE_BACKEND", "inductor"))
    parser.add_argument("--compile-mode", default=os.environ.get("SDXL_COMPILE_MODE", "reduce-overhead"))
    parser.add_argument("--fullgraph", action="store_true", default=False)
    parser.add_argument("--dynamic", action="store_true", default=False)

    args = parser.parse_args()

    _print_env()
    dtype = _torch_dtype(args.dtype)
    print(
        f"mode={args.mode}, model={args.model}, dtype={dtype}, steps={args.steps}, cfg={args.guidance}, "
        f"size={args.width}x{args.height}"
    )

    if torch.cuda.is_available():
        # 测试友好：尽量让 kernel 选型稳定 + 提升 matmul 性能
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = _load_pipe(args.model, dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    negative = ""

    # baseline warmup（在相同 sdpa_ctx() 行为下）
    for i in range(max(0, args.warmup)):
        r = _run_once(
            pipe,
            prompt=prompt,
            negative_prompt=negative,
            steps=min(args.steps, 2),
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed + i,
        )
        print(f"warmup[{i}] {r.elapsed_s:.3f}s")

    if args.mode == "compile":
        ok = _try_compile_unet(
            pipe,
            compile_backend=args.compile_backend,
            compile_mode=args.compile_mode,
            fullgraph=bool(args.fullgraph),
            dynamic=bool(args.dynamic),
        )
        if ok:
            try:
                # compile 后再 warmup 一次，覆盖编译/图捕获开销
                r2 = _run_once(
                    pipe,
                    prompt=prompt,
                    negative_prompt=negative,
                    steps=min(args.steps, 2),
                    guidance=args.guidance,
                    height=args.height,
                    width=args.width,
                    seed=args.seed,
                )
                print(f"post-compile warmup {r2.elapsed_s:.3f}s")
            except Exception as e:
                print(f"[compile] 运行期失败，自动回退 baseline：{repr(e)}")
                args.mode = "baseline"
                pipe = _load_pipe(args.model, dtype=dtype)
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    try:
        result = _run_once(
            pipe,
            prompt=prompt,
            negative_prompt=negative,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
    except Exception as e:
        # 如果 compile 路径在正式跑时失败，再兜底回退一次
        if args.mode == "compile":
            print(f"[compile] 正式推理失败，自动回退 baseline：{repr(e)}")
            args.mode = "baseline"
            pipe = _load_pipe(args.model, dtype=dtype)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            result = _run_once(
                pipe,
                prompt=prompt,
                negative_prompt=negative,
                steps=args.steps,
                guidance=args.guidance,
                height=args.height,
                width=args.width,
                seed=args.seed,
            )
        else:
            raise
    print(f"Time taken ({args.mode}): {result.elapsed_s:.3f}s")

    # 保存输出（目录不存在则尽量创建）
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    except Exception:
        pass
    try:
        result.image.save(args.out)
        print("Saved:", args.out)
    except Exception as e:
        print("Save failed:", e)


if __name__ == "__main__":
    main()



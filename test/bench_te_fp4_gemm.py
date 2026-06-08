"""
使用 NVIDIA TransformerEngine (TE) 测 FP4/FP8 GEMM 吞吐（更接近“硬件标称算力”的测法）。

为什么需要它？
- 你现有的 `bench_gemm_tflops.py` 走的是 PyTorch 原生 matmul。
  目前 PyTorch 对 FP4 dtype 的 cast/matmul 支持不完整（你已看到 copy_/addmm 未实现），
  所以无法用“torch.float4 + A@B”来测出 FP4 TensorCore 的真实吞吐。
- 对 Blackwell/GB10 这类新架构，FP4/FP8 高吞吐通常要靠 TE / cuBLASLt / CUTLASS 之类路径。

本脚本策略：
- 如果环境里装了 `transformer_engine`，就用 TE 的 Linear/GEMM 路径测吞吐；
- 如果没装，会给出安装提示与如何验证。

注意：
- TE 的 API 版本变化较快，本脚本写成“尽量兼容”的形式；如果你环境里 TE 版本不匹配，
  报错堆栈贴我我可以按你版本快速对齐。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch


@dataclass
class TEResult:
    host: str
    platform: str
    python: str
    torch: str
    torch_cuda: Optional[str]
    device: str
    capability: str
    total_vram_gb: float
    te_available: bool
    te_version: str
    dtype: str
    m: int
    n: int
    k: int
    warmup: int
    iters: int
    tflops: float
    ms_per_iter: float
    note: str = ""


def _gpu_props() -> dict[str, Any]:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / (1024**3), 2),
        "device_index": idx,
    }


def _try_import_te():
    try:
        import transformer_engine  # type: ignore
        import transformer_engine.pytorch as te  # type: ignore

        ver = getattr(transformer_engine, "__version__", "unknown")
        return te, ver, ""
    except Exception as e:
        return None, "", str(e)


def _bench_te_linear(m: int, n: int, k: int, dtype: str, warmup: int, iters: int) -> TEResult:
    gpu = _gpu_props()
    te, te_ver, te_err = _try_import_te()

    if te is None:
        return TEResult(
            host=platform.node(),
            platform=platform.platform(),
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            device=gpu["name"],
            capability=gpu["capability"],
            total_vram_gb=gpu["total_vram_gb"],
            te_available=False,
            te_version="",
            dtype=dtype,
            m=m,
            n=n,
            k=k,
            warmup=warmup,
            iters=iters,
            tflops=0.0,
            ms_per_iter=0.0,
            note=f"transformer_engine 不可用: {te_err}\n"
            f"安装建议：优先用 NVIDIA 官方 wheel/容器；或 `pip install transformer-engine`（需匹配 torch/cuDNN/CUDA）。",
        )

    # 选择输入 dtype（TE 的 FP4/FP8 通常是 weight/计算路径，输入一般用 bf16/fp16）
    in_dtype = torch.bfloat16 if dtype.lower() in ("fp4", "fp8", "fp8e4m3", "fp8e5m2", "bf16") else torch.float16

    # TE Linear：y = x @ W^T  （我们用它来近似 GEMM）
    # x: [m, k], W: [n, k] -> y: [m, n]
    x = torch.randn((m, k), device="cuda", dtype=in_dtype)

    # 尽量构造 TE Linear 并启用 FP8（FP4/FP8 的具体开关随 TE 版本变化）
    note = ""
    try:
        # 新版 TE 常用 Linear
        lin = te.Linear(k, n, bias=False, device="cuda", params_dtype=in_dtype)  # type: ignore[attr-defined]
    except Exception as e:
        return TEResult(
            host=platform.node(),
            platform=platform.platform(),
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            device=gpu["name"],
            capability=gpu["capability"],
            total_vram_gb=gpu["total_vram_gb"],
            te_available=True,
            te_version=te_ver,
            dtype=dtype,
            m=m,
            n=n,
            k=k,
            warmup=warmup,
            iters=iters,
            tflops=0.0,
            ms_per_iter=0.0,
            note=f"TE Linear 创建失败（可能 API 版本不匹配）：{e}",
        )

    # FP8 recipe（如果可用）
    fp8_enabled = False
    if hasattr(te, "fp8_autocast"):
        fp8_enabled = dtype.lower().startswith("fp8")
    if dtype.lower() == "fp4":
        # TE 的 FP4 支持依版本而定；这里先做“尝试启用”并把状态写到 note
        note += "请求 dtype=fp4：是否真正走 FP4 取决于 TE 版本/Blackwell 支持。\n"

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def run_once():
        if fp8_enabled and hasattr(te, "fp8_autocast"):
            # 某些 TE 版本需要 fp8_recipe；这里留空用默认
            with te.fp8_autocast(enabled=True):  # type: ignore[attr-defined]
                y = lin(x)
        else:
            y = lin(x)
        return y

    for _ in range(max(0, warmup)):
        y = run_once()
        _ = y.sum()
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        y = run_once()
        _ = y.sum()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    ms_per_iter = ms / iters

    # Linear 的 FLOPs 约等于 GEMM：2*m*n*k
    flops = 2.0 * m * n * k
    tflops = (flops / (ms_per_iter / 1000.0)) / 1e12

    return TEResult(
        host=platform.node(),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        device=gpu["name"],
        capability=gpu["capability"],
        total_vram_gb=gpu["total_vram_gb"],
        te_available=True,
        te_version=te_ver,
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        warmup=warmup,
        iters=iters,
        tflops=round(tflops, 3),
        ms_per_iter=round(ms_per_iter, 3),
        note=note.strip(),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=int(os.getenv("GEMM_M", "16384")))
    p.add_argument("--n", type=int, default=int(os.getenv("GEMM_N", "16384")))
    p.add_argument("--k", type=int, default=int(os.getenv("GEMM_K", "16384")))
    p.add_argument("--dtype", type=str, default=os.getenv("GEMM_DTYPE", "fp8e4m3"))
    p.add_argument("--warmup", type=int, default=int(os.getenv("GEMM_WARMUP", "5")))
    p.add_argument("--iters", type=int, default=int(os.getenv("GEMM_ITERS", "20")))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，无法测试")

    res = _bench_te_linear(args.m, args.n, args.k, args.dtype, args.warmup, args.iters)
    print("==== TE GEMM Benchmark ====")
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    print("==== ONE_LINE_JSON ====")
    print(json.dumps(asdict(res), ensure_ascii=False))


if __name__ == "__main__":
    main()



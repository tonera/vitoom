"""
测量 GPU 上矩阵乘 (GEMM) 的有效吞吐（TFLOPS）。

用途：
- 对比两台机器（GB10 vs 3090）在 BF16/FP16（以及可选 FP8/FP4）下的实际 GEMM 吞吐。
- 这更接近“硬件/内核实际算力是否被用上”，比端到端 diffusion 推理更可控。

说明（重要）：
- PyTorch 是否“真的能跑 FP4 TensorCore GEMM”取决于：PyTorch/驱动/CUDA 是否暴露 FP4 dtype + 内核。
  如果当前栈不支持，本脚本会明确提示，并仍输出 BF16/FP16（以及 FP8）作为对照。
- 默认会做少量 warmup；如果你只想测冷启动，把 --warmup 设为 0。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch


@dataclass
class RunInfo:
    host: str
    platform: str
    python: str
    torch: str
    torch_cuda: Optional[str]
    device: str
    capability: str
    total_vram_gb: float
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


def _resolve_dtype(name: str) -> tuple[Optional[torch.dtype], str]:
    """
    将用户输入的 dtype 名称解析为 torch.dtype。
    返回 (dtype或None, 说明字符串)
    """
    name = name.lower()

    if name in ("bf16", "bfloat16"):
        return torch.bfloat16, ""
    if name in ("fp16", "float16", "half"):
        return torch.float16, ""
    if name in ("fp32", "float32"):
        return torch.float32, ""

    # FP8（是否支持取决于 GPU 架构 + torch 版本）
    if name in ("fp8e4m3", "float8_e4m3fn"):
        dt = getattr(torch, "float8_e4m3fn", None)
        return dt, "" if dt is not None else "torch 未暴露 float8_e4m3fn"
    if name in ("fp8e5m2", "float8_e5m2"):
        dt = getattr(torch, "float8_e5m2", None)
        return dt, "" if dt is not None else "torch 未暴露 float8_e5m2"

    # FP4（PyTorch 可能暴露 float4_*，但未必支持 matmul 内核）
    if name in ("fp4", "float4"):
        dt = getattr(torch, "float4_e2m1fn_x2", None)
        return dt, "" if dt is not None else "torch 未暴露 float4_e2m1fn_x2"

    return None, f"未知 dtype: {name}"


def _gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # 强制走 matmul 路径（避免某些 fuse 误差）
    return A @ B


def bench_gemm(m: int, n: int, k: int, dtype_name: str, warmup: int, iters: int) -> RunInfo:
    assert torch.cuda.is_available(), "需要 CUDA GPU"

    dtype, note = _resolve_dtype(dtype_name)
    gpu = _gpu_props()

    # 如果 dtype 不可用，直接返回 “不可测” 结果
    if dtype is None:
        return RunInfo(
            host=platform.node(),
            platform=platform.platform(),
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            device=gpu["name"],
            capability=gpu["capability"],
            total_vram_gb=gpu["total_vram_gb"],
            dtype=dtype_name,
            m=m,
            n=n,
            k=k,
            warmup=warmup,
            iters=iters,
            tflops=0.0,
            ms_per_iter=0.0,
            note=note or "dtype 不可用",
        )

    # 基础设置：避免 TF32 影响（对 bf16/fp16 不相关，但保持一致）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 创建矩阵
    # 对 float8/float4：一些 torch 版本可能不允许直接 rand 该 dtype；先用 fp16 生成再 cast
    A = torch.randn((m, k), device="cuda", dtype=torch.float16)
    B = torch.randn((k, n), device="cuda", dtype=torch.float16)
    try:
        A = A.to(dtype)
        B = B.to(dtype)
    except Exception as e:
        return RunInfo(
            host=platform.node(),
            platform=platform.platform(),
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            device=gpu["name"],
            capability=gpu["capability"],
            total_vram_gb=gpu["total_vram_gb"],
            dtype=dtype_name,
            m=m,
            n=n,
            k=k,
            warmup=warmup,
            iters=iters,
            tflops=0.0,
            ms_per_iter=0.0,
            note=f"无法 cast 到 {dtype_name}: {e}",
        )

    # 编译/第一次运行开销很大，先 warmup
    for _ in range(max(0, warmup)):
        C = _gemm(A, B)
        # 防止被优化掉
        _ = C.sum()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        C = _gemm(A, B)
        _ = C.sum()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    ms_per_iter = ms / iters

    # FLOPs: 2*m*n*k
    flops = 2.0 * m * n * k
    tflops = (flops / (ms_per_iter / 1000.0)) / 1e12

    return RunInfo(
        host=platform.node(),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        device=gpu["name"],
        capability=gpu["capability"],
        total_vram_gb=gpu["total_vram_gb"],
        dtype=dtype_name,
        m=m,
        n=n,
        k=k,
        warmup=warmup,
        iters=iters,
        tflops=round(tflops, 3),
        ms_per_iter=round(ms_per_iter, 3),
        note=note,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=int(os.getenv("GEMM_M", "16384")))
    p.add_argument("--n", type=int, default=int(os.getenv("GEMM_N", "16384")))
    p.add_argument("--k", type=int, default=int(os.getenv("GEMM_K", "16384")))
    p.add_argument("--dtype", type=str, default=os.getenv("GEMM_DTYPE", "bf16"))
    p.add_argument("--warmup", type=int, default=int(os.getenv("GEMM_WARMUP", "5")))
    p.add_argument("--iters", type=int, default=int(os.getenv("GEMM_ITERS", "20")))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，无法测试")

    info = bench_gemm(args.m, args.n, args.k, args.dtype, args.warmup, args.iters)

    print("==== GEMM Benchmark ====")
    print(json.dumps(asdict(info), ensure_ascii=False, indent=2))
    print("==== ONE_LINE_JSON ====")
    print(json.dumps(asdict(info), ensure_ascii=False))


if __name__ == "__main__":
    main()



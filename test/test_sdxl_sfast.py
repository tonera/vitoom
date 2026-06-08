"""
SDXL + stable-fast(sfast) 单项加速测试脚本

目标：
- 先把“单机单次推理”的 sfast 加速跑通（仅 SDXL base）
- 支持 baseline / sfast 两种模式，但建议你按需单独跑（你要求：一个一个测）
- 固定 seed + 同一组参数，便于对比耗时与生成结果

使用示例（在项目根目录）：
    # baseline
    python test/test_sdxl_sfast.py --mode baseline

    # sfast
    python test/test_sdxl_sfast.py --mode sfast

安装 stable-fast（常见方式，若你内部有镜像/分支请替换 URL）：
    pip install -U "git+https://github.com/chengzeyi/stable-fast.git"
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import dataclasses
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from diffusers import StableDiffusionXLPipeline


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


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
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
    print(
        "SDP backends:",
        "flash=", torch.backends.cuda.flash_sdp_enabled() if torch.cuda.is_available() else "N/A",
        "mem_efficient=", torch.backends.cuda.mem_efficient_sdp_enabled() if torch.cuda.is_available() else "N/A",
        "math=", torch.backends.cuda.math_sdp_enabled() if torch.cuda.is_available() else "N/A",
    )
    print("============")


def _resolve_device(name: str) -> torch.device:
    name = (name or "").strip().lower()
    if name in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --device cuda 但当前 torch.cuda.is_available() 为 False。")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {name} (use auto/cuda/cpu)")


def _get_ptxas_version() -> tuple[int, int] | None:
    """
    Returns (major, minor) parsed from `ptxas --version`, or None if not found/unparseable.
    Example output: "ptxas: NVIDIA (R) Ptx optimizing assembler  Version 12.4.131"
    """
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return None
    try:
        out = subprocess.check_output([ptxas, "--version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    # Different CUDA releases print different formats. Support a few common ones:
    # - "Version 12.4.131"
    # - "Cuda compilation tools, release 13.0, V13.0.88"
    patterns = [
        r"Version\s+(\d+)\.(\d+)",          # classic
        r"release\s+(\d+)\.(\d+)",          # CUDA toolkit style
        r"\bV(\d+)\.(\d+)\.",               # V13.0.88 style (take major/minor)
        r"\bV(\d+)\.(\d+)\b",               # V13.0 (fallback)
    ]
    for pat in patterns:
        m = re.search(pat, out, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None

def _get_ptxas_path() -> str | None:
    return shutil.which("ptxas")

def _get_ptxas_version_raw() -> str | None:
    ptxas = _get_ptxas_path()
    if not ptxas:
        return None
    try:
        return subprocess.check_output([ptxas, "--version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None


def _ptxas_supports_gpu_name(gpu_name: str) -> bool | None:
    """
    Best-effort check by scanning `ptxas --help` output.
    Returns True/False if determinable, None if ptxas unavailable/unreadable.
    """
    ptxas = _get_ptxas_path()
    if not ptxas:
        return None
    try:
        out = subprocess.check_output([ptxas, "--help"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    return (gpu_name in out)


def _get_triton_bundled_ptxas() -> str | None:
    """
    Triton wheel 往往自带一个 ptxas（用于保证在无完整 CUDA toolkit 时也能工作）。
    你的报错里 Repro command 就是用这个路径：
      .../site-packages/triton/backends/nvidia/bin/ptxas
    """
    try:
        import triton  # type: ignore

        triton_dir = os.path.dirname(getattr(triton, "__file__", "") or "")
        if not triton_dir:
            return None
        cand = os.path.join(triton_dir, "backends", "nvidia", "bin", "ptxas")
        return cand if os.path.exists(cand) else None
    except Exception:
        return None


def _get_ptxas_version_of(path_str: str) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output([path_str, "--version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    patterns = [
        r"Version\s+(\d+)\.(\d+)",
        r"release\s+(\d+)\.(\d+)",
        r"\bV(\d+)\.(\d+)\.",
        r"\bV(\d+)\.(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, out, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _ptxas_supports_gpu_name_of(path_str: str, gpu_name: str) -> bool | None:
    try:
        out = subprocess.check_output([path_str, "--help"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    return (gpu_name in out)


def _preflight_triton_if_needed(*, device: torch.device, want_triton: bool) -> None:
    """
    Triton 会调用 ptxas。新 GPU（例如出现 sm_121a）需要足够新的 CUDA/ptxas。
    这里做一个“友好失败”，把内部 Triton codegen error 变成可操作的提示。
    """
    if not want_triton:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    cc = torch.cuda.get_device_capability()
    # Always print what *this process* sees, to catch "ptxas shadowing".
    ptxas_path = _get_ptxas_path()
    ptxas_ver = _get_ptxas_version()
    ptxas_raw = _get_ptxas_version_raw()
    triton_ptxas = _get_triton_bundled_ptxas()
    triton_ptxas_ver = _get_ptxas_version_of(triton_ptxas) if triton_ptxas else None
    print(f"[preflight] ptxas path: {ptxas_path}")
    print(f"[preflight] ptxas version: {ptxas_ver}")
    if ptxas_raw is not None and ptxas_ver is None:
        # Avoid false-negative: parsing failed but ptxas exists.
        first_line = (ptxas_raw.strip().splitlines() or [""])[0]
        print(f"[preflight] ptxas version(raw): {first_line}")
    if triton_ptxas:
        print(f"[preflight] triton bundled ptxas: {triton_ptxas}")
        print(f"[preflight] triton bundled ptxas version: {triton_ptxas_ver}")
    # Blackwell/新架构常见会报 sm_12x(a)；旧 ptxas 会直接不认识这个 gpu-name
    if cc[0] >= 12:
        if ptxas_path is None:
            print(
                "[preflight] 你开启了 --sfast-triton，但未能检测到可用的 ptxas。\n"
                "  - 请确认 CUDA toolkit 已安装且 ptxas 在 PATH 中（或正确设置 CUDA_HOME）。\n"
                "  - 对于计算能力 >= 12 的 GPU，通常需要 CUDA 13.x（或至少包含支持该 SM 的 ptxas）。"
            )
            sys.exit(2)
        # If we can't parse version, don't hard-fail. We'll rely on the gpu-name support check below.
        if ptxas_ver is not None and ptxas_ver[0] < 13:
            print(
                "[preflight] 你开启了 --sfast-triton，但当前 ptxas 版本过旧，可能不支持该 GPU 架构。\n"
                f"  - detected compute capability: {cc}\n"
                f"  - detected ptxas version: {ptxas_ver[0]}.{ptxas_ver[1]}\n"
                "  - 建议：升级到 CUDA 13.x（确保 ptxas 支持 sm_12x/12x a），并升级到支持该架构的 triton 版本。\n"
                "  - 你也可以先只用 --sfast-cuda-graph（通常也能带来明显收益）。"
            )
            sys.exit(2)
        # Extra guard for the exact error you saw: ptxas doesn't accept sm_121a
        # (even if toolkit looks new). If it can't be determined, we don't block.
        if cc[0] == 12 and cc[1] == 1:
            # Triton 实际会优先用它自己的 ptxas（你给的 Repro command 就是这个路径）。
            # 所以优先检查 triton bundled ptxas，其次才检查 PATH 下的 ptxas。
            supported = None
            if triton_ptxas:
                supported = _ptxas_supports_gpu_name_of(triton_ptxas, "sm_121a")
            if supported is None:
                supported = _ptxas_supports_gpu_name("sm_121a")
            if supported is False:
                print(
                    "[preflight] 检测到你的 GPU 可能会触发 Triton 使用 sm_121a，但当前 ptxas 的 help 输出里不包含 sm_121a。\n"
                    "  - 这会导致 Triton 报：ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'\n"
                    "  - 解决路径：\n"
                    "    1) 升级/更换 triton wheel（让 triton/backends/nvidia/bin/ptxas 支持 sm_121a，或让 triton 选择 sm_121 而非 sm_121a）\n"
                    "    2) 临时手动用系统 ptxas 覆盖 triton 自带 ptxas（风险自担，但通常可用）：\n"
                    "         TRITON_PTXAS=$(python -c \"import os,triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends','nvidia','bin','ptxas'))\")\n"
                    "         mv $TRITON_PTXAS ${TRITON_PTXAS}.bak && ln -s /usr/local/cuda/bin/ptxas $TRITON_PTXAS\n"
                    "       然后重试。\n"
                    "    3) 或者先别开 --sfast-triton，只用 --sfast-cuda-graph\n"
                    "\n"
                    "（如果你愿意，我可以根据你当前 triton 版本进一步给出最稳的版本组合建议。）"
                )
                sys.exit(2)


def _load_pipe(model: str, *, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    # local_files_only=True：匹配你当前 test/test_sdxl.py 的假设（模型在 resources/ 下）
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _apply_sfast(pipe: StableDiffusionXLPipeline) -> StableDiffusionXLPipeline:
    """
    尝试以“最少假设”的方式应用 stable-fast(sfast)。

    stable-fast 的 API 在不同版本/分支可能略有差异：
    - 这里优先尝试 SDXL 专用 compiler，其次尝试通用的 pipeline compiler
    - 若导入失败，会给出明确的安装提示
    """
    try:
        # 推荐：新版统一入口（支持 SD / SDXL / SVD 等）
        from sfast.compilers.diffusion_pipeline_compiler import (  # type: ignore
            compile as sfast_compile,
            CompilationConfig,
        )

        cfg = CompilationConfig.Default()
        print("[sfast] compiler = sfast.compilers.diffusion_pipeline_compiler")
        print("[sfast] default config:", dataclasses.asdict(cfg))
        return sfast_compile(pipe, cfg)
    except Exception:
        pass

    try:
        # 兼容：老入口（内部会 re-export diffusion_pipeline_compiler）
        from sfast.compilers.stable_diffusion_pipeline_compiler import (  # type: ignore
            compile as sfast_compile,
            CompilationConfig,
        )

        cfg = CompilationConfig.Default()
        print("[sfast] compiler = sfast.compilers.stable_diffusion_pipeline_compiler (deprecated)")
        print("[sfast] default config:", dataclasses.asdict(cfg))
        return sfast_compile(pipe, cfg)
    except Exception:
        pass

    # 再兜底：看看是否存在更“扁平”的入口（不同 fork 可能这样组织）
    try:
        import sfast  # type: ignore

        if hasattr(sfast, "optimize"):
            return sfast.optimize(pipe)  # type: ignore[misc]
        if hasattr(sfast, "compile"):
            return sfast.compile(pipe)  # type: ignore[misc]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "未检测到 stable-fast(sfast)。建议安装：\n"
            '  pip install -U "git+https://github.com/chengzeyi/stable-fast.git"\n'
            "若你使用内部 fork/镜像，请替换为对应 git 地址。"
        ) from e

    raise RuntimeError(
        "已安装 sfast 但未找到可用的编译入口（API 可能与此脚本假设不一致）。\n"
        "请把你使用的 stable-fast 分支 README / 典型用法贴出来，我可以把 _apply_sfast() 适配到你那版。"
    )


def _run_once(
    pipe: StableDiffusionXLPipeline,
    *,
    device: torch.device,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    seed: int,
) -> RunResult:
    generator = torch.Generator(device=device).manual_seed(seed)
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        )
    _sync_if_cuda(device)
    t1 = time.perf_counter()
    return RunResult(elapsed_s=t1 - t0, image=out.images[0])


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return float((s[m - 1] + s[m]) / 2.0)


def _percentile(xs: Sequence[float], p: float) -> float:
    """p in [0,100]. No numpy dependency; linear interpolation."""
    if not xs:
        return 0.0
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if c == f:
        return float(s[f])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


def _format_stats(times: Sequence[float]) -> str:
    if not times:
        return "n=0"
    return (
        f"n={len(times)}, mean={_mean(times):.3f}s, med={_median(times):.3f}s, "
        f"p90={_percentile(times, 90):.3f}s, p95={_percentile(times, 95):.3f}s"
    )


def _ensure_parent_dir(path_str: str) -> None:
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _get_cached_size(fn) -> int | None:
    cached = getattr(fn, "_cached", None)
    if isinstance(cached, dict):
        return len(cached)
    return None


def _safe_version(mod) -> str:
    return getattr(mod, "__version__", "unknown")


def _print_sfast_runtime_env(pipe: StableDiffusionXLPipeline) -> None:
    print("=== SFAST DIAG ===")
    try:
        import sfast  # type: ignore

        print("[sfast] module:", getattr(sfast, "__file__", None))
        print("[sfast] version:", getattr(sfast, "__version__", "unknown"))
        try:
            import sfast._C  # type: ignore

            print("[sfast] _C:", getattr(sfast._C, "__file__", None))
        except Exception as e:
            print("[sfast] _C import failed:", e)
    except Exception as e:
        print("[sfast] not importable:", e)

    # xformers / triton availability
    try:
        import xformers  # type: ignore

        print("[xformers] version:", _safe_version(xformers))
        print("[xformers] module:", getattr(xformers, "__file__", None))
    except Exception as e:
        print("[xformers] not available:", e)

    try:
        import triton  # type: ignore

        print("[triton] version:", _safe_version(triton))
        print("[triton] module:", getattr(triton, "__file__", None))
    except Exception as e:
        print("[triton] not available:", e)

    # torch.ops injections (whether sfast kernels are actually registered)
    try:
        has_triton_ops = hasattr(torch.ops, "sfast_triton")
        has_xformers_ops = hasattr(torch.ops, "sfast_xformers")
        print("[torch.ops] has sfast_triton:", has_triton_ops)
        print("[torch.ops] has sfast_xformers:", has_xformers_ops)
        if has_triton_ops:
            cand = [
                "contiguous",
                "clone",
                "reshape",
                "group_norm",
                "group_norm_silu",
                "layer_norm",
                "_convolution",
            ]
            present = [n for n in cand if hasattr(torch.ops.sfast_triton, n)]
            print("[torch.ops.sfast_triton] present:", present)
        if has_xformers_ops:
            cand = ["memory_efficient_attention"]
            present = [n for n in cand if hasattr(torch.ops.sfast_xformers, n)]
            print("[torch.ops.sfast_xformers] present:", present)
    except Exception as e:
        print("[torch.ops] inspect failed:", e)

    # Attention processor types (helps decide whether baseline already uses SDPA/flash/xformers)
    try:
        unet = getattr(pipe, "unet", None)
        attn_procs = getattr(unet, "attn_processors", None)
        if isinstance(attn_procs, dict) and attn_procs:
            types = sorted({type(v).__module__ + "." + type(v).__name__ for v in attn_procs.values()})
            print("[unet.attn_processors] unique types:")
            for t in types:
                print("  -", t)
        else:
            print("[unet.attn_processors] not available")
    except Exception as e:
        print("[unet.attn_processors] inspect failed:", e)

    # SFAST_* env vars
    envs = sorted((k, v) for k, v in os.environ.items() if k.startswith("SFAST_"))
    if envs:
        print("[env] SFAST_*:")
        for k, v in envs:
            print(f"  - {k}={v}")
    print("==================")


def _print_sfast_effect(pipe: StableDiffusionXLPipeline) -> None:
    # 主要观察：forward 是否被 lazy_trace / cuda graph wrapper 包了一层（它们都会带 _cached 字典）
    unet_f = getattr(pipe.unet, "forward", None)
    vae_f = getattr(pipe.vae, "forward", None)
    te_f = getattr(getattr(pipe, "text_encoder", None), "forward", None)
    te2_f = getattr(getattr(pipe, "text_encoder_2", None), "forward", None)

    def _one(name: str, f) -> None:
        if f is None:
            print(f"[sfast] {name}: <none>")
            return
        cached_n = _get_cached_size(f)
        cached_str = "n/a" if cached_n is None else str(cached_n)
        print(f"[sfast] {name}: type={type(f)}, _cached={cached_str}")

    _one("unet.forward", unet_f)
    _one("vae.forward", vae_f)
    _one("text_encoder.forward", te_f)
    _one("text_encoder_2.forward", te2_f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "sfast"], default="sfast")
    parser.add_argument("--model", default="resources/models/xl-base")
    parser.add_argument("--dtype", default=os.environ.get("SDXL_DTYPE", "bf16"))
    parser.add_argument("--device", default=os.environ.get("SDXL_DEVICE", "auto"), help="auto/cuda/cpu")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1, help="正式计时 runs 次数（建议 >= 3）")
    parser.add_argument("--same-seed", action="store_true", help="每次 run 固定同一个 seed（更利于对比）")
    parser.add_argument("--prompt", default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.")
    parser.add_argument("--negative", default="")
    parser.add_argument("--tf32", action="store_true", help="CUDA 下开启 TF32（更贴近线上默认/更快）")
    parser.add_argument("--out", default="outputs/_tmp_sdxl_sfast.png")
    parser.add_argument("--no-save", action="store_true", help="不保存图片（纯计时）")
    parser.add_argument("--dump-sfast", action="store_true", help="打印更详细的 sfast/attention/ops 诊断信息")

    # sfast knobs（默认保持保守，避免质量/稳定性问题；你需要“更猛”的优化时显式打开）
    parser.add_argument("--sfast-cuda-graph", action="store_true", help="开启 CUDA Graph（通常增益最大）")
    parser.add_argument("--sfast-triton", action="store_true", help="开启 Triton pass（建议配合 --sfast-cuda-graph）")
    parser.add_argument("--sfast-xformers", action="store_true", help="开启 xFormers 注意力（需要已安装 xformers）")
    parser.add_argument("--sfast-no-jit-freeze", action="store_true", help="关闭 JIT freeze（更兼容 LoRA/动态权重，但会慢一点）")
    parser.add_argument("--sfast-no-lowp-gemm", action="store_true", help="关闭 lowp GEMM 偏好（更稳，但可能慢）")
    args = parser.parse_args()

    _print_env()

    dtype = _torch_dtype(args.dtype)
    device = _resolve_device(args.device)
    print(
        f"mode={args.mode}, model={args.model}, device={device}, dtype={dtype}, "
        f"steps={args.steps}, cfg={args.guidance}, warmup={args.warmup}, runs={args.runs}"
    )

    _preflight_triton_if_needed(device=device, want_triton=bool(args.sfast_triton))

    if device.type == "cuda" and torch.cuda.is_available():
        # TF32 会影响数值（通常可接受），但显著影响性能；这里用显式开关避免“悄悄变快/变慢”的误解
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    pipe = _load_pipe(args.model, dtype=dtype)
    pipe = pipe.to(device)
    if args.dump_sfast:
        _print_sfast_runtime_env(pipe)

    for i in range(max(0, args.warmup)):
        r = _run_once(
            pipe,
            device=device,
            prompt=args.prompt,
            negative_prompt=args.negative,
            steps=min(args.steps, 2),
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed + i,
        )
        print(f"warmup[{i}] {r.elapsed_s:.3f}s")

    if args.mode == "sfast":
        print("Applying stable-fast(sfast) ...")
        # 先把 config 明确打印出来，避免“以为开了但其实没开”
        try:
            from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig  # type: ignore

            cfg = CompilationConfig.Default()
            cfg.enable_cuda_graph = bool(args.sfast_cuda_graph)
            cfg.enable_triton = bool(args.sfast_triton)
            cfg.enable_xformers = bool(args.sfast_xformers)
            if args.sfast_no_jit_freeze:
                cfg.enable_jit_freeze = False
            if args.sfast_no_lowp_gemm:
                cfg.prefer_lowp_gemm = False

            print("[sfast] requested config:", dataclasses.asdict(cfg))
        except Exception as e:
            cfg = None
            print("[sfast] cannot import CompilationConfig to show config:", e)

        t_compile0 = time.perf_counter()
        if cfg is not None:
            # 优先走新版入口（更确定）
            try:
                from sfast.compilers.diffusion_pipeline_compiler import compile as sfast_compile  # type: ignore

                print("[sfast] compiler = sfast.compilers.diffusion_pipeline_compiler")
                pipe = sfast_compile(pipe, cfg)
            except Exception:
                pipe = _apply_sfast(pipe)
        else:
            pipe = _apply_sfast(pipe)

        _sync_if_cuda(device)
        t_compile1 = time.perf_counter()
        print(f"sfast compile walltime: {t_compile1 - t_compile0:.3f}s")
        if args.dump_sfast:
            _print_sfast_runtime_env(pipe)
        _print_sfast_effect(pipe)
        # sfast 编译后再做一次 warmup（覆盖编译/图捕获等一次性开销）
        r2 = _run_once(
            pipe,
            device=device,
            prompt=args.prompt,
            negative_prompt=args.negative,
            steps=min(args.steps, 2),
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
        print(f"sfast post-compile warmup {r2.elapsed_s:.3f}s")
        _print_sfast_effect(pipe)

    times: list[float] = []
    last: RunResult | None = None
    for i in range(max(1, args.runs)):
        seed = args.seed if args.same_seed else (args.seed + i)
        last = _run_once(
            pipe,
            device=device,
            prompt=args.prompt,
            negative_prompt=args.negative,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=seed,
        )
        times.append(last.elapsed_s)
        print(f"run[{i}] {last.elapsed_s:.3f}s (seed={seed})")

    print(f"Time stats ({args.mode}): {_format_stats(times)}")

    # 保存输出（目录不存在则尽量创建）
    if not args.no_save and last is not None:
        try:
            _ensure_parent_dir(args.out)
            last.image.save(args.out)
            print("Saved:", args.out)
        except Exception as e:
            print("Save failed:", e)


if __name__ == "__main__":
    main()

# resources/models/SDXLRonghua_v40.safetensors
# python test.py --mode sfast --steps 30 --height 1024 --width 1024 --dtype bf16 --out outputs/sdxl_sfast.png

# python test.py --mode baseline --runs 5 --warmup 1 --same-seed --tf32 --dump-sfast --no-save
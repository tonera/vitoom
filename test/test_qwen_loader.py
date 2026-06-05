import argparse
import gc
import time
import warnings
from dataclasses import dataclass

import torch
from diffusers import QwenImagePipeline

from inference.common.monitor import print_gpu_memory_usage


model_dir = "/home/tonera/models"

# 你要对比的两个模型路径
fp4_model_name = f"{model_dir}/QWEN_IMAGE_fp4_w_AbliteratedTE_Diffusers"
bf16_model_name = f"{model_dir}/Qwen-Image"


@dataclass
class LoadResult:
    label: str
    model_path: str
    from_pretrained_s: float
    pretouch_cpu_s: float
    pin_cpu_s: float
    to_device_s: float
    total_s: float


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup_cuda() -> None:
    """尽量释放显存（同进程内测试仍会受 allocator/cache 影响）。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 某些场景下能进一步回收
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _warmup_cuda(device: str) -> None:
    """避免把 CUDA context 初始化算进第一个模型的 to(device) 时间里。"""
    if device.startswith("cuda") and torch.cuda.is_available():
        _ = torch.empty(1, device=device)
        torch.cuda.synchronize()


def _cuda_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return 0
    return 0


def _is_gb10(device: str) -> bool:
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return False
    idx = _cuda_index(device)
    props = torch.cuda.get_device_properties(idx)
    name = (props.name or "").upper()
    # 经验特征：GB10 / compute capability 12.1
    return ("GB10" in name) or (props.major == 12 and props.minor == 1)


def _pin_module_cpu_params_and_buffers(module: torch.nn.Module) -> None:
    """
    将 module 内位于 CPU 的参数/缓冲区复制到 pinned memory（in-place 替换）。

    目的：在 GB10 这类“pageable many-small H2D 极慢”的平台上，
    让后续迁移到 CUDA 走 pinned 路径，显著降低小块 H2D 开销。
    """
    with torch.no_grad():
        for sub in module.modules():
            for name, p in list(sub._parameters.items()):
                if p is None or not isinstance(p, torch.Tensor):
                    continue
                if p.device.type != "cpu" or p.numel() == 0:
                    continue
                sub._parameters[name] = torch.nn.Parameter(p.detach().pin_memory(), requires_grad=p.requires_grad)
            for name, b in list(sub._buffers.items()):
                if b is None or not isinstance(b, torch.Tensor):
                    continue
                if b.device.type != "cpu" or b.numel() == 0:
                    continue
                sub._buffers[name] = b.detach().pin_memory()


def _pin_pipeline_cpu_tensors(pipe: object) -> None:
    from inference.common.torch_transfer_utils import DEFAULT_PIPELINE_COMPONENT_ATTRS

    for attr in DEFAULT_PIPELINE_COMPONENT_ATTRS:
        if not hasattr(pipe, attr):
            continue
        m = getattr(pipe, attr)
        if m is None or not isinstance(m, torch.nn.Module):
            continue
        _pin_module_cpu_params_and_buffers(m)


def _move_pipeline_modules_to_device(pipe: object, device: str, *, non_blocking: bool) -> None:
    """
    diffusers 的 pipeline.to(...) 不一定支持 non_blocking。
    这里直接对常见组件的 nn.Module 调用 .to(device, non_blocking=...).
    """
    from inference.common.torch_transfer_utils import DEFAULT_PIPELINE_COMPONENT_ATTRS

    for attr in DEFAULT_PIPELINE_COMPONENT_ATTRS:
        if not hasattr(pipe, attr):
            continue
        m = getattr(pipe, attr)
        if m is None or not isinstance(m, torch.nn.Module):
            continue
        m.to(device, non_blocking=non_blocking)


def _maybe_nvtx_range_push(device: str, name: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(name)
        except Exception:
            pass


def _maybe_nvtx_range_pop(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass


def _patch_accelerate_pin_memory(*, enabled: bool) -> bool:
    """
    在不改 diffusers/transformers 源码的前提下，尽量把 from_pretrained 阶段的 CPU->CUDA 拷贝
    从 pageable 变成 pinned（针对 GB10 many-small H2D cliff）。

    原理：diffusers/transformers 在启用 device_map/low_cpu_mem_usage/量化时，
    常通过 accelerate 的 `set_module_tensor_to_device` 把权重逐个放到目标 device。
    我们在这里 monkey patch：如果 value 是 CPU tensor 且目标是 CUDA，则先尝试 pin_memory()。
    """
    if not enabled:
        return False
    try:
        import accelerate.utils.modeling as am  # type: ignore[import-not-found]
    except Exception:
        return False

    if getattr(am.set_module_tensor_to_device, "_vitoom_patched", False):
        return True

    orig = am.set_module_tensor_to_device

    def wrapped_set_module_tensor_to_device(  # type: ignore[no-untyped-def]
        module,
        tensor_name,
        device,
        value=None,
        dtype=None,
        fp16_statistics=None,
        **kwargs,
    ):
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            dev_str = str(device)
            if dev_str.startswith("cuda"):
                # 关键：pageable -> pinned，规避 GB10 many-small H2D pathological slow path
                try:
                    if not value.is_pinned():
                        value = value.pin_memory()
                except Exception:
                    pass
        return orig(
            module,
            tensor_name,
            device,
            value=value,
            dtype=dtype,
            fp16_statistics=fp16_statistics,
            **kwargs,
        )

    wrapped_set_module_tensor_to_device._vitoom_patched = True  # type: ignore[attr-defined]
    am.set_module_tensor_to_device = wrapped_set_module_tensor_to_device  # type: ignore[assignment]
    return True


def _patch_safetensors_force_cpu(*, enabled: bool) -> bool:
    """
    强制 safetensors 在 load 时使用 CPU（即使调用方请求 device='cuda'）。

    背景：
    - 你已经用 nsys 证明 fp4 的 HtoD memcpy 大量发生在 from_pretrained 阶段（~50s）。
    - accelerate 的 set_module_tensor_to_device patch 对你这条路径影响不大，说明很多 HtoD 可能发生在
      safetensors 的 `load_file(..., device='cuda')` 或 `safe_open(..., device='cuda')` 这一层。

    这个 patch 的目标是：让 from_pretrained 阶段尽量不做 HtoD，把权重留在 CPU，
    然后再由我们显式的 `.to(cuda)`（可 pinned + non_blocking）来完成搬运。

    注意：
    - 会增加 CPU 峰值内存压力（因为不再直接在 GPU 上构建/加载）。
    - 如果调用方强依赖 “直接在 cuda 上 load”（少见，但可能），可能改变行为；这是测试脚本级别的诊断开关。
    """
    if not enabled:
        return False
    try:
        import safetensors.safe_open as _so  # type: ignore[attr-defined]
    except Exception:
        _so = None  # type: ignore[assignment]

    try:
        import safetensors.torch as st  # type: ignore[import-not-found]
    except Exception:
        return False

    # Patch safetensors.torch.load_file
    if not getattr(st.load_file, "_vitoom_patched", False):
        orig_load_file = st.load_file

        def load_file_force_cpu(filename, device="cpu", **kwargs):  # type: ignore[no-untyped-def]
            if isinstance(device, str) and device.startswith("cuda"):
                device = "cpu"
            return orig_load_file(filename, device=device, **kwargs)

        load_file_force_cpu._vitoom_patched = True  # type: ignore[attr-defined]
        load_file_force_cpu._vitoom_orig = orig_load_file  # type: ignore[attr-defined]
        st.load_file = load_file_force_cpu  # type: ignore[assignment]

    # Patch safetensors.safe_open (used by some loaders)
    # `safe_open` is a function in safetensors module; import style varies.
    try:
        import safetensors  # type: ignore[import-not-found]
    except Exception:
        safetensors = None  # type: ignore[assignment]

    if safetensors is not None and hasattr(safetensors, "safe_open"):
        so = safetensors.safe_open
        if not getattr(so, "_vitoom_patched", False):
            orig_so = so

            def safe_open_force_cpu(filename, framework="pt", device="cpu", **kwargs):  # type: ignore[no-untyped-def]
                if isinstance(device, str) and device.startswith("cuda"):
                    device = "cpu"
                return orig_so(filename, framework=framework, device=device, **kwargs)

            safe_open_force_cpu._vitoom_patched = True  # type: ignore[attr-defined]
            safe_open_force_cpu._vitoom_orig = orig_so  # type: ignore[attr-defined]
            safetensors.safe_open = safe_open_force_cpu  # type: ignore[assignment]

    return True


def _from_pretrained_with_dtype(
    *,
    model_path: str,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool,
    device_map: str | None,
    disable_mmap: bool,
) -> QwenImagePipeline:
    """
    diffusers 在不同版本里对 dtype/torch_dtype 的支持不一致：
    - 有的版本提示 torch_dtype 弃用（但仍可用）
    - 有的 pipeline（你这里的 QwenImagePipeline）会忽略 dtype=... 并打印“will be ignored”
    为了保证 dtype 真正生效：优先尝试 torch_dtype（并过滤弃用 warning），失败再 fallback 到 dtype。
    """
    extra_kwargs: dict[str, object] = {}
    if device_map is not None:
        extra_kwargs["device_map"] = device_map

    def _call(*, use_torch_dtype: bool, use_device_map: bool, use_disable_mmap: bool) -> QwenImagePipeline:
        kw: dict[str, object] = {"low_cpu_mem_usage": low_cpu_mem_usage}
        if use_torch_dtype:
            kw["torch_dtype"] = dtype
        else:
            kw["dtype"] = dtype
        if use_device_map and device_map is not None:
            kw["device_map"] = device_map
        if use_disable_mmap:
            kw["disable_mmap"] = disable_mmap
        return QwenImagePipeline.from_pretrained(model_path, **kw)

    # 1) 优先 torch_dtype（你当前环境最可能生效），并优先带上 device_map（若有）
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
            )
            return _call(use_torch_dtype=True, use_device_map=True, use_disable_mmap=True)
    except (TypeError, NotImplementedError) as e:
        # disable_mmap / device_map / torch_dtype 兼容性差：逐步回退
        _ = e
        pass

    # 2) fallback：dtype（新版本可能只接受 dtype），仍尝试带上 device_map
    try:
        return _call(use_torch_dtype=False, use_device_map=True, use_disable_mmap=True)
    except NotImplementedError as e:
        # 典型报错：device_map="cpu" 不支持（只支持 balanced/cuda）
        msg = str(e).lower()
        if device_map is not None and "cpu not supported" in msg:
            return _call(use_torch_dtype=False, use_device_map=False, use_disable_mmap=True)
        raise
    except TypeError:
        # 3) 某些 pipeline/版本不支持 device_map；最后再试一次不带它
        try:
            return _call(use_torch_dtype=False, use_device_map=False, use_disable_mmap=True)
        except TypeError:
            # 4) 某些 pipeline/版本不支持 disable_mmap；最后再试一次不带它
            return _call(use_torch_dtype=False, use_device_map=False, use_disable_mmap=False)


def time_load_model(
    *,
    label: str,
    model_path: str,
    device: str,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool,
    pin_memory: bool,
    from_pretrained_device_map: str | None,
    disable_mmap: bool,
) -> LoadResult:
    print("\n" + "=" * 80)
    print(f"开始测试：{label}")
    print(f"模型路径：{model_path}")
    print(f"device={device}, dtype={dtype}")
    print(f"low_cpu_mem_usage={low_cpu_mem_usage}")
    print(f"pin_memory={pin_memory}")
    print(f"from_pretrained.device_map={from_pretrained_device_map}")
    print(f"disable_mmap={disable_mmap}")

    print_gpu_memory_usage(f"[{label}] 载入前")
    _sync_if_cuda(device)

    t0 = time.perf_counter()
    t_from0 = time.perf_counter()
    _maybe_nvtx_range_push(device, f"{label}:from_pretrained")
    pipe = _from_pretrained_with_dtype(
        model_path=model_path,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=from_pretrained_device_map,
        disable_mmap=disable_mmap,
    )
    _maybe_nvtx_range_pop(device)
    t_from1 = time.perf_counter()
    print_gpu_memory_usage(f"[{label}] from_pretrained 后(仍可能主要在CPU)")

    t_pretouch_s = 0.0
    if time_load_model.pretouch_cpu:  # type: ignore[attr-defined]
        # 在 .to(device) 前先 pretouch CPU tensor，降低缺页/懒加载导致的长尾。
        from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

        t_mat0 = time.perf_counter()
        _maybe_nvtx_range_push(device, f"{label}:pretouch_cpu")
        pretouch_pipeline_cpu_tensors(pipe)
        _maybe_nvtx_range_pop(device)
        t_mat1 = time.perf_counter()
        t_pretouch_s = t_mat1 - t_mat0
        print_gpu_memory_usage(f"[{label}] pretouch_cpu 后")

    t_pin_s = 0.0
    if pin_memory:
        t_pin0 = time.perf_counter()
        _maybe_nvtx_range_push(device, f"{label}:pin_cpu")
        _pin_pipeline_cpu_tensors(pipe)
        _maybe_nvtx_range_pop(device)
        t_pin1 = time.perf_counter()
        t_pin_s = t_pin1 - t_pin0
        print_gpu_memory_usage(f"[{label}] pin_memory(cpu) 后")

    t_to0 = time.perf_counter()
    _maybe_nvtx_range_push(device, f"{label}:to_device")
    if pin_memory and device.startswith("cuda"):
        # pinned -> cuda 建议 non_blocking=True，再用 synchronize 让计时可比
        _move_pipeline_modules_to_device(pipe, device, non_blocking=True)
    else:
        pipe = pipe.to(device)
    _sync_if_cuda(device)
    _maybe_nvtx_range_pop(device)
    t_to1 = time.perf_counter()
    print_gpu_memory_usage(f"[{label}] to({device}) 后")

    t1 = time.perf_counter()

    # 清理，避免影响下一个模型（仍无法做到完全隔离；最公平是分别起新进程）
    del pipe
    _cleanup_cuda()
    print_gpu_memory_usage(f"[{label}] 清理后")

    res = LoadResult(
        label=label,
        model_path=model_path,
        from_pretrained_s=t_from1 - t_from0,
        pretouch_cpu_s=t_pretouch_s,
        pin_cpu_s=t_pin_s,
        to_device_s=t_to1 - t_to0,
        total_s=t1 - t0,
    )
    print(
        f"[{label}] from_pretrained: {res.from_pretrained_s:.3f}s | "
        f"pretouch_cpu: {res.pretouch_cpu_s:.3f}s | "
        f"pin_cpu: {res.pin_cpu_s:.3f}s | "
        f"to_device: {res.to_device_s:.3f}s | total: {res.total_s:.3f}s"
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 fp4 vs bf16 的模型加载耗时")
    parser.add_argument("--device", default="cuda", help='例如 "cuda" / "cuda:0" / "cpu"')
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="加载时希望使用的浮点精度（注意：这不是量化位宽）",
    )
    parser.add_argument(
        "--order",
        default="fp4,bf16",
        help='测试顺序，例如 "fp4,bf16" 或 "bf16,fp4"',
    )
    parser.add_argument(
        "--pretouch-cpu",
        action="store_true",
        help="在 .to(device) 之前先 pretouch CPU tensors，并单独计时（用于验证惰性/mmap 加载是否把耗时挪到了 .to）",
    )
    parser.add_argument(
        "--disable-mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "尝试传给 diffusers/safetensors 的 disable_mmap（禁用 mmap）。"
            "用于验证 mmap pagefault 是否放大 GB10 的加载耗时。"
            "注意：并非所有 pipeline/版本都支持；不支持时会自动忽略。"
        ),
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "传给 from_pretrained 的 low_cpu_mem_usage（默认 True；可用 --no-low-cpu-mem-usage 关闭惰性加载策略）。"
            "注意：fp4/量化模型会强制为 True。"
        ),
    )
    parser.add_argument(
        "--pin-memory",
        choices=["off", "on", "auto"],
        default="auto",
        help=(
            "在迁移到 CUDA 之前对 CPU tensors 执行 pin_memory，以规避 GB10 上 pageable many-small H2D 极慢问题。"
            "auto：仅在检测到 GB10 且 device 为 cuda 时开启。"
        ),
    )
    parser.add_argument(
        "--from-pretrained-device",
        choices=["auto", "cpu"],
        default="auto",
        help=(
            "控制 from_pretrained 阶段是否允许触达 GPU。"
            "cpu：尽量强制 device_map='cpu'，避免 fp4 在 from_pretrained 内部发生 many-small H2D。"
        ),
    )
    parser.add_argument(
        "--accelerate-pin",
        choices=["off", "on", "auto"],
        default="auto",
        help=(
            "monkey patch accelerate 的 set_module_tensor_to_device：当 CPU->CUDA 时先 pin_memory()。"
            "用于缓解 GB10 many-small pageable H2D cliff（尤其是 fp4 from_pretrained 阶段）。"
        ),
    )
    parser.add_argument(
        "--safetensors-force-cpu",
        choices=["off", "on", "auto"],
        default="auto",
        help=(
            "monkey patch safetensors：当 load 请求 device='cuda' 时强制回退到 CPU。"
            "用于把 from_pretrained 内部 H2D 搬运推迟到显式 .to(cuda)（可配合 pin_memory/non_blocking）。"
        ),
    )
    args = parser.parse_args()

    device = args.device
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"将测试两个模型的 load 时间（顺序：{args.order}）")
    print("提示：同进程顺序测试会受缓存影响；如需最公平对比，建议分别运行两次脚本。")

    _warmup_cuda(device)

    # 在开始加载前尽早 patch（否则 from_pretrained 内部已经做完 H2D 了）
    if args.accelerate_pin == "on":
        ok = _patch_accelerate_pin_memory(enabled=True)
        print(f"accelerate_pin=on patch={'OK' if ok else 'SKIP'}")
    elif args.accelerate_pin == "off":
        _patch_accelerate_pin_memory(enabled=False)
        print("accelerate_pin=off")
    else:
        auto_en = _is_gb10(device) and device.startswith("cuda")
        ok = _patch_accelerate_pin_memory(enabled=auto_en)
        print(f"accelerate_pin=auto enabled={auto_en} patch={'OK' if ok else 'SKIP'}")

    # safetensors 强制 CPU：更“对症”地避免 from_pretrained 阶段 H2D
    if args.safetensors_force_cpu == "on":
        ok = _patch_safetensors_force_cpu(enabled=True)
        print(f"safetensors_force_cpu=on patch={'OK' if ok else 'SKIP'}")
    elif args.safetensors_force_cpu == "off":
        _patch_safetensors_force_cpu(enabled=False)
        print("safetensors_force_cpu=off")
    else:
        # 默认：仅在 GB10+cuda 时开启；并且更偏向 fp4 场景
        auto_en = _is_gb10(device) and device.startswith("cuda")
        ok = _patch_safetensors_force_cpu(enabled=auto_en)
        print(f"safetensors_force_cpu=auto enabled={auto_en} patch={'OK' if ok else 'SKIP'}")

    name_map = {
        "fp4": ("fp4", fp4_model_name),
        "bf16": ("bf16", bf16_model_name),
    }
    keys = [k.strip() for k in args.order.split(",") if k.strip()]
    unknown = [k for k in keys if k not in name_map]
    if unknown:
        raise ValueError(f"未知的 order key：{unknown}，只支持 fp4/bf16")

    # 通过函数属性把开关传进 time_load_model（避免改很多函数签名）
    time_load_model.pretouch_cpu = bool(args.pretouch_cpu)  # type: ignore[attr-defined]

    # fp4/4bit 路径通常依赖 bitsandbytes；只在需要时导入，避免不测 fp4 时被依赖阻塞
    if "fp4" in keys:
        try:
            import bitsandbytes  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "检测到需要测试 fp4 模型，但当前环境未安装 bitsandbytes。"
                "请先安装 bitsandbytes，或把 --order 设为仅测试 bf16。"
            ) from e

    results: list[LoadResult] = []
    for k in keys:
        # low_cpu_mem_usage：fp4/量化强制 True；bf16 才按用户参数走
        low_cpu_mem_usage = bool(args.low_cpu_mem_usage)
        if k == "fp4" and not low_cpu_mem_usage:
            print(
                "注意：检测到需要测试 fp4/量化模型，diffusers 要求 low_cpu_mem_usage=True；"
                "已忽略 --no-low-cpu-mem-usage 并对 fp4 强制开启。"
            )
            low_cpu_mem_usage = True

        # pin_memory：off/on/auto（auto 仅在 GB10 + cuda 开启）
        pin_memory: bool
        if args.pin_memory == "on":
            pin_memory = True
        elif args.pin_memory == "off":
            pin_memory = False
        else:
            pin_memory = _is_gb10(device)

        # from_pretrained device_map 策略：
        # - auto：不强制（保持与 diffusers 默认一致）
        # - cpu：尝试 device_map='cpu'（注意：某些 pipeline/版本不支持，会自动回退）
        from_pretrained_device_map: str | None
        if args.from_pretrained_device == "cpu":
            from_pretrained_device_map = "cpu"
        else:
            from_pretrained_device_map = None

        label, path = name_map[k]
        results.append(
            time_load_model(
                label=label,
                model_path=path,
                device=device,
                dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                pin_memory=pin_memory,
                from_pretrained_device_map=from_pretrained_device_map,
                disable_mmap=bool(args.disable_mmap),
            )
        )

    print("\n" + "-" * 80)
    print("汇总：")
    print(
        f"{'label':<8} {'from_pretrained(s)':>18} {'pretouch(s)':>14} {'pin_cpu(s)':>11} {'to_device(s)':>12} {'total(s)':>10}"
    )
    for r in results:
        print(
            f"{r.label:<8} {r.from_pretrained_s:>18.3f} {r.pretouch_cpu_s:>14.3f} {r.pin_cpu_s:>11.3f} "
            f"{r.to_device_s:>12.3f} {r.total_s:>10.3f}"
        )


if __name__ == "__main__":
    main()

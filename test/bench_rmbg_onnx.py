import argparse
import ctypes
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import onnxruntime as ort

try:
    import pynvml  # nvidia-ml-py3
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def pick_providers(force_cpu: bool):
    if force_cpu:
        return ["CPUExecutionProvider"]
    avail = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in avail:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def missing_cuda_deps() -> list[str]:
    """
    onnxruntime-gpu(>=1.23) 的 CUDA EP 通常依赖 CUDA 12.* + cuDNN 9.*。
    如果这些动态库缺失，ORT 会在创建 session 时报错并回退到 CPU。
    """
    required = ["libcublasLt.so.12", "libcudnn.so.9", "libcudart.so.12"]
    missing: list[str] = []
    for lib in required:
        try:
            ctypes.CDLL(lib)
        except Exception:
            missing.append(lib)
    return missing


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    # 避免 exp 溢出
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_minmax_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def preprocess(pil_img: Image.Image, h: int, w: int, norm: str):
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size
    resized = img.resize((w, h), Image.BILINEAR)

    x = np.asarray(resized).astype(np.float32) / 255.0  # HWC [0,1]
    x = np.transpose(x, (2, 0, 1))[None, ...]           # 1CHW

    if norm == "bria":
        x = x - 0.5
    elif norm == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        x = (x - mean) / std
    else:
        raise ValueError(f"Unknown norm: {norm}")

    return x.astype(np.float32), (orig_h, orig_w)


def postprocess_mask(y: np.ndarray, orig_hw):
    y = np.array(y)
    y = np.squeeze(y)      # (1,1,H,W)/(1,H,W)/(H,W)
    if y.ndim == 3:        # (C,H,W)
        y = y[0]
    y = safe_minmax_norm(y)

    mask = Image.fromarray((y * 255).astype(np.uint8), mode="L")
    mask = mask.resize((orig_hw[1], orig_hw[0]), Image.BILINEAR)
    return mask


class GpuMemSampler:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.pid = os.getpid()
        self._stop = threading.Event()
        self.samples = []
        self.ok = False

        if not _NVML_OK:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.ok = True
        except Exception:
            self.ok = False

    def _read_pid_mem_mb(self):
        if not self.ok:
            return None
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            for p in procs:
                if int(p.pid) == int(self.pid):
                    return float(p.usedGpuMemory) / (1024 * 1024)
            return 0.0
        except Exception:
            return None

    def start(self, interval_s: float = 0.01):
        if not self.ok:
            return
        self.samples = []
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                v = self._read_pid_mem_mb()
                if v is not None:
                    self.samples.append(v)
                time.sleep(interval_s)

        self.th = threading.Thread(target=_loop, daemon=True)
        self.th.start()

    def stop(self):
        if not self.ok:
            return
        self._stop.set()
        if hasattr(self, "th"):
            self.th.join(timeout=1.0)

    def summary(self):
        if not self.ok or not self.samples:
            return {"vram_pid_mb_before": None, "vram_pid_mb_peak": None, "vram_pid_mb_after": None}
        return {
            "vram_pid_mb_before": float(self.samples[0]),
            "vram_pid_mb_peak": float(max(self.samples)),
            "vram_pid_mb_after": float(self.samples[-1]),
        }


def infer_one(
    sess: ort.InferenceSession,
    pil_img: Image.Image,
    norm: str,
    invert: bool,
    output_index: int,
    postprocess: str,
):
    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    if not outs:
        raise RuntimeError("ONNX model has no outputs")

    shape = inp.shape  # e.g. [1,3,1024,1024] or [None,3,'unk','unk']
    H = int(shape[2]) if isinstance(shape[2], int) else 1024
    W = int(shape[3]) if isinstance(shape[3], int) else 1024

    x, orig_hw = preprocess(pil_img, H, W, norm=norm)
    # output_index=-1 表示最后一个输出（对齐官方 PyTorch 示例：model(...)[-1]）
    out = outs[output_index]
    y = sess.run([out.name], {inp.name: x})[0]

    # RMBG-2.0 官方示例是 sigmoid
    if postprocess == "sigmoid":
        y = sigmoid_np(np.asarray(y))
    elif postprocess == "minmax":
        # postprocess_mask 内部会做 minmax
        pass
    else:
        raise ValueError(f"Unknown postprocess: {postprocess}")

    mask = postprocess_mask(y, orig_hw)

    if invert:
        mask = Image.eval(mask, lambda p: 255 - p)

    rgba = pil_img.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="输入图片 URL")
    parser.add_argument("--output-dir", default="/home/tonera/website/output")
    parser.add_argument("--model-dir", default=".", help="包含多个 .onnx 的目录（默认当前目录）")
    parser.add_argument("--models", nargs="*", default=None, help="可选：显式指定若干 .onnx 路径；不填则自动扫描 model*.onnx")
    parser.add_argument("--repeat", type=int, default=1, help="每个模型计时重复次数（取平均）")
    parser.add_argument("--warmup", type=int, default=1, help="每个模型预热次数（不计时）")
    # RMBG-2.0 官方 PyTorch 示例是 ImageNet normalize
    parser.add_argument("--norm", choices=["bria", "imagenet"], default="imagenet")
    parser.add_argument("--invert", action="store_true", help="如果前景/背景反了，加这个反相 alpha")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（不测显存）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 序号（NVML 采样用）")
    parser.add_argument("--log-file", default=None, help="日志文件路径（默认写到 output-dir/rmbg_bench.log）")
    parser.add_argument("--output-index", type=int, default=-1, help="选择第几个输出（默认 -1=最后一个输出，贴近官方示例）")
    parser.add_argument(
        "--postprocess",
        choices=["sigmoid", "minmax"],
        default="sigmoid",
        help="输出后处理：sigmoid(官方) 或 minmax(兼容某些已归一化输出)",
    )
    parser.add_argument("--ort-log-severity", type=int, default=3, help="onnxruntime 日志级别：0=verbose 1=info 2=warning 3=error 4=fatal")
    parser.add_argument(
        "--no-cuda-precheck",
        action="store_true",
        help="关闭 CUDA 依赖预检（默认会检查 libcublasLt/libcudnn/libcudart，缺失则自动用 CPU，避免刷屏）",
    )
    args = parser.parse_args()

    # 下载图片
    resp = requests.get(args.url, timeout=60)
    resp.raise_for_status()
    from io import BytesIO
    pil_img = Image.open(BytesIO(resp.content)).convert("RGBA")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file) if args.log_file else (out_dir / "rmbg_bench.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("rmbg_bench")

    # onnxruntime 自检（Torch CUDA 可用 ≠ ORT CUDA provider 可用）
    try:
        ort_ver = getattr(ort, "__version__", "unknown")
        avail = ort.get_available_providers()
        device = ort.get_device()
        logger.info("onnxruntime version=%s device=%s available_providers=%s", ort_ver, device, avail)
        print(f"onnxruntime version={ort_ver} device={device} available_providers={avail}")
    except Exception:
        logger.exception("Failed to query onnxruntime runtime info")

    force_cpu = bool(args.cpu)
    if not force_cpu and not args.no_cuda_precheck:
        miss = missing_cuda_deps()
        if miss:
            force_cpu = True
            msg = (
                "检测到 ORT CUDA 依赖缺失（将回退 CPU）: "
                + ", ".join(miss)
                + ". 你当前日志里的 libcublasLt.so.12 缺失就属于这一类。"
            )
            print("WARNING:", msg)
            logger.warning(msg)

    if args.models:
        model_paths = [Path(p) for p in args.models]
    else:
        model_paths = sorted(Path(args.model_dir).glob("model*.onnx"))

    if not model_paths:
        raise SystemExit("没找到模型：请用 --models 指定，或把 onnx 放在 --model-dir 下且文件名匹配 model*.onnx")

    providers = pick_providers(force_cpu=force_cpu)
    print("Providers:", providers)
    if not args.cpu and "CUDAExecutionProvider" not in providers:
        print("警告：没检测到 CUDAExecutionProvider，将在 CPU 上运行（显存统计可能为空）")
        logger.warning("CUDAExecutionProvider not available; will run on CPU (VRAM stats may be empty).")

    results = []
    for mp in model_paths:
        one = {
            "model": str(mp),
            "providers": providers,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "norm": args.norm,
            "invert": bool(args.invert),
            "output_index": args.output_index,
            "postprocess": args.postprocess,
            "status": "ok",
        }
        print(f"\n==> Model: {mp.name}")
        logger.info("Start model=%s", mp.name)

        # 创建 session（失败则跳过）
        sess = None
        try:
            so = ort.SessionOptions()
            so.log_severity_level = int(args.ort_log_severity)
            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(mp), sess_options=so, providers=providers)
            one["session_init_ms"] = (time.perf_counter() - t0) * 1000.0
            # 记录输入输出签名，方便排查不同 onnx 的差异
            one["input_name"] = sess.get_inputs()[0].name if sess.get_inputs() else None
            one["input_shape"] = sess.get_inputs()[0].shape if sess.get_inputs() else None
            one["output_names"] = [o.name for o in sess.get_outputs()]
            one["output_shapes"] = [o.shape for o in sess.get_outputs()]
            logger.info("IO model=%s input=%s outputs=%s", mp.name, one["input_shape"], one["output_shapes"])
        except Exception as e:
            one["status"] = "failed_init"
            one["error"] = f"{type(e).__name__}: {e}"
            one["traceback"] = traceback.format_exc()
            logger.exception("Failed to init session for model=%s", mp.name)
            results.append(one)
            continue

        # 预热
        try:
            for _ in range(args.warmup):
                _ = infer_one(
                    sess,
                    pil_img,
                    norm=args.norm,
                    invert=args.invert,
                    output_index=args.output_index,
                    postprocess=args.postprocess,
                )
        except Exception as e:
            one["status"] = "failed_warmup"
            one["error"] = f"{type(e).__name__}: {e}"
            one["traceback"] = traceback.format_exc()
            logger.exception("Warmup failed for model=%s", mp.name)
            results.append(one)
            continue

        # 计时 + 显存采样（按当前进程）
        sampler = GpuMemSampler(gpu_index=args.gpu)
        times = []
        out_img = None
        try:
            for _ in range(args.repeat):
                sampler.start(interval_s=0.01)
                t1 = time.perf_counter()
                try:
                    out_img = infer_one(
                        sess,
                        pil_img,
                        norm=args.norm,
                        invert=args.invert,
                        output_index=args.output_index,
                        postprocess=args.postprocess,
                    )
                finally:
                    # 无论成功与否都要 stop，避免后台线程泄露
                    sampler.stop()
                dt = (time.perf_counter() - t1) * 1000.0
                times.append(dt)
        except Exception as e:
            one["status"] = "failed_infer"
            one["error"] = f"{type(e).__name__}: {e}"
            one["traceback"] = traceback.format_exc()
            logger.exception("Infer failed for model=%s", mp.name)

        mem = sampler.summary()
        one.update(mem)

        one["infer_ms_all"] = [float(x) for x in times]
        one["infer_ms_avg"] = float(sum(times) / len(times)) if times else None

        # 保存输出
        if out_img is not None:
            try:
                out_path = out_dir / f"rmbg_{mp.stem}.png"
                out_img.save(out_path)
                one["output"] = str(out_path)
                print(f"saved: {out_path}")
                logger.info("Saved output=%s", out_path)
            except Exception as e:
                one["status"] = "failed_save"
                one["error"] = f"{type(e).__name__}: {e}"
                one["traceback"] = traceback.format_exc()
                logger.exception("Save failed for model=%s", mp.name)
        else:
            logger.warning("No output image for model=%s (status=%s)", mp.name, one.get("status"))

        if "session_init_ms" in one and one["session_init_ms"] is not None:
            if one.get("infer_ms_avg") is not None:
                print(f"session_init_ms={one['session_init_ms']:.2f} infer_ms_avg={one['infer_ms_avg']:.2f}")
            else:
                print(f"session_init_ms={one['session_init_ms']:.2f} infer_ms_avg=None")
        else:
            print("session_init_ms=None infer_ms_avg=None")
        print(f"vram_pid_mb_before={one['vram_pid_mb_before']} peak={one['vram_pid_mb_peak']} after={one['vram_pid_mb_after']}")

        results.append(one)

    # 汇总写盘
    report_path = out_dir / "rmbg_bench_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {report_path}")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    main()
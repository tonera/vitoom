"""
检测 SDXL 单文件模型（*.safetensors）的 prediction_type：epsilon / v_prediction

优先级（从高到低）：
1) 读取 safetensors header 里的 metadata（若存在且包含 prediction_type/ss_prediction_type 等）
2) 通过文件名做弱启发式（例如包含 vpred/v_prediction）
3) （可选）quick visual test：在服务器有 torch+diffusers 时，分别按两种 prediction_type 生成小图对比

用法示例：
  python test/tools/detect_sdxl_prediction_type.py --model /path/to/model.safetensors
  python test/tools/detect_sdxl_prediction_type.py --model /path/to/model.safetensors --json
  python test/tools/detect_sdxl_prediction_type.py --model /path/to/model.safetensors --quick-visual-test --out-dir /tmp/predtype_test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import inspect
import struct
import sys
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 确保能从任意工作目录运行：把项目根目录加入 sys.path
# file: <repo>/test/tools/detect_sdxl_prediction_type.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PRED_EPS = "epsilon"
PRED_V = "v_prediction"


def _normalize_prediction_type(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    # 常见写法归一
    if s in {"epsilon", "eps", "e"}:
        return PRED_EPS
    if s in {"v_prediction", "v-prediction", "vpred", "v_pred", "v", "vp"}:
        return PRED_V
    return None


def _read_safetensors_header(path: Path, max_header_bytes: int = 64 * 1024 * 1024) -> Dict[str, Any]:
    """
    safetensors 文件格式开头：
    - 前 8 字节：little-endian u64，表示 header JSON 的字节长度
    - 接着 header_len 字节：JSON（包含 tensor offsets 和可选 __metadata__）
    """
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("文件过小，无法读取 safetensors header 长度（前 8 字节）")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len <= 0:
            raise ValueError(f"非法 header_len={header_len}")
        if header_len > max_header_bytes:
            raise ValueError(f"header_len={header_len} 超过上限 {max_header_bytes}，疑似不是标准 safetensors 或文件损坏")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"header 未读满：期望 {header_len} 实际 {len(header_bytes)}")
    try:
        return json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"header JSON 解析失败：{e}") from e


def _extract_metadata(header: Dict[str, Any]) -> Dict[str, Any]:
    md = header.get("__metadata__")
    return md if isinstance(md, dict) else {}


def _heuristic_from_filename(path: Path) -> Tuple[Optional[str], str]:
    name = path.name.lower()
    if re.search(r"(v[_\- ]?pred|v[_\- ]?prediction|\bvpred\b)", name):
        return PRED_V, "filename contains vpred/v_prediction"
    if re.search(r"(epsilon|\beps\b)", name):
        return PRED_EPS, "filename contains epsilon/eps"
    return None, "no strong keyword in filename"


@dataclass
class DetectionResult:
    model_path: str
    file_size_bytes: int
    detected_prediction_type: Optional[str]
    confidence: str
    method: str
    evidence: Dict[str, Any]
    notes: str
    timestamp: str


def _read_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _detect_diffusers_pipeline_class(model_dir: Path) -> Optional[str]:
    """
    目录模型常见会在 model_index.json 写 _class_name。
    例如：StableDiffusionXLPipeline / FluxPipeline / ...
    """
    mi = _read_json_if_exists(model_dir / "model_index.json")
    if not mi:
        return None
    v = mi.get("_class_name") or mi.get("class_name") or mi.get("pipeline_class")
    return str(v) if v else None


def _looks_like_sdxl_diffusers_dir(model_dir: Path) -> bool:
    cls = (_detect_diffusers_pipeline_class(model_dir) or "").lower()
    if "stablediffusionxlpipeline".lower() in cls:
        return True
    # 一些 repo 可能不写 class_name；用结构兜底判断
    if (model_dir / "unet").exists() and (model_dir / "vae").exists() and (model_dir / "text_encoder").exists():
        return True
    return False


def _check_sdxl_dir_components(model_dir: Path) -> Tuple[bool, str]:
    """
    仅做轻量校验，避免把不完整目录拿去 from_pretrained 导致迷惑报错。
    """
    required_dirs = ["unet", "vae", "scheduler", "tokenizer", "text_encoder"]
    missing = [d for d in required_dirs if not (model_dir / d).exists()]
    if missing:
        return False, f"missing required subdirs: {missing}"

    # unet 权重文件（常见两种命名）
    unet_ok = any(
        (model_dir / "unet" / fn).exists()
        for fn in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "model.safetensors", "model.bin")
    )
    if not unet_ok:
        return False, "missing unet weights (diffusion_pytorch_model.safetensors/bin or model.safetensors/bin)"

    vae_ok = any(
        (model_dir / "vae" / fn).exists()
        for fn in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "model.safetensors", "model.bin")
    )
    if not vae_ok:
        return False, "missing vae weights (diffusion_pytorch_model.safetensors/bin or model.safetensors/bin)"

    return True, "ok"


def detect_from_metadata(model_path: Path) -> Optional[DetectionResult]:
    header = _read_safetensors_header(model_path)
    md = _extract_metadata(header)

    # 重点查找的 metadata key（不同生态会写不同字段）
    candidate_keys = [
        "prediction_type",
        "ss_prediction_type",  # kohya/ss 常见
        "ss_pred_type",
        "pred_type",
    ]

    found: Dict[str, Any] = {}
    for k in candidate_keys:
        if k in md:
            found[k] = md.get(k)

    # 有些会把信息塞到更长的字段里（例如 training_comment / description 等）
    scan_blob_keys = ["ss_training_comment", "training_comment", "comment", "description", "notes"]
    scanned_blobs: Dict[str, str] = {}
    for k in scan_blob_keys:
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            scanned_blobs[k] = v

    # 1) 直接字段
    for k, v in found.items():
        pred = _normalize_prediction_type(v)
        if pred:
            return DetectionResult(
                model_path=str(model_path),
                file_size_bytes=model_path.stat().st_size,
                detected_prediction_type=pred,
                confidence="high",
                method="safetensors_metadata",
                evidence={
                    "metadata_key": k,
                    "metadata_value": v,
                    "metadata_keys_present": sorted(list(md.keys()))[:200],  # 避免爆输出
                },
                notes="metadata 中存在明确字段，通常比 scheduler_config.json 更可信",
                timestamp=datetime.now().isoformat(),
            )

    # 2) 文本 blob 模糊扫描（中等可信）
    blob_hit = None
    blob_key = None
    for k, text in scanned_blobs.items():
        t = text.lower()
        if "v_prediction" in t or "v-prediction" in t or "vpred" in t:
            blob_hit, blob_key = PRED_V, k
            break
        if "epsilon" in t or re.search(r"\beps\b", t):
            blob_hit, blob_key = PRED_EPS, k
            break

    if blob_hit:
        return DetectionResult(
            model_path=str(model_path),
            file_size_bytes=model_path.stat().st_size,
            detected_prediction_type=blob_hit,
            confidence="medium",
            method="safetensors_metadata_blob_scan",
            evidence={
                "metadata_blob_key": blob_key,
                "metadata_blob_excerpt": (scanned_blobs.get(blob_key, "")[:300] + "...") if blob_key else "",
            },
            notes="从 metadata 的描述文本中推断，可能被手工编辑/不规范；建议结合 quick-visual-test 复核",
            timestamp=datetime.now().isoformat(),
        )

    return None


def detect_prediction_type(model_path: Path) -> DetectionResult:
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    # 目录形式（diffusers）
    if model_path.is_dir():
        if not _looks_like_sdxl_diffusers_dir(model_path):
            cls = _detect_diffusers_pipeline_class(model_path)
            return DetectionResult(
                model_path=str(model_path),
                file_size_bytes=0,
                detected_prediction_type=None,
                confidence="n/a",
                method="skip_not_sdxl",
                evidence={"pipeline_class": cls},
                notes="该目录看起来不是 SDXL（或缺少 SDXL 结构）；prediction_type 检测仅对 SDXL 有意义，已跳过",
                timestamp=datetime.now().isoformat(),
            )

        ok, reason = _check_sdxl_dir_components(model_path)
        if not ok:
            cls = _detect_diffusers_pipeline_class(model_path)
            return DetectionResult(
                model_path=str(model_path),
                file_size_bytes=0,
                detected_prediction_type=None,
                confidence="unknown",
                method="sdxl_dir_incomplete",
                evidence={"pipeline_class": cls, "reason": reason},
                notes="目录看似 SDXL，但组件不完整；无法加载做 quick-latent-test。请补齐目录或用单文件模型。",
                timestamp=datetime.now().isoformat(),
            )

        # 静态信息方面：尝试从 unet 的 safetensors metadata 里找（若有）
        unet_candidates = [
            model_path / "unet" / "diffusion_pytorch_model.safetensors",
            model_path / "unet" / "model.safetensors",
        ]
        for p in unet_candidates:
            if p.exists() and p.is_file() and p.suffix.lower() == ".safetensors":
                md_res = detect_from_metadata(p)
                if md_res and md_res.detected_prediction_type:
                    # 把路径替换成“模型目录”，但保留证据指向具体权重文件
                    md_res.model_path = str(model_path)
                    md_res.evidence["weights_file"] = str(p)
                    return md_res

        return DetectionResult(
            model_path=str(model_path),
            file_size_bytes=0,
            detected_prediction_type=None,
            confidence="unknown",
            method="unknown",
            evidence={"reason": "diffusers_dir_no_metadata_hit"},
            notes="目录模型未在 unet 权重 metadata 中发现 prediction_type；建议用 --quick-latent-test 自动校准并缓存结果",
            timestamp=datetime.now().isoformat(),
        )

    # 单文件形式（safetensors）
    if model_path.suffix.lower() != ".safetensors":
        raise ValueError("当前脚本仅支持 *.safetensors 单文件，或 diffusers 目录形式模型")

    # 1) metadata
    md_res = detect_from_metadata(model_path)
    if md_res:
        return md_res

    # 2) filename heuristic
    pred, reason = _heuristic_from_filename(model_path)
    if pred:
        return DetectionResult(
            model_path=str(model_path),
            file_size_bytes=model_path.stat().st_size,
            detected_prediction_type=pred,
            confidence="low",
            method="filename_heuristic",
            evidence={"reason": reason, "filename": model_path.name},
            notes="仅文件名启发式：请尽量用 metadata 或 quick-visual-test 复核",
            timestamp=datetime.now().isoformat(),
        )

    # 3) unknown
    return DetectionResult(
        model_path=str(model_path),
        file_size_bytes=model_path.stat().st_size,
        detected_prediction_type=None,
        confidence="unknown",
        method="unknown",
        evidence={"reason": reason, "filename": model_path.name},
        notes="未发现可靠 metadata，文件名也无关键字；建议用 --quick-visual-test 或离线人工确认",
        timestamp=datetime.now().isoformat(),
    )


def _print_human(res: DetectionResult) -> None:
    sz_gb = res.file_size_bytes / (1024**3)
    print(f"model: {res.model_path}")
    print(f"size: {res.file_size_bytes} bytes ({sz_gb:.3f} GiB)")
    print(f"prediction_type: {res.detected_prediction_type}")
    print(f"confidence: {res.confidence}")
    print(f"method: {res.method}")
    if res.evidence:
        print("evidence:")
        print(textwrap.indent(json.dumps(res.evidence, ensure_ascii=False, indent=2), "  "))
    print(f"notes: {res.notes}")


def _run_quick_visual_test(
    model_path: Path,
    out_dir: Path,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    offline: bool,
    cache_dir: Optional[str],
    original_config: Optional[str],
) -> None:
    """
    用 diffusers 直接从单文件加载（需要服务器有 torch+diffusers）。
    分别覆写 scheduler.config.prediction_type 为 epsilon / v_prediction，并各生成一张小图用于人工对比。
    """
    # 关键：如果服务器不能联网，建议开启 offline，避免 diffusers/transformers/hf_hub 尝试下载
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 torch 或 torch 导入失败：{e}")

    try:
        from diffusers import StableDiffusionXLPipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 diffusers 或导入失败：{e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("quick-visual-test env:")
    print(f"  torch: {getattr(torch, '__version__', 'unknown')}")
    print(f"  device: {device}")
    if device == "cuda":
        try:
            print(f"  cuda: {torch.version.cuda}")
            print(f"  gpu: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # 为了尽量快：默认 fp16 + 低分辨率 + 较少 steps
    dtype = torch.float16 if device == "cuda" else torch.float32
    # 兼容不同 diffusers 版本：from_single_file 参数可能不同
    def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sig = inspect.signature(fn)
            allowed = set(sig.parameters.keys())
            return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        except Exception:
            # 兜底：如果无法获取 signature，就原样返回（让 diffusers 自己报错）
            return {k: v for k, v in kwargs.items() if v is not None}

    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "variant": None,
        "use_safetensors": True,
        # 关键：离线模式下强制只读本地文件，禁止联网
        "local_files_only": True if offline else None,
        "cache_dir": cache_dir,
        "original_config": original_config,
    }

    try:
        pipe = StableDiffusionXLPipeline.from_single_file(str(model_path), **_filter_kwargs(StableDiffusionXLPipeline.from_single_file, kwargs))
    except Exception as e:
        extra = ""
        if offline:
            extra = (
                "\n提示：你启用了 --offline。若报缺少 tokenizer/config 等文件，说明这些资产未在本机缓存；"
                "需要提前把对应 HuggingFace 缓存目录拷到服务器，或改用本项目已有的本地 diffusers 目录模型做实测。"
            )
        raise RuntimeError(f"from_single_file 加载失败：{e}{extra}") from e
    pipe = pipe.to(device)

    # 一些服务器环境没有 xformers；不强制开启
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    def gen(pred_type: str) -> Path:
        pipe.scheduler.config.prediction_type = pred_type  # 关键：覆写解释方式
        g = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=5.0,
            generator=g,
        ).images[0]
        out_path = out_dir / f"sdxl_predtype_{pred_type}_steps{steps}_s{seed}_{width}x{height}.png"
        image.save(out_path)
        return out_path

    p1 = gen(PRED_EPS)
    p2 = gen(PRED_V)

    print("quick-visual-test outputs:")
    print(f"  epsilon: {p1}")
    print(f"  v_prediction: {p2}")
    print("如何判定：通常“更清晰、更稳定、更不崩坏”的那张对应正确的 prediction_type（请结合你模型的常用 scheduler/采样器）。")


def _run_quick_latent_test(
    model_path: Path,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    offline: bool,
    cache_dir: Optional[str],
    original_config: Optional[str],
) -> None:
    """
    不出图、不保存文件：只生成最终 latent（output_type="latent"）并做数值统计。
    目的：给“业务流程”一个自动化的、尽量短的校准手段（仍然是启发式，不保证 100%）。
    """
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 torch 或 torch 导入失败：{e}")

    try:
        from diffusers import StableDiffusionXLPipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 diffusers 或导入失败：{e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sig = inspect.signature(fn)
            allowed = set(sig.parameters.keys())
            return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        except Exception:
            return {k: v for k, v in kwargs.items() if v is not None}

    # 单文件 vs 目录：分别用 from_single_file / from_pretrained
    if model_path.is_dir():
        kwargs_dir: Dict[str, Any] = {
            "torch_dtype": dtype,
            "variant": None,
            "use_safetensors": True,
            "local_files_only": True if offline else None,
            "cache_dir": cache_dir,
        }
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path),
            **_filter_kwargs(StableDiffusionXLPipeline.from_pretrained, kwargs_dir),
        )
    else:
        kwargs_file: Dict[str, Any] = {
            "torch_dtype": dtype,
            "variant": None,
            "use_safetensors": True,
            "local_files_only": True if offline else None,
            "cache_dir": cache_dir,
            "original_config": original_config,
        }
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            **_filter_kwargs(StableDiffusionXLPipeline.from_single_file, kwargs_file),
        )
    pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # 重要：不同 diffusers/模型组合下，部分组件可能仍保持 float32，导致 Half != float 的 matmul/linear 报错。
    # 在 CUDA+fp16 场景下，尽量把关键组件 dtype 对齐；并在推理时启用 autocast 做兜底。
    def _maybe_cast_component(name: str) -> None:
        m = getattr(pipe, name, None)
        if m is None:
            return
        try:
            # 统一到目标 dtype（通常 fp16）
            m.to(device=device, dtype=dtype)
        except Exception:
            try:
                m.to(device=device)
            except Exception:
                pass

    for comp in ("text_encoder", "text_encoder_2", "unet", "vae"):
        _maybe_cast_component(comp)

    # 可选：复用项目里的 VAE dtype 修复器（即使本测试不 decode，也不影响）
    try:
        from inference.image.runtime.vae_dtype_fixer import ensure_vae_dtype  # type: ignore

        ensure_vae_dtype(pipe)
    except Exception:
        pass

    def _latent_stats(latents: "torch.Tensor") -> Dict[str, Any]:
        x = latents.detach()
        if x.device.type != "cpu":
            x = x.float().cpu()
        finite = torch.isfinite(x)
        finite_ratio = float(finite.float().mean().item())
        # 只在 finite 的元素上统计，避免 NaN/Inf 污染
        xf = x[finite] if finite_ratio > 0 else x.flatten()[:0]
        stats: Dict[str, Any] = {
            "shape": list(latents.shape),
            "dtype": str(latents.dtype),
            "device": str(latents.device),
            "finite_ratio": finite_ratio,
        }
        if xf.numel() > 0:
            stats.update(
                {
                    "mean": float(xf.mean().item()),
                    "std": float(xf.std(unbiased=False).item()),
                    "abs_mean": float(xf.abs().mean().item()),
                    "abs_max": float(xf.abs().max().item()),
                }
            )
        return stats

    def _decode_image_stats(latents: "torch.Tensor") -> Dict[str, Any]:
        """
        不保存图片：仅把 latent decode 成 [0,1] 图像张量并统计分布特征。
        这通常比纯 latent 统计更能区分“prediction_type 用错导致的结果偏灰/偏糊/对比度异常”等问题。
        """
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return {"available": False, "reason": "pipe.vae is None"}

        try:
            scaling = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
        except Exception:
            scaling = 1.0

        try:
            # diffusers 约定：decode 前要除以 scaling_factor
            z = latents
            if scaling and scaling != 1.0:
                z = z / scaling

            # 避免 dtype mismatch：用 autocast（若在 cuda 上）
            if device == "cuda" and dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    dec = vae.decode(z, return_dict=False)[0]
            else:
                dec = vae.decode(z, return_dict=False)[0]

            img = (dec / 2 + 0.5).clamp(0, 1)
        except Exception as e:
            return {"available": False, "reason": f"decode_failed: {type(e).__name__}: {e}"}

        try:
            x = img.detach()
            if x.device.type != "cpu":
                x = x.float().cpu()
            finite = torch.isfinite(x)
            nan_ratio = float(torch.isnan(x).float().mean().item())
            inf_ratio = float(torch.isinf(x).float().mean().item())
            finite_ratio = float(finite.float().mean().item())
            xf = x[finite] if finite_ratio > 0 else x.flatten()[:0]

            stats: Dict[str, Any] = {
                "available": True,
                "scaling_factor": scaling,
                "finite_ratio": finite_ratio,
                "nan_ratio": nan_ratio,
                "inf_ratio": inf_ratio,
                "shape": list(img.shape),
            }
            if xf.numel() > 0:
                # 统计整体
                stats.update(
                    {
                        "mean": float(xf.mean().item()),
                        "std": float(xf.std(unbiased=False).item()),
                        "min": float(xf.min().item()),
                        "max": float(xf.max().item()),
                    }
                )
                # 饱和比例（越高越像“崩了/被夹死”）
                eps = 1e-3
                sat_low = float((xf <= eps).float().mean().item())
                sat_high = float((xf >= 1 - eps).float().mean().item())
                stats["sat_low_ratio"] = sat_low
                stats["sat_high_ratio"] = sat_high
                stats["sat_ratio"] = sat_low + sat_high
            return stats
        except Exception as e:
            return {"available": False, "reason": f"stats_failed: {type(e).__name__}: {e}"}

    def _score(stats: Dict[str, Any]) -> float:
        """
        一个非常粗的启发式评分：更偏好 finite_ratio 高、abs_max 不爆、std 合理的结果。
        评分仅用于“自动推荐”，不是严格数学判据。
        """
        fr = float(stats.get("finite_ratio", 0.0))
        abs_max = float(stats.get("abs_max", 1e9))
        std = float(stats.get("std", 0.0))
        # 惩罚 NaN/Inf
        s = fr * 100.0
        # 惩罚爆炸（阈值是经验值，主要用于区分明显错误的 prediction_type）
        if abs_max > 1000:
            s -= 1000
        elif abs_max > 200:
            s -= 200
        elif abs_max > 50:
            s -= 50
        # std 太小/太大都不理想（latent 的“平/爆”通常能区分 epsilon vs v_prediction）
        if std <= 0:
            s -= 100
        elif std < 0.10:
            s -= 35
        elif std < 0.18:
            s -= 18
        elif std < 0.25:
            s -= 8
        elif std > 8:
            s -= 35
        elif std > 4:
            s -= 15

        # abs_max 过大也常见于用错解释方式
        if abs_max > 50:
            s -= 50
        elif abs_max > 20:
            s -= 20
        elif abs_max > 10:
            s -= 10
        return float(s)

    def _score_with_decode(latent_stats: Dict[str, Any], img_stats: Dict[str, Any]) -> float:
        """
        在基础 latent 健康度上，加上 decode 图像统计的启发式打分。
        目标：自动区分“明显不对”的 prediction_type，而不是追求完美审美。
        """
        s = _score(latent_stats)
        if not img_stats.get("available"):
            return s

        fr = float(img_stats.get("finite_ratio", 0.0))
        s += fr * 50.0

        # decode 出 NaN/Inf：强惩罚
        nan_r = float(img_stats.get("nan_ratio", 0.0))
        inf_r = float(img_stats.get("inf_ratio", 0.0))
        if nan_r > 0 or inf_r > 0:
            s -= 100.0 * (nan_r + inf_r)

        img_std = float(img_stats.get("std", 0.0))
        img_mean = float(img_stats.get("mean", 0.5))
        sat = float(img_stats.get("sat_ratio", 0.0))

        # 对比度太低往往是错误/不收敛（但也可能是 prompt 太简单，所以只做轻惩罚）
        if img_std < 0.05:
            s -= 20
        elif img_std < 0.08:
            s -= 10
        elif img_std > 0.35:
            s -= 5  # 过分“花”也略扣一点

        # 平均亮度太偏（非常粗的异常检测）
        if img_mean < 0.15 or img_mean > 0.85:
            s -= 20
        elif img_mean < 0.25 or img_mean > 0.75:
            s -= 8

        # 饱和（大量 0 或 1）通常是明显错误
        if sat > 0.30:
            s -= 40
        elif sat > 0.15:
            s -= 20
        elif sat > 0.05:
            s -= 5

        return float(s)

    def _gen_latents(pred_type: str) -> Tuple[Dict[str, Any], float]:
        pipe.scheduler.config.prediction_type = pred_type
        g = torch.Generator(device=device).manual_seed(seed)
        # 在 CUDA+fp16 下用 autocast 兜底，减少 dtype mismatch 风险
        if device == "cuda" and dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance_scale=5.0,
                    generator=g,
                    output_type="latent",
                )
        else:
            out = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=5.0,
                generator=g,
                output_type="latent",
            )
        latents = out.images  # output_type="latent" 时，这里就是 latent tensor
        st = _latent_stats(latents)
        img_st = _decode_image_stats(latents)
        st["decoded_image_stats"] = img_st
        return st, _score_with_decode(st, img_st)

    print("quick-latent-test env:")
    print(f"  device: {device}")
    try:
        print(f"  torch_dtype: {dtype}")
        print(f"  pipe.unet.dtype: {getattr(getattr(pipe, 'unet', None), 'dtype', None)}")
        print(f"  pipe.vae.dtype: {getattr(getattr(pipe, 'vae', None), 'dtype', None)}")
        print(f"  pipe.text_encoder.dtype: {getattr(getattr(pipe, 'text_encoder', None), 'dtype', None)}")
        print(f"  pipe.text_encoder_2.dtype: {getattr(getattr(pipe, 'text_encoder_2', None), 'dtype', None)}")
    except Exception:
        pass
    print(f"  steps: {steps}, size: {width}x{height}, seed: {seed}")

    eps_stats, eps_score = _gen_latents(PRED_EPS)
    v_stats, v_score = _gen_latents(PRED_V)

    print("quick-latent-test stats:")
    print("  epsilon:")
    print(textwrap.indent(json.dumps({"score": eps_score, **eps_stats}, ensure_ascii=False, indent=2), "    "))
    print("  v_prediction:")
    print(textwrap.indent(json.dumps({"score": v_score, **v_stats}, ensure_ascii=False, indent=2), "    "))

    if eps_score == v_score:
        rec = None
        conf = "unknown"
    else:
        rec = PRED_EPS if eps_score > v_score else PRED_V
        conf = "low"
    print(f"quick-latent-test recommendation: {rec} (confidence={conf})")
    print("说明：这是启发式自动推荐，用于后台校准/缓存；若两者差异不大，建议以 metadata 或白名单为准。")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect SDXL prediction_type for single-file safetensors model.")
    p.add_argument("--model", required=True, help="单文件模型路径（*.safetensors）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本化处理）")

    # 可选：实测生成两张小图，肉眼对比
    p.add_argument("--quick-latent-test", action="store_true", help="需要 torch+diffusers；不出图，只跑少量步数输出 latent 统计并自动推荐")
    p.add_argument("--quick-visual-test", action="store_true", help="需要 torch+diffusers；生成两张图对比 epsilon vs v_prediction")
    p.add_argument(
        "--fast",
        action="store_true",
        help="快速预设：若未显式指定 steps/width/height，则使用 steps=4、128x128（更快，通常也够区分）",
    )
    p.add_argument("--offline", action="store_true", help="强制离线模式（禁止 hf_hub/transformers 联网下载；若缺文件会直接失败）")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="（可选）传给 diffusers.from_single_file 的 cache_dir，用于指定本地缓存目录（离线服务器建议显式指定）",
    )
    p.add_argument(
        "--original-config",
        default=None,
        help="（可选）传给 diffusers.from_single_file 的 original_config（例如 sd_xl_base.yaml），有些单文件需要它来补齐结构信息",
    )
    p.add_argument("--out-dir", default="./outputs/predtype_test", help="quick-visual-test 输出目录")
    p.add_argument("--prompt", default="a photo of a cat, high quality", help="quick-visual-test prompt")
    p.add_argument("--steps", type=int, default=None, help="steps（默认 12；若 --fast 则默认 4）")
    p.add_argument("--width", type=int, default=None, help="width（默认 512；若 --fast 则默认 128）")
    p.add_argument("--height", type=int, default=None, help="height（默认 512；若 --fast 则默认 128）")
    p.add_argument("--seed", type=int, default=0, help="quick-visual-test seed")
    return p.parse_args()


def main() -> None:
    args = build_args()
    model_path = Path(args.model).expanduser().resolve()

    # 预设：只在用户未显式传参时生效
    if args.steps is None:
        args.steps = 4 if args.fast else 12
    if args.width is None:
        args.width = 128 if args.fast else 512
    if args.height is None:
        args.height = 128 if args.fast else 512

    try:
        res = detect_prediction_type(model_path)
        if args.json:
            print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
        else:
            _print_human(res)

        # 如果已判定“不是 SDXL”或“目录不完整”，则不应该继续做 quick test（否则会强行加载 SDXL pipeline 报错）
        if res.method in {"skip_not_sdxl", "sdxl_dir_incomplete"}:
            if args.quick_latent_test or args.quick_visual_test:
                print(f"skip quick test because method={res.method}")
            return

        if args.quick_latent_test:
            _run_quick_latent_test(
                model_path=model_path,
                prompt=args.prompt,
                steps=args.steps,
                width=args.width,
                height=args.height,
                seed=args.seed,
                offline=args.offline,
                cache_dir=args.cache_dir,
                original_config=args.original_config,
            )

        if args.quick_visual_test:
            out_dir = Path(args.out_dir).expanduser().resolve()
            _run_quick_visual_test(
                model_path=model_path,
                out_dir=out_dir,
                prompt=args.prompt,
                steps=args.steps,
                width=args.width,
                height=args.height,
                seed=args.seed,
                offline=args.offline,
                cache_dir=args.cache_dir,
                original_config=args.original_config,
            )
    except Exception as e:
        # 让 CLI 的错误更友好，也方便你在服务器上批量跑
        msg = f"{type(e).__name__}: {e}"
        if args.json:
            print(json.dumps({"error": msg, "model": str(model_path)}, ensure_ascii=False, indent=2))
        else:
            print(msg)
        raise SystemExit(2)


if __name__ == "__main__":
    main()


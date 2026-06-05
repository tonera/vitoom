#!/usr/bin/env python3
"""
模型类型（family）检测脚本（用于排查/回归检测规则）。

支持：
- 目录（优先读取 model_index.json → 映射到 family）
- 单文件 checkpoint（.safetensors；读取权重 keys 做启发式判别）

示例：
  python scripts/detect_model_type.py resources/models/Z-Image-Turbo
  python scripts/detect_model_type.py resources/models/xxx/model.safetensors
  python scripts/detect_model_type.py --scan resources/models
  python scripts/detect_model_type.py --json resources/models/Z-Image-Turbo
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 让脚本能直接 import inference/common 下的模块（保持与 test/* 脚本一致）
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

try:
    import safetensors.torch  # type: ignore

    SAFETENSORS_AVAILABLE = True
except Exception:
    SAFETENSORS_AVAILABLE = False


@dataclass(frozen=True)
class DetectResult:
    path: str
    kind: str  # "file" | "dir"
    family: Optional[str]
    reason: str


def _detect_family_from_model_index(class_name: str) -> Optional[str]:
    """
    尽量复用项目内 catalog 的映射关系，保证 family 命名与线上一致。
    """
    try:
        from common.model_catalog import get_catalog  # type: ignore

        cat = get_catalog()
        picked = cat.choose_pipeline_ref(str(class_name or ""), is_img2img=False)
        if picked:
            family, _pref = picked
            return str(family)
    except Exception:
        pass

    # 兜底：简单关键字
    cn = str(class_name or "").lower()
    if "zimage" in cn:
        return "zimage"
    if "qwen" in cn and "edit" in cn:
        return "qwen.edit"
    if "qwen" in cn:
        return "qwen"
    if "flux" in cn:
        return "flux"
    if "stable" in cn and "xl" in cn:
        return "sdxl"
    if "stable" in cn and "diffusion" in cn:
        return "sd15"
    return None


def detect_from_dir(model_dir: Path) -> DetectResult:
    model_index = model_dir / "model_index.json"
    if model_index.exists():
        try:
            data = json.loads(model_index.read_text(encoding="utf-8"))
            class_name = str(data.get("_class_name") or "")
            fam = _detect_family_from_model_index(class_name)
            if fam:
                return DetectResult(str(model_dir), "dir", fam, f"model_index.json:_class_name={class_name}")
            return DetectResult(str(model_dir), "dir", None, f"model_index.json:_class_name={class_name} (unmapped)")
        except Exception as e:
            return DetectResult(str(model_dir), "dir", None, f"read model_index.json failed: {e}")

    # 兜底：目录名启发式
    name = model_dir.name.lower()
    if "z-image" in name or "zimage" in name:
        return DetectResult(str(model_dir), "dir", "zimage", "dirname contains zimage")
    if "qwen" in name and "edit" in name:
        return DetectResult(str(model_dir), "dir", "qwen.edit", "dirname contains qwen+edit")
    if "qwen" in name:
        return DetectResult(str(model_dir), "dir", "qwen", "dirname contains qwen")
    if "flux" in name:
        return DetectResult(str(model_dir), "dir", "flux", "dirname contains flux")
    if "sdxl" in name or ("xl" in name and "sd" in name):
        return DetectResult(str(model_dir), "dir", "sdxl", "dirname contains sdxl/xl")
    if "sd15" in name or "sd-15" in name or "stable-diffusion-v1" in name:
        return DetectResult(str(model_dir), "dir", "sd15", "dirname contains sd15")

    return DetectResult(str(model_dir), "dir", None, "no model_index.json and no dirname hint")


def _keys_flags(keys: list[str]) -> dict[str, bool]:
    # 注意：这里故意把 zimage 的特征放在 flux 判别之前，避免 “text_encoders + not cond_stage_model”
    # 这种较宽松的 flux 规则误判 zimage。
    flags = {
        "has_text_encoder_2": False,
        "has_conditioner": False,
        "has_double_blocks": False,
        "has_single_transformer_blocks": False,
        "has_context_embedder": False,
        "has_text_encoders": False,
        "has_cond_stage_model": False,
        "has_input_blocks": False,
        # zimage family（关键差异：Qwen3 text encoder 前缀）
        "has_zimage_qwen3_prefix": False,
        # qwen family（Qwen-Image 单文件常见的 DiT/Transformer 结构信号）
        "has_qwen_time_text_embed": False,
        "has_qwen_pos_embed": False,
        "has_qwen_txt_in": False,
        "has_qwen_img_in": False,
        "has_qwen_transformer_blocks": False,
    }

    for k in keys:
        kl = k.lower()
        if "text_encoder_2" in kl or "text_encoder2" in kl:
            flags["has_text_encoder_2"] = True
        if "conditioner" in kl:
            flags["has_conditioner"] = True

        if "double_blocks" in k:
            flags["has_double_blocks"] = True
        if "single_transformer_blocks" in k:
            flags["has_single_transformer_blocks"] = True
        if "context_embedder" in kl:
            flags["has_context_embedder"] = True

        if "text_encoders" in kl:
            flags["has_text_encoders"] = True
        if "cond_stage_model" in kl:
            flags["has_cond_stage_model"] = True
        if "input_blocks" in k:
            flags["has_input_blocks"] = True

        # zimage: single-file converter 明确依赖这些前缀
        if k.startswith("text_encoders.qwen3_4b.") or k.startswith("text_encoders.qwen3_4b.transformer."):
            flags["has_zimage_qwen3_prefix"] = True

        # qwen-image / DiT：这些前缀在 SD15/SDXL UNet checkpoint 中通常不会出现
        if "model.diffusion_model.time_text_embed." in k:
            flags["has_qwen_time_text_embed"] = True
        if "model.diffusion_model.pos_embed" in k:
            flags["has_qwen_pos_embed"] = True
        if "model.diffusion_model.txt_in" in k:
            flags["has_qwen_txt_in"] = True
        if "model.diffusion_model.img_in" in k:
            flags["has_qwen_img_in"] = True
        if "model.diffusion_model.transformer_blocks" in k:
            flags["has_qwen_transformer_blocks"] = True

    return flags


def detect_from_safetensors(file_path: Path) -> DetectResult:
    if not SAFETENSORS_AVAILABLE:
        # 兜底：文件名启发式
        name = file_path.name.lower()
        if "z-image" in name or "zimage" in name:
            return DetectResult(str(file_path), "file", "zimage", "filename contains zimage (no safetensors)")
        if "flux" in name:
            return DetectResult(str(file_path), "file", "flux", "filename contains flux (no safetensors)")
        if "sdxl" in name:
            return DetectResult(str(file_path), "file", "sdxl", "filename contains sdxl (no safetensors)")
        return DetectResult(str(file_path), "file", None, "safetensors not installed; cannot inspect keys")

    try:
        with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:  # type: ignore
            keys = list(f.keys())
            if not keys:
                return DetectResult(str(file_path), "file", None, "empty keys")

            flags = _keys_flags(keys)

            # ===== zimage（必须优先于 flux）=====
            if flags["has_zimage_qwen3_prefix"]:
                return DetectResult(str(file_path), "file", "zimage", "keys contain text_encoders.qwen3_4b.*")

            # ===== sdxl =====
            if flags["has_text_encoder_2"] or flags["has_conditioner"]:
                return DetectResult(str(file_path), "file", "sdxl", "keys contain text_encoder_2/conditioner")

            # ===== qwen =====
            # 典型 Qwen-Image 单文件经常只有 model.diffusion_model.*（DiT/Transformer），没有 UNet 的 input_blocks。
            # 为避免误判：
            # - 优先：命中 >=2 个 DiT 结构信号
            # - 兜底：几乎全是 model.diffusion_model.* 且包含 time_text_embed，并且不具备 SDXL/SD15/Flux 的关键特征
            qwen_hits = sum(
                [
                    1 if flags["has_qwen_time_text_embed"] else 0,
                    1 if flags["has_qwen_pos_embed"] else 0,
                    1 if flags["has_qwen_txt_in"] else 0,
                    1 if flags["has_qwen_img_in"] else 0,
                    1 if flags["has_qwen_transformer_blocks"] else 0,
                ]
            )
            diffusion_prefix_cnt = sum(1 for k in keys if k.startswith("model.diffusion_model."))
            diffusion_prefix_ratio = diffusion_prefix_cnt / max(len(keys), 1)
            qwen_fallback_ok = (
                flags["has_qwen_time_text_embed"]
                and diffusion_prefix_ratio >= 0.8
                and (not flags["has_input_blocks"])
                and (not flags["has_conditioner"])
                and (not flags["has_cond_stage_model"])
                and (not flags["has_text_encoders"])
                and (not flags["has_double_blocks"])
                and (not flags["has_single_transformer_blocks"])
            )

            if (qwen_hits >= 2 and (not flags["has_input_blocks"]) and (not flags["has_double_blocks"])) or qwen_fallback_ok:
                reason_bits = []
                if flags["has_qwen_time_text_embed"]:
                    reason_bits.append("time_text_embed")
                if flags["has_qwen_pos_embed"]:
                    reason_bits.append("pos_embed")
                if flags["has_qwen_txt_in"]:
                    reason_bits.append("txt_in")
                if flags["has_qwen_img_in"]:
                    reason_bits.append("img_in")
                if flags["has_qwen_transformer_blocks"]:
                    reason_bits.append("transformer_blocks")
                reason = f"keys look like DiT/Transformer ({', '.join(reason_bits)})"
                if qwen_fallback_ok and qwen_hits < 2:
                    reason += f" + diffusion_prefix_ratio={diffusion_prefix_ratio:.2f} (fallback)"
                return DetectResult(str(file_path), "file", "qwen", reason)

            # ===== flux =====
            if flags["has_double_blocks"]:
                return DetectResult(str(file_path), "file", "flux", "keys contain double_blocks")
            if flags["has_single_transformer_blocks"] and flags["has_context_embedder"]:
                return DetectResult(str(file_path), "file", "flux", "keys contain single_transformer_blocks + context_embedder")
            if flags["has_text_encoders"] and (not flags["has_cond_stage_model"]):
                return DetectResult(str(file_path), "file", "flux", "keys contain text_encoders and not cond_stage_model")

            # ===== sd15 =====
            if flags["has_input_blocks"] and flags["has_cond_stage_model"]:
                return DetectResult(str(file_path), "file", "sd15", "keys contain input_blocks + cond_stage_model")

            # ===== fallback：UNet 输入通道（尽量与 PipelineDetector 对齐）=====
            for k in keys:
                if "input_blocks.0.0.weight" in k or "model.diffusion_model.input_blocks.0.0.weight" in k:
                    try:
                        w = f.get_tensor(k)
                        if len(w.shape) >= 2:
                            ch = int(w.shape[0])
                            if ch == 9:
                                return DetectResult(str(file_path), "file", "sdxl", "unet input_channels=9")
                            if ch == 4:
                                return DetectResult(str(file_path), "file", "sd15", "unet input_channels=4 (fallback)")
                    except Exception:
                        continue

            return DetectResult(str(file_path), "file", None, "no matching key patterns")
    except Exception as e:
        return DetectResult(str(file_path), "file", None, f"safe_open failed: {e}")


def detect_path(p: Path) -> DetectResult:
    if not p.exists():
        return DetectResult(str(p), "file", None, "path not found")
    if p.is_dir():
        return detect_from_dir(p)
    if p.is_file():
        suf = p.suffix.lower()
        if suf == ".safetensors":
            return detect_from_safetensors(p)
        if suf == ".ckpt":
            # 目前不做 ckpt 解析，避免引入 torch 依赖；仍然保留 filename 兜底
            name = p.name.lower()
            if "z-image" in name or "zimage" in name:
                return DetectResult(str(p), "file", "zimage", "filename contains zimage (.ckpt)")
            if "flux" in name:
                return DetectResult(str(p), "file", "flux", "filename contains flux (.ckpt)")
            if "sdxl" in name:
                return DetectResult(str(p), "file", "sdxl", "filename contains sdxl (.ckpt)")
            if "sd15" in name or "stable-diffusion-v1" in name:
                return DetectResult(str(p), "file", "sd15", "filename contains sd15 (.ckpt)")
            return DetectResult(str(p), "file", None, "ckpt unsupported (only filename heuristic)")
        return DetectResult(str(p), "file", None, f"unsupported suffix: {p.suffix}")
    return DetectResult(str(p), "file", None, "unknown path type")


def _iter_scan(root: Path):
    for item in sorted(root.iterdir()):
        # 只扫描“看起来像模型”的候选：目录 / safetensors / ckpt
        if item.is_dir():
            yield item
            continue
        if item.is_file() and item.suffix.lower() in {".safetensors", ".ckpt"}:
            yield item


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="检测模型类型（family）：目录(model_index.json) / 单文件(.safetensors)")
    ap.add_argument("paths", nargs="+", help="模型目录或 checkpoint 文件路径")
    ap.add_argument("--scan", action="store_true", help="把每个 path 当作目录扫描其一级子项并逐个检测")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出（适合脚本调用）")
    args = ap.parse_args(argv)

    targets: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / raw
        if args.scan and p.exists() and p.is_dir():
            targets.extend(list(_iter_scan(p)))
        else:
            targets.append(p)

    results = [detect_path(p) for p in targets]

    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        for r in results:
            fam = r.family or "UNKNOWN"
            print(f"{fam:8s}  {r.path}  ({r.reason})")

    # 如果存在无法识别的条目，返回非 0，便于 CI/批处理发现问题
    return 0 if all(r.family for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())


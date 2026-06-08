"""
RBG（去背景）处理器：参考 aiservice/RMBG-1.4/RbgProcessor.py

依赖：
- briarmbg（BriaRMBG）
- torch, torchvision
- pillow, numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize

from .logger import get_logger

logger = get_logger(__name__)


def preprocess_image(im: np.ndarray, model_input_size: List[int]) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = (
        F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear")
        .type(torch.uint8)
    )
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image(result: torch.Tensor, im_size: Tuple[int, int]) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=list(im_size), mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


@dataclass
class RbgConfig:
    model_dir: str
    backend: Literal["auto", "rmbg1", "rmbg2"] = "auto"
    # RMBG-2.0 默认走 transformers 的本地目录加载（用户不联网更友好）
    # - 如果传的是一个本地目录（例如 resources/weights/RMBG-2.0），就会从该目录加载。
    # - 如果你希望直接从 HuggingFace 拉取，可把 model_dir 设置为 "briaai/RMBG-2.0" 并 local_files_only=False。
    local_files_only: bool = True
    batch_size_cuda: int = 4
    input_size: Tuple[int, int] = (1024, 1024)


class RbgProcessor:
    def __init__(self, cfg: RbgConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = Path(cfg.model_dir)
        # backend 自动推断：目录存在且包含 transformers 模型的常见文件时，优先 rmbg2
        backend = cfg.backend
        if backend == "auto":
            if model_path.exists() and model_path.is_dir():
                has_tf = (model_path / "config.json").exists()
                has_weights = any(
                    (model_path / n).exists()
                    for n in (
                        "model.safetensors",
                        "pytorch_model.bin",
                        "model.fp16.safetensors",
                    )
                )
                backend = "rmbg2" if (has_tf and has_weights) else "rmbg1"
            else:
                backend = "rmbg2" if "/" in cfg.model_dir or cfg.model_dir.count("-") else "rmbg1"

        self.backend: Literal["rmbg1", "rmbg2"] = backend  # type: ignore[assignment]

        if self.backend == "rmbg1":
            # fail-fast：固定从项目内置 runtime 导入（不做任何兜底/兼容）
            # 需要文件存在：inference/image/runtime/briarmbg.py
            # 且 runtime 目录为可导入包（runtime/__init__.py）
            try:
                # 说明：本项目推理侧通常把 `inference/` 加入 sys.path，
                # 因此顶层模块名是 `image`/`common`，而不是 `inference.*`。
                from image.runtime.briarmbg import BriaRMBG  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "RBG(rmbg1) import failed: "
                    "expected 'image.runtime.briarmbg.BriaRMBG'. "
                    "Please ensure 'inference/image/runtime/briarmbg.py' exists "
                    "and 'inference/image/runtime/__init__.py' exists."
                ) from e

            if not model_path.exists():
                raise FileNotFoundError(f"RMBG-1.4 model_dir not found: {cfg.model_dir}")

            self.net = BriaRMBG.from_pretrained(str(model_path), local_files_only=True)
            self.net.to(self.device)
            if self.device.type == "cuda":
                try:
                    self.net.half()
                except Exception:
                    pass
            self.net.eval()
            self._tf_model = None
            self._tf_transform = None
            return

        # RMBG-2.0（transformers / safetensors）
        try:
            from transformers import AutoModelForImageSegmentation  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("RBG(rmbg2) requires package 'transformers'") from e

        # 允许 model_dir 既可以是本地目录，也可以是 HF model id
        model_ref = cfg.model_dir
        if model_path.exists() and model_path.is_dir():
            model_ref = str(model_path)

        self._tf_transform = transforms.Compose(
            [
                transforms.Resize(cfg.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self._tf_model = AutoModelForImageSegmentation.from_pretrained(
            model_ref,
            trust_remote_code=True,
            local_files_only=bool(cfg.local_files_only),
        )
        self._tf_model.eval()
        self._tf_model.to(self.device)
        if self.device.type == "cuda":
            try:
                self._tf_model.half()
            except Exception:
                pass
        self.net = None  # type: ignore[assignment]

    def generate(self, pil_images: List[Image.Image]) -> List[Image.Image]:
        ret: List[Image.Image] = []
        logger.info(
            f"RBG: backend={self.backend} images={len(pil_images)} device={self.device.type}"
        )

        if self.backend == "rmbg2":
            return self._generate_rmbg2(pil_images)

        preprocessed: List[torch.Tensor] = []
        metas: List[Tuple[Tuple[int, int], Image.Image]] = []
        for img in pil_images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            pil_rgb = img.convert("RGB")
            np_rgb = np.array(pil_rgb)
            tensor = preprocess_image(np_rgb, [self.cfg.input_size[0], self.cfg.input_size[1]])
            preprocessed.append(tensor)
            metas.append((np_rgb.shape[0:2], pil_rgb))

        if not preprocessed:
            return ret

        batch_size = self.cfg.batch_size_cuda if self.device.type == "cuda" else 1
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        with torch.inference_mode():
            for start in range(0, len(preprocessed), batch_size):
                end = min(start + batch_size, len(preprocessed))
                batch = torch.cat(preprocessed[start:end], dim=0).to(
                    self.device, dtype=dtype, non_blocking=True
                )
                outputs, _ = self.net(batch)
                d1 = outputs[0]  # [B,1,H,W]
                for i in range(d1.shape[0]):
                    orig_im_size, orig_pil = metas[start + i]
                    result_image = postprocess_image(d1[i : i + 1], orig_im_size)
                    mask_pil = Image.fromarray(result_image)  # L
                    canvas = Image.new("RGBA", orig_pil.size, (0, 0, 0, 0))
                    canvas.paste(orig_pil, mask=mask_pil)
                    ret.append(canvas)

        return ret

    def _generate_rmbg2(self, pil_images: List[Image.Image]) -> List[Image.Image]:
        assert self._tf_model is not None
        assert self._tf_transform is not None

        ret: List[Image.Image] = []
        preprocessed: List[torch.Tensor] = []
        metas: List[Tuple[Tuple[int, int], Image.Image]] = []

        for img in pil_images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            pil_rgba = img.convert("RGBA")
            pil_rgb = pil_rgba.convert("RGB")
            orig_h, orig_w = pil_rgb.size[1], pil_rgb.size[0]
            tensor = self._tf_transform(pil_rgb).unsqueeze(0)  # 1CHW
            preprocessed.append(tensor)
            metas.append(((orig_h, orig_w), pil_rgb))

        if not preprocessed:
            return ret

        batch_size = self.cfg.batch_size_cuda if self.device.type == "cuda" else 1
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        with torch.inference_mode():
            for start in range(0, len(preprocessed), batch_size):
                end = min(start + batch_size, len(preprocessed))
                batch = torch.cat(preprocessed[start:end], dim=0).to(self.device, dtype=dtype)

                out = self._tf_model(batch)
                # 官方示例：model(input)[-1].sigmoid()
                logits = out[-1] if isinstance(out, (tuple, list)) else out
                mask = torch.sigmoid(logits)  # [B,1,H,W] (通常)

                for i in range(mask.shape[0]):
                    (orig_h, orig_w), orig_pil = metas[start + i]
                    m = mask[i].detach().float()
                    if m.ndim == 3:
                        m = m[0]  # [H,W]
                    m = m.clamp(0, 1)
                    m_pil = Image.fromarray((m.cpu().numpy() * 255).astype(np.uint8), mode="L")
                    m_pil = m_pil.resize((orig_w, orig_h), Image.BILINEAR)

                    canvas = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
                    canvas.paste(orig_pil, mask=m_pil)
                    ret.append(canvas)

        return ret


def resolve_rmbg_model_dir(weights_dir: str) -> str:
    """
    尝试从 models_dir 推断 RMBG-1.4 模型目录。
    支持：
    - {models_dir}/RMBG-1.4
    - {models_dir}/rmbg/RMBG-1.4
    - 直接 {models_dir} 本身就是模型目录
    """
    wd = Path(weights_dir)
    candidates = [
        wd / "RMBG-1.4",
        wd / "rmbg" / "RMBG-1.4",
        wd,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return str(p)
    return str(wd / "RMBG-1.4")


def resolve_rmbg2_model_dir(models_dir: str) -> str:
    """
    推断 RMBG-2.0（transformers）模型目录（基于 models_dir）。
    支持：
    - {models_dir}/RMBG-2.0
    - {models_dir}/rmbg/RMBG-2.0
    - 直接 {models_dir} 本身就是模型目录
    """
    wd = Path(models_dir)
    candidates = [
        wd / "RMBG-2.0",
        wd / "rmbg" / "RMBG-2.0",
        wd,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return str(p)
    return str(wd / "RMBG-2.0")


def resolve_rmbg_backend_and_dir(models_dir: str) -> Tuple[Literal["rmbg2", "rmbg1"], str]:
    """
    自动选择去背景模型：
    - 若存在 RMBG-2.0 目录则优先使用
    - 否则回落到 RMBG-1.4（briarmbg）
    """
    rmbg2 = Path(resolve_rmbg2_model_dir(models_dir))
    if rmbg2.exists() and rmbg2.is_dir() and (rmbg2 / "config.json").exists():
        return "rmbg2", str(rmbg2)
    return "rmbg1", resolve_rmbg_model_dir(models_dir)



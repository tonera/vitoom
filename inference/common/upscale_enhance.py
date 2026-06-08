"""
图片超分 + 人脸增强：参考 aiservice/diffusers/tonera/UpscaleEnhance.py

依赖：
- realesrgan, basicsr
- gfpgan
- opencv-python, numpy, pillow, torch

权重目录约定（默认从 inference_config.models_dir 推断）：
- {models_dir}/Real-ESRGAN/RealESRGAN_x2plus.pth / RealESRGAN_x4plus.pth / RealESRGAN_x4plus_anime_6B.pth / realesr-animevideov3.pth
- {models_dir}/Real-ESRGAN/GFPGANv1.4.pth 或 RestoreFormer.pth

兼容兜底：
- 仍支持旧的 {weights_dir} 或 {weights_dir}/weights 目录结构。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .logger import get_logger
from .face_enhancers import FaceEnhancerBuildConfig, build_face_enhancer

logger = get_logger(__name__)

def _install_torchvision_functional_tensor_shim() -> None:
    """
    兼容 basicsr/realesrgan 对旧 torchvision API 的依赖：
    - 旧代码：from torchvision.transforms.functional_tensor import rgb_to_grayscale
    - 新版 torchvision：rgb_to_grayscale 位于 torchvision.transforms.functional

    说明：不再依赖 `sitecustomize.py` 是否被自动加载；在真正 import 依赖前主动注入 shim。
    """
    import sys
    import types

    name = "torchvision.transforms.functional_tensor"
    # 若已存在（可能来自其他 shim 或半残留），也尽量覆盖/补齐缺失属性，避免继续报缺模块/缺符号
    existing = sys.modules.get(name)

    try:
        from torchvision.transforms import functional as F  # type: ignore
    except Exception:
        return

    shim = existing if isinstance(existing, types.ModuleType) else types.ModuleType(name)
    # 最常见的缺失符号：basicsr 旧版本会 import 这个
    if hasattr(F, "rgb_to_grayscale"):
        shim.rgb_to_grayscale = getattr(F, "rgb_to_grayscale")  # type: ignore[attr-defined]

    # 兜底：若未来/其他依赖从 functional_tensor 取更多符号，
    # 统一转发到 torchvision.transforms.functional（新版位置）。
    def __getattr__(attr: str):  # type: ignore[override]
        v = getattr(F, attr, None)
        if v is None:
            raise AttributeError(attr)
        return v

    shim.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[name] = shim


@dataclass
class UpscaleEnhanceConfig:
    weights_dir: str
    mode: str = "normal"  # normal/video/anime
    arch: str = "clean"  # clean/original/RestoreFormer
    upscale: int = 0  # 0/1/2/4
    face_enhance: bool = False


class UpscaleEnhance:
    def __init__(self, cfg: UpscaleEnhanceConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # normalize config
        if self.cfg.upscale not in (0, 1, 2, 4):
            self.cfg.upscale = 2
        if self.cfg.arch not in ("RestoreFormer", "clean", "original"):
            self.cfg.arch = "clean"

        self._upsampler = None
        self._face_enhancer = None

    def _resolve_weight(self, fname: str) -> str:
        wd = Path(self.cfg.weights_dir)
        # 新目录：{models_dir}/Real-ESRGAN
        candidates = [
            wd / "Real-ESRGAN" / fname,
            wd / "Real-ESRGAN" / "weights" / fname,
            # roop 一体化目录（本项目常用）：{models_dir}/roop
            wd / "roop" / fname,
            # 旧目录兼容：{weights_dir} / {weights_dir}/weights
            wd / fname,
            wd / "weights" / fname,
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        # default first candidate
        return str(candidates[0])

    def _get_net_model(self):
        _install_torchvision_functional_tensor_shim()
        # imports are heavy; delay
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

        model = None
        load_name = "RealESRGAN_x2plus"
        if self.cfg.mode == "anime":
            load_name = "RealESRGAN_x4plus_anime_6B"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        elif self.cfg.mode == "video":
            load_name = "realesr-animevideov3"
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        else:
            if self.cfg.upscale == 4:
                load_name = "RealESRGAN_x4plus"
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            else:
                load_name = "RealESRGAN_x2plus"
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        return load_name, model

    def _load_upscaler(self):
        try:
            _install_torchvision_functional_tensor_shim()
            from realesrgan import RealESRGANer  # type: ignore
        except Exception as e:  # pragma: no cover
            # 常见场景：realesrgan/basicsr 依赖的 torchvision 版本不匹配（functional_tensor 被移除）
            raise RuntimeError(
                "upscale import failed. Please ensure 'realesrgan' dependencies are satisfied. "
                "If you see 'torchvision.transforms.functional_tensor' missing, upgrade/downgrade "
                "torch/torchvision or rely on project 'sitecustomize.py' shim. "
                f"Original error: {e}"
            ) from e

        load_name, net_model = self._get_net_model()
        model_path = self._resolve_weight(load_name + ".pth")

        tile = int(os.environ.get("TONERA_SR_TILE", "500"))
        tile_pad = int(os.environ.get("TONERA_SR_TILE_PAD", "10"))
        pre_pad = int(os.environ.get("TONERA_SR_PRE_PAD", "0"))
        half = False

        try:
            upscaler = RealESRGANer(
                scale=self.cfg.upscale,
                model_path=model_path,
                dni_weight=None,
                model=net_model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=half,
                device="cuda" if self.device == "cuda" else "cpu",
            )
        except TypeError:
            upscaler = RealESRGANer(
                scale=self.cfg.upscale,
                model_path=model_path,
                dni_weight=None,
                model=net_model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=half,
                gpu_id=0 if self.device == "cuda" else None,
            )
        return upscaler

    def _load_face_enhance(self, upsampler):
        """
        构建人脸增强器（可插拔后端）。

        默认使用 CodeFormer；通过环境变量切换：
        - VITOOM_FACE_ENHANCER=codeformer|gfpgan（默认 codeformer）
        """
        build_cfg = FaceEnhancerBuildConfig(
            weights_dir=self.cfg.weights_dir,
            arch=self.cfg.arch,
            upscale=self.cfg.upscale if self.cfg.upscale >= 1 else 1,
            bg_upsampler=upsampler,
        )
        return build_face_enhancer(build_cfg)

    def _maybe_downscale(self, img: Image.Image) -> Image.Image:
        try:
            safe_pixels = int(os.environ.get("TONERA_UPSCALE_SAFE_PIXELS", "4000000"))
            w, h = img.size
            if w * h <= safe_pixels:
                return img
            ratio = (w * h / safe_pixels) ** 0.5
            nw, nh = max(1, int(w / ratio)), max(1, int(h / ratio))
            return img.resize((nw, nh), Image.BICUBIC)
        except Exception:
            return img

    def run(self, images: List[Image.Image]) -> List[Image.Image]:
        if not images:
            return []

        t0 = time.time()
        results: List[Image.Image] = []

        # prepare upsampler/face enhancer once per call
        if self.cfg.upscale in (0, 1):
            self._upsampler = None
        else:
            lt = time.time()
            self._upsampler = self._load_upscaler()
            logger.info(f"Upscaler loaded in {round(time.time()-lt,3)}s")

        if self.cfg.face_enhance:
            lt = time.time()
            try:
                self._face_enhancer = self._load_face_enhance(self._upsampler)
                if self._face_enhancer is None:
                    raise RuntimeError("face_enhance backend is not available")
                backend_name = getattr(self._face_enhancer, "__class__", type("X", (), {})).__name__
                logger.info(f"FaceEnhancer loaded backend={backend_name} in {round(time.time()-lt,3)}s")
            except Exception as e:
                logger.warning(f"FaceEnhancer init failed, disable face_enhance for this call: {e}")
                self._face_enhancer = None
        else:
            self._face_enhancer = None

        for img in images:
            try:
                proc_img = self._maybe_downscale(img)
                bgr = cv2.cvtColor(np.asarray(proc_img), cv2.COLOR_RGB2BGR)

                if self.cfg.face_enhance and self._face_enhancer is not None:
                    out = self._face_enhancer.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
                    out_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                else:
                    if self._upsampler is None or self.cfg.upscale in (0, 1):
                        out_img = proc_img
                    else:
                        out, _ = self._upsampler.enhance(bgr, outscale=self.cfg.upscale)
                        out_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

                results.append(out_img)
            except Exception as e:
                logger.warning(f"upscale/face_enhance failed, returning original: {e}")
                results.append(img)

        # release per-call refs
        self._face_enhancer = None
        self._upsampler = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info(
            f"UpscaleEnhance total={round(time.time()-t0,3)}s images={len(images)} "
            f"mode={self.cfg.mode} arch={self.cfg.arch} upscale={self.cfg.upscale} face={self.cfg.face_enhance}"
        )
        return results



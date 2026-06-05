from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from common.logger import get_logger

logger = get_logger(__name__)


def _install_torchvision_functional_tensor_shim() -> None:
    """
    兼容 gfpgan/basicsr 旧版本对 torchvision API 的依赖。
    项目根目录有 sitecustomize.py，但这里再做一次“就地兜底”，避免某些运行方式未触发 sitecustomize。
    """
    try:
        import sys
        import types
    except Exception:
        return

    name = "torchvision.transforms.functional_tensor"
    existing = sys.modules.get(name)
    try:
        from torchvision.transforms import functional as F  # type: ignore
        import torchvision.transforms as T  # type: ignore
    except Exception:
        return

    shim = existing if isinstance(existing, types.ModuleType) else types.ModuleType(name)
    if hasattr(F, "rgb_to_grayscale"):
        shim.rgb_to_grayscale = getattr(F, "rgb_to_grayscale")  # type: ignore[attr-defined]

    def __getattr__(attr: str):  # type: ignore[override]
        v = getattr(F, attr, None)
        if v is None:
            raise AttributeError(attr)
        return v

    shim.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[name] = shim
    try:
        setattr(T, "functional_tensor", shim)
    except Exception:
        pass


def _prepare_gfpgan_aux_weights(weights_dir: str) -> None:
    """
    预置 facexlib/GFPGAN 所需的辅助权重，避免运行时下载到不可控目录。

    目标位置（facexlib 历史默认）：{cwd}/gfpgan/weights/
    来源优先：{models_dir}/roop/
    """
    default_dir = os.path.abspath(os.path.join(os.getcwd(), "gfpgan", "weights"))
    os.makedirs(default_dir, exist_ok=True)

    wd = Path(weights_dir)
    roop_dir = wd / "roop"
    files = ["detection_Resnet50_Final.pth", "parsing_parsenet.pth", "parsing_bisenet.pth"]
    for fname in files:
        dst = os.path.join(default_dir, fname)
        if os.path.isfile(dst):
            continue
        src = (roop_dir / fname) if roop_dir.is_dir() else (wd / fname)
        if not src.is_file():
            continue
        try:
            os.symlink(str(src), dst)
        except Exception:
            try:
                import shutil

                shutil.copy2(str(src), dst)
            except Exception:
                pass


class GFPGANEnhancer:
    def __init__(
        self,
        *,
        model_path: str,
        weights_dir: str,
        arch: str,
        upscale: int,
        bg_upsampler: Optional[Any] = None,
    ) -> None:
        self.model_path = model_path
        self.weights_dir = weights_dir
        self.arch = arch or "clean"
        self.upscale = int(upscale or 1)
        self.bg_upsampler = bg_upsampler
        self._impl = None

    def _ensure(self):
        if self._impl is not None:
            return
        _install_torchvision_functional_tensor_shim()
        try:
            from gfpgan import GFPGANer  # type: ignore
        except Exception:
            # 兼容某些版本的 import 路径
            from gfpgan.utils import GFPGANer  # type: ignore

        try:
            _prepare_gfpgan_aux_weights(self.weights_dir)
        except Exception:
            pass

        self._impl = GFPGANer(
            model_path=self.model_path,
            upscale=max(1, int(self.upscale)),
            arch=self.arch,
            channel_multiplier=2,
            bg_upsampler=self.bg_upsampler,
        )

    def enhance(
        self,
        bgr: np.ndarray,
        *,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        self._ensure()
        try:
            _, _, out = self._impl.enhance(  # type: ignore[union-attr]
                bgr,
                has_aligned=bool(has_aligned),
                only_center_face=bool(only_center_face),
                paste_back=bool(paste_back),
            )
            return out if out is not None else bgr
        except Exception as e:
            logger.warning(f"GFPGAN enhance failed: {e}")
            return bgr

    def close(self) -> None:
        self._impl = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


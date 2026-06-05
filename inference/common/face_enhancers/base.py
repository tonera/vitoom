from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np


class FaceEnhancer(Protocol):
    """
    人脸增强器接口（与现有 GFPGAN 调用方式对齐）。

    约定：
    - 输入/输出：OpenCV BGR ndarray（uint8）
    - paste_back=True：返回“贴回原图后的整图”
    """

    def enhance(
        self,
        bgr: np.ndarray,
        *,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FaceEnhancerBuildConfig:
    weights_dir: str
    arch: str
    upscale: int
    bg_upsampler: Optional[Any] = None
    # codeformer fidelity weight (w)
    codeformer_w: float = 0.5


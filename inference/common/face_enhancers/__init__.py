"""
人脸增强后端（可插拔）。

当前约定：
- 输入/输出均为 OpenCV BGR ndarray（uint8），便于与现有 UpscaleEnhance/GFPGAN 兼容。
- 通过环境变量选择实现：默认 CodeFormer，可回退 GFPGAN。
"""

from .base import FaceEnhancer, FaceEnhancerBuildConfig
from .factory import build_face_enhancer

__all__ = ["FaceEnhancer", "FaceEnhancerBuildConfig", "build_face_enhancer"]


"""
Wan2 系列视频推理 handlers（当前专用实现）

后续扩展其他模型家族时，建议新增并行的子包，例如：
- video/handlers/hunyuan/...
- video/handlers/seedance/...
并在 VideoInferrer 中做 family/registry 路由。
"""

from .mkv_handler import Wan2MkvHandler
from .s2v_handler import Wan2S2vHandler
from .inp_handler import Wan2InpHandler
from .ccv_handler import Wan2CcvHandler

__all__ = ["Wan2MkvHandler", "Wan2S2vHandler", "Wan2InpHandler", "Wan2CcvHandler"]


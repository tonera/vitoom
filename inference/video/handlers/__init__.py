"""
视频推理 handlers
"""

from .video_handler import VideoHandler
from .wan2.mkv_handler import Wan2MkvHandler
from .wan2.s2v_handler import Wan2S2vHandler
from .wan2.inp_handler import Wan2InpHandler
from .wan2.ccv_handler import Wan2CcvHandler

__all__ = [
    "VideoHandler",
    "Wan2MkvHandler",
    "Wan2S2vHandler",
    "Wan2InpHandler",
    "Wan2CcvHandler",
]


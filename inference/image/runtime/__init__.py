"""
inference.image.runtime
运行时工具集合（推理管线 patch / 资源管理 / prompt 辅助等）
"""

from .nunchaku_transformer_fixer import (
    configure_hybrid_offload,
    enable_qwenimage_nunchaku_compat,
    patch_apply_rotary_emb_qwen,
    patch_nunchaku_qwenembedrope_forward,
    set_qwen_video_fhw_from_size,
)



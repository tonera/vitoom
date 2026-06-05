"""
SDPA (Scaled Dot-Product Attention) helpers.

目标：
- 在支持新架构（如 sm121）的环境里，避免触发 mem_efficient/cutlass 路径导致的 fmha_cutlass* FATAL 刷屏
- 尽量保留 flash attention 的性能；若 flash 不可用则回退到 math

用法：
    from inference.common.sdpa_utils import sdpa_ctx

    with sdpa_ctx():
        ... 你的推理代码 ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def sdpa_ctx() -> Iterator[None]:
    """
    一个“局部生效”的 SDPA 上下文：
    - 允许：FLASH_ATTENTION、MATH
    - 禁用：MEM_EFFICIENT（cutlass）

    注意：这是“每次推理包一层”的方案，不做任何全局开关设置。
    """
    # 只在“新 API 不存在/导入失败”时才 fallback。
    # 不能用裸 except 包住整个 yield：否则推理过程中的异常会被误捕获，
    # 进而走到 fallback 再 yield 一次，触发：
    #   RuntimeError: generator didn't stop after throw()
    try:
        # PyTorch 新API（推荐）：torch.nn.attention.sdpa_kernel
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        # 兼容旧版本 torch：不再使用 torch.backends.cuda.sdp_kernel（已 deprecated，且在异常路径下可能触发
        # `RuntimeError: generator didn't stop after throw()`），改用“临时切换全局开关 + try/finally 恢复”。
        import torch

        old_flash = torch.backends.cuda.flash_sdp_enabled()
        old_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        old_math = torch.backends.cuda.math_sdp_enabled()
        try:
            # 尽量保留 flash；禁用 mem_efficient（cutlass）；保留 math 兜底
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            yield
        finally:
            torch.backends.cuda.enable_flash_sdp(bool(old_flash))
            torch.backends.cuda.enable_mem_efficient_sdp(bool(old_mem))
            torch.backends.cuda.enable_math_sdp(bool(old_math))
        return

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
        yield



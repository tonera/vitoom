from __future__ import annotations

"""
CPU tensor pretouch 工具（可选）。

动机：
- 某些环境下 safetensors 的 mmap/懒加载会让 `.to("cuda")` 触发缺页风暴；
- 通过在 CPU 侧预触达 tensor 页，可降低迁移到 CUDA 的抖动与耗时。
"""

import os

import torch


def _resolve_page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 4096


def _pretouch_tensor(tensor: torch.Tensor, *, page_size: int) -> None:
    storage = tensor.untyped_storage()
    storage_size = len(storage)
    if storage_size == 0:
        return

    checksum = 0
    for offset in range(0, storage_size, page_size):
        checksum += int(storage[offset])
    checksum += int(storage[storage_size - 1])
    _ = checksum


def pretouch_module_cpu_tensors(module: torch.nn.Module) -> None:
    """
    预触达 module 内位于 CPU 的参数与缓冲区页（in-place，无 clone）。
    """
    page_size = _resolve_page_size()
    with torch.no_grad():
        for sub in module.modules():
            for tensor in list(sub._parameters.values()) + list(sub._buffers.values()):
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.device.type != "cpu" or tensor.numel() == 0:
                    continue
                _pretouch_tensor(tensor, page_size=page_size)


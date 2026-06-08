from __future__ import annotations

"""
推理版的最小 offloading/block swap 支持。

当前 anima_runtime 的默认目标是“可移植 + 最少依赖”，因此这里只保留
`ModelOffloader` 的必要逻辑（来自 sd-scripts 实现思路的精简版）。
"""

from concurrent.futures import ThreadPoolExecutor
import gc
from typing import Callable, Optional, Union

import torch
import torch.nn as nn


def _clean_memory_on_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def weights_to_device(layer: nn.Module, device: torch.device) -> None:
    for module in layer.modules():
        if hasattr(module, "weight") and getattr(module, "weight") is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


def _swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module) -> None:
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    jobs = []
    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for name, m_cuda in layer_to_cuda.named_modules():
        if hasattr(m_cuda, "weight") and getattr(m_cuda, "weight") is not None:
            m_cpu = modules_to_cpu.get(name)
            if m_cpu is not None and getattr(m_cpu, "weight") is not None and m_cpu.weight.shape == m_cuda.weight.shape:
                jobs.append((m_cpu, m_cuda, m_cpu.weight.data, m_cuda.weight.data))
            else:
                if m_cuda.weight.data.device.type != device.type:
                    m_cuda.weight.data = m_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()
    stream = torch.Stream(device="cuda")
    with torch.cuda.stream(stream):
        for m_cpu, m_cuda, cpu_view, cuda_view in jobs:
            cuda_view.record_stream(stream)
            m_cpu.weight.data = cuda_view.data.to("cpu", non_blocking=True)
        stream.synchronize()
        for m_cpu, m_cuda, cpu_view, cuda_view in jobs:
            cuda_view.copy_(m_cuda.weight.data, non_blocking=True)
            m_cuda.weight.data = cuda_view
    stream.synchronize()
    torch.cuda.current_stream().synchronize()


def _swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module) -> None:
    # 非 CUDA 的简单 fallback（不追求极致性能）
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__
    for m_cpu, m_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(m_cpu, "weight") and getattr(m_cpu, "weight") is not None:
            m_cpu.weight.data = m_cuda.weight.data.to("cpu")
            m_cuda.weight.data = m_cuda.weight.data.to(device)
    _synchronize_device(device)


class Offloader:
    def __init__(self, num_blocks: int, blocks_to_swap: int, device: torch.device):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures: dict[int, object] = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module) -> None:
        if self.cuda_available:
            _swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            _swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu: int, block_idx_to_cuda: int) -> None:
        def move(bidx_cpu, bidx_cuda):
            self.swap_weight_devices(blocks[bidx_cpu], blocks[bidx_cuda])
            return bidx_cuda

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(move, block_idx_to_cpu, block_idx_to_cuda)

    def _wait_blocks_move(self, block_idx: int) -> None:
        fut = self.futures.pop(block_idx, None)
        if fut is None:
            return
        fut.result()


class ModelOffloader(Offloader):
    """
    推理 forward-only 的 block swap。
    """

    def __init__(self, blocks: Union[list[nn.Module], nn.ModuleList], blocks_to_swap: int, device: torch.device):
        super().__init__(len(blocks), blocks_to_swap, device)
        self.forward_only = True

    def prepare_block_devices_before_forward(self, blocks: Union[list[nn.Module], nn.ModuleList]) -> None:
        if not self.blocks_to_swap:
            return

        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)

        # 前面一段常驻 device
        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weights_to_device(b, self.device)

        # 后面 swap 的放 CPU（权重）
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # buffers on device
            weights_to_device(b, torch.device("cpu"))

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int) -> None:
        if not self.blocks_to_swap:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks(self, blocks: Union[list[nn.Module], nn.ModuleList], block_idx: int) -> None:
        if not self.blocks_to_swap:
            return
        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)


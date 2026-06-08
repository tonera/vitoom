"""
diffsynth 加载加速（不修改 site-packages 的方案）

背景：
- diffsynth 默认用 safetensors.safe_open()（mmap）读取权重，然后再把模型 .to("cuda")
- 在一些“统一内存/大文件/大量分片”的机器上，mmap + 大规模拷贝会出现很低的有效带宽（比如 ~100MB/s）

思路：
- 不做任何“自动探测”，直接 patch DiffSynth-Studio v2 的稳定入口：
  `diffsynth.core.loader.file.load_state_dict` / `load_state_dict_from_safetensors`
  （对应你本地源码：`DiffSynth-Studio/diffsynth/core/loader/file.py`）。
  - 可选：disable_mmap=True 时，先把 safetensors 文件读成 bytes，再用 safetensors.torch.load(data) 解析（避免 mmap page fault）
  - 可选：target_device="cuda" 时，解析后把 tensor 迁移到目标设备（可配合 pin_memory + non_blocking）

用法（推荐用环境变量控制，便于在不同机器上开关）：
  export DIFFSYNTH_FASTLOAD=1
  export DIFFSYNTH_DISABLE_MMAP=1
  export DIFFSYNTH_TARGET_DEVICE=cuda
  export DIFFSYNTH_PIN_MEMORY=1
"""

from __future__ import annotations

import os


def patch_diffsynth_fastload(
    *,
    disable_mmap: bool = False,
    target_device: str = "cpu",
    pin_memory: bool = False,
) -> None:
    """
    对 diffsynth 的权重加载函数做 monkey patch。

    - disable_mmap=True：safetensors 走 “read bytes -> safetensors.torch.load(data)” 路径，避免 mmap
    - target_device="cuda"：加载后的 tensor 迁移到 cuda（可能更快/也可能更吃内存峰值，视机器而定）
    - pin_memory=True：当 target_device 是 cuda 时，对 CPU tensor 先 pin，再 non_blocking 拷贝
    """
    import torch
    import diffsynth.core.loader.file as lf  # type: ignore[import-not-found]
    from safetensors.torch import load as safetensors_load_from_bytes
    from safetensors.torch import load_file as safetensors_load_file

    orig_load_state_dict = lf.load_state_dict
    orig_load_state_dict_from_safetensors = lf.load_state_dict_from_safetensors

    def _move_tensor(t, *, torch_dtype, device: str):
        if torch_dtype is not None:
            t = t.to(torch_dtype)
        if device and device != "cpu":
            # 只有从 CPU -> CUDA 时 pin_memory/non_blocking 才有意义
            if pin_memory:
                try:
                    t = t.pin_memory()
                    t = t.to(device, non_blocking=True)
                except Exception:
                    t = t.to(device)
            else:
                t = t.to(device)
        return t

    def load_state_dict_fast(file_path: str, torch_dtype=None, device: str = "cpu"):
        # 不影响调用方签名，但强制使用 target_device
        device = target_device or device

        # safetensors：可选 disable_mmap；其他格式：torch.load(map_location=...)
        if file_path.endswith(".safetensors"):
            if disable_mmap:
                # read bytes -> safetensors.torch.load(data)（CPU）-> (optional) move
                with open(file_path, "rb") as f:
                    data = f.read()
                state_dict = safetensors_load_from_bytes(data)  # tensors on cpu
                for k in list(state_dict.keys()):
                    state_dict[k] = _move_tensor(state_dict[k], torch_dtype=torch_dtype, device=device)
                return state_dict
            # 使用 safetensors 官方接口（通常比逐 key safe_open 更快）
            state_dict = safetensors_load_file(file_path, device=device)
            if torch_dtype is not None:
                for k in list(state_dict.keys()):
                    state_dict[k] = state_dict[k].to(torch_dtype)
            return state_dict

        # 非 safetensors：尽量保持与 diffsynth 旧实现一致
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
        if torch_dtype is not None:
            for k, v in list(state_dict.items()):
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.to(torch_dtype)
        return state_dict

    # Apply patches
    lf.load_state_dict = load_state_dict_fast
    if disable_mmap:
        # 对齐 diffsynth.core.loader.file 的接口：让其内部 safetensors 路径也走我们这条逻辑
        lf.load_state_dict_from_safetensors = lambda file_path, torch_dtype=None, device="cpu": load_state_dict_fast(  # type: ignore[assignment]
            file_path, torch_dtype=torch_dtype, device=device
        )
    else:
        # 未关闭 mmap 时，不改原实现
        lf.load_state_dict_from_safetensors = orig_load_state_dict_from_safetensors


def auto_patch_from_env() -> bool:
    """
    从环境变量自动启用 patch。返回是否已启用。

    - DIFFSYNTH_FASTLOAD=1：开启
    - DIFFSYNTH_DISABLE_MMAP=1：关闭 mmap
    - DIFFSYNTH_TARGET_DEVICE=cuda/cpu：加载后 tensor 目标设备
    - DIFFSYNTH_PIN_MEMORY=1：CPU->CUDA 先 pin 再 non_blocking
    """
    if os.getenv("DIFFSYNTH_FASTLOAD", "").strip() not in ("1", "true", "TRUE", "yes", "YES"):
        return False

    disable_mmap = os.getenv("DIFFSYNTH_DISABLE_MMAP", "").strip() in ("1", "true", "TRUE", "yes", "YES")
    target_device = os.getenv("DIFFSYNTH_TARGET_DEVICE", "cpu").strip() or "cpu"
    pin_memory = os.getenv("DIFFSYNTH_PIN_MEMORY", "").strip() in ("1", "true", "TRUE", "yes", "YES")

    patch_diffsynth_fastload(
        disable_mmap=disable_mmap,
        target_device=target_device,
        pin_memory=pin_memory,
    )
    return True



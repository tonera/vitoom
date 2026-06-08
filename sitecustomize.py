"""
运行期兼容补丁（自动生效）

背景：
- `realesrgan -> basicsr` 的部分版本会在导入时执行：
  `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
- 但在较新的 `torchvision` 版本中，`functional_tensor` 模块已被移除/重构，
  导致 `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`

方案：
- Python 在启动时会自动尝试导入 `sitecustomize`（只要当前工作目录/项目根目录在 sys.path）。
- 这里通过向 `sys.modules` 注入一个同名模块，提供 `rgb_to_grayscale` 的兼容实现，
  避免强制用户降级 `torch/torchvision`。

注意：
- 该补丁是“最小面”实现：只提供当前报错所需的符号。
"""

from __future__ import annotations

import sys
import types


def _install_torchvision_functional_tensor_shim() -> None:
    name = "torchvision.transforms.functional_tensor"
    existing = sys.modules.get(name)

    try:
        # 新版 torchvision 中对应实现通常在 functional 里
        from torchvision.transforms import functional as F  # type: ignore
        import torchvision.transforms as T  # type: ignore
    except Exception:
        return

    shim = existing if isinstance(existing, types.ModuleType) else types.ModuleType(name)
    if hasattr(F, "rgb_to_grayscale"):
        shim.rgb_to_grayscale = getattr(F, "rgb_to_grayscale")  # type: ignore[attr-defined]

    # 兜底：缺什么符号就转发到 torchvision.transforms.functional
    def __getattr__(attr: str):  # type: ignore[override]
        v = getattr(F, attr, None)
        if v is None:
            raise AttributeError(attr)
        return v

    shim.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[name] = shim
    # 同时挂到父包上，避免某些路径找不到
    try:
        setattr(T, "functional_tensor", shim)
    except Exception:
        pass


_install_torchvision_functional_tensor_shim()

# awq模型会触发，以后文本模型上线可使用
def _install_transformers_pytorch_gelu_tanh_shim() -> None:
    try:
        import transformers.activations as A  # type: ignore
    except Exception:
        return
    if getattr(A, "PytorchGELUTanh", None) is not None:
        return
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        return

    class PytorchGELUTanh(nn.Module):  # type: ignore[valid-type]
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.gelu(x, approximate="tanh")

    A.PytorchGELUTanh = PytorchGELUTanh  # type: ignore[attr-defined]


_install_transformers_pytorch_gelu_tanh_shim()



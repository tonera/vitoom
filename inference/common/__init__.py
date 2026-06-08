"""
推理器公共模块
提供所有推理器共用的功能
"""

# 在导入任何依赖前注入 torchvision 兼容 shim，确保 GFPGAN/basicsr 可正常 import。
# 解决报错：
# - ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
def _install_torchvision_functional_tensor_shim() -> None:
    try:
        import sys
        import types
    except Exception:
        return

    name = "torchvision.transforms.functional_tensor"
    existing = sys.modules.get(name)
    try:
        from torchvision.transforms import functional as F  # type: ignore
        import torchvision.transforms as T  # type: ignore
    except Exception:
        return

    shim = existing if isinstance(existing, types.ModuleType) else types.ModuleType(name)

    if hasattr(F, "rgb_to_grayscale"):
        shim.rgb_to_grayscale = getattr(F, "rgb_to_grayscale")  # type: ignore[attr-defined]

    def __getattr__(attr: str):  # type: ignore[override]
        v = getattr(F, attr, None)
        if v is None:
            raise AttributeError(attr)
        return v

    shim.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[name] = shim
    # 让父包也能通过属性访问到（部分 import 路径会依赖它）
    try:
        setattr(T, "functional_tensor", shim)
    except Exception:
        pass


_install_torchvision_functional_tensor_shim()

from .logger import get_logger
from .config_loader import load_startup_config, StartupConfig, load_inference_config, InferenceConfig
from .system_monitor import SystemMonitor
from .api_client import APIClient
from .ws_client import WebSocketClient
from .message_queue import MessageQueue
from .message_cache import MessageCache
from .task_processor import TaskProcessor
from .base_inferrer import BaseInferrer
from .signal_handler import SignalHandler
from .result_handler import ResultHandler
from .hf_weight_detector import infer_variant_and_use_safetensors
try:
    # torch 可能在轻量测试环境中缺失；避免 import common 直接炸掉
    from .torch_transfer_utils import (
        maybe_pretouch_pipeline_cpu_tensors,
        pretouch_pipeline_cpu_tensors,
        resolve_pin_memory,
        should_pretouch,
    )
except Exception:  # pragma: no cover
    pretouch_pipeline_cpu_tensors = None  # type: ignore[assignment]
    maybe_pretouch_pipeline_cpu_tensors = None  # type: ignore[assignment]
    should_pretouch = None  # type: ignore[assignment]
    resolve_pin_memory = None  # type: ignore[assignment]
__all__ = [
    "get_logger",
    "load_startup_config",
    "StartupConfig",
    "load_inference_config",
    "InferenceConfig",
    "SystemMonitor",
    "APIClient",
    "WebSocketClient",
    "MessageQueue",
    "MessageCache",
    "TaskProcessor",
    "BaseInferrer",
    "SignalHandler",
    "ResultHandler",
    "pretouch_pipeline_cpu_tensors",
    "maybe_pretouch_pipeline_cpu_tensors",
    "should_pretouch",
    "resolve_pin_memory",
    "infer_variant_and_use_safetensors",
]


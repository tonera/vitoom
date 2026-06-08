from dataclasses import dataclass
import torch

from common.model_registry import MODEL_REGISTRY


@dataclass
class DevicePlan:
    device: str
    torch_dtype: torch.dtype
    preferred: torch.device


class DevicePlanner:
    """
    负责设备、dtype 与低显存模式的统一规划
    """

    def _preferred_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _bf16_supported(self) -> bool:
        if not hasattr(torch, "bfloat16"):
            return False
        cuda_ok = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        mps_ok = hasattr(torch.backends, "mps") and getattr(torch.backends.mps, "is_available", lambda: False)()
        return cuda_ok or mps_ok

    def plan(self, params) -> DevicePlan:
        preferred = self._preferred_device()
        device_str = preferred.type

        # dtype 选择：默认 fp16，非 sd15/sdxl 且硬件支持则用 bf16
        torch_dtype = torch.float16
        fam = MODEL_REGISTRY.to_family(getattr(params, "family", None))
        if self._bf16_supported() and fam and fam not in ("sd15", "sdxl"):
            torch_dtype = torch.bfloat16

        device = device_str

        return DevicePlan(
            device=device,
            torch_dtype=torch_dtype,
            preferred=preferred,
        )


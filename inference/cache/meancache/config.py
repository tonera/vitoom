from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class MeanCacheConfig:
    """
    Configuration for MeanCache.

    Notes:
    - `rel_l1_thresh`: threshold for the online L_K metric (lower=quality, higher=speed)
    - `skip_budget`: maximum fraction of steps to skip (0.0-0.5)
    - `start_step`: protect early steps (structure formation)
    - `end_step`: stop applying caching at this step (exclusive); -1 means until end
    - `cache_device`: where to store cached velocities/JVP to save VRAM ('cpu' recommended)
    """

    # 项目内实测更合适的一组默认值（参见 test.py 的 B 分支调参结果）
    rel_l1_thresh: float = 0.80
    skip_budget: float = 0.50
    start_step: int = 2
    end_step: int = -1

    cache_device: Union[str, torch.device] = "cpu"

    enable_pssp: bool = True
    peak_threshold: float = 1.0
    gamma: float = 1.0
    # Maximum allowed accumulated error (used by both simple thresholding and PSSP gating).
    # Upstream defaults to 0.5, but different pipelines may need larger values.
    max_accumulated_error: float = 1.0

    # How to treat batched inputs (x.shape[0] > 1):
    # - Most Diffusers pipelines use batch dimension for *real batch*, not CFG branches.
    # - Some implementations batch CFG as batch=2 (cond/uncond). Set this True if you
    #   are sure the model forward receives [cond, uncond] in the batch dimension.
    assume_cfg_batch: bool = False

    max_cache_span: int = 3
    debug: bool = False
    preset_name: str = "Custom"

    def resolve_cache_device(self) -> Union[str, torch.device]:
        if isinstance(self.cache_device, torch.device):
            return self.cache_device
        if self.cache_device == "cuda":
            return torch.device("cuda")
        return "cpu"


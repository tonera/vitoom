"""
MeanCache: training-free inference acceleration for Flow Matching / ODE samplers.

This is a ComfyUI-independent extraction of the core algorithm from
`facok/comfyui-meancache-z`:
- JVP-based velocity correction (average velocity extrapolation)
- Online L_K stability metric for skip decisions
- Optional PSSP scheduling for compute budget allocation

Upstream project:
`https://github.com/facok/comfyui-meancache-z/tree/master`
"""

from .config import MeanCacheConfig
from .engine import MeanCacheEngine
from .diffusers_patch import apply_meancache_on_pipe

__all__ = [
    "MeanCacheConfig",
    "MeanCacheEngine",
    "apply_meancache_on_pipe",
]


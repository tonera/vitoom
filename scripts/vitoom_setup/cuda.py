"""NVIDIA CUDA availability checks."""

from __future__ import annotations

import shutil
import subprocess


def cuda_is_available() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "NVIDIA" in (result.stdout or "")

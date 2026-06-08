import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TurboImports:
    """
    A thin import facade so handlers don't hardcode TurboDiffusion module layout.

    Design:
    - Prefer pip-installed packages (recommended by user).
    - If user still vendors sources into the repo, that path can be added separately by caller.
    """

    create_model: Any
    tensor_kwargs: dict
    VIDEO_RES_SIZE_INFO: Any
    Wan2pt1VAEInterface: Any
    UMT5EncoderModel: Any


def import_turbodiffusion_core() -> TurboImports:
    """
    Import TurboDiffusion "core" pieces needed for inference.

    We intentionally do NOT import any `serve` / TUI modules here.
    """
    try:
        # Upstream code references some modules as top-level imports: `rcm`, `ops`, `SLA`, `imaginaire`.
        # Depending on how you installed TurboDiffusion, `rcm` may or may not be available as a top-level package.
        # We try:
        # 1) pip-installed top-level `rcm`
        # 2) fallback to vendored upstream sources under this repo: <repo>/inference/turbodiffusion
        try:
            from rcm.datasets.utils import VIDEO_RES_SIZE_INFO  # type: ignore
            from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface  # type: ignore
            from rcm.utils.umt5 import UMT5EncoderModel  # type: ignore
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") != "rcm":
                raise
            repo_root = Path(__file__).resolve().parents[4]
            vendored_root = (repo_root / "inference" / "turbodiffusion").resolve()
            if vendored_root.exists():
                sp = str(vendored_root)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            from rcm.datasets.utils import VIDEO_RES_SIZE_INFO  # type: ignore
            from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface  # type: ignore
            from rcm.utils.umt5 import UMT5EncoderModel  # type: ignore

        # Upstream provides a convenient model factory. The exact module path may differ across releases,
        # so we try a few known locations.
        try:
            from turbodiffusion.inference.modify_model import create_model, tensor_kwargs  # type: ignore
        except Exception:
            # Some users still vendor the upstream repo into `inference/turbodiffusion` inside this project.
            from inference.turbodiffusion.inference.modify_model import create_model, tensor_kwargs  # type: ignore

        return TurboImports(
            create_model=create_model,
            tensor_kwargs=tensor_kwargs,
            VIDEO_RES_SIZE_INFO=VIDEO_RES_SIZE_INFO,
            Wan2pt1VAEInterface=Wan2pt1VAEInterface,
            UMT5EncoderModel=UMT5EncoderModel,
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            "TurboDiffusion core import failed. Expected a pip install like:\n"
            "- `pip install turbodiffusion --no-build-isolation`\n"
            "Import error detail:\n"
            f"- Exception: {type(e).__name__}: {e}\n"
            "Full traceback:\n"
            f"{tb}"
        ) from e


"""释放 H3 对象。只丢掉引用，不对 QuantizedTensor 调用 .to('cpu')。"""

from __future__ import annotations

import gc
from typing import Any, Optional

from video.handlers.minimax_h3.models import H3Models


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_h3_models(models: H3Models | None) -> None:
    if models is None:
        return
    models.processor = None
    free_cuda()


async def release_h3_models_async(
    models: Any,
    *,
    log: Any = None,
    run_blocking: Optional[Any] = None,
) -> None:
    del log
    if run_blocking is not None:
        await run_blocking(release_h3_models, models)
        return
    release_h3_models(models)

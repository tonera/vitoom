from __future__ import annotations

from functools import lru_cache

from .builders import build_catalog
from .validators import validate_catalog
from .types import ModelCatalog


@lru_cache(maxsize=1)
def get_catalog() -> ModelCatalog:
    # 只在第一次调用时构建并校验（启动期/首次用到时）
    from common.model_families import ALL_FAMILY_SPECS

    cat = build_catalog(list(ALL_FAMILY_SPECS))
    validate_catalog(cat)
    return cat


from __future__ import annotations

import importlib
import pkgutil

from common.model_catalog.types import ModelFamilySpec


def _discover_family_specs() -> list[ModelFamilySpec]:
    """
    自动发现本包内所有 family spec 模块（`SPEC: ModelFamilySpec`）。
    新增家族时只需要新增一个文件 `common/model_families/<family>.py`，无需再改本文件。
    """
    specs: list[ModelFamilySpec] = []
    module_names = sorted([m.name for m in pkgutil.iter_modules(__path__)])  # type: ignore[name-defined]
    for name in module_names:
        if name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{name}")
        spec = getattr(mod, "SPEC", None)
        if isinstance(spec, ModelFamilySpec):
            specs.append(spec)
    return specs


# 统一的家族 spec 列表（Catalog 构建入口）
ALL_FAMILY_SPECS: list[ModelFamilySpec] = _discover_family_specs()

__all__ = ["ALL_FAMILY_SPECS"]


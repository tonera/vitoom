"""
模型家族注册表（ModelRegistry）——兼容层（对外 API 保持稳定）。

现行改造目标：
- 单一事实源（SSOT）迁移到 `common.model_catalog`（纯代码 Catalog + per-family spec）
- 本模块仅作为薄封装，避免上层业务到处改 import
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from common.model_catalog import get_catalog


@dataclass(frozen=True)
class ModelIndexRule:
    family: str
    pipeline_text2img: str
    pipeline_img2img: Optional[str] = None


class ModelRegistry:
    def __init__(self):
        self._catalog = get_catalog()

    def to_family(self, v: object) -> str:
        return self._catalog.to_family(v)

    def model_index_rule(self, class_name: str) -> Optional[ModelIndexRule]:
        r = self._catalog.model_index_rule(str(class_name or "").strip())
        if r is None:
            return None
        return ModelIndexRule(
            family=r.family,
            pipeline_text2img=r.pipeline_text2img.class_name,
            pipeline_img2img=r.pipeline_img2img.class_name if r.pipeline_img2img else None,
        )

    def choose_pipeline_name(self, class_name: str, *, is_img2img: bool) -> Optional[tuple[str, str]]:
        """
        Returns: (family, pipeline_name)
        """
        rule = self.model_index_rule(class_name)
        if rule is None:
            return None
        if is_img2img and rule.pipeline_img2img:
            return rule.family, rule.pipeline_img2img
        return rule.family, rule.pipeline_text2img

    def is_flowmatch_family(self, family_or_alias: object) -> bool:
        fam = self.to_family(family_or_alias)
        return bool(fam and fam in self._catalog.flowmatch_families)


MODEL_REGISTRY = ModelRegistry()


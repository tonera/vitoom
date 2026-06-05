from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PipelineRef:
    """
    延迟解析的 Pipeline 类引用（纯代码配置，不引入 YAML）：
    - module: python module path（如 "diffusers"）
    - attr: 类名（如 "Flux2Pipeline"）
    """

    module: str
    attr: str

    def resolve(self):
        import importlib

        mod = importlib.import_module(self.module)
        try:
            return getattr(mod, self.attr)
        except AttributeError as e:
            raise ImportError(f"Pipeline class not found: {self.module}:{self.attr}") from e

    @property
    def class_name(self) -> str:
        return self.attr


@dataclass(frozen=True)
class ModelIndexRule:
    """
    model_index.json 的 _class_name -> pipeline 选择规则。
    - family：canonical family
    - pipeline_text2img：text2img pipeline
    - pipeline_img2img：img2img pipeline（可选）
    """

    family: str
    pipeline_text2img: PipelineRef
    pipeline_img2img: Optional[PipelineRef] = None


@dataclass(frozen=True)
class ModelFamilySpec:
    """
    模型家族单一事实源（SSOT）。
    目标：新增家族时只新增/修改一份 spec（纯代码），其余消费者自动装配。
    """

    family: str
    aliases: set[str] = field(default_factory=set)

    # file-detect 场景兜底默认 pipeline（如 sd15/sdxl/flux/zimage）
    default_text2img: Optional[PipelineRef] = None
    default_img2img: Optional[PipelineRef] = None

    # model_index.json 场景：_class_name -> rule
    model_index_rules: dict[str, ModelIndexRule] = field(default_factory=dict)

    # scheduler policy / 特性
    is_flowmatch: bool = False


@dataclass(frozen=True)
class ModelCatalog:
    families: dict[str, ModelFamilySpec]
    alias_to_family: dict[str, str]
    model_index_rules: dict[str, ModelIndexRule]
    flowmatch_families: set[str]

    def to_family(self, v: object) -> str:
        s = str(v).strip().lower() if v is not None else ""
        if not s:
            return ""
        if s in self.families:
            return s
        return self.alias_to_family.get(s, s)

    def model_index_rule(self, class_name: str) -> Optional[ModelIndexRule]:
        if not class_name:
            return None
        return self.model_index_rules.get(str(class_name).strip())

    def choose_pipeline_ref(self, class_name: str, *, is_img2img: bool) -> Optional[tuple[str, PipelineRef]]:
        """
        Returns: (family, pipeline_ref)
        """
        rule = self.model_index_rule(class_name)
        if rule is None:
            return None
        if is_img2img and rule.pipeline_img2img is not None:
            return rule.family, rule.pipeline_img2img
        return rule.family, rule.pipeline_text2img

    def default_pipeline_ref(self, family: str, *, is_img2img: bool) -> Optional[PipelineRef]:
        fam = self.to_family(family)
        spec = self.families.get(fam)
        if not spec:
            return None
        if is_img2img and spec.default_img2img is not None:
            return spec.default_img2img
        return spec.default_text2img


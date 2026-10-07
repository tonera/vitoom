from __future__ import annotations

from common.Constant import MODEL_KREA2
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="krea2",
    aliases={m.lower() for m in MODEL_KREA2},
    default_text2img=PipelineRef("diffusers", "Krea2Pipeline"),
    model_index_rules={
        "Krea2Pipeline": ModelIndexRule(
            family="krea2",
            pipeline_text2img=PipelineRef("diffusers", "Krea2Pipeline"),
        ),
    },
    is_flowmatch=True,
)

from __future__ import annotations

from common.Constant import MODEL_CHROMA
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="chroma",
    aliases={m.lower() for m in MODEL_CHROMA},
    default_text2img=PipelineRef("diffusers", "ChromaPipeline"),
    model_index_rules={
        "ChromaPipeline": ModelIndexRule(
            family="chroma",
            pipeline_text2img=PipelineRef("diffusers", "ChromaPipeline"),
        ),
    },
)


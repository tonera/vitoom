from __future__ import annotations

from common.Constant import MODEL_AURA
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="aura",
    aliases={m.lower() for m in MODEL_AURA},
    model_index_rules={
        "AuraFlowPipeline": ModelIndexRule(
            family="aura",
            pipeline_text2img=PipelineRef("diffusers", "AuraFlowPipeline"),
        ),
    },
)


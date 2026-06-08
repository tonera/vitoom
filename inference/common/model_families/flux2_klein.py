from __future__ import annotations

from common.Constant import MODEL_FLUX2_KLEIN
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="flux2_klein",
    aliases={m.lower() for m in MODEL_FLUX2_KLEIN},
    default_text2img=PipelineRef("diffusers", "Flux2KleinPipeline"),
    model_index_rules={
        "Flux2KleinPipeline": ModelIndexRule(
            family="flux2_klein",
            pipeline_text2img=PipelineRef("diffusers", "Flux2KleinPipeline"),
        ),
        "Flux2KleinKVPipeline": ModelIndexRule(
            family="flux2_klein",
            pipeline_text2img=PipelineRef("diffusers", "Flux2KleinKVPipeline"),
        ),
    },
    is_flowmatch=True,
)


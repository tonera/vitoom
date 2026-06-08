from __future__ import annotations

from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="diffusion",
    aliases={"diffusion"},
    default_text2img=PipelineRef("diffusers", "DiffusionPipeline"),
    model_index_rules={
        "DiffusionPipeline": ModelIndexRule(
            family="diffusion",
            pipeline_text2img=PipelineRef("diffusers", "DiffusionPipeline"),
        ),
    },
)


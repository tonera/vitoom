from __future__ import annotations

from common.Constant import MODEL_15
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="sd15",
    aliases={m.lower() for m in MODEL_15} | {"sd21"},
    default_text2img=PipelineRef("diffusers", "StableDiffusionPipeline"),
    default_img2img=PipelineRef("diffusers", "StableDiffusionImg2ImgPipeline"),
    model_index_rules={
        "StableDiffusionPipeline": ModelIndexRule(
            family="sd15",
            pipeline_text2img=PipelineRef("diffusers", "StableDiffusionPipeline"),
            pipeline_img2img=PipelineRef("diffusers", "StableDiffusionImg2ImgPipeline"),
        ),
    },
)


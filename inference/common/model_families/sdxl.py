from __future__ import annotations

from common.Constant import MODEL_SDXL
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="sdxl",
    aliases={m.lower() for m in MODEL_SDXL},
    default_text2img=PipelineRef("diffusers", "StableDiffusionXLPipeline"),
    default_img2img=PipelineRef("diffusers", "StableDiffusionXLImg2ImgPipeline"),
    model_index_rules={
        "StableDiffusionXLPipeline": ModelIndexRule(
            family="sdxl",
            pipeline_text2img=PipelineRef("diffusers", "StableDiffusionXLPipeline"),
            pipeline_img2img=PipelineRef("diffusers", "StableDiffusionXLImg2ImgPipeline"),
        ),
        # controlnet 变体（目前仅 text2img 名称）
        "StableDiffusionXLControlNetPipeline": ModelIndexRule(
            family="sdxl",
            pipeline_text2img=PipelineRef("diffusers", "StableDiffusionXLControlNetPipeline"),
        ),
    },
)


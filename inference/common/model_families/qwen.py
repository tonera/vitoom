from __future__ import annotations

from common.Constant import MODEL_QWEN
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="qwen",
    aliases={m.lower() for m in MODEL_QWEN},
    default_text2img=PipelineRef("diffusers", "QwenImagePipeline"),
    default_img2img=PipelineRef("diffusers", "QwenImageImg2ImgPipeline"),
    model_index_rules={
        "QwenImagePipeline": ModelIndexRule(
            family="qwen",
            pipeline_text2img=PipelineRef("diffusers", "QwenImagePipeline"),
            pipeline_img2img=PipelineRef("diffusers", "QwenImageImg2ImgPipeline"),
        ),
    },
)


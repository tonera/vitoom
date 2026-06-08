from __future__ import annotations

from common.Constant import MODEL_Z_IMAGE
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="zimage",
    aliases={m.lower() for m in MODEL_Z_IMAGE},
    default_text2img=PipelineRef("diffusers", "ZImagePipeline"),
    default_img2img=PipelineRef("diffusers", "ZImageImg2ImgPipeline"),
    model_index_rules={
        "ZImagePipeline": ModelIndexRule(
            family="zimage",
            pipeline_text2img=PipelineRef("diffusers", "ZImagePipeline"),
            pipeline_img2img=PipelineRef("diffusers", "ZImageImg2ImgPipeline"),
        ),
    },
    is_flowmatch=True,
)


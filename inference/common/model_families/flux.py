from __future__ import annotations

from common.Constant import MODEL_FLUX
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="flux",
    aliases={m.lower() for m in MODEL_FLUX},
    default_text2img=PipelineRef("diffusers", "FluxPipeline"),
    default_img2img=PipelineRef("diffusers", "FluxImg2ImgPipeline"),
    model_index_rules={
        "FluxPipeline": ModelIndexRule(
            family="flux",
            pipeline_text2img=PipelineRef("diffusers", "FluxPipeline"),
            pipeline_img2img=PipelineRef("diffusers", "FluxImg2ImgPipeline"),
        ),
        "FluxControlPipeline": ModelIndexRule(
            family="flux",
            pipeline_text2img=PipelineRef("diffusers", "FluxControlPipeline"),
        ),
        "FluxFillPipeline": ModelIndexRule(
            family="flux",
            pipeline_text2img=PipelineRef("diffusers", "FluxFillPipeline"),
        ),
    },
    is_flowmatch=True,
)


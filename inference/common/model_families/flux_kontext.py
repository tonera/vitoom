from __future__ import annotations

from common.Constant import MODEL_FLUX_KONTEXT
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="flux_kontext",
    aliases={m.lower() for m in MODEL_FLUX_KONTEXT},
    default_text2img=PipelineRef("diffusers", "FluxKontextPipeline"),
    model_index_rules={
        "FluxKontextPipeline": ModelIndexRule(
            family="flux_kontext",
            pipeline_text2img=PipelineRef("diffusers", "FluxKontextPipeline"),
        ),
    },
    is_flowmatch=True,
)


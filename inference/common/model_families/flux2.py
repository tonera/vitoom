from __future__ import annotations

from common.Constant import MODEL_FLUX2
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="flux2",
    aliases={m.lower() for m in MODEL_FLUX2},
    default_text2img=PipelineRef("diffusers", "Flux2Pipeline"),
    model_index_rules={
        "Flux2Pipeline": ModelIndexRule(
            family="flux2",
            pipeline_text2img=PipelineRef("diffusers", "Flux2Pipeline"),
        ),
    },
    is_flowmatch=True,
)


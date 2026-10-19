from __future__ import annotations

from common.Constant import MODEL_FLUX3_KLEIN
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


_PIPELINE = PipelineRef("image.runtime.flux3_klein_pipeline", "Flux3KleinPipeline")

SPEC = ModelFamilySpec(
    family="flux3_klein",
    aliases={m.lower() for m in MODEL_FLUX3_KLEIN},
    default_text2img=_PIPELINE,
    model_index_rules={
        "Flux3ModularPipeline": ModelIndexRule(
            family="flux3_klein",
            pipeline_text2img=_PIPELINE,
        ),
        "Flux3AutoBlocks": ModelIndexRule(
            family="flux3_klein",
            pipeline_text2img=_PIPELINE,
        ),
    },
    is_flowmatch=True,
)

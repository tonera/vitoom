from __future__ import annotations

from common.Constant import MODEL_HUNYUAN
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="hunyuan",
    aliases={m.lower() for m in MODEL_HUNYUAN},
    model_index_rules={
        "HunyuanVideoPipeline": ModelIndexRule(
            family="hunyuan",
            pipeline_text2img=PipelineRef("diffusers", "HunyuanVideoPipeline"),
        ),
    },
)


from __future__ import annotations

from common.Constant import MODEL_ANIMA
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="anima",
    aliases={m.lower() for m in MODEL_ANIMA},
    # 单文件走 AnimaDiffusersPipeline。旧 bundle 的 model_index 类名仍是 AnimaPipeline。
    default_text2img=PipelineRef("image.runtime.anima_diffusers_pipeline", "AnimaDiffusersPipeline"),
    model_index_rules={
        "AnimaPipeline": ModelIndexRule(
            family="anima",
            pipeline_text2img=PipelineRef("third_party.anima_runtime.pipeline", "AnimaPipeline"),
        ),
        "AnimaModularPipeline": ModelIndexRule(
            family="anima",
            pipeline_text2img=PipelineRef("image.runtime.anima_diffusers_pipeline", "AnimaDiffusersPipeline"),
        ),
        "AnimaAutoBlocks": ModelIndexRule(
            family="anima",
            pipeline_text2img=PipelineRef("image.runtime.anima_diffusers_pipeline", "AnimaDiffusersPipeline"),
        ),
    },
)


from __future__ import annotations

from common.Constant import MODEL_ANIMA
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="anima",
    aliases={m.lower() for m in MODEL_ANIMA},
    # Anima 是非 diffusers pipeline：由 third_party/anima_runtime/pipeline.py 提供适配器
    default_text2img=PipelineRef("third_party.anima_runtime.pipeline", "AnimaPipeline"),
    model_index_rules={
        # 允许未来给 Anima bundle 放一个轻量 model_index.json（不要求 diffusers 组件齐全）
        "AnimaPipeline": ModelIndexRule(
            family="anima",
            pipeline_text2img=PipelineRef("third_party.anima_runtime.pipeline", "AnimaPipeline"),
        ),
    },
)


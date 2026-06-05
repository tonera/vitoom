from __future__ import annotations

from common.Constant import MODEL_QWEN_EDIT
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="qwen.edit",
    aliases={m.lower() for m in MODEL_QWEN_EDIT},
    default_text2img=PipelineRef("diffusers", "QwenImageEditPipeline"),
    model_index_rules={
        "QwenImageEditPipeline": ModelIndexRule(
            family="qwen.edit",
            pipeline_text2img=PipelineRef("diffusers", "QwenImageEditPipeline"),
        ),
        "QwenImageEditPlusPipeline": ModelIndexRule(
            family="qwen.edit",
            pipeline_text2img=PipelineRef("diffusers", "QwenImageEditPlusPipeline"),
        ),
    },
)


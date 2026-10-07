from __future__ import annotations

from common.Constant import MODEL_QWEN_IMAGE_21
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


# 文生图和编辑共用 QwenImage21Pipeline：编辑时把参考图传给 image。
_PIPELINE = PipelineRef("diffusers", "QwenImage21Pipeline")

SPEC = ModelFamilySpec(
    family="qwen.image21",
    aliases={m.lower() for m in MODEL_QWEN_IMAGE_21},
    default_text2img=_PIPELINE,
    default_img2img=_PIPELINE,
    model_index_rules={
        "QwenImage21Pipeline": ModelIndexRule(
            family="qwen.image21",
            pipeline_text2img=_PIPELINE,
            pipeline_img2img=_PIPELINE,
        ),
    },
    is_flowmatch=True,
)

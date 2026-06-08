from __future__ import annotations

from common.Constant import MODEL_WAN
from common.model_catalog.types import ModelFamilySpec, ModelIndexRule, PipelineRef


SPEC = ModelFamilySpec(
    family="wan",
    aliases={m.lower() for m in MODEL_WAN},
    model_index_rules={
        # 可能在部分环境不存在（未安装/版本差异）；仅在被选择时才 resolve
        "WanPipeline": ModelIndexRule(
            family="wan",
            pipeline_text2img=PipelineRef("diffusers", "WanPipeline"),
        ),
    },
)


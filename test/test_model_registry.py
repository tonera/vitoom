from pathlib import Path
import sys

# 添加项目路径，便于导入 inference 下的模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from common.model_registry import MODEL_REGISTRY


def test_registry_to_family_aliases():
    assert MODEL_REGISTRY.to_family("Pony") == "sdxl"
    assert MODEL_REGISTRY.to_family("Flux.2 D") == "flux2"
    assert MODEL_REGISTRY.to_family("flux2_klein") == "flux2_klein"


def test_registry_choose_pipeline_name():
    fam, pipe = MODEL_REGISTRY.choose_pipeline_name("Flux2KleinPipeline", is_img2img=False)
    assert fam == "flux2_klein"
    assert pipe == "Flux2KleinPipeline"

    fam, pipe = MODEL_REGISTRY.choose_pipeline_name("Flux2KleinKVPipeline", is_img2img=False)
    assert fam == "flux2_klein"
    assert pipe == "Flux2KleinKVPipeline"

    fam, pipe = MODEL_REGISTRY.choose_pipeline_name("StableDiffusionXLPipeline", is_img2img=True)
    assert fam == "sdxl"
    assert pipe == "StableDiffusionXLImg2ImgPipeline"


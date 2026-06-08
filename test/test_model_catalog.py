from pathlib import Path
import sys

# 添加项目路径，便于导入 inference 下的模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from common.model_catalog import get_catalog
from image.inference_param_specs import pick_spec, SPECS


def test_catalog_builds_and_normalizes():
    cat = get_catalog()
    assert cat.to_family("Pony") in {"sdxl", "pony"}  # 依赖 Constant 别名集合；最少保证不抛异常
    assert cat.to_family("Flux.2 D") == "flux2"
    assert cat.to_family("flux2_klein") == "flux2_klein"


def test_catalog_choose_pipeline_ref():
    cat = get_catalog()
    fam, pref = cat.choose_pipeline_ref("StableDiffusionPipeline", is_img2img=True)
    assert fam == "sd15"
    assert pref.class_name == "StableDiffusionImg2ImgPipeline"

    fam, pref = cat.choose_pipeline_ref("Flux2KleinKVPipeline", is_img2img=False)
    assert fam == "flux2_klein"
    assert pref.class_name == "Flux2KleinKVPipeline"


def test_inference_param_specs_auto_discovery():
    # 确保不再依赖手写 SPECS 列表
    assert SPECS and len(SPECS) >= 3
    assert pick_spec("flux2", "Flux2Pipeline") is not None
    assert pick_spec("flux2_klein", "Flux2KleinKVPipeline") is not None


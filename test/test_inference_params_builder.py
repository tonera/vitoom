"""
测试 build_inference_params 函数
打印返回值的所有 key，便于检查是否符合预期
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "inference"))

from schemas import InferenceRequestParams
from image.inference_params_builder import build_inference_params


class MockPipeline:
    """模拟 Pipeline 类"""
    def __init__(self, class_name: str):
        self.class_name = class_name
    
    @property
    def __name__(self):
        return self.class_name
    
    @property
    def __class__(self):
        return type(self.class_name, (), {})


def print_params_keys(pipeline_name: str, params: dict, title: str = ""):
    """打印参数字典的所有 key"""
    print(f"\n{'='*80}")
    if title:
        print(f"测试场景: {title}")
    print(f"Pipeline: {pipeline_name}")
    print(f"{'='*80}")
    print(f"返回的 keys ({len(params)} 个):")
    print("-" * 80)
    
    # 按字母顺序排序
    sorted_keys = sorted(params.keys())
    for i, key in enumerate(sorted_keys, 1):
        value_type = type(params[key]).__name__
        value_preview = str(params[key])[:50] if not isinstance(params[key], (dict, list)) else f"<{value_type}>"
        print(f"{i:2d}. {key:30s} ({value_type:15s}) = {value_preview}")
    
    print("-" * 80)
    print(f"总计: {len(params)} 个参数\n")


def test_basic_scenarios():
    """测试基本场景"""
    
    # 基础 request_params
    base_params = InferenceRequestParams(
        action="MK",
        job_type="image",
        id="test-001",
        user_id="user-001",
        task_id="task-001",
        prompt="a beautiful landscape",
        negative_prompt="blurry, low quality",
        width=1024,
        height=1024,
        guidance_scale=7.5,
        seed=42,
        num_inference_steps=30,
        strength=0.5,
        generate_num=1,
        family="sdxl"
    )
    
    pipeline_params = {}
    
    # 测试场景1: StableDiffusionXLPipeline
    print("\n" + "="*80)
    print("场景1: StableDiffusionXLPipeline (类)")
    print("="*80)
    pipeline_class = type("StableDiffusionXLPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("StableDiffusionXLPipeline", result, "文生图 - SDXL")
    
    # 测试场景2: StableDiffusionXLImg2ImgPipeline
    print("\n" + "="*80)
    print("场景2: StableDiffusionXLImg2ImgPipeline (类)")
    print("="*80)
    base_params.url = "https://example.com/image.jpg"
    pipeline_class = type("StableDiffusionXLImg2ImgPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("StableDiffusionXLImg2ImgPipeline", result, "图生图 - SDXL")
    
    # 测试场景3: AutoPipelineForText2Image (SDXL)
    print("\n" + "="*80)
    print("场景3: AutoPipelineForText2Image (SDXL)")
    print("="*80)
    base_params.url = None
    base_params.family = "sdxl"
    pipeline_class = type("AutoPipelineForText2Image", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("AutoPipelineForText2Image", result, "AutoPipeline - SDXL 文生图")
    
    # 测试场景4: AutoPipelineForText2Image (SD15)
    print("\n" + "="*80)
    print("场景4: AutoPipelineForText2Image (SD15)")
    print("="*80)
    base_params.family = "sd15"
    pipeline_class = type("AutoPipelineForText2Image", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("AutoPipelineForText2Image", result, "AutoPipeline - SD15 文生图")
    
    # 测试场景5: AutoPipelineForImage2Image (SDXL)
    print("\n" + "="*80)
    print("场景5: AutoPipelineForImage2Image (SDXL)")
    print("="*80)
    base_params.url = "https://example.com/image.jpg"
    base_params.family = "sdxl"
    pipeline_class = type("AutoPipelineForImage2Image", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("AutoPipelineForImage2Image", result, "AutoPipeline - SDXL 图生图")
    
    # 测试场景6: AutoPipelineForImage2Image (SD15)
    print("\n" + "="*80)
    print("场景6: AutoPipelineForImage2Image (SD15)")
    print("="*80)
    base_params.family = "sd15"
    pipeline_class = type("AutoPipelineForImage2Image", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("AutoPipelineForImage2Image", result, "AutoPipeline - SD15 图生图")
    
    # 测试场景7: FluxPipeline
    print("\n" + "="*80)
    print("场景7: FluxPipeline")
    print("="*80)
    base_params.url = None
    base_params.family = "flux"
    pipeline_class = type("FluxPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("FluxPipeline", result, "Flux 文生图")
    
    # 测试场景8: FluxImg2ImgPipeline
    print("\n" + "="*80)
    print("场景8: FluxImg2ImgPipeline")
    print("="*80)
    base_params.url = "https://example.com/image.jpg"
    pipeline_class = type("FluxImg2ImgPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("FluxImg2ImgPipeline", result, "Flux 图生图")
    
    # 测试场景9: FluxInpaintPipeline
    print("\n" + "="*80)
    print("场景9: FluxInpaintPipeline")
    print("="*80)
    base_params.image_file2 = "https://example.com/mask.jpg"
    pipeline_class = type("FluxInpaintPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("FluxInpaintPipeline", result, "Flux Inpaint")
    
    # 测试场景10: FluxControlPipeline
    print("\n" + "="*80)
    print("场景10: FluxControlPipeline")
    print("="*80)
    base_params.url = None
    base_params.image_file2 = "https://example.com/control.jpg"
    pipeline_class = type("FluxControlPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("FluxControlPipeline", result, "Flux ControlNet")
    
    # 测试场景11: QwenImagePipeline
    print("\n" + "="*80)
    print("场景11: QwenImagePipeline")
    print("="*80)
    base_params.family = "qwen"
    base_params.image_file2 = None
    pipeline_class = type("QwenImagePipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("QwenImagePipeline", result, "Qwen 文生图")
    
    # 测试场景12: StableDiffusionPipeline (SD15)
    print("\n" + "="*80)
    print("场景12: StableDiffusionPipeline")
    print("="*80)
    base_params.family = "sd15"
    pipeline_class = type("StableDiffusionPipeline", (), {})
    result = build_inference_params(pipeline_class, base_params, pipeline_params)
    print_params_keys("StableDiffusionPipeline", result, "SD15 文生图")


def test_keep_size_scenarios():
    """测试 keep_size 规则"""
    
    print("\n\n" + "="*80)
    print("测试 keep_size 规则")
    print("="*80)
    
    # 场景1: keep_size='init_images'
    print("\n" + "="*80)
    print("场景: keep_size='init_images'")
    print("="*80)
    params = InferenceRequestParams(
        action="MK",
        job_type="image",
        id="test-002",
        user_id="user-001",
        task_id="task-002",
        prompt="test prompt",
        width=512,
        height=512,
        keep_size="init_images",
        url="https://example.com/image.jpg"
    )
    pipeline_class = type("StableDiffusionXLPipeline", (), {})
    result = build_inference_params(pipeline_class, params, {})
    print_params_keys("StableDiffusionXLPipeline", result, "keep_size=init_images")
    print(f"设置的 width: {result.get('width')}, height: {result.get('height')}")
    
    # 场景2: keep_size='image_file2'
    print("\n" + "="*80)
    print("场景: keep_size='image_file2'")
    print("="*80)
    params = InferenceRequestParams(
        action="MK",
        job_type="image",
        id="test-003",
        user_id="user-001",
        task_id="task-003",
        prompt="test prompt",
        width=512,
        height=512,
        keep_size="image_file2",
        image_file2="https://example.com/mask.jpg"
    )
    pipeline_class = type("StableDiffusionXLPipeline", (), {})
    result = build_inference_params(pipeline_class, params, {})
    print_params_keys("StableDiffusionXLPipeline", result, "keep_size=image_file2")
    print(f"设置的 width: {result.get('width')}, height: {result.get('height')}")


def test_special_cases():
    """测试特殊情况"""
    
    print("\n\n" + "="*80)
    print("测试特殊情况")
    print("="*80)
    
    # 场景1: seed=0 (不生成generator)
    print("\n" + "="*80)
    print("场景: seed=0")
    print("="*80)
    params = InferenceRequestParams(
        action="MK",
        job_type="image",
        id="test-004",
        user_id="user-001",
        task_id="task-004",
        prompt="test prompt",
        seed=0
    )
    pipeline_class = type("StableDiffusionXLPipeline", (), {})
    result = build_inference_params(pipeline_class, params, {})
    print_params_keys("StableDiffusionXLPipeline", result, "seed=0")
    print(f"generator 是否存在: {'generator' in result}")
    
    # 场景2: seed>0 (生成generator)
    print("\n" + "="*80)
    print("场景: seed>0")
    print("="*80)
    params.seed = 42
    result = build_inference_params(pipeline_class, params, {})
    print_params_keys("StableDiffusionXLPipeline", result, "seed=42")
    print(f"generator 是否存在: {'generator' in result}")
    if 'generator' in result:
        print(f"generator 类型: {type(result['generator'])}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("build_inference_params 函数测试")
    print("="*80)
    
    try:
        # 测试基本场景
        test_basic_scenarios()
        
        # 测试 keep_size 规则
        test_keep_size_scenarios()
        
        # 测试特殊情况
        test_special_cases()
        
        print("\n" + "="*80)
        print("测试完成!")
        print("="*80)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


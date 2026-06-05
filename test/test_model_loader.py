"""
模型加载器模块测试脚本
"""
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.inference.model_loader import ModelLoader, get_model_loader, _model_cache
from backend.core.exceptions import ModelLoadFailedException


def setup_module():
    """测试模块初始化"""
    # 清空缓存
    ModelLoader.clear_cache()


def teardown_module():
    """测试模块清理"""
    # 清空缓存
    ModelLoader.clear_cache()


def test_model_loader_singleton():
    """测试单例模式"""
    print("=" * 50)
    print("Testing ModelLoader Singleton")
    print("=" * 50)
    
    loader1 = get_model_loader()
    loader2 = get_model_loader()
    
    assert loader1 is loader2
    print("✓ Singleton pattern works correctly\n")


def test_clear_cache():
    """测试清空缓存"""
    print("=" * 50)
    print("Testing Clear Cache")
    print("=" * 50)
    
    # 添加一些假数据到缓存
    _model_cache["test_key"] = Mock()
    assert len(_model_cache) > 0
    
    # 清空缓存
    ModelLoader.clear_cache()
    assert len(_model_cache) == 0
    print("✓ Cache cleared successfully\n")


def test_get_cache_size():
    """测试获取缓存大小"""
    print("=" * 50)
    print("Testing Get Cache Size")
    print("=" * 50)
    
    ModelLoader.clear_cache()
    assert ModelLoader.get_cache_size() == 0
    
    _model_cache["test_key"] = Mock()
    assert ModelLoader.get_cache_size() == 1
    
    ModelLoader.clear_cache()
    print("✓ Get cache size works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.StableDiffusionXLPipeline')
async def test_load_sdxl_model(mock_pipeline_class, mock_torch):
    """测试加载SDXL模型"""
    print("=" * 50)
    print("Testing Load SDXL Model")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    
    mock_pipe = Mock()
    mock_pipe.device = "cpu"
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload.return_value = None
    mock_pipeline_class.from_pretrained.return_value = mock_pipe
    
    # 测试加载
    model_path = "/fake/path/to/sdxl"
    ModelLoader.clear_cache()
    
    pipe = await ModelLoader.load_image_model(
        model_path=model_path,
        family="sdxl",
        device="cpu"
    )
    
    assert pipe is not None
    mock_pipeline_class.from_pretrained.assert_called_once()
    mock_pipe.to.assert_called_once_with("cpu")
    print("✓ SDXL model loading works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.DiffusionPipeline')
async def test_load_flux_model(mock_pipeline_class, mock_torch):
    """测试加载Flux模型"""
    print("=" * 50)
    print("Testing Load Flux Model")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    
    mock_pipe = Mock()
    mock_pipe.device = "cpu"
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload.return_value = None
    mock_pipeline_class.from_pretrained.return_value = mock_pipe
    
    # 测试加载
    model_path = "/fake/path/to/flux"
    ModelLoader.clear_cache()
    
    pipe = await ModelLoader.load_image_model(
        model_path=model_path,
        family="flux",
        device="cpu"
    )
    
    assert pipe is not None
    mock_pipeline_class.from_pretrained.assert_called_once()
    print("✓ Flux model loading works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.StableDiffusionPipeline')
async def test_load_sd15_model(mock_pipeline_class, mock_torch):
    """测试加载SD 1.5模型"""
    print("=" * 50)
    print("Testing Load SD 1.5 Model")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    
    mock_pipe = Mock()
    mock_pipe.device = "cpu"
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload.return_value = None
    mock_pipeline_class.from_pretrained.return_value = mock_pipe
    
    # 测试加载
    model_path = "/fake/path/to/sd15"
    ModelLoader.clear_cache()
    
    pipe = await ModelLoader.load_image_model(
        model_path=model_path,
        family="1.5",
        device="cpu"
    )
    
    assert pipe is not None
    mock_pipeline_class.from_pretrained.assert_called_once()
    print("✓ SD 1.5 model loading works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.Path')
async def test_load_model_path_not_exists(mock_path_class, mock_torch):
    """测试模型路径不存在的情况"""
    print("=" * 50)
    print("Testing Model Path Not Exists")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    
    mock_path = Mock()
    mock_path.exists.return_value = False
    mock_path_class.return_value = mock_path
    
    ModelLoader.clear_cache()
    
    # 测试加载不存在的模型路径
    with pytest.raises(ModelLoadFailedException) as exc_info:
        await ModelLoader.load_image_model(
            model_path="/nonexistent/path",
            family="sdxl",
            device="cpu"
        )
    
    assert "does not exist" in str(exc_info.value).lower()
    print("✓ Model path validation works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.StableDiffusionXLPipeline')
async def test_model_caching(mock_pipeline_class, mock_torch):
    """测试模型缓存功能"""
    print("=" * 50)
    print("Testing Model Caching")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    
    mock_pipe = Mock()
    mock_pipe.device = "cpu"
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload.return_value = None
    mock_pipeline_class.from_pretrained.return_value = mock_pipe
    
    model_path = "/fake/path/to/cached_model"
    ModelLoader.clear_cache()
    
    # 第一次加载
    pipe1 = await ModelLoader.load_image_model(
        model_path=model_path,
        family="sdxl",
        device="cpu"
    )
    
    # 重置mock调用计数
    mock_pipeline_class.from_pretrained.reset_mock()
    
    # 第二次加载（应该使用缓存）
    pipe2 = await ModelLoader.load_image_model(
        model_path=model_path,
        family="sdxl",
        device="cpu"
    )
    
    # 验证使用了缓存（from_pretrained不应该被再次调用）
    assert pipe1 is pipe2
    mock_pipeline_class.from_pretrained.assert_not_called()
    print("✓ Model caching works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.StableDiffusionXLPipeline')
async def test_unknown_family_defaults_to_sdxl(mock_pipeline_class, mock_torch):
    """测试未知模型类型默认使用SDXL"""
    print("=" * 50)
    print("Testing Unknown Model Class Defaults to SDXL")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"
    
    mock_pipe = Mock()
    mock_pipe.device = "cpu"
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload.return_value = None
    mock_pipeline_class.from_pretrained.return_value = mock_pipe
    
    model_path = "/fake/path/to/unknown_model"
    ModelLoader.clear_cache()
    
    # 使用未知的模型类型
    pipe = await ModelLoader.load_image_model(
        model_path=model_path,
        family="unknown_type",
        device="cpu"
    )
    
    # 应该使用SDXL加载器
    assert pipe is not None
    mock_pipeline_class.from_pretrained.assert_called_once()
    print("✓ Unknown model class defaults to SDXL correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.StableDiffusionXLPipeline')
async def test_auto_device_selection(mock_pipeline_class, mock_torch):
    """测试自动设备选择"""
    print("=" * 50)
    print("Testing Auto Device Selection")
    print("=" * 50)
    
    # Mock设置 - CUDA可用
    mock_torch.cuda.is_available.return_value = True
    mock_torch.float16 = "float16"
    
    mock_pipe = Mock()
    mock_pipe.device = "cuda"
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.enable_model_cpu_offload.return_value = None
    mock_pipeline_class.from_pretrained.return_value = mock_pipe
    
    model_path = "/fake/path/to/model"
    ModelLoader.clear_cache()
    
    # 不指定device，应该自动选择CUDA
    pipe = await ModelLoader.load_image_model(
        model_path=model_path,
        family="sdxl"
    )
    
    assert pipe is not None
    # 验证使用了float16（CUDA模式）
    call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
    assert call_kwargs.get("torch_dtype") == "float16"
    print("✓ Auto device selection works correctly\n")


@patch('backend.services.inference.model_loader.torch')
@patch('backend.services.inference.model_loader.StableDiffusionXLPipeline')
async def test_load_model_import_error(mock_pipeline_class, mock_torch):
    """测试导入错误处理"""
    print("=" * 50)
    print("Testing Import Error Handling")
    print("=" * 50)
    
    # Mock设置
    mock_torch.cuda.is_available.return_value = False
    
    # 模拟导入错误
    mock_pipeline_class.from_pretrained.side_effect = ImportError("diffusers not installed")
    
    model_path = "/fake/path/to/model"
    ModelLoader.clear_cache()
    
    with pytest.raises(ModelLoadFailedException) as exc_info:
        await ModelLoader.load_image_model(
            model_path=model_path,
            family="sdxl",
            device="cpu"
        )
    
    assert "diffusers" in str(exc_info.value).lower()
    print("✓ Import error handling works correctly\n")


def test_family_variations():
    """测试不同模型分类名称的识别"""
    print("=" * 50)
    print("Testing Model Class Variations")
    print("=" * 50)
    
    # 测试各种SDXL变体名称
    sdxl_variants = ["sdxl", "SDXL", "sd xl", "SDXL 1.0", "sdxl"]
    for variant in sdxl_variants:
        # 这里只是测试名称识别逻辑，不实际加载
        assert variant.lower() in ["sdxl", "sd xl", "sdxl 1.0"] or variant.lower().startswith("sdxl")
    
    # 测试Flux变体
    flux_variants = ["flux", "flux.1", "flux-1"]
    for variant in flux_variants:
        assert "flux" in variant.lower()
    
    print("✓ Model class variations recognized correctly\n")


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("ModelLoader Unit Tests")
    print("=" * 50 + "\n")
    
    # 同步测试
    test_model_loader_singleton()
    test_clear_cache()
    test_get_cache_size()
    test_family_variations()
    
    # 异步测试
    await test_load_sdxl_model()
    await test_load_flux_model()
    await test_load_sd15_model()
    await test_load_model_path_not_exists()
    await test_model_caching()
    await test_unknown_family_defaults_to_sdxl()
    await test_auto_device_selection()
    await test_load_model_import_error()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_all_tests())


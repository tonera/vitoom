"""
创建测试用的模型记录
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database import Model
from backend.utils import generate_uuid
from backend.core.logger import get_app_logger

logger = get_app_logger(__name__)

# python test/create_test_model.py -id model_888888892 -name RealVisXL_V4.0.safetensors -class sdxl
def create_test_image_model(model_id=None, model_name=None, family=None):
    """创建一个测试用的image类型模型
    
    Args:
        model_id: 模型ID，默认为 model_888888892
        model_name: 模型名称，默认为 RealVisXL_V4.0.safetensors
        family: 模型类别，默认为 sdxl
    """
    model_id = model_id or "model_888888892"
    model_name = model_name or "RealVisXL_V4.0.safetensors"
    family = family or "sdxl"
    
    # 检查模型是否已存在
    existing_model = Model.get_by_id(model_id)
    if existing_model:
        print(f"Model {model_id} already exists")
        return existing_model
    
    # 创建模型记录
    try:
        model_dict = Model.create(
            id=model_id,
            name=model_name,
            model_type="image",
            storage_mode="local",
            local_path="",  # 测试路径
            status="active",
            family=family,
            is_local_model=True,
            description="Test image model for inference service",
            is_editable=True,
            model_config={
                "resolution": "1K",
                "steps": 50,
                "guidance_scale": 7.5
            }
        )
        
        if model_dict:
            print(f"✓ Test model created successfully:")
            print(f"  - ID: {model_dict['id']}")
            print(f"  - Name: {model_dict['name']}")
            print(f"  - Type: {model_dict['type']}")
            print(f"  - Status: {model_dict['status']}")
            print(f"  - Local Path: {model_dict.get('local_path', 'N/A')}")
            return model_dict
        else:
            print(f"❌ Failed to create model: returned None")
            return None
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="创建测试用的Image模型",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-id", "--id",
        type=str,
        help="模型ID (例如: model_888888892)"
    )
    parser.add_argument(
        "-name", "--name",
        type=str,
        help="模型名称 (例如: RealVisXL_V4.0.safetensors)"
    )
    parser.add_argument(
        "-class", "--class",
        dest="family",
        type=str,
        help="模型类别 (例如: sdxl)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("创建测试用的Image模型")
    print("="*60)
    
    # 显示使用的参数
    if args.id or args.name or args.family:
        print("\n使用命令行参数:")
        if args.id:
            print(f"  - ID: {args.id}")
        if args.name:
            print(f"  - Name: {args.name}")
        if args.family:
            print(f"  - Class: {args.family}")
        print()
    
    model = create_test_image_model(
        model_id=args.id,
        model_name=args.name,
        family=args.family
    )
    
    if model:
        print("\n✓ 测试模型创建成功！")
        print(f"\n可以在创建任务时使用 model_id: {model['id']}")
    else:
        print("\n❌ 测试模型创建失败")


if __name__ == "__main__":
    main()

#sdxl目录 python test/create_test_model.py -id model_888888893 -name Pony-Diffusion-V6-XL-for-Anime -class sdxl
#sdxl-v python test/create_test_model.py -id model_888888894 -name NoobAI-XL-Vpred-v1.0-cyberfix.safetensors
#flux目录 python test/create_test_model.py -id model_888888895 -name FLUX.1-dev -class flux
#flux文件 python test/create_test_model.py -id model_888888896 -name flux1_v40Fp8.safetensors -class flux
#qwen目录 python test/create_test_model.py -id model_888888897 -name Qwen-Image -class qwen
#zimage目录 python test/create_test_model.py -id model_888888898 -name Z-Image-Turbo -class zimage
#aura录 python test/create_test_model.py -id model_888888899 -name pony-v7-base-fp8_scaled -class aura

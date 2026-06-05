#!/usr/bin/env python3
"""
大模型文件键名对比工具
对比多个模型文件的键名差异，只显示不同的键名
"""
import sys
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠️  警告: safetensors 未安装，无法读取 .safetensors 文件")


def load_safetensors_keys(file_path: Path) -> Set[str]:
    """从 safetensors 文件加载键名"""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 未安装")
    
    try:
        with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:
            return set(f.keys())
    except Exception as e:
        raise ValueError(f"读取 safetensors 文件失败: {e}")


def load_ckpt_keys(file_path: Path) -> Set[str]:
    """从 ckpt 文件加载键名（需要 torch）"""
    try:
        import torch
    except ImportError:
        raise ImportError("torch 未安装，无法读取 .ckpt 文件")
    
    try:
        checkpoint = torch.load(file_path, map_location="cpu")
        
        # ckpt 文件可能是字典，键可能是 'state_dict' 或其他
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                return set(checkpoint["state_dict"].keys())
            elif "model" in checkpoint:
                return set(checkpoint["model"].keys())
            else:
                # 尝试直接使用顶层键
                return set(checkpoint.keys())
        else:
            raise ValueError("ckpt 文件格式不支持")
    except Exception as e:
        raise ValueError(f"读取 ckpt 文件失败: {e}")


def load_model_keys(file_path: Path) -> Set[str]:
    """根据文件扩展名加载模型键名"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"不是文件: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".safetensors":
        return load_safetensors_keys(file_path)
    elif suffix == ".ckpt":
        return load_ckpt_keys(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .safetensors 和 .ckpt")


def compare_model_keys(model_paths: List[Path]) -> Dict[str, Dict[str, Set[str]]]:
    """
    对比多个模型的键名
    
    Returns:
        {
            "all_keys": set,  # 所有键名的并集
            "common_keys": set,  # 所有模型都有的键名
            "unique_keys": {model_name: set},  # 每个模型独有的键名
            "model_keys": {model_name: set}  # 每个模型的所有键名
        }
    """
    model_keys_dict: Dict[str, Set[str]] = {}
    
    # 加载所有模型的键名
    print("📂 正在加载模型键名...")
    for model_path in model_paths:
        try:
            keys = load_model_keys(model_path)
            model_name = model_path.name
            model_keys_dict[model_name] = keys
            print(f"  ✓ {model_name}: {len(keys)} 个键")
        except Exception as e:
            print(f"  ❌ {model_path.name}: 加载失败 - {e}")
            sys.exit(1)
    
    if len(model_keys_dict) < 2:
        raise ValueError("至少需要 2 个模型文件进行对比")
    
    # 计算所有键名的并集
    all_keys = set()
    for keys in model_keys_dict.values():
        all_keys.update(keys)
    
    # 计算所有模型都有的键名（交集）
    common_keys = set.intersection(*model_keys_dict.values())
    
    # 计算每个模型独有的键名
    unique_keys: Dict[str, Set[str]] = {}
    for model_name, keys in model_keys_dict.items():
        # 该模型有但其他模型都没有的键
        other_keys = set()
        for other_name, other_key_set in model_keys_dict.items():
            if other_name != model_name:
                other_keys.update(other_key_set)
        unique_keys[model_name] = keys - other_keys
    
    return {
        "all_keys": all_keys,
        "common_keys": common_keys,
        "unique_keys": unique_keys,
        "model_keys": model_keys_dict
    }


def print_comparison_result(result: Dict[str, Dict[str, Set[str]]], model_paths: List[Path]):
    """打印对比结果"""
    all_keys = result["all_keys"]
    common_keys = result["common_keys"]
    unique_keys = result["unique_keys"]
    model_keys = result["model_keys"]
    
    print("\n" + "=" * 80)
    print("📊 模型键名对比结果")
    print("=" * 80)
    
    # 统计信息
    print(f"\n📈 统计信息:")
    print(f"  总键数（并集）: {len(all_keys)}")
    print(f"  共同键数（交集）: {len(common_keys)}")
    print(f"  差异键数: {len(all_keys) - len(common_keys)}")
    
    for model_name in model_keys.keys():
        unique_count = len(unique_keys[model_name])
        print(f"  {model_name}: {len(model_keys[model_name])} 个键，其中 {unique_count} 个独有")
    
    # 显示共同键（可选，如果用户想看）
    if len(common_keys) > 0:
        print(f"\n✅ 共同键（{len(common_keys)} 个，已省略）")
    
    # 显示每个模型的独有键
    print(f"\n🔍 差异键名详情:")
    print("-" * 80)
    
    has_differences = False
    for model_name, unique_key_set in unique_keys.items():
        if unique_key_set:
            has_differences = True
            print(f"\n📌 {model_name} 独有的键（{len(unique_key_set)} 个）:")
            for key in sorted(unique_key_set):
                print(f"  + {key}")
    
    # 显示键名存在但内容可能不同的情况
    # 找出在所有模型中都存在但可能结构不同的键
    print(f"\n🔎 键名存在性对比:")
    print("-" * 80)
    
    # 为每个键显示它在哪些模型中存在
    key_presence: Dict[str, List[str]] = defaultdict(list)
    for model_name, keys in model_keys.items():
        for key in keys:
            key_presence[key].append(model_name)
    
    # 找出不在所有模型中都存在的键
    missing_keys: Dict[str, List[str]] = {}
    for key, present_in in key_presence.items():
        if len(present_in) < len(model_keys):
            missing_in = [name for name in model_keys.keys() if name not in present_in]
            missing_keys[key] = missing_in
    
    if missing_keys:
        has_differences = True
        print(f"\n⚠️  以下键名在某些模型中缺失（{len(missing_keys)} 个）:")
        for key, missing_in in sorted(missing_keys.items()):
            present_in = [name for name in model_keys.keys() if name not in missing_in]
            print(f"  • {key}")
            print(f"    存在: {', '.join(present_in)}")
            print(f"    缺失: {', '.join(missing_in)}")
    
    if not has_differences:
        print("\n✅ 所有模型的键名完全相同！")


def main():
    if len(sys.argv) < 3:
        print("用法: python compare_model_keys.py <模型路径1> <模型路径2> [模型路径3] ...")
        print("\n示例:")
        print("  python compare_model_keys.py model1.safetensors model2.safetensors")
        print("  python compare_model_keys.py model1.ckpt model2.ckpt model3.safetensors")
        sys.exit(1)
    
    # 解析模型路径
    model_paths: List[Path] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.is_absolute():
            # 如果不是绝对路径，尝试相对于项目根目录
            project_root = Path(__file__).parent.parent
            path = project_root / arg
        
        model_paths.append(path)
    
    # 验证文件存在
    for path in model_paths:
        if not path.exists():
            print(f"❌ 文件不存在: {path}")
            sys.exit(1)
    
    try:
        # 对比模型键名
        result = compare_model_keys(model_paths)
        
        # 打印结果
        print_comparison_result(result, model_paths)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


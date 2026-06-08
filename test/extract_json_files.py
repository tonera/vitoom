"""
提取diffusers格式模型目录下的所有JSON文件
扫描指定目录下的所有子目录（diffusers格式模型），找出所有JSON文件并复制到输出目录
"""
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def is_diffusers_model_dir(dir_path: Path) -> bool:
    """
    判断是否是diffusers格式模型目录
    
    Args:
        dir_path: 目录路径
    
    Returns:
        True如果是diffusers格式模型目录
    """
    if not dir_path.is_dir():
        return False
    
    # 检查是否有model_index.json（diffusers格式的标志）
    model_index = dir_path / "model_index.json"
    if model_index.exists():
        return True
    
    # 或者检查是否有常见的diffusers组件目录
    common_components = ["unet", "text_encoder", "vae", "scheduler", "tokenizer"]
    has_components = any((dir_path / comp).exists() for comp in common_components)
    return has_components


def find_json_files(directory: Path) -> List[Path]:
    """
    递归查找目录下的所有JSON文件
    
    Args:
        directory: 要搜索的目录
    
    Returns:
        JSON文件路径列表
    """
    json_files = []
    
    if not directory.exists() or not directory.is_dir():
        return json_files
    
    # 递归查找所有.json文件
    for json_file in directory.rglob("*.json"):
        if json_file.is_file():
            json_files.append(json_file)
    
    return sorted(json_files)


def copy_json_files(source_dir: Path, json_files: List[Path], output_base: Path, model_dir_name: str) -> int:
    """
    复制JSON文件到输出目录，保持相对路径结构
    
    Args:
        source_dir: 源目录（模型目录）
        json_files: JSON文件列表
        output_base: 输出基础目录
        model_dir_name: 模型目录名称
    
    Returns:
        复制的文件数量
    """
    copied_count = 0
    
    for json_file in json_files:
        try:
            # 计算相对路径（相对于模型目录）
            relative_path = json_file.relative_to(source_dir)
            
            # 构建输出路径
            output_path = output_base / model_dir_name / relative_path
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(json_file, output_path)
            copied_count += 1
            
            print(f"  Copied: {relative_path}")
            
        except Exception as e:
            print(f"  Error copying {json_file}: {e}")
    
    return copied_count


def extract_json_from_models(source_directory: Path, output_directory: Path) -> None:
    """
    从指定目录下的所有diffusers格式模型目录中提取JSON文件
    
    Args:
        source_directory: 源目录（包含模型目录的目录）
        output_directory: 输出目录
    """
    if not source_directory.exists():
        print(f"Error: Source directory does not exist: {source_directory}")
        return
    
    if not source_directory.is_dir():
        print(f"Error: Not a directory: {source_directory}")
        return
    
    print(f"Scanning directory: {source_directory}")
    print(f"Output directory: {output_directory}")
    print("=" * 80)
    
    # 扫描所有子目录
    model_dirs = []
    for item in sorted(source_directory.iterdir()):
        if item.is_dir() and is_diffusers_model_dir(item):
            model_dirs.append(item)
    
    if not model_dirs:
        print("No diffusers format model directories found")
        return
    
    print(f"Found {len(model_dirs)} diffusers model directory(ies)")
    print("=" * 80)
    
    total_files = 0
    
    # 处理每个模型目录
    for i, model_dir in enumerate(model_dirs, 1):
        model_name = model_dir.name
        print(f"\n[{i}/{len(model_dirs)}] Processing: {model_name}")
        
        # 查找所有JSON文件
        json_files = find_json_files(model_dir)
        
        if not json_files:
            print(f"  No JSON files found in {model_name}")
            continue
        
        print(f"  Found {len(json_files)} JSON file(s)")
        
        # 复制JSON文件
        copied_count = copy_json_files(model_dir, json_files, output_directory, model_name)
        total_files += copied_count
        
        print(f"  Copied {copied_count} file(s)")
    
    print("\n" + "=" * 80)
    print(f"Total: {total_files} JSON file(s) extracted from {len(model_dirs)} model directory(ies)")
    print(f"Output directory: {output_directory}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Extract JSON files from diffusers format model directories")
    parser.add_argument("source", type=str, help="Source directory containing model directories")
    parser.add_argument("output", type=str, nargs="?", default="outputs",
                       help="Output directory (default: outputs)")
    args = parser.parse_args()
    
    source_directory = Path(args.source).resolve()
    output_directory = Path(args.output).resolve()
    
    # 确保输出目录存在
    output_directory.mkdir(parents=True, exist_ok=True)
    
    extract_json_from_models(source_directory, output_directory)


if __name__ == "__main__":
    main()


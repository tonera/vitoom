"""
导出模型文件键名到JSON
扫描指定目录下所有的 .ckpt 或 .safetensors 文件，将文件名和键名导出到JSON文件
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors library not installed, only .safetensors files will be processed")


def get_keys_from_safetensors(file_path: Path) -> List[str]:
    """
    从safetensors文件读取所有键名
    
    Args:
        file_path: safetensors文件路径
    
    Returns:
        键名列表
    """
    if not SAFETENSORS_AVAILABLE:
        return []
    
    try:
        with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            return keys
    except Exception as e:
        print(f"Warning: Failed to read keys from {file_path}: {e}")
        return []


def get_keys_from_ckpt(file_path: Path) -> List[str]:
    """
    从ckpt文件读取所有键名（需要加载整个文件，可能很慢）
    
    Args:
        file_path: ckpt文件路径
    
    Returns:
        键名列表
    """
    try:
        import torch
        checkpoint = torch.load(file_path, map_location="cpu")
        
        # ckpt文件可能是字典，包含state_dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                keys = list(checkpoint["state_dict"].keys())
            else:
                # 直接是state_dict
                keys = list(checkpoint.keys())
        else:
            keys = []
        
        # 清理内存
        del checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return keys
    except Exception as e:
        print(f"Warning: Failed to read keys from {file_path}: {e}")
        return []


def scan_directory(directory: Path) -> List[Path]:
    """
    扫描目录，找出所有模型文件（只扫描一级目录）
    
    Args:
        directory: 要扫描的目录
    
    Returns:
        模型文件路径列表
    """
    model_files = []
    
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        return model_files
    
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return model_files
    
    # 只扫描一级目录
    for ext in [".safetensors", ".ckpt"]:
        model_files.extend(directory.glob(f"*{ext}"))
    
    # 排序
    model_files.sort()
    
    return model_files


def export_keys_to_json(directory: Path, output_file: Path) -> None:
    """
    导出所有模型文件的键名到JSON文件
    
    Args:
        directory: 要扫描的目录
        output_file: 输出JSON文件路径
    """
    print(f"Scanning directory: {directory}")
    print("=" * 80)
    
    # 扫描模型文件
    model_files = scan_directory(directory)
    
    if not model_files:
        print("No model files found")
        return
    
    print(f"Found {len(model_files)} model file(s)")
    print("=" * 80)
    
    # 导出数据
    export_data = {
        "directory": str(directory),
        "total_files": len(model_files),
        "files": []
    }
    
    for i, file_path in enumerate(model_files, 1):
        print(f"[{i}/{len(model_files)}] Processing: {file_path.name}")
        
        file_info = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "relative_path": str(file_path.relative_to(directory)) if file_path.is_relative_to(directory) else str(file_path),
            "extension": file_path.suffix,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "keys": [],
            "key_count": 0,
            "error": None
        }
        
        try:
            if file_path.suffix == ".safetensors":
                keys = get_keys_from_safetensors(file_path)
            elif file_path.suffix == ".ckpt":
                keys = get_keys_from_ckpt(file_path)
            else:
                keys = []
            
            file_info["keys"] = keys
            file_info["key_count"] = len(keys)
            print(f"  Found {len(keys)} keys")
            
        except Exception as e:
            file_info["error"] = str(e)
            print(f"  Error: {e}")
        
        export_data["files"].append(file_info)
    
    # 保存到JSON文件
    print("=" * 80)
    print(f"Writing to: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully exported {len(export_data['files'])} files to {output_file}")
        print(f"Total keys: {sum(f['key_count'] for f in export_data['files'])}")
        
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Export model file keys to JSON")
    parser.add_argument("directory", type=str, help="Directory to scan")
    parser.add_argument("output", type=str, help="Output JSON file path")
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    output_file = Path(args.output).resolve()
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    export_keys_to_json(directory, output_file)


if __name__ == "__main__":
    main()


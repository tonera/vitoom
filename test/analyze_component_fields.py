"""
分析模型配置文件中特定组件字段的异同
按_class_name归类，分析text_encoder, text_encoder_2, unet, vae, transformer字段
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_indexes(directory: Path) -> Dict[str, Dict[str, Any]]:
    """加载所有model_index.json文件"""
    model_indexes = {}
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory does not exist: {directory}")
        return model_indexes
    
    for model_index_file in directory.rglob("model_index.json"):
        try:
            model_name = model_index_file.parent.name
            
            with open(model_index_file, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            model_indexes[model_name] = model_index
            
        except Exception as e:
            print(f"Warning: Failed to load {model_index_file}: {e}")
    
    return model_indexes


def group_by_class_name(model_indexes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """按_class_name分组"""
    groups = defaultdict(list)
    
    for model_name, model_index in model_indexes.items():
        if isinstance(model_index, dict):
            class_name = model_index.get("_class_name", "Unknown")
            groups[class_name].append(model_name)
    
    return dict(groups)


def analyze_component_fields(model_indexes: Dict[str, Dict[str, Any]], 
                            model_names: List[str],
                            target_fields: List[str]) -> Dict[str, Any]:
    """分析特定组件字段"""
    field_analysis = {}
    
    for field in target_fields:
        field_values = []
        
        for model_name in model_names:
            model_index = model_indexes.get(model_name, {})
            value = model_index.get(field)
            if value is not None:
                field_values.append({
                    "model": model_name,
                    "value": value
                })
        
        # 分析值的异同
        unique_values = {}
        for item in field_values:
            val = item["value"]
            val_str = json.dumps(val, sort_keys=True) if isinstance(val, (list, dict)) else str(val)
            if val_str not in unique_values:
                unique_values[val_str] = []
            unique_values[val_str].append(item["model"])
        
        field_analysis[field] = {
            "presence_count": len(field_values),
            "presence_rate": len(field_values) / len(model_names) if model_names else 0,
            "unique_value_count": len(unique_values),
            "is_consistent": len(unique_values) == 1 if field_values else False,
            "values": unique_values
        }
    
    return field_analysis


def print_component_analysis(groups: Dict[str, List[str]], 
                             model_indexes: Dict[str, Dict[str, Any]],
                             target_fields: List[str],
                             output_file: Path = None):
    """打印组件字段分析报告"""
    report_lines = []
    
    report_lines.append("=" * 100)
    report_lines.append("模型组件字段分析报告")
    report_lines.append("=" * 100)
    report_lines.append(f"\n分析字段: {', '.join(target_fields)}")
    report_lines.append(f"总共找到 {len(model_indexes)} 个模型")
    report_lines.append(f"共 {len(groups)} 种不同的 _class_name")
    report_lines.append("\n" + "=" * 100)
    
    # 按_class_name分析
    for class_name, model_names in sorted(groups.items(), key=lambda x: -len(x[1])):
        report_lines.append(f"\n\n### 【{class_name}】 ({len(model_names)} 个模型)")
        report_lines.append("-" * 100)
        report_lines.append(f"模型列表: {', '.join(sorted(model_names))}")
        report_lines.append("")
        
        analysis = analyze_component_fields(model_indexes, model_names, target_fields)
        
        for field in target_fields:
            field_info = analysis.get(field, {})
            presence_count = field_info.get("presence_count", 0)
            presence_rate = field_info.get("presence_rate", 0) * 100
            unique_count = field_info.get("unique_value_count", 0)
            is_consistent = field_info.get("is_consistent", False)
            values = field_info.get("values", {})
            
            if presence_count == 0:
                report_lines.append(f"\n{field}: ❌ 不存在（所有模型都没有此字段）")
            elif is_consistent:
                # 值一致
                val_str = list(values.keys())[0]
                val = json.loads(val_str) if (val_str.startswith('[') or val_str.startswith('{')) else val_str
                report_lines.append(f"\n{field}: ✓ 一致 ({presence_rate:.0f}% 的模型有此字段)")
                report_lines.append(f"  值: {val}")
            else:
                # 值不一致
                report_lines.append(f"\n{field}: ⚠ 不一致 ({presence_rate:.0f}% 的模型有此字段，有 {unique_count} 种不同值)")
                for val_str, models in values.items():
                    val = json.loads(val_str) if (val_str.startswith('[') or val_str.startswith('{')) else val_str
                    report_lines.append(f"  值: {val}")
                    report_lines.append(f"    出现在: {', '.join(sorted(models)[:5])}{'...' if len(models) > 5 else ''}")
    
    # 输出报告
    report_text = "\n".join(report_lines)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存到: {output_file}")
    else:
        print(report_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Analyze component fields in model configs")
    parser.add_argument("directory", type=str, help="Directory containing model directories")
    parser.add_argument("--output", "-o", type=str, help="Output report file path (optional)")
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    output_file = Path(args.output).resolve() if args.output else None
    
    # 要分析的字段
    target_fields = ["text_encoder", "text_encoder_2", "unet", "vae", "transformer"]
    
    print(f"Loading model_index.json files from: {directory}")
    
    # 加载所有model_index.json文件
    model_indexes = load_model_indexes(directory)
    
    if not model_indexes:
        print("No model_index.json files found")
        return
    
    print(f"Loaded {len(model_indexes)} model_index.json files")
    
    # 按_class_name分组
    groups = group_by_class_name(model_indexes)
    
    # 生成分析报告
    print_component_analysis(groups, model_indexes, target_fields, output_file)


if __name__ == "__main__":
    main()
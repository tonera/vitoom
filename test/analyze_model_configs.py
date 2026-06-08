"""
分析diffusers模型配置文件
按_class_name归类，并找出同类模型中其他键值的差异
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_indexes(directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    加载所有model_index.json文件
    
    Args:
        directory: 包含模型目录的目录
    
    Returns:
        字典：{模型名称: model_index内容}
    """
    model_indexes = {}
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory does not exist: {directory}")
        return model_indexes
    
    # 扫描所有model_index.json文件
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
    """
    按_class_name分组
    
    Args:
        model_indexes: 模型索引字典
    
    Returns:
        字典：{_class_name: [模型名称列表]}
    """
    groups = defaultdict(list)
    
    for model_name, model_index in model_indexes.items():
        if isinstance(model_index, dict):
            class_name = model_index.get("_class_name", "Unknown")
            groups[class_name].append(model_name)
    
    return dict(groups)


def analyze_key_differences(model_indexes: Dict[str, Dict[str, Any]], 
                            model_names: List[str]) -> Dict[str, Any]:
    """
    分析一组模型中键值的差异
    
    Args:
        model_indexes: 模型索引字典
        model_names: 要分析的模型名称列表
    
    Returns:
        分析结果字典
    """
    if not model_names:
        return {}
    
    # 要忽略的键
    ignored_keys = {
        "_diffusers_version",
        "force_zeros_for_empty_prompt",
        "_name_or_path"
    }
    
    # 收集所有键
    all_keys: Set[str] = set()
    key_values: Dict[str, List[Any]] = defaultdict(list)
    key_presence: Dict[str, int] = defaultdict(int)
    
    for model_name in model_names:
        model_index = model_indexes.get(model_name, {})
        if not isinstance(model_index, dict):
            continue
        
        model_keys = set(model_index.keys())
        # 过滤掉要忽略的键
        model_keys = model_keys - ignored_keys
        all_keys.update(model_keys)
        
        for key in model_keys:
            value = model_index[key]
            key_values[key].append({
                "model": model_name,
                "value": value
            })
            key_presence[key] += 1
    
    # 分析结果
    result = {
        "total_models": len(model_names),
        "all_keys": sorted(all_keys),
        "common_keys": [],  # 所有模型都有的键
        "optional_keys": [],  # 部分模型有的键
        "key_details": {}  # 每个键的详细信息
    }
    
    for key in sorted(all_keys):
        presence_count = key_presence[key]
        values = key_values[key]
        
        # 检查值是否一致
        unique_values = set()
        for item in values:
            # 将值转换为可哈希的形式进行比较
            val = item["value"]
            if isinstance(val, (dict, list)):
                val_str = json.dumps(val, sort_keys=True)
            else:
                val_str = str(val)
            unique_values.add(val_str)
        
        is_common = presence_count == len(model_names)
        is_consistent = len(unique_values) == 1
        
        key_info = {
            "presence_count": presence_count,
            "presence_rate": presence_count / len(model_names),
            "is_common": is_common,
            "is_consistent": is_consistent,
            "unique_value_count": len(unique_values),
            "values": values
        }
        
        result["key_details"][key] = key_info
        
        if is_common:
            result["common_keys"].append(key)
        else:
            result["optional_keys"].append(key)
    
    return result


def print_analysis_report(groups: Dict[str, List[str]], 
                         model_indexes: Dict[str, Dict[str, Any]],
                         output_file: Path = None):
    """
    打印分析报告
    
    Args:
        groups: 按_class_name分组的字典
        model_indexes: 模型索引字典
        output_file: 可选的输出文件路径
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("Diffusers模型配置文件分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"\n总共找到 {len(model_indexes)} 个模型")
    report_lines.append(f"共 {len(groups)} 种不同的 _class_name")
    report_lines.append("\n" + "=" * 80)
    
    # 按_class_name分组统计
    report_lines.append("\n【按 _class_name 分组统计】")
    report_lines.append("-" * 80)
    for class_name, model_names in sorted(groups.items(), key=lambda x: -len(x[1])):
        report_lines.append(f"\n{class_name}: {len(model_names)} 个模型")
        for model_name in sorted(model_names):
            report_lines.append(f"  - {model_name}")
    
    # 详细分析每个组
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("【详细差异分析】")
    report_lines.append("=" * 80)
    
    for class_name, model_names in sorted(groups.items(), key=lambda x: -len(x[1])):
        report_lines.append(f"\n\n### {class_name} ({len(model_names)} 个模型)")
        report_lines.append("-" * 80)
        
        analysis = analyze_key_differences(model_indexes, model_names)
        
        report_lines.append(f"\n总键数: {len(analysis['all_keys'])}")
        report_lines.append(f"共同键数: {len(analysis['common_keys'])}")
        report_lines.append(f"可选键数: {len(analysis['optional_keys'])}")
        
        # 共同键
        if analysis['common_keys']:
            report_lines.append(f"\n【共同键】（所有 {len(model_names)} 个模型都有）:")
            for key in analysis['common_keys']:
                key_info = analysis['key_details'][key]
                if key_info['is_consistent']:
                    # 值一致，只显示一个值
                    value = key_info['values'][0]['value']
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=2, ensure_ascii=False)
                        report_lines.append(f"  ✓ {key}: (值一致)")
                        report_lines.append(f"    {value_str}")
                    else:
                        report_lines.append(f"  ✓ {key}: {value}")
                else:
                    # 值不一致
                    report_lines.append(f"  ⚠ {key}: (值不一致，有 {key_info['unique_value_count']} 种不同值)")
                    # 显示不同的值
                    seen_values = {}
                    for item in key_info['values']:
                        val = item['value']
                        val_str = json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else str(val)
                        if val_str not in seen_values:
                            seen_values[val_str] = []
                        seen_values[val_str].append(item['model'])
                    
                    for val_str, models in seen_values.items():
                        if isinstance(val_str, str) and val_str.startswith('{') or val_str.startswith('['):
                            report_lines.append(f"    值: {val_str[:100]}... (出现在: {', '.join(models[:3])}{'...' if len(models) > 3 else ''})")
                        else:
                            report_lines.append(f"    值: {val_str} (出现在: {', '.join(models[:3])}{'...' if len(models) > 3 else ''})")
        
        # 可选键
        if analysis['optional_keys']:
            report_lines.append(f"\n【可选键】（只在部分模型中出现）:")
            for key in sorted(analysis['optional_keys']):
                key_info = analysis['key_details'][key]
                presence_rate = key_info['presence_rate'] * 100
                report_lines.append(f"  • {key}: 出现在 {key_info['presence_count']}/{len(model_names)} 个模型 ({presence_rate:.1f}%)")
                
                # 显示哪些模型有这个键
                models_with_key = [item['model'] for item in key_info['values']]
                report_lines.append(f"    模型: {', '.join(sorted(models_with_key)[:5])}{'...' if len(models_with_key) > 5 else ''}")
                
                # 如果值一致，也显示值
                if key_info['is_consistent']:
                    value = key_info['values'][0]['value']
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=4, ensure_ascii=False)
                        report_lines.append(f"    值: {value_str[:200]}...")
                    else:
                        report_lines.append(f"    值: {value}")
    
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
    parser = argparse.ArgumentParser(description="Analyze diffusers model config files")
    parser.add_argument("directory", type=str, help="Directory containing model directories with model_index.json")
    parser.add_argument("--output", "-o", type=str, help="Output report file path (optional)")
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    output_file = Path(args.output).resolve() if args.output else None
    
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
    print_analysis_report(groups, model_indexes, output_file)


if __name__ == "__main__":
    main()


"""
图片模型类型检测脚本
扫描目录并自动检测模型类型
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors library not installed, single file detection may be limited")


def scan_directory(directory: Path) -> List[Tuple[str, Path]]:
    """
    扫描目录，列出所有文件和目录
    
    Args:
        directory: 要扫描的目录路径
    
    Returns:
        列表，每个元素是 (类型, 路径) 元组，类型为"目录"或"文件"
    """
    items = []
    
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        return items
    
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return items
    
    # 扫描目录下的所有项目
    for item in sorted(directory.iterdir()):
        if item.is_dir():
            items.append(("目录", item))
        elif item.is_file():
            items.append(("文件", item))
    
    return items


def _detect_from_filename(file_path: Path) -> Optional[str]:
    """
    从文件名推断模型类型（启发式规则）
    
    Args:
        file_path: 文件路径（文件或目录）
    
    Returns:
        模型类型或None
    """
    filename_lower = file_path.name.lower()
    stem_lower = file_path.stem.lower()
    
    # 特殊模型类型（优先级最高）
    if "z-image" in filename_lower or "zimage" in filename_lower:
        return "z-image"
    
    if "pony-v7" in filename_lower or "ponyv7" in filename_lower or "pony_v7" in filename_lower:
        return "pony-v7"
    
    # Flux特征（先检查，因为有些文件名可能同时包含flux和xl）
    if "flux" in filename_lower:
        return "flux"
    
    # SDXL特征（优先级高，但排除flux和特殊模型）
    # 检查明确的SDXL标识
    if any(keyword in filename_lower for keyword in ["sdxl", "sd-xl", "sd_xl"]):
        return "sdxl"
    
    # 检查XL标识（但需要更严格的规则）
    # 如果文件名包含XL且不包含flux，很可能是SDXL
    if "xl" in filename_lower and "flux" not in filename_lower:
        # 进一步检查是否是SDXL相关的关键词
        sdxl_keywords = [
            "xl", "x-xl", "xl-", "-xl", "xl_", "_xl",
            "juggernaut-xl", "pony-xl", "noobai-xl", "realvisxl",
            "animagine-xl", "cyberrealistic", "dreamshaperxl",
            "duchaiten", "dynavisionxl", "hassakuxl", "illustriousxl",
            "novaanimexl", "ponydiffusion", "prefectponyxl",
            "protovisionxl", "realvisxl", "sdxl", "wai", "zavychromaxl"
        ]
        # 检查文件名是否包含这些关键词
        if any(keyword in filename_lower for keyword in sdxl_keywords):
            return "sdxl"
    
    # SD15/SD21特征
    if any(keyword in filename_lower for keyword in ["sd15", "sd-15", "sd_15", "stable-diffusion-v1"]):
        return "sd15"
    if any(keyword in filename_lower for keyword in ["sd21", "sd-21", "sd_21", "stable-diffusion-v2"]):
        return "sd21"
    
    return None


def detect_model_type_from_file(file_path: Path) -> Optional[str]:
    """
    从单个checkpoint文件中检测模型类型
    
    Args:
        file_path: 模型文件路径
    
    Returns:
        模型类型: "sdxl", "sd15", "sd21", "flux", "qwen-image", 或 None（无法确定）
    """
    if not file_path.exists() or not file_path.is_file():
        return None
    
    # 只处理safetensors和ckpt文件
    if file_path.suffix not in [".safetensors", ".ckpt"]:
        return None
    
    # 方法0: 先检查文件名（快速启发式）
    filename_type = _detect_from_filename(file_path)
    
    try:
        # 方法1: 读取safetensors文件的键名和元数据
        if file_path.suffix == ".safetensors" and SAFETENSORS_AVAILABLE:
            try:
                with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    
                    if not keys:
                        # 如果无法读取键，使用文件名推断
                        return filename_type
                    
                    # 检查UNet输入通道数（最可靠的方法）
                    unet_input_channels = None
                    for key in keys:
                        if "input_blocks.0.0.weight" in key or "model.diffusion_model.input_blocks.0.0.weight" in key:
                            try:
                                weight = f.get_tensor(key)
                                if len(weight.shape) >= 2:
                                    unet_input_channels = weight.shape[0]
                                    break
                            except:
                                pass
                    
                    # 根据UNet输入通道数判断
                    if unet_input_channels == 9:
                        # SDXL的UNet输入通道是9
                        return "sdxl"
                    elif unet_input_channels == 4:
                        # SD15/SD21/Flux的UNet输入通道是4
                        # 需要进一步区分
                        
                        # 优先检查文件名提示（文件名通常很可靠）
                        if filename_type:
                            return filename_type
                        
                        # 检查Flux特征（transformer结构，需要更严格的匹配）
                        has_transformer = any(
                            "transformer" in key.lower() and (
                                "blocks" in key.lower() or
                                "layers" in key.lower() or
                                "norm" in key.lower() or
                                "attn" in key.lower()
                            )
                            for key in keys
                        )
                        
                        # 检查Flux的pos_embed特征（更严格）
                        has_flux_pos_embed = any(
                            ("pos_embed" in key.lower() or "time_pos_embed" in key.lower()) and
                            ("transformer" in key.lower() or "time" in key.lower())
                            for key in keys
                        )
                        
                        # 检查Flux的rope特征
                        has_rope = any("rope" in key.lower() for key in keys)
                        
                        # 只有同时满足多个Flux特征才判断为Flux
                        flux_score = sum([has_transformer, has_flux_pos_embed, has_rope])
                        if flux_score >= 2:
                            return "flux"
                        
                        # 检查SDXL特征（即使输入通道是4，也可能有SDXL的其他特征）
                        has_sdxl_keys = any(
                            "conditioner" in key.lower() or
                            "text_encoder_2" in key.lower() or
                            "conditioner.embedders" in key.lower()
                            for key in keys
                        )
                        if has_sdxl_keys:
                            return "sdxl"
                        
                        # 默认SD15
                        return "sd15"
                    
                    # 如果无法确定输入通道数，使用其他特征
                    # 优先使用文件名提示
                    if filename_type:
                        return filename_type
                    
                    # 检查Flux特征（需要更严格的匹配）
                    has_transformer = any(
                        "transformer" in key.lower() and (
                            "blocks" in key.lower() or
                            "layers" in key.lower() or
                            "attn" in key.lower()
                        )
                        for key in keys
                    )
                    has_flux_pos_embed = any(
                        ("pos_embed" in key.lower() or "time_pos_embed" in key.lower()) and
                        ("transformer" in key.lower() or "time" in key.lower())
                        for key in keys
                    )
                    flux_score = sum([has_transformer, has_flux_pos_embed])
                    if flux_score >= 2:
                        return "flux"
                    
                    # 检查SDXL特征
                    has_sdxl_keys = any(
                        "conditioner" in key.lower() or
                        "text_encoder_2" in key.lower()
                        for key in keys
                    )
                    if has_sdxl_keys:
                        return "sdxl"
                    
                    # 检查是否有基本的UNet结构
                    has_unet = any("model.diffusion_model" in key or "unet" in key.lower() for key in keys)
                    if has_unet:
                        return "sd15"  # 默认SD15
                        
            except Exception as e:
                print(f"Warning: Failed to read safetensors file {file_path}: {e}")
                # 如果读取失败，使用文件名推断
                return filename_type
        
        # 方法2: 对于ckpt文件，使用文件名推断
        elif file_path.suffix == ".ckpt":
            return filename_type
            
    except Exception as e:
        print(f"Warning: Failed to detect model type from file {file_path}: {e}")
        return filename_type
    
    return filename_type


def _is_non_image_model(dir_path: Path) -> bool:
    """
    判断是否是非图片模型（文本模型、视频模型等）
    
    Args:
        dir_path: 目录路径
    
    Returns:
        True如果是非图片模型
    """
    dirname_lower = dir_path.name.lower()
    
    # 文本模型特征
    text_model_keywords = [
        "llama", "qwen", "mistral", "gpt", "gptq", "awq", "gguf",
        "instruct", "chat", "text", "language", "llm"
    ]
    if any(keyword in dirname_lower for keyword in text_model_keywords):
        # 排除Qwen-Image系列
        if "qwen-image" not in dirname_lower:
            return True
    
    # 视频模型特征
    video_model_keywords = [
        "video", "t2v", "i2v", "ti2v", "wan2", "hunyuanvideo",
        "framepack", "animate"
    ]
    if any(keyword in dirname_lower for keyword in video_model_keywords):
        return True
    
    # 其他非图片模型
    other_non_image_keywords = [
        "translation", "opus", "antelope", "buffalo", "gpt2", "promptist"
    ]
    if any(keyword in dirname_lower for keyword in other_non_image_keywords):
        return True
    
    return False


def detect_model_type_from_directory(dir_path: Path) -> Optional[str]:
    """
    从目录中检测模型类型（通过model_index.json或config.json）
    
    Args:
        dir_path: 模型目录路径
    
    Returns:
        模型类型: "sdxl", "sd15", "sd21", "flux", "qwen-image", 或 None（无法确定）
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    
    # 先检查是否是非图片模型
    if _is_non_image_model(dir_path):
        return None
    
    # 检查文件名特征
    filename_type = _detect_from_filename(dir_path)
    
    # 方法1: 查找model_index.json文件（diffusers格式）
    model_index_path = dir_path / "model_index.json"
    if model_index_path.exists():
        try:
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            detected = _detect_from_model_index(model_index, dir_path)
            if detected:
                return detected
            
            # 检查Qwen-Image
            if isinstance(model_index, dict):
                class_name = model_index.get("_class_name", "")
                if "qwen" in class_name.lower() and "image" in class_name.lower():
                    return "qwen-image"
                
                # 检查components中的Qwen特征
                components = model_index.get("components", {})
                if isinstance(components, dict):
                    if any("qwen" in k.lower() for k in components.keys()):
                        if "image" in dir_path.name.lower():
                            return "qwen-image"
                        
        except Exception as e:
            print(f"Warning: Failed to read model_index.json from {dir_path}: {e}")
    
    # 方法2: 查找config.json文件（备用方法）
    config_path = dir_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            detected = _detect_from_config(config)
            if detected:
                return detected
            
            # 检查Qwen-Image
            if isinstance(config, dict):
                model_type = config.get("model_type", "")
                if "qwen" in model_type.lower() and "image" in model_type.lower():
                    return "qwen-image"
                    
        except Exception as e:
            print(f"Warning: Failed to read config.json from {dir_path}: {e}")
    
    # 方法3: 检查目录结构特征（在文件名检测之后）
    # 如果文件名已经识别出特殊类型，直接返回
    if filename_type in ["z-image", "pony-v7", "qwen-image"]:
        return filename_type
    
    # SDXL通常有text_encoder_2目录
    text_encoder_2_path = dir_path / "text_encoder_2"
    if text_encoder_2_path.exists() and text_encoder_2_path.is_dir():
        return "sdxl"
    
    # Flux通常有transformer目录（但需要排除其他特殊模型）
    transformer_path = dir_path / "transformer"
    if transformer_path.exists() and transformer_path.is_dir():
        # 如果文件名提示是其他模型，优先相信文件名
        if filename_type and filename_type != "flux":
            return filename_type
        return "flux"
    
    # Qwen-Image特征
    if "qwen" in dir_path.name.lower() and "image" in dir_path.name.lower():
        return "qwen-image"
    
    # Z-Image特征
    if "z-image" in dir_path.name.lower() or "zimage" in dir_path.name.lower():
        return "z-image"
    
    # Pony-V7特征
    if "pony-v7" in dir_path.name.lower() or "ponyv7" in dir_path.name.lower() or "pony_v7" in dir_path.name.lower():
        return "pony-v7"
    
    # 如果文件名有提示，使用文件名
    if filename_type:
        return filename_type
    
    return None


def _detect_from_model_index(model_index: Dict[str, Any], dir_path: Optional[Path] = None) -> Optional[str]:
    """
    从model_index.json内容推断模型类型
    
    Args:
        model_index: model_index.json的内容（字典）
        dir_path: 目录路径（可选，用于辅助判断）
    
    Returns:
        模型类型字符串
    """
    if not isinstance(model_index, dict):
        return None
    
    # 方法1: 检查_class_name字段（最直接）
    class_name = model_index.get("_class_name", "")
    if isinstance(class_name, str):
        class_name_lower = class_name.lower()
        
        # 特殊模型类型
        if "z-image" in class_name_lower or "zimage" in class_name_lower:
            return "z-image"
        if "pony-v7" in class_name_lower or "ponyv7" in class_name_lower:
            return "pony-v7"
        if "qwen" in class_name_lower and "image" in class_name_lower:
            return "qwen-image"
        
        if "sdxl" in class_name_lower or "stable diffusion xl" in class_name_lower:
            return "sdxl"
        elif "flux" in class_name_lower:
            return "flux"
        elif "stable diffusion" in class_name_lower:
            # 检查版本
            if "v2" in class_name_lower or "2.1" in class_name_lower:
                return "sd21"
            else:
                return "sd15"
    
    # 方法2: 检查components字段
    components = model_index.get("components", {})
    if isinstance(components, dict):
        component_keys = list(components.keys())
        component_keys_lower = [k.lower() for k in component_keys]
        
        # 检查Z-Image特征
        if any("z-image" in k.lower() or "zimage" in k.lower() for k in component_keys):
            return "z-image"
        
        # 检查Pony-V7特征
        if any("pony-v7" in k.lower() or "ponyv7" in k.lower() for k in component_keys):
            return "pony-v7"
        
        # SDXL有两个text_encoder（text_encoder和text_encoder_2）
        text_encoder_count = sum(1 for k in component_keys_lower if "text_encoder" in k)
        if text_encoder_count >= 2:
            return "sdxl"
        
        # Flux有transformer组件（但需要更严格的检查，避免误判）
        # 检查是否有transformer且不是其他模型的特征
        has_transformer = any("transformer" in k for k in component_keys_lower)
        if has_transformer:
            # 进一步检查是否是Flux（Flux通常有特定的transformer结构）
            # 如果目录名提示是其他模型，优先相信目录名
            if dir_path:
                dirname_lower = dir_path.name.lower()
                if "z-image" in dirname_lower:
                    return "z-image"
                if "pony-v7" in dirname_lower or "ponyv7" in dirname_lower:
                    return "pony-v7"
            return "flux"
        
        # 检查tokenizer数量（SDXL有两个）
        tokenizer_count = sum(1 for k in component_keys_lower if "tokenizer" in k)
        if tokenizer_count >= 2:
            return "sdxl"
    
    # 方法3: 检查scheduler类型
    if isinstance(components, dict) and "scheduler" in components:
        scheduler = components.get("scheduler", {})
        if isinstance(scheduler, dict):
            scheduler_class = scheduler.get("_class_name", "")
            if isinstance(scheduler_class, str):
                scheduler_lower = scheduler_class.lower()
                if "flux" in scheduler_lower:
                    return "flux"
    
    # 方法4: 检查是否有特定的组件组合
    if isinstance(components, dict):
        # SDXL特有的组件
        has_text_encoder_2 = any("text_encoder_2" in k.lower() for k in components.keys())
        if has_text_encoder_2:
            return "sdxl"
    
    return None


def _detect_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    从config.json内容推断模型类型
    
    Args:
        config: config.json的内容（字典）
    
    Returns:
        模型类型字符串
    """
    if not isinstance(config, dict):
        return None
    
    # 检查模型架构
    architectures = config.get("_class_name", "") or config.get("architectures", [])
    
    if isinstance(architectures, list):
        architectures = " ".join(architectures)
    else:
        architectures = str(architectures)
    
    architectures_lower = architectures.lower()
    
    if "sdxl" in architectures_lower or "xl" in architectures_lower:
        return "sdxl"
    elif "flux" in architectures_lower:
        return "flux"
    elif "stable-diffusion" in architectures_lower:
        # 检查版本
        if "v2" in architectures_lower or "2.1" in architectures_lower:
            return "sd21"
        else:
            return "sd15"
    
    # 检查其他特征
    if "sample_size" in config:
        sample_size = config.get("sample_size", 512)
        if isinstance(sample_size, list):
            sample_size = sample_size[0] if sample_size else 512
        
        # SDXL通常是1024
        if sample_size >= 1024:
            return "sdxl"
    
    return None


def detect_model_type_from_file_keys(file_path: Path) -> Optional[str]:
    """
    通过读取文件权重键值进行分类（仅基于文件内容，不依赖文件名）
    
    Args:
        file_path: 模型文件路径
    
    Returns:
        模型类型: "sdxl", "sd15", "sd21", "flux", "z-image", "pony-v7", 或 None
    """
    if not file_path.exists() or not file_path.is_file():
        return None
    
    # 只处理safetensors文件（ckpt文件需要特殊处理，暂时跳过）
    if file_path.suffix != ".safetensors":
        return None
    
    if not SAFETENSORS_AVAILABLE:
        return None
    
    try:
        with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            if not keys:
                return None
            
            # 统计不同类型的键
            key_stats = {
                "unet": [],
                "text_encoder": [],
                "text_encoder_2": [],
                "vae": [],
                "transformer": [],
                "conditioner": [],
                "controlnet": [],
                "double_blocks": [],  # Flux特有特征（double_blocks结构）
                "single_transformer_blocks": [],  # Flux特有特征（single_transformer_blocks结构）
                "context_embedder": [],  # Flux特有特征
                "text_encoders": [],   # Flux使用text_encoders
                "cond_stage_model": [],  # SD15/SD21使用cond_stage_model
                "input_blocks": [],    # SD15/SD21的UNet结构
            }
            
            # 分类键
            for key in keys:
                key_lower = key.lower()
                if "double_blocks" in key:
                    key_stats["double_blocks"].append(key)
                elif "single_transformer_blocks" in key:
                    key_stats["single_transformer_blocks"].append(key)
                elif "context_embedder" in key:
                    key_stats["context_embedder"].append(key)
                elif "text_encoders" in key:
                    key_stats["text_encoders"].append(key)
                elif "cond_stage_model" in key:
                    key_stats["cond_stage_model"].append(key)
                elif "input_blocks" in key:
                    key_stats["input_blocks"].append(key)
                elif "model.diffusion_model" in key or "unet" in key_lower:
                    key_stats["unet"].append(key)
                elif "text_encoder" in key_lower:
                    if "text_encoder_2" in key_lower or "text_encoder2" in key_lower:
                        key_stats["text_encoder_2"].append(key)
                    else:
                        key_stats["text_encoder"].append(key)
                elif "vae" in key_lower:
                    key_stats["vae"].append(key)
                elif "transformer" in key_lower:
                    key_stats["transformer"].append(key)
                elif "conditioner" in key_lower:
                    key_stats["conditioner"].append(key)
                elif "controlnet" in key_lower:
                    key_stats["controlnet"].append(key)
            
            # 分析模型类型
            
            # 1. SDXL特征：有两个text_encoder或有conditioner
            if len(key_stats["text_encoder_2"]) > 0 or len(key_stats["conditioner"]) > 0:
                return "sdxl"
            
            # 2. Flux特征：检查double_blocks（Flux最独特的特征）
            if len(key_stats["double_blocks"]) > 0:
                return "flux"
            
            # 3. Flux特征：检查single_transformer_blocks和context_embedder（Flux的另一种结构）
            if len(key_stats["single_transformer_blocks"]) > 0 and len(key_stats["context_embedder"]) > 0:
                # 检查是否是nunchaku变体
                nunchaku_keys = [k for k in keys if "nunchaku" in k.lower()]
                if len(nunchaku_keys) > 0:
                    return "flux-nunchaku"
                return "flux"
            
            # 4. Flux特征：检查text_encoders（Flux使用text_encoders，SD15使用cond_stage_model）
            if len(key_stats["text_encoders"]) > 0 and len(key_stats["cond_stage_model"]) == 0:
                return "flux"
            
            # 5. SD15特征：有input_blocks和cond_stage_model
            if len(key_stats["input_blocks"]) > 0 and len(key_stats["cond_stage_model"]) > 0:
                # 检查UNet输入通道数确认
                unet_input_channels = None
                for key in key_stats["input_blocks"]:
                    if "input_blocks.0.0.weight" in key or "model.diffusion_model.input_blocks.0.0.weight" in key:
                        try:
                            weight = f.get_tensor(key)
                            if len(weight.shape) >= 2:
                                unet_input_channels = weight.shape[0]
                                break
                        except:
                            pass
                
                if unet_input_channels == 9:
                    return "sdxl"
                elif unet_input_channels == 4:
                    return "sd15"
                else:
                    # 即使无法确定输入通道，有input_blocks和cond_stage_model也倾向于SD15
                    return "sd15"
            
            # 6. 检查UNet输入通道数（备用方法）
            unet_input_channels = None
            for key in key_stats["unet"]:
                if "input_blocks.0.0.weight" in key or "model.diffusion_model.input_blocks.0.0.weight" in key:
                    try:
                        weight = f.get_tensor(key)
                        if len(weight.shape) >= 2:
                            unet_input_channels = weight.shape[0]
                            break
                    except:
                        pass
            
            if unet_input_channels == 9:
                return "sdxl"
            elif unet_input_channels == 4:
                # 如果输入通道是4，但没有明确的Flux特征，默认SD15
                return "sd15"
            
            # 7. 如果无法确定输入通道数，使用其他特征
            # 检查是否有基本的UNet结构
            if len(key_stats["unet"]) > 0:
                return "sd15"  # 默认SD15
            
    except Exception as e:
        print(f"Warning: Failed to analyze file keys from {file_path}: {e}")
        return None
    
    return None


def detect_model_type_from_json_config(dir_path: Path) -> Optional[str]:
    """
    通过读取JSON配置文件进行分类（仅基于JSON内容，不依赖目录名或目录结构）
    
    Args:
        dir_path: 模型目录路径
    
    Returns:
        模型类型: "sdxl", "sd15", "sd21", "flux", "z-image", "pony-v7", "qwen-image", 或 None
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return None
    
    # 先检查是否是非图片模型（视频模型等）
    if _is_non_image_model(dir_path):
        return None
    
    # 方法1: 读取model_index.json（优先级最高）
    model_index_path = dir_path / "model_index.json"
    if model_index_path.exists():
        try:
            with open(model_index_path, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            if isinstance(model_index, dict):
                # 检查_class_name
                class_name = model_index.get("_class_name", "")
                if isinstance(class_name, str):
                    class_name_lower = class_name.lower()
                    
                    # 检查是否是视频模型（优先检查，避免误判）
                    video_keywords = ["video", "t2v", "i2v", "ti2v", "wan", "hunyuanvideo"]
                    if any(keyword in class_name_lower for keyword in video_keywords):
                        return None  # 视频模型，不返回类型
                    
                    # 特殊模型类型
                    if "z-image" in class_name_lower or "zimage" in class_name_lower:
                        return "z-image"
                    if "pony-v7" in class_name_lower or "ponyv7" in class_name_lower:
                        return "pony-v7"
                    if "qwen" in class_name_lower and "image" in class_name_lower:
                        return "qwen-image"
                    
                    if "sdxl" in class_name_lower or "stable diffusion xl" in class_name_lower:
                        return "sdxl"
                    if "flux" in class_name_lower:
                        return "flux"
                    if "stable diffusion" in class_name_lower:
                        if "v2" in class_name_lower or "2.1" in class_name_lower:
                            return "sd21"
                        return "sd15"
                
                # 详细分析components
                components = model_index.get("components", {})
                if isinstance(components, dict):
                    component_keys = list(components.keys())
                    component_keys_lower = [k.lower() for k in component_keys]
                    
                    # 检查是否是视频模型组件
                    video_component_keywords = ["video", "t2v", "i2v", "ti2v", "wan", "hunyuan"]
                    if any(any(keyword in k.lower() for keyword in video_component_keywords) for k in component_keys):
                        return None  # 视频模型，不返回类型
                    
                    # 检查特殊模型类型的组件
                    if any("z-image" in k.lower() or "zimage" in k.lower() for k in component_keys):
                        return "z-image"
                    if any("pony-v7" in k.lower() or "ponyv7" in k.lower() for k in component_keys):
                        return "pony-v7"
                    if any("qwen" in k.lower() for k in component_keys) and any("image" in k.lower() for k in component_keys):
                        return "qwen-image"
                    
                    # 统计组件
                    has_text_encoder_2 = any("text_encoder_2" in k.lower() for k in component_keys)
                    has_transformer = any("transformer" in k.lower() for k in component_keys)
                    has_conditioner = any("conditioner" in k.lower() for k in component_keys)
                    
                    # SDXL: 有两个text_encoder或conditioner
                    if has_text_encoder_2 or has_conditioner:
                        return "sdxl"
                    
                    # Flux: 有transformer组件（但需要排除视频模型）
                    if has_transformer:
                        # 检查transformer的详细配置
                        transformer_comp = components.get("transformer", {})
                        if isinstance(transformer_comp, dict):
                            transformer_class = transformer_comp.get("_class_name", "")
                            transformer_class_lower = transformer_class.lower()
                            
                            # 再次检查是否是视频模型
                            if any(keyword in transformer_class_lower for keyword in video_keywords):
                                return None  # 视频模型
                            
                            if "flux" in transformer_class_lower:
                                return "flux"
                        # 如果没有明确的视频模型特征，才判断为flux
                        return "flux"
                    
                    # 检查tokenizer数量
                    tokenizer_count = sum(1 for k in component_keys_lower if "tokenizer" in k)
                    if tokenizer_count >= 2:
                        return "sdxl"
                    
        except Exception as e:
            print(f"Warning: Failed to read model_index.json from {dir_path}: {e}")
    
    # 方法2: 读取各个组件的config.json
    config_files = [
        ("unet", dir_path / "unet" / "config.json"),
        ("transformer", dir_path / "transformer" / "config.json"),
        ("text_encoder", dir_path / "text_encoder" / "config.json"),
        ("text_encoder_2", dir_path / "text_encoder_2" / "config.json"),
    ]
    
    for component_name, config_path in config_files:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if isinstance(config, dict):
                    # 检查模型架构
                    architectures = config.get("architectures", [])
                    if isinstance(architectures, list):
                        architectures_str = " ".join(architectures).lower()
                    else:
                        architectures_str = str(architectures).lower()
                    
                    # 检查_class_name
                    class_name = config.get("_class_name", "")
                    if isinstance(class_name, str):
                        class_name_lower = class_name.lower()
                        
                        if "z-image" in class_name_lower or "zimage" in class_name_lower:
                            return "z-image"
                        if "pony-v7" in class_name_lower or "ponyv7" in class_name_lower:
                            return "pony-v7"
                        if "flux" in class_name_lower:
                            return "flux"
                        if "sdxl" in class_name_lower or "xl" in class_name_lower:
                            return "sdxl"
                    
                    # 检查架构特征
                    if "flux" in architectures_str:
                        return "flux"
                    if "sdxl" in architectures_str or "xl" in architectures_str:
                        return "sdxl"
                    
                    # 检查transformer组件的特定特征
                    if component_name == "transformer":
                        # Flux的transformer有特定配置
                        if "num_layers" in config or "num_heads" in config:
                            # 检查是否是Flux的特征
                            if "rope" in str(config).lower() or "pos_embed" in str(config).lower():
                                return "flux"
                    
                    # 检查text_encoder_2的存在（SDXL特征）
                    if component_name == "text_encoder_2":
                        return "sdxl"
                    
            except Exception as e:
                print(f"Warning: Failed to read config.json from {config_path}: {e}")
    
    # 方法3: 读取主config.json
    main_config_path = dir_path / "config.json"
    if main_config_path.exists():
        try:
            with open(main_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            detected = _detect_from_config(config)
            if detected:
                return detected
            
            # 检查特殊模型类型
            if isinstance(config, dict):
                model_type = config.get("model_type", "")
                class_name = config.get("_class_name", "")
                
                combined = (model_type + " " + class_name).lower()
                
                if "z-image" in combined or "zimage" in combined:
                    return "z-image"
                if "pony-v7" in combined or "ponyv7" in combined:
                    return "pony-v7"
                if "qwen" in combined and "image" in combined:
                    return "qwen-image"
                    
        except Exception as e:
            print(f"Warning: Failed to read main config.json from {dir_path}: {e}")
    
    return None


def format_output(item_type: str, path: Path, base_path: Path, model_type: Optional[str] = None) -> str:
    """
    格式化输出行
    
    Args:
        item_type: 类型（"目录"或"文件"）
        path: 项目路径
        base_path: 基础路径（用于计算相对路径）
        model_type: 检测到的模型类型（可选）
    
    Returns:
        格式化的字符串，如果model_type为None且是目录，可能返回None（表示跳过）
    """
    # 计算相对路径
    try:
        relative_path = path.relative_to(base_path)
    except ValueError:
        relative_path = path
    
    result = f"{item_type} {relative_path}"
    
    if model_type:
        result += f" [{model_type}]"
    
    return result


def debug_file_keys(file_path: Path, max_keys: int = 50):
    """
    调试函数：输出文件的键名（用于分析）
    
    Args:
        file_path: 模型文件路径
        max_keys: 最多输出的键数量
    """
    if not file_path.exists() or not file_path.is_file():
        print(f"File does not exist: {file_path}")
        return
    
    if file_path.suffix != ".safetensors":
        print(f"Not a safetensors file: {file_path}")
        return
    
    if not SAFETENSORS_AVAILABLE:
        print("safetensors library not available")
        return
    
    try:
        with safetensors.torch.safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"\n=== Keys in {file_path.name} (total: {len(keys)}) ===")
            
            # 分类键
            transformer_keys = [k for k in keys if "transformer" in k.lower()]
            unet_keys = [k for k in keys if "model.diffusion_model" in k or "unet" in k.lower()]
            text_encoder_keys = [k for k in keys if "text_encoder" in k.lower()]
            
            print(f"\nTransformer keys ({len(transformer_keys)}):")
            for k in transformer_keys[:max_keys]:
                print(f"  {k}")
            if len(transformer_keys) > max_keys:
                print(f"  ... and {len(transformer_keys) - max_keys} more")
            
            print(f"\nUNet keys (first {min(max_keys, len(unet_keys))}):")
            for k in unet_keys[:max_keys]:
                print(f"  {k}")
            if len(unet_keys) > max_keys:
                print(f"  ... and {len(unet_keys) - max_keys} more")
            
            print(f"\nText Encoder keys ({len(text_encoder_keys)}):")
            for k in text_encoder_keys[:max_keys]:
                print(f"  {k}")
            
            # 检查输入通道
            for key in unet_keys:
                if "input_blocks.0.0.weight" in key or "model.diffusion_model.input_blocks.0.0.weight" in key:
                    try:
                        weight = f.get_tensor(key)
                        if len(weight.shape) >= 2:
                            print(f"\nUNet input channels: {weight.shape[0]} (from key: {key})")
                            break
                    except:
                        pass
            
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Detect image generation model types")
    parser.add_argument("directory", type=str, nargs="?", help="Directory to scan")
    parser.add_argument("--detect", action="store_true", 
                       help="Enable model type detection (slower but more informative)")
    parser.add_argument("--deep", action="store_true",
                       help="Use deep detection method (analyze file keys and JSON configs only, no filename/dirname heuristics)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Scan recursively")
    parser.add_argument("--debug-file", type=str,
                       help="Debug: print keys from a specific safetensors file")
    args = parser.parse_args()
    
    # 调试模式：输出文件的键名
    if args.debug_file:
        debug_file_keys(Path(args.debug_file))
        return
    
    if not args.directory:
        parser.print_help()
        return
    
    base_path = Path(args.directory).resolve()
    
    if not base_path.exists():
        print(f"Error: Directory does not exist: {base_path}")
        return
    
    print(f"Scanning directory: {base_path}")
    if args.deep:
        print("Using deep detection method (analyzing file keys and JSON configs only)")
    print("=" * 80)
    
    # 扫描目录
    if args.recursive:
        # 递归扫描
        items = []
        for item in base_path.rglob("*"):
            if item.is_dir():
                items.append(("目录", item))
            elif item.is_file() and item.suffix in [".safetensors", ".ckpt"]:
                items.append(("文件", item))
        items.sort(key=lambda x: str(x[1]))
    else:
        # 只扫描一级
        items = scan_directory(base_path)
    
    # 输出结果
    for item_type, item_path in items:
        model_type = None
        
        if args.detect:
            if args.deep:
                # 使用深度检测方法（仅基于文件内容和JSON配置）
                if item_type == "目录":
                    model_type = detect_model_type_from_json_config(item_path)
                    # 如果是非图片模型，跳过输出
                    if model_type is None and _is_non_image_model(item_path):
                        continue
                elif item_type == "文件":
                    model_type = detect_model_type_from_file_keys(item_path)
                    # 只输出图片模型文件
                    if model_type is None and item_path.suffix not in [".safetensors", ".ckpt"]:
                        continue
            else:
                # 使用常规检测方法（包含文件名和目录名启发式规则）
                if item_type == "目录":
                    model_type = detect_model_type_from_directory(item_path)
                    # 如果是非图片模型，跳过输出
                    if model_type is None and _is_non_image_model(item_path):
                        continue
                elif item_type == "文件":
                    model_type = detect_model_type_from_file(item_path)
                    # 只输出图片模型文件
                    if model_type is None and item_path.suffix not in [".safetensors", ".ckpt"]:
                        continue
        
        output_line = format_output(item_type, item_path, base_path, model_type)
        print(output_line)
    
    print("=" * 80)
    print(f"Total: {len(items)} items")


if __name__ == "__main__":
    main()


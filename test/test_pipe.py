import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline
from typing import Any
import argparse
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def _is_single_file(model_path: str) -> bool:
    """
    判断模型路径是单个文件还是目录
    
    Args:
        model_path: 模型路径
    
    Returns:
        True如果是单个文件，False如果是目录
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    return path.is_file()


def _load_sdxl_model(model_path: str, device: str) -> Any:
    """
    加载SDXL模型
    支持两种格式：
    1. diffusers格式（目录结构）
    2. 单个checkpoint/safetensors文件
    """
    try:
        is_single_file = _is_single_file(model_path)
        
        if is_single_file:
            print(f"Loading SDXL model from single file: {model_path}")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        else:
            print(f"Loading SDXL model from directory: {model_path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
        )
        
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()  # 优化内存使用
        
        return pipe
    except ImportError:
        raise Exception("diffusers library not installed")
    except Exception as e:
        raise Exception(f"Failed to load SDXL model: {e}")


def _load_sd15_model(model_path: str, device: str) -> Any:
    """
    加载Stable Diffusion 1.5模型
    支持两种格式：
    1. diffusers格式（目录结构）
    2. 单个checkpoint/safetensors文件
    """
    try:
        is_single_file = _is_single_file(model_path)
        
        if is_single_file:
            print(f"Loading SD 1.5 model from single file: {model_path}")
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        else:
            print(f"Loading SD 1.5 model from directory: {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()
        
        return pipe
    except ImportError:
        raise Exception("diffusers library not installed")
    except Exception as e:
        raise Exception(f"Failed to load SD 1.5 model: {e}")


def _load_flux_model(model_path: str, device: str) -> Any:
    """
    加载Flux模型
    支持两种格式：
    1. diffusers格式（目录结构）
    2. 单个checkpoint/safetensors文件
    """
    try:
        is_single_file = _is_single_file(model_path)
        
        if is_single_file:
            print(f"Loading Flux model from single file: {model_path}")
            # Flux模型使用DiffusionPipeline.from_single_file
            pipe = DiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        else:
            print(f"Loading Flux model from directory: {model_path}")
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()
        
        return pipe
    except ImportError:
        raise Exception("diffusers library not installed")
    except Exception as e:
        raise Exception(f"Failed to load Flux model: {e}")

# 通过命令行参数选择模型，和模型类别，加载模型
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test image generation models")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Model path (can be a directory for diffusers format or a single .safetensors/.ckpt file)")
    parser.add_argument("--family", type=str, required=True, 
                       choices=["sdxl", "sd15", "flux"],
                       help="Model class: sdxl, sd15, or flux")
    args = parser.parse_args()
    
    try:
        if args.family == "sdxl":
            pipe = _load_sdxl_model(args.model_path, device)
        elif args.family == "sd15":
            pipe = _load_sd15_model(args.model_path, device)
        elif args.family == "flux":
            pipe = _load_flux_model(args.model_path, device)
        else:
            print("Invalid model class")
            exit(1)
            
            print("\n" + "=" * 50)
            print("Model loaded successfully!")
            print("=" * 50)
            print(f"Model type: {type(pipe).__name__}")
            print(f"Device: {device}")
            
            # 显示模型信息
            if hasattr(pipe, 'unet'):
                print(f"UNet: {type(pipe.unet).__name__}")
            if hasattr(pipe, 'text_encoder'):
                print(f"Text Encoder: {type(pipe.text_encoder).__name__}")
            if hasattr(pipe, 'vae'):
                print(f"VAE: {type(pipe.vae).__name__}")
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

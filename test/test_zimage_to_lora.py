# inference/diffsynth

from inference.diffsynth.pipelines.z_image import (
    ZImagePipeline, ModelConfig,
    ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
)

model_zimage = "/home/tonera/models/Z-Image"
model_zimage_turbo = "/home/tonera/models/Z-Image-Turbo"
model_encoders = "/home/tonera/models/General-Image-Encoders"
model_zimage_i2l = "/home/tonera/models/Z-Image-i2L"

import glob
import os
from safetensors.torch import save_file
import torch
from PIL import Image

# Use `vram_config` to enable LoRA hot-loading
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# diffsynth 的 ModelManager 不会自动展开 path 里的通配符（*）。
# 必须先用 glob 展开成“真实文件路径列表”，否则会把 state_dict 置为 None 触发 TypeError。
transformer_files = sorted(glob.glob(os.path.join(model_zimage, "transformer", "*.safetensors")))
text_encoder_files = sorted(glob.glob(os.path.join(model_zimage_turbo, "text_encoder", "*.safetensors")))
vae_files = sorted(glob.glob(os.path.join(model_zimage_turbo, "vae", "diffusion_pytorch_model*.safetensors")))

siglip_files = sorted(glob.glob(os.path.join(model_encoders, "SigLIP2-G384", "model*.safetensors")))
dino_files = sorted(glob.glob(os.path.join(model_encoders, "DINOv3-7B", "model*.safetensors")))

i2l_files = sorted(glob.glob(os.path.join(model_zimage_i2l, "model*.safetensors")))

if not transformer_files:
    raise FileNotFoundError(f"未找到 Z-Image transformer 权重: {model_zimage}/transformer/*.safetensors")
if not text_encoder_files:
    raise FileNotFoundError(f"未找到 Z-Image-Turbo text_encoder 权重: {model_zimage_turbo}/text_encoder/*.safetensors")
if not vae_files:
    raise FileNotFoundError(f"未找到 Z-Image-Turbo vae 权重: {model_zimage_turbo}/vae/diffusion_pytorch_model*.safetensors")
if not siglip_files:
    raise FileNotFoundError(f"未找到 General-Image-Encoders SigLIP2 权重: {model_encoders}/SigLIP2-G384/model*.safetensors")
if not dino_files:
    raise FileNotFoundError(f"未找到 General-Image-Encoders DINOv3 权重: {model_encoders}/DINOv3-7B/model*.safetensors")
if not i2l_files:
    raise FileNotFoundError(f"未找到 Z-Image-i2L 权重: {model_zimage_i2l}/model*.safetensors")

# Load models
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 本地模型文件不要用 model_id=（那是用来下载的）；请用 path=
        ModelConfig(path=transformer_files, **vram_config),
        ModelConfig(path=text_encoder_files[0] if len(text_encoder_files) == 1 else text_encoder_files),
        ModelConfig(path=vae_files[0] if len(vae_files) == 1 else vae_files),
        ModelConfig(path=siglip_files[0] if len(siglip_files) == 1 else siglip_files),
        ModelConfig(path=dino_files[0] if len(dino_files) == 1 else dino_files),
        ModelConfig(path=i2l_files[0] if len(i2l_files) == 1 else i2l_files),
    ],
    tokenizer_config=ModelConfig(path=os.path.join(model_zimage_turbo, "tokenizer")),
)

# Load images（离线可跑：直接生成几张简单图片作为 i2L 输入）
images = [Image.open(f"/home/tonera/sakimi/{i+1}.jpg") for i in range(5)]

# Image to LoRA
with torch.no_grad():
    embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
    lora = ZImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]
save_file(lora, "lora.safetensors")

# Generate images
prompt = "a girl"
negative_prompt = "泛黄，发绿，模糊，低分辨率，低质量图像，扭曲的肢体，诡异的外观，丑陋，AI感，噪点，网格感，JPEG压缩条纹，异常的肢体，水印，乱码，意义不明的字符"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=0, cfg_scale=4, num_inference_steps=50,
    positive_only_lora=lora,
    sigma_shift=8
)
image.save("/home/tonera/website/output/image.jpg")
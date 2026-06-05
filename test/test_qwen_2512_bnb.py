import torch
from diffusers import DiffusionPipeline,QwenImagePipeline

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision

from inference.image.runtime import configure_hybrid_offload, enable_qwenimage_nunchaku_compat

model_dir = "/home/tonera/models"
model_name = f"{model_dir}/Qwen-Image-2512-unsloth-bnb-4bit"
torch_dtype = torch.bfloat16
device = "cuda" 

transformer_path = f"{model_dir}/Nunchaku-Qwen-Image-2512/nunchaku_qwen_image_2512_balance_{get_precision()}.safetensors"
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_path)

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    transformer=transformer
)
# pipe.enable_sequential_cpu_offload()

# uncomment if you run out of memory
# pipe.enable_model_cpu_offload() 


aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
width, height = aspect_ratios["16:9"]

prompt = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, "
    "with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. "
    "She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit "
    "in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, "
    "posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: "
    "unpretentious composition, yet brimming with vivid, fresh, youthful charm."
)
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
generator = torch.Generator(device=("cpu" if device == "cuda" else "cpu")).manual_seed(42)

enable_qwenimage_nunchaku_compat(pipe, transformer, prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, latent_scale_factor=16)
configure_hybrid_offload(pipe, transformer, device=device, enable_vae_tiling=True)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=20,
    true_cfg_scale=4.0,
    generator=generator,
).images[0]


image.save("/home/tonera/website/output/qwen-image-2512-sdnq-uint4-svd-r32.png")
# bnb 37s
# nunchaku 16s
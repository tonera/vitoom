import os
import torch
from PIL import Image
from modelscope import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
from inference.image.runtime import configure_hybrid_offload, enable_qwenimage_nunchaku_compat



model_dir = "/home/tonera/models"
model_name = f"{model_dir}/Qwen-Image-Edit-2511"
device = "cuda" 
transformer_path = f"{model_dir}/Nunchaku-Qwen-Image-EDIT-2511/nunchaku_qwen_image_2511_balance_{get_precision()}.safetensors"
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_path)


pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16, transformer=transformer)

print("pipeline loaded")
# pipeline.enable_sequential_cpu_offload()
# pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = load_image("http://192.168.0.107:8080/zimage_nunchaku_test.png")
# image2 = load_image("http://192.168.0.107:8080/test_z_image.png")
prompt = "the  girls is crying"
negative_prompt = ""
width=1024
height=1024
enable_qwenimage_nunchaku_compat(pipeline, transformer, prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, latent_scale_factor=16)
configure_hybrid_offload(pipeline, transformer, device=device, enable_vae_tiling=True)



inputs = {
    "image": [image1,],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": negative_prompt,
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "width": width,
    "height": height,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("/home/tonera/website/output/output_image_edit_2511.png")
    print("image saved at", os.path.abspath("/home/tonera/website/output/output_image_edit_2511.png"))

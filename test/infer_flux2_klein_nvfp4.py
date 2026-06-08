import os
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel,Flux2Pipeline

model_dir = "/home/tonera/models"
base_model = f"{model_dir}/FLUX.2-dev"
model_path = f"{model_dir}/FLUX.2-dev-NVFP4/flux2-dev-nvfp4.safetensors"
prompt = "A cat holding a sign that says hello world"
height, width, guidance_scale, steps, seed = 1024, 1024, 4.0, 4, 0
dtype = torch.bfloat16

pipe = Flux2Pipeline.from_single_file(
    f"{model_path}", 
    torch_dtype=dtype, 
)

pipe.to("cuda")
img = pipe(
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=steps,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
output = "output/flux2_klein.png"
os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
img.save(output)
print(output)
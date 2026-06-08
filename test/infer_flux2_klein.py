import os
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from nunchaku import NunchakuFlux2Transformer2DModel
from nunchaku import NunchakuQwenEncoderModel


model_dir = "/home/tonera/models"
model_path = f"{model_dir}/FLUX.2-klein-9B-Nunchaku"
prompt = "A cat holding a sign that says hello world"
prompt = "aztodio,living room,1girl,perfect face,brown long hair,wide hips,narrow waist,blush,nude,red sofa.lying the sofa,unconscious, fainted,1 boy,,big penis,rough sex,deep insertion ,wet pussy, pussy juice,hands on waist,impact lines,cum, cum in pussy, cum on body,pull out,grab breasts"
height, width, guidance_scale, steps, seed = 1024, 1024, 4.0, 4, 0
dtype = torch.bfloat16

transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
    f"/home/tonera/models/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
    torch_dtype=dtype,
)
text_encoder = NunchakuQwenEncoderModel.from_pretrained(
    f"/home/tonera/weights/Qwen3-text-Nunchaku/svdq-int4-Qwen3-text-Nunchaku.safetensors",
)
pipe = Flux2KleinPipeline.from_pretrained(
    f"{model_path}", 
    torch_dtype=dtype, 
    transformer=transformer,
    text_encoder=text_encoder,
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
output = "/home/tonera/website/output/flux2_klein_tt.png"
os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
img.save(output)
print(output)
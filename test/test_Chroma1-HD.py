import torch
from diffusers import ChromaPipeline

model_id = "lodestones/Chroma1-HD"
ckpt_path = "https://huggingface.co/lodestones/Chroma1-HD/blob/main/Chroma1-HD.safetensors"
transformer = ChromaTransformer2DModel.from_single_file(ckpt_path, torch_dtype=torch.bfloat16)
pipe = ChromaPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt = [
    "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
]
image = pipe(prompt, negative_prompt=negative_prompt).images[0]
image.save("chroma.png")
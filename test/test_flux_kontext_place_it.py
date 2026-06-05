import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision
from inference.common.teacache import teacache_forward,set_tea_cache

num_inference_steps = 30

# model_dir = "/home/tonera/models"
# weight_dir = "/home/tonera/weights"

model_dir = "/home/tonera/models"
weight_dir = "/home/tonera/weights"
loras_dir = "/home/tonera/loras"

transformer_path = f"{weight_dir}/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
text_encoder_2_path = f"{weight_dir}/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(text_encoder_2_path)


# test a
# NunchakuFluxTransformer2dModel.forward = teacache_forward
# transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_path)
# pipeline = FluxKontextPipeline.from_pretrained(
#     f"{model_dir}/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16, text_encoder_2=text_encoder_2
# ).to("cuda")
# set_tea_cache(pipeline.transformer,num_inference_steps)
# transformer.update_lora_params(
#     # f"{loras_dir}/place_it.safetensors"
#     f"{loras_dir}/kontext_big_breasts_and_butts.safetensors"
# )
# transformer.set_lora_strength(1) 

# test b
pipeline = FluxKontextPipeline.from_pretrained(
    f"{model_dir}/FLUX.1-Kontext-dev",  torch_dtype=torch.bfloat16, text_encoder_2=text_encoder_2
).to("cuda")
set_tea_cache(pipeline.transformer,num_inference_steps)


image = load_image(
    "http://192.168.0.105:8888/outputs/2026/01/29/eec185add2a94e75a187b9168355d464_0.jpeg"
).convert("RGB")

prompt = "a full body photograph of a beautiful woman "
image = pipeline(image=image, prompt=prompt, guidance_scale=2.5,num_inference_steps=num_inference_steps,).images[0]
image.save("/home/tonera/website/output/flux-kontext-dev-place_it.png")
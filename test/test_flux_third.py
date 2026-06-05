""" 第三方flux不量化+pretouch_pipeline_cpu_tensors加速+ pin_memory=True """

import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel

from diffusers import FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
import time
from inference.common.monitor import print_gpu_memory_usage
from inference.common.sdpa_utils import sdpa_ctx
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

from safetensors.torch import load_file 
# config='weights/svdq-int4-flux.1-dev/config.json'

start_time = time.time()
print_gpu_memory_usage("开始") 
model_dir = "/home/tonera/models"
weight_dir = "/home/tonera/weights"
output_dir = "/home/tonera/project/vitoom/resources/outputs"
base_model = f'{model_dir}/pixelwave_flux1_dev_bf16_03.safetensors'
# base_model = "models/flux.1-dev-SRPO-bf16.safetensors"

transformer = FluxTransformer2DModel.from_single_file(
    base_model,
    torch_dtype=torch.bfloat16,
    # offload=True,
    config=f"{model_dir}/FLUX.1-dev/transformer/config.json",
    pin_memory=True
)
print_gpu_memory_usage("量化")
# 量化 transformer
# quantize_(transformer, int8_weight_only())
 
text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(f"{weight_dir}/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",pin_memory=True)
print_gpu_memory_usage("加载text_encoder_2后")
# "models/FLUX.1-dev-bnb-4bit",
pipe = FluxPipeline.from_pretrained(
    f"{model_dir}/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    text_encoder_2=text_encoder_2,
    transformer=transformer,
)
print_gpu_memory_usage("创建pipeline后")
pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name}加速"))

pipe.to("cuda")
print_gpu_memory_usage("移动到GPU后")

# pipe.enable_model_cpu_offload()

num_inference_steps=20
prompt = "night,1girl is sitting on a car hood,dynamic pose, wide hips,narrow waist,very beauty,big breasts, cleavage"
print_gpu_memory_usage("推理前")


image = pipe(
    prompt,
    negative_prompt="lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry, logo",
    guidance_scale=5,
    num_inference_steps=num_inference_steps,
    width=1024, height=1024,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

print_gpu_memory_usage("推理后")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

image.save(f"{output_dir}/flux_dev_third.png")
print_gpu_memory_usage("保存后")

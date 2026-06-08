""" flux pretouch_pipeline_cpu_tensors加速+ pin_memory=True  disable_mmap不需要开"""
import torch
from diffusers import FluxPipeline
from datetime import datetime
from nunchaku import NunchakuFluxTransformer2dModel,NunchakuT5EncoderModel
from nunchaku.utils import get_precision
from inference.common.teacache import teacache_forward,set_tea_cache
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
from inference.common.sdpa_utils import sdpa_ctx
import time
model_dir = "/home/tonera/models"
weight_dir = "/home/tonera/weights"
output_dir = "/home/tonera/website/output"
num_inference_steps = 30

def _precondition_pipeline_before_to_cuda(pipe) -> None:
    """测试用：在 `pipe.to("cuda")` 前做 CPU 侧 pretouch，以对比迁移耗时。"""
    pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name}加速"))

start_time = time.time()
# 输出当前时间 
print(f"Start:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
# NunchakuFluxTransformer2dModel.forward = teacache_forward
# # disable_mmap=True 在nuchaku上有点用，大约省500ms到1s,但是在其他模型上全部比不用差
# transformer = NunchakuFluxTransformer2dModel.from_pretrained(
#     f"{weight_dir}/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
#     torch_dtype=torch.bfloat16,
#     device=torch.device("cuda"),
#     # disable_mmap=True,
# )
# transformer.set_attention_impl("nunchaku-fp16")
# set_tea_cache(transformer,num_inference_steps)
print(f"Load text_encoder_2:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
    f"{weight_dir}/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",
    device=torch.device("cuda"),
    torch_dtype=torch.bfloat16,
    pin_memory=True,  
    # disable_mmap=True,
)

print(f"Load transformer:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# 这个正常
# pipeline = FluxPipeline.from_pretrained(
#     f"{model_dir}/FLUX.1-dev", transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16
# )

# 这个会报错
pipeline = FluxPipeline.from_pretrained(
    f"{model_dir}/FLUX.1-dev",  text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16
)
set_tea_cache(pipeline.transformer,num_inference_steps)
# pipeline = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
# pipeline = FluxPipeline.from_pretrained(f"{model_dir}/FLUX.1-dev",torch_dtype=torch.bfloat16)

print(f"Load pipeline:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# _precondition_pipeline_before_to_cuda(pipeline)
# pipeline.to("cuda")
pipeline.enable_model_cpu_offload()

print(f"To cuda:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Generate image:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

prompt = "A cat holding a sign that says hello world"
with sdpa_ctx():
    image = pipeline(
        prompt,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        width=1024, height=1024,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

print(f"Save image:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
image.save(f"{output_dir}/flux.1-dev-{precision}.png")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

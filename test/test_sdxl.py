import torch
from diffusers import StableDiffusionXLPipeline
import time
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
from inference.common.fbcache_sdxl import apply_cache_on_pipe

start_time = time.time()
num_inference_steps = 30
model_dir = "/home/tonera/models"
weight_dir = "/home/tonera/weights"


pipeline = StableDiffusionXLPipeline.from_pretrained(
    "resources/models/xl-base",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    local_files_only=True,
)
pretouch_pipeline_cpu_tensors(pipeline, on_component=lambda name: print(f"{name} 加速加载"))
pipeline.to("cuda")
apply_cache_on_pipe(pipeline, residual_diff_threshold=0.1, verbose=True)
# pipeline.enable_xformers_memory_efficient_attention()

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

print(
    "SDP开关状态:",
    "flash=", torch.backends.cuda.flash_sdp_enabled(),
    "mem_efficient=", torch.backends.cuda.mem_efficient_sdp_enabled(),
    "math=", torch.backends.cuda.math_sdp_enabled(),
)

image = pipeline(prompt=prompt, guidance_scale=5.0, num_inference_steps=num_inference_steps).images[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")
# Time taken: 12.150806903839111 seconds pretouch_pipeline_cpu_tensors不影响时间

import torch
from diffusers import ZImagePipeline
import time
start_time = time.time()
from inference.common.monitor import print_gpu_memory_usage
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
from nunchaku.utils import get_precision, is_turing
from nunchaku import NunchakuZImageTransformer2DModel
precision = get_precision()
rank = 32
dtype = torch.bfloat16
# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
print_gpu_memory_usage("载入pipeline前")
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    f"/home/tonera/weights/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors", torch_dtype=dtype
)

pipe = ZImagePipeline.from_pretrained(
    "resources/models/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True,
    transformer=transformer,
)
pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name} 加速加载"))
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
negative_promp="bad quality,worst quality,worst detail,censor"
print_gpu_memory_usage("推理前")
# 2. Generate Image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_promp,
    height=1024,
    width=1024,
    num_inference_steps=8,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("/home/tonera/website/output/test_z_image.png")
print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

# pipe.enable_model_cpu_offload() Time taken: 80.42123818397522 seconds
# pipe.to("cuda") Time taken: 57.934117555618286 seconds
# pretouch_pipeline_cpu_tensors + pipe.to("cuda")  Time taken: 23.24919605255127 seconds 已保留=24.40GB, 峰值=21.61GB
# pretouch_pipeline_cpu_tensors + pipe.to("cuda") + 关闭 low_cpu_mem_usage=True,Time taken: 21.21004366874695 seconds

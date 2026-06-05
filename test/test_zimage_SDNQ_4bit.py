import torch
from diffusers import ZImagePipeline
import time
start_time = time.time()
from inference.common.monitor import print_gpu_memory_usage
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
print_gpu_memory_usage("载入pipeline前")


pipe = ZImagePipeline.from_pretrained("resources/models/Z-Image-Turbo-SDNQ-uint4-svd-r32", torch_dtype=torch.bfloat16)

# # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
# if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
#     pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
#     pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
#     pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds

# pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name} 加速加载"))
pipe.to("cuda")
# pipe.enable_model_cpu_offload()

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.manual_seed(42),
).images[0]
# image.save("z-image-turbo-sdnq-uint4-svd-r32.png")



image.save("/home/tonera/website/output/test_z_image_sdnq.png")
print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

# pipe.to("cuda")
# [16:57:51] [推理后] GPU内存: 6.80GB/119.64GB (5.7%), 已保留=11.88GB, 峰值=9.18GB 减少了内存
# Time taken: 29.206205129623413 seconds

# pipe.enable_model_cpu_offload() 显存节约太bT了 
# [17:00:32] [推理后] GPU内存: 0.01GB/119.64GB (0.0%), 已保留=0.04GB, 峰值=3.74GB
# Time taken: 40.44870090484619 seconds

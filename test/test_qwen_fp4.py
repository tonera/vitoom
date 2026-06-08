from diffusers import DiffusionPipeline,QwenImagePipeline
import torch
import bitsandbytes
model_dir = "/home/tonera/models"
import time
start_time = time.time()
num_inference_steps=10
from inference.common.monitor import print_gpu_memory_usage
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

model_name = f"{model_dir}/QWEN_IMAGE_fp4_w_AbliteratedTE_Diffusers"
# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

print_gpu_memory_usage("载入pipeline前")
# disable_mmap=True不可用
pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
print_gpu_memory_usage("载入pipeline后")
# pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name}加速"))
pipe = pipe.to(device)
print_gpu_memory_usage(f"To {device}")
# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
negative_prompt = " "
print_gpu_memory_usage("推理前")
image = pipe(
    prompt=prompt ,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=num_inference_steps,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")


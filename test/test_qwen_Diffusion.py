from diffusers import DiffusionPipeline
import torch
from inference.common.monitor import print_gpu_memory_usage
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
model_name = "/home/tonera/models/Qwen-Image"
import time
start_time = time.time()
# model_name="/home/tonera/aimodels/models/Qwen-Image"

# Load the pipeline - 参考官方示例代码
# https://huggingface.co/Qwen/Qwen-Image
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

print_gpu_memory_usage("载入pipeline前")
# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# 按照官方示例加载，但添加low_cpu_mem_usage以降低内存峰值
# 注意：官方使用torch_dtype，虽然可能已废弃，但为了兼容性我们先用dtype
pipe = DiffusionPipeline.from_pretrained(
    model_name, 
    torch_dtype=torch_dtype,  # 使用官方示例的参数名
    local_files_only=True,
    disable_mmap=True
)

print_gpu_memory_usage("载入pipeline后")


print("统一内存架构：直接加载到GPU（不使用CPU卸载以提升性能）")
# GB10载入GPU极其缓慢，使用这个函数直接 pretouch
pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name}加速"))

pipe = pipe.to(device)
print_gpu_memory_usage("加载到GPU完成")


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

negative_prompt = " " # using an empty string if you do not have specific concept to remove
# 官方推荐 1328, 1328 = 1:1

print_gpu_memory_usage("推理前")
torch.cuda.synchronize()
with torch.inference_mode():
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]
torch.cuda.synchronize()
print_gpu_memory_usage("推理后")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

# transformer加速
# [09:40:16] [加载到GPU完成] GPU内存: 53.79GB/119.64GB (45.0%), 已保留=54.01GB, 峰值=53.79GB
# [09:40:16] [推理前] GPU内存: 53.79GB/119.64GB (45.0%), 已保留=54.01GB, 峰值=53.79GB
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [02:15<00:00,  4.53s/it]
# [09:42:35] [推理后] GPU内存: 53.81GB/119.64GB (45.0%), 已保留=64.37GB, 峰值=62.25GB

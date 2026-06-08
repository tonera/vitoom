from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import glob
model_dir = "/home/tonera/models/Qwen-Image-Distill-Full"
qwen_dir = "/home/tonera/models/Qwen-Image"
import time
start_time = time.time()
from inference.common.monitor import print_gpu_memory_usage

# # # # # # # # # # # # # 此版本可用 # # # # # # # # # # # # 30步 73秒

transformer_files = sorted(glob.glob(f"{model_dir}/diffusion_pytorch_model-*-of-*.safetensors"))
text_encoder_files = sorted(glob.glob(f"{qwen_dir}/text_encoder/model*.safetensors"))
vae_files = sorted(glob.glob(f"{qwen_dir}/vae/diffusion_pytorch_model*.safetensors"))

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "cpu", #disk ->cpu
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cuda", #cpu -> cuda
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}


print_gpu_memory_usage("开始")
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Distill-Full", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        # ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        # ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(path=transformer_files,**vram_config),
        ModelConfig(path=text_encoder_files,**vram_config),
        ModelConfig(path=vae_files[0] if len(vae_files) == 1 else vae_files,**vram_config),
    ],
    tokenizer_config=ModelConfig(path=f"{qwen_dir}/tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
print_gpu_memory_usage("推理前")

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=15, cfg_scale=1)


print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

# "onload_device": "cpu",
# [16:04:43] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.52GB, 峰值=19.82GB
# Time taken: 74.36785697937012 seconds

# "onload_device": "cuda"
# [16:06:07] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.57GB, 峰值=19.80GB
# Time taken: 73.57300996780396 seconds

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig  # type: ignore[import-not-found]
import torch
import glob
from inference.common.diffsynth_fastload import auto_patch_from_env
from inference.common.monitor import print_gpu_memory_usage
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
model_dir = "/home/tonera/models/Qwen-Image"
import time
start_time = time.time()

# 可选：通过环境变量启用 diffsynth 加载加速（不需要改 site-packages）
# export DIFFSYNTH_FASTLOAD=1
# export DIFFSYNTH_DISABLE_MMAP=1
# export DIFFSYNTH_TARGET_DEVICE=cuda
# export DIFFSYNTH_PIN_MEMORY=1
auto_patch_from_env()
print_gpu_memory_usage("开始")
# diffsynth 的 ModelManager 不会自动展开 path 里的通配符（*）。
# 必须先用 glob 展开成“真实文件路径列表”，否则会把 state_dict 置为 None 触发 TypeError。
transformer_files = sorted(glob.glob(f"{model_dir}/transformer/diffusion_pytorch_model-*-of-*.safetensors"))
text_encoder_files = sorted(glob.glob(f"{model_dir}/text_encoder/model*.safetensors"))
vae_files = sorted(glob.glob(f"{model_dir}/vae/diffusion_pytorch_model*.safetensors"))

if not transformer_files:
    raise FileNotFoundError(f"未找到 transformer 权重分片: {model_dir}/transformer/diffusion_pytorch_model-*-of-*.safetensors")
if not text_encoder_files:
    raise FileNotFoundError(f"未找到 text_encoder 权重: {model_dir}/text_encoder/model*.safetensors")
if not vae_files:
    raise FileNotFoundError(f"未找到 vae 权重: {model_dir}/vae/diffusion_pytorch_model*.safetensors")

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

# . .venv/bin/activate
print_gpu_memory_usage("pipeline")
load_start = time.time()
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # diffsynth 的 ModelConfig：本地目录/文件要用 path=，不要把本地路径塞到 model_id=
        ModelConfig(path=transformer_files,**vram_config),
        ModelConfig(path=text_encoder_files,**vram_config),
        # vae 通常是单文件，但这里也用 glob 兼容不同命名
        ModelConfig(path=vae_files[0] if len(vae_files) == 1 else vae_files,**vram_config),
    ],
    tokenizer_config=ModelConfig(path=f"{model_dir}/tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
load_end = time.time()
print(f"from_pretrained time: {load_end - load_start} seconds")
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"

infer_start = time.time()
image = pipe(prompt, seed=0, num_inference_steps=10)
infer_end = time.time()
print(f"inference time: {infer_end - infer_start} seconds")


print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

# 四项全开 - max:83G
# [10:14:58] [推理后] GPU内存: 53.81GB/119.64GB (45.0%), 已保留=71.57GB, 峰值=64.54GB
# Time taken: 178.81533312797546 seconds

# 开三项，关一项DIFFSYNTH_PIN_MEMORY=0 63G+
# [10:17:46] [推理后] GPU内存: 53.81GB/119.64GB (45.0%), 已保留=71.59GB, 峰值=64.54GB
# Time taken: 137.24148750305176 seconds

# 开两项 关两项:DIFFSYNTH_PIN_MEMORY=0 DIFFSYNTH_TARGET_DEVICE=cpu 63G+
# [10:20:52] [推理后] GPU内存: 53.81GB/119.64GB (45.0%), 已保留=71.57GB, 峰值=64.54GB
# Time taken: 146.45258378982544 seconds

# 启用 vram_config 33G 
# [10:45:52] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.52GB, 峰值=19.82GB
# Time taken: 99.0356113910675 seconds

# 启用 vram_config 开三项，关一项DIFFSYNTH_PIN_MEMORY=0 
# inference time: 96.00239872932434 seconds
# [10:51:26] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.52GB, 峰值=19.82GB
# Time taken: 96.65002083778381 seconds
# # 

# 启用 vram_config 开三项，关一项DIFFSYNTH_PIN_MEMORY=0  offload_device=cpu onload_device=cuda
# from_pretrained time: 4.570971965789795 seconds
# inference time: 90.86987662315369 seconds
# [10:56:11] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.57GB, 峰值=19.80GB
# Time taken: 95.44233512878418 seconds


# 启用 vram_config 开三项，关一项DIFFSYNTH_PIN_MEMORY=0  offload_device=cpu onload_device=cuda
# 1.DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_3 python -u test.py
# from_pretrained time: 3.9995315074920654 seconds
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:21<00:00,  8.14s/it]
# inference time: 90.13676738739014 seconds
# [11:21:28] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.57GB, 峰值=19.80GB
# Time taken: 94.1378288269043 seconds

# 2. export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2   # 或 flash_attention_3
# from_pretrained time: 4.031761407852173 seconds
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:22<00:00,  8.22s/it]
# inference time: 90.92094016075134 seconds
# [11:24:11] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.57GB, 峰值=19.80GB
# Time taken: 94.95424675941467 seconds

# 3. export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
# from_pretrained time: 3.985262393951416 seconds
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:22<00:00,  8.25s/it]
# inference time: 90.75562119483948 seconds
# [11:26:36] [推理后] GPU内存: 0.02GB/119.64GB (0.0%), 已保留=0.57GB, 峰值=19.80GB
# Time taken: 94.74238443374634 seconds



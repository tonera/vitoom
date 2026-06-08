import torch
from diffusers import QwenImagePipeline
from datetime import datetime
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig


import time

# 在GB10和3090显卡上测试.GB10是blackwell架构，采用fp4量化，3090是int4量化
tag = get_precision()
is_gb10 = True
if is_gb10:
    from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
    from inference.common.monitor import print_gpu_memory_usage
    model_dir = "/home/tonera/models"
    weight_dir = "/home/tonera/weights"
else:
    from tonera.monitor import print_gpu_memory_usage
    model_dir = "/home/tonera/aimodels/models"
    weight_dir = "/home/tonera/project/aiservice/diffusers/weights"
start_time = time.time()

# Load the model
print_gpu_memory_usage("载入transformer")
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"{weight_dir}/nunchaku-qwen-image/svdq-{tag}_r32-qwen-image.safetensors",
    torch_dtype=torch.bfloat16,
    pin_memory=True,
)
# currently, you need to use this pipeline to offload the model to CPU
print_gpu_memory_usage("载入pipeline")
pipe = QwenImagePipeline.from_pretrained(
    f"{model_dir}/Qwen-Image", 
    # transformer=transformer, 
    torch_dtype=torch.bfloat16)
print_gpu_memory_usage("To cuda")

if get_gpu_memory() > 24:
    print("载入GPU")
    if is_gb10:
        pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name}加速"))
    # pipe.to("cuda")
    # pipe.enable_model_cpu_offload()
    # transformer.set_offload(
    #     False, use_pin_memory=True, num_blocks_on_gpu=1
    # )
    pipe.to("cuda")
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(True)
    pipe._exclude_from_cpu_offload.append("transformer")
    pipe.enable_sequential_cpu_offload()

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",  # for english prompt,
    "zh": "超清，4K，电影级构图",  # for chinese prompt,
}
print_gpu_memory_usage("推理前")
# Generate image
prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""
negative_prompt = " "  # using an empty string if you do not have specific concept to remove

# torch.cuda.synchronize()
with torch.inference_mode():
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        guidance_scale=4.0,
        generator=torch.Generator("cpu").manual_seed(1)
    ).images[0]
# torch.cuda.synchronize()
print_gpu_memory_usage("推理后")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")


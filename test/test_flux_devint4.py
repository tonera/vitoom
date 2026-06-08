import torch
from diffusers import FluxPipeline
from nunchaku.utils import get_precision
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from inference.common.teacache import teacache_forward,set_tea_cache
from datetime import datetime
num_inference_steps=50
import time
from inference.common.monitor import print_gpu_memory_usage
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

def _precondition_pipeline_before_to_cuda(pipe) -> None:
    """测试用：在 `pipe.to("cuda")` 前做 CPU 侧 pretouch，以对比迁移耗时。"""
    pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name}加速"))


model_dir = "/home/tonera/models"
weight_dir = "/home/tonera/weights"
output_dir = "/home/tonera/project/vitoom/resources/outputs"
precision = get_precision()


start_time = time.time()
# quant_config = TransformersBitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# text_encoder_2_4bit = T5EncoderModel.from_pretrained(
#     "weights/t5-nf4",
#     # quantization_config=quant_config,
#     torch_dtype=torch.bfloat16,
# )

print(f"Start:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
base_model = f"{model_dir}/FLUX.1-dev-bnb-4bit"
pipeline = FluxPipeline.from_pretrained(
    base_model, 
    dtype=torch.bfloat16)
_precondition_pipeline_before_to_cuda(pipeline)
print(f"To cuda:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
pipeline.to("cuda")
print(f"Generate image:当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_gpu_memory_usage("推理前")
prompt = "A cat holding a sign that says hello world"
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
    image = pipeline(
        prompt,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        width=1024, height=1024,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
print_gpu_memory_usage("推理后")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

image.save(f"{output_dir}/flux_dev.png")
print_gpu_memory_usage("保存后")



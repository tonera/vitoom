import torch

# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

from diffusers import FluxPipeline
from nunchaku.utils import get_precision
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from inference.common.teacache import teacache_forward,set_tea_cache

num_inference_steps=50
import time
from inference.common.monitor import print_gpu_memory_usage
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig


model_dir = "/home/tonera/models"
# model_dir = "/home/tonera/aimodels/models"
weight_dir = "/home/tonera/weights"
output_dir = "/home/tonera/project/vitoom/resources/outputs"
precision = get_precision()
start_time = time.time()

NunchakuFluxTransformer2dModel.forward = teacache_forward
# transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"{weight_dir}/svdq-int4-flux.1-dev")
transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"{weight_dir}/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",pin_memory="auto")
transformer.set_attention_impl("nunchaku-fp16")
set_tea_cache(transformer,num_inference_steps)

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

text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(f"{weight_dir}/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors")

# base_model = "models/FLUX.1-dev-bnb-4bit"
base_model = f"{model_dir}/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    base_model, 
    transformer=transformer, 
    text_encoder_2=text_encoder_2,
    torch_dtype=torch.bfloat16).to("cuda")

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


# pipe = FluxPipeline.from_pretrained(base_model,torch_dtype=torch.bfloat16,disable_mmap=True)
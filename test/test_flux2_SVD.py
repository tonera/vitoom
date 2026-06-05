import torch
import diffusers
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
import time
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
start_time = time.time()
num_inference_steps = 30
print(f"开始")
pipe = diffusers.Flux2Pipeline.from_pretrained("resources/models/FLUX.2-dev-SDNQ-uint4-svd-r32", torch_dtype=torch.bfloat16)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"from_pretrained用时: {elapsed_time} seconds")
pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name} 加速加载"))
# Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
# if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
#     pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
#     pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
#     pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds

# pipe.enable_model_cpu_offload()

pipe.to("cuda")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"to cuda用时: {elapsed_time} seconds")
prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

image = pipe(
    prompt=prompt,
    generator=torch.manual_seed(42),
    num_inference_steps=num_inference_steps,
    guidance_scale=4,
).images[0]
# image.save("flux-2-dev-sdnq-uint4-svd-r32.png")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"推理用时: {elapsed_time} seconds")

# to cuda用时: 19.85730767250061 seconds
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [04:19<00:00,  8.64s/it]
# 推理用时: 286.30479288101196 seconds
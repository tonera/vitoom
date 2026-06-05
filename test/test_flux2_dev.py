import torch
from transformers import Mistral3ForConditionalGeneration
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from transformers import BitsAndBytesConfig

repo_id = "resources/models/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16
import time
start_time = time.time()
print(f"开始")


transformer = Flux2Transformer2DModel.from_pretrained(
  repo_id, subfolder="transformer", torch_dtype=torch_dtype, device_map="cpu"
)
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
  repo_id, subfolder="text_encoder", dtype=torch_dtype, device_map="cpu"
)



end_time = time.time()
elapsed_time = end_time - start_time
print(f"载入 transformer和text_encoder用时: {elapsed_time} seconds")
pipe = Flux2Pipeline.from_pretrained(
  repo_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()
# pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name} 加速加载"))
# pipe.to(device)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"to device用时: {elapsed_time} seconds")

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell"

image = pipe(
  prompt=prompt,
  generator=torch.Generator(device=device).manual_seed(42),
  num_inference_steps=28, # 28 is a good trade-off
  guidance_scale=4,
).images[0]
end_time = time.time()
elapsed_time = end_time - start_time
print(f"推理用时: {elapsed_time} seconds")
# image.save("/home/tonera/website/output/flux2_output.png")



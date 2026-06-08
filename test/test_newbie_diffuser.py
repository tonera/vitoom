import torch
from modelscope import NewbiePipeline
from modelscope import AutoModel
import time
start_time = time.time()
from inference.common.monitor import print_gpu_memory_usage

device = "cuda"
model_path = "models/NewBie-image-Exp0.1-Diffusers"
print_gpu_memory_usage("text_encoder_2")
text_encoder_2 = AutoModel.from_pretrained(model_path, subfolder="text_encoder_2", trust_remote_code=True, torch_dtype=torch.bfloat16)
print_gpu_memory_usage("from_pretrained")
pipe = NewbiePipeline.from_pretrained(model_path, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16)
del text_encoder_2

# Enable memory optimizations.
pipe.enable_model_cpu_offload(device=device)

prompt = """
  <character_1>
  <n>$character_1$</n>
  <gender>1girl</gender>
  <appearance>chibi, red_eyes, blue_hair, long_hair, hair_between_eyes, head_tilt, tareme, closed_mouth</appearance>
  <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, blue_skirt, miniskirt, pleated_skirt, blue_hat, mini_hat, thighhighs, grey_thighhighs, black_shoes, mary_janes</clothing>
  <expression>happy, smile</expression>
  <action>standing, holding, holding_briefcase</action>
  <position>center_left</position>
  </character_1>

  <character_2>
  <n>$character_2$</n>
  <gender>1girl</gender>
  <appearance>chibi, red_eyes, pink_hair, long_hair, very_long_hair, multi-tied_hair, open_mouth</appearance>
  <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, red_skirt, miniskirt, pleated_skirt, hair_bow, multiple_hair_bows, white_bow, ribbon_trim, ribbon-trimmed_bow, white_thighhighs, black_shoes, mary_janes, bow_legwear, bare_arms</clothing>
  <expression>happy, smile</expression>
  <action>standing, holding, holding_briefcase, waving</action>
  <position>center_right</position>
  </character_2>

  <general_tags>
  <count>2girls, multiple_girls</count>
  <style>anime_style, digital_art</style>
  <background>white_background, simple_background</background>
  <atmosphere>cheerful</atmosphere>
  <quality>high_resolution, detailed</quality>
  <objects>briefcase</objects>
  <other>alternate_costume</other>
  </general_tags>
"""

negative_prompt = "blurry, worst quality, low quality, deformed hands, bad anatomy, extra limbs, poorly drawn face, mutated, extra eyes, bad proportions"

print_gpu_memory_usage("text_encoder_2 to divice ")

pipe.text_encoder_2 = pipe.text_encoder_2.to(device)

print_gpu_memory_usage("推理前 ")
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    guidance_scale=2.5,
    num_inference_steps=30,
    generator=torch.manual_seed(42),
).images[0]

print_gpu_memory_usage("推理后 ")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")



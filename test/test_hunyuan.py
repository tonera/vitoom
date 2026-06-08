
import torch

dtype = torch.bfloat16
device = "cuda:0"
from diffusers import HunyuanVideo15ImageToVideoPipeline, attention_backend
from diffusers.utils import export_to_video, load_image


model_dir = "/home/tonera/models"
generator = torch.Generator(device=device).manual_seed(1)
image = load_image("http://192.168.0.106:8080/wan_i2v_input.JPG")
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."


pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(f"{model_dir}/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()


# with attention_backend("flash_hub"): # or `"flash_hub"` if you are not on H100/H800
video = pipe(
    prompt=prompt,
    image=image,
    generator=generator,
    num_frames=121,
    num_inference_steps=12,
).frames[0]
export_to_video(video, "/home/tonera/website/output/hunyuan_output.mp4", fps=24)



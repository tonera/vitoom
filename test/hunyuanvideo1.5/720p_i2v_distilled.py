#example 1
import torch

dtype = torch.bfloat16
device = "cuda:0"
from diffusers import HunyuanVideo15ImageToVideoPipeline, attention_backend
from diffusers.utils import export_to_video, load_image

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device=device).manual_seed(1)
image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
with attention_backend("_flash_3_hub"): # or `"flash_hub" if you're not using Hopper GPUs
    video = pipe(
        prompt=prompt,
        image=image,
        generator=generator,
        num_frames=121,
        num_inference_steps=50,
    ).frames[0]
export_to_video(video, "output.mp4", fps=24)


#example 2
import torch

dtype = torch.bfloat16
device = "cuda:0"
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device=device).manual_seed(1)
image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

video = pipe(
    prompt=prompt,
    image=image,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)

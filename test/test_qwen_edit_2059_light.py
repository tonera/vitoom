import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

import time
start_time = time.time()
from inference.common.monitor import print_gpu_memory_usage


# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

num_inference_steps = 4  # you can also use the 8-step model to improve the quality
rank = 32  # you can also use the rank=128 model to improve the quality
model_path = f"resources/weights/nunchaku-qwen-image-edit-2509/svdq-fp4_r32-qwen-image-edit-2509-lightningv2.0-4steps.safetensors"

# Load the model
print_gpu_memory_usage("载入transformer前")
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
print_gpu_memory_usage("载入transformer后")

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "resources/models/Qwen-Image-Edit-2509", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)
print_gpu_memory_usage("载入pipeline后")
if get_gpu_memory() > 18:
    # pipeline.enable_model_cpu_offload()
    pretouch_pipeline_cpu_tensors(pipeline, on_component=lambda name: print(f"{name} 加速加载"))
    pipeline.to("cuda")
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

image1 = load_image("http://192.168.0.102:8080/flux_dev_majicflus_v10.png")
image1 = image1.convert("RGB")
# image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png")
# image2 = image2.convert("RGB")
# image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png")
# image3 = image3.convert("RGB")

prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
inputs = {
    "image": [image1],
    "prompt": prompt,
    "true_cfg_scale": 1.0,
    "num_inference_steps": num_inference_steps,
}
print_gpu_memory_usage("推理前")
output = pipeline(**inputs)
output_image = output.images[0]
# output_image.save(f"qwen-image-edit-2509-lightning-r{rank}-{num_inference_steps}steps.png")


print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")


# pipeline.enable_model_cpu_offload() Time taken: 122.43736863136292 seconds
# pipeline.to("cuda") Time taken: 91.92518663406372 seconds , 已保留=37.46GB, 峰值=35.52GB
# pipeline.to("cuda") + pretouch_pipeline_cpu_tensors Time taken: 29.6326904296875 seconds 已保留=37.46GB, 峰值=35.52GB
import json
import os
import platform
import time
from datetime import datetime

import torch
from diffusers import QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import (  # type: ignore[import-not-found]
    NunchakuQwenImageTransformer2DModel,
)
from nunchaku.utils import get_precision

# # # # # # # # # # # # # 此版本可用 # # # # # # # # # # # #  pin_memory=True 开 pretouch_pipeline_cpu_tensors 30步76.8086秒

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_available": True,
        "device_index": idx,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / (1024**3), 2),
    }


def _mem_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_mem_free_gb": round(free / (1024**3), 2),
        "cuda_mem_total_gb": round(total / (1024**3), 2),
        "cuda_mem_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
        "cuda_mem_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
        "cuda_mem_max_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 2),
        "cuda_mem_max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
    }


def print_gpu_memory_usage(tag: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{_now_iso()}] [{tag}] CUDA不可用")
        return
    s = _mem_snapshot()
    print(
        f"[{_now_iso()}] [{tag}] GPU显存: "
        f"allocated={s['cuda_mem_allocated_gb']}G reserved={s['cuda_mem_reserved_gb']}G "
        f"max_allocated={s['cuda_mem_max_allocated_gb']}G max_reserved={s['cuda_mem_max_reserved_gb']}G "
        f"free/total={s['cuda_mem_free_gb']}G/{s['cuda_mem_total_gb']}G"
    )


def _resolve_torch_dtype(name: str) -> torch.dtype:
    name = (name or "").strip().lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"不支持的 VITOOM_TORCH_DTYPE={name}（仅支持 bf16/fp16）")


# ========= 配置（两台机器用环境变量覆盖）=========
# 量化类型：get_precision() 返回 "int4" 或 "fp4"
tag = str(get_precision())
is_gb10 = tag.lower() == "fp4"

# ===== dtype 规则（重要）=====
# 你使用的是 nunchaku 的 svdq 量化 transformer（tag=fp4/int4）。
# 这类权重通常“运行时 dtype”是固定的（实践中为 bf16），强行设为 fp16 会在 nunchaku 的
# from_pretrained() 里触发 dtype 断言失败（你已经遇到 AssertionError）。
#
# 因此：当 tag=fp4/int4 时，我们强制 transformer/pipeline 都用 bf16。
requested_pipe_dtype = os.getenv("VITOOM_TORCH_DTYPE", "bf16")
requested_transformer_dtype = os.getenv("VITOOM_TRANSFORMER_TORCH_DTYPE", requested_pipe_dtype)
if tag.lower() in ("fp4", "int4"):
    if (requested_pipe_dtype or "").strip().lower() not in ("bf16", "bfloat16") or (
        (requested_transformer_dtype or "").strip().lower() not in ("bf16", "bfloat16")
    ):
        print(
            f"提示：检测到量化 transformer tag={tag}，强制使用 bf16（忽略 VITOOM_TORCH_DTYPE={requested_pipe_dtype} / "
            f"VITOOM_TRANSFORMER_TORCH_DTYPE={requested_transformer_dtype}），否则会触发 nunchaku dtype 断言失败。"
        )
    pipe_torch_dtype = torch.bfloat16
    transformer_torch_dtype = torch.bfloat16
else:
    # 非量化权重时才允许自由切换
    pipe_torch_dtype = _resolve_torch_dtype(requested_pipe_dtype)
    transformer_torch_dtype = _resolve_torch_dtype(requested_transformer_dtype)
generator_device = os.getenv("VITOOM_GENERATOR_DEVICE", "cpu").strip().lower()  # cpu/cuda
if generator_device not in ("cpu", "cuda"):
    raise ValueError("VITOOM_GENERATOR_DEVICE 仅支持 cpu 或 cuda")
if generator_device == "cuda" and not torch.cuda.is_available():
    print("警告：VITOOM_GENERATOR_DEVICE=cuda 但 CUDA 不可用，已回退到 cpu")
    generator_device = "cpu"

# 路径：优先用环境变量覆盖，避免脚本里写死
default_model_dir = "/home/tonera/models" if is_gb10 else "/home/tonera/aimodels/models"
default_weight_dir = "/home/tonera/weights" if is_gb10 else "/home/tonera/project/aiservice/diffusers/weights"
model_dir = os.getenv("VITOOM_MODEL_DIR", default_model_dir).rstrip("/")
weight_dir = os.getenv("VITOOM_WEIGHT_DIR", default_weight_dir).rstrip("/")

# 是否把整管道搬到 GPU（大显存机器），否则走 sequential cpu offload
gpu_total_gb = _gpu_info().get("total_vram_gb", 0.0) or 0.0
load_to_gpu_threshold_gb = float(os.getenv("VITOOM_LOAD_TO_GPU_THRESHOLD_GB", "30"))
force_load_to_gpu = os.getenv("VITOOM_FORCE_LOAD_TO_GPU", "").strip().lower() in ("1", "true", "yes")
force_cpu_offload = os.getenv("VITOOM_FORCE_CPU_OFFLOAD", "").strip().lower() in ("1", "true", "yes")
load_to_gpu = (gpu_total_gb >= load_to_gpu_threshold_gb) or force_load_to_gpu
if force_cpu_offload:
    load_to_gpu = False

# 推理参数
steps = int(os.getenv("VITOOM_STEPS", "10"))
width = int(os.getenv("VITOOM_WIDTH", "1024"))
height = int(os.getenv("VITOOM_HEIGHT", "1024"))
seed = int(os.getenv("VITOOM_SEED", "1"))

prompt = os.getenv(
    "VITOOM_PROMPT",
    "Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author.",
)
negative_prompt = os.getenv("VITOOM_NEGATIVE_PROMPT", " ")

# ========= 运行（只测冷启动一次，不做预热）=========
run_id = _now_iso()
env_info = {
    "run_id": run_id,
    "host": platform.node(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "precision_tag": tag,
    "is_gb10_guess": is_gb10,
    "pipe_torch_dtype": str(pipe_torch_dtype).replace("torch.", ""),
    "transformer_torch_dtype": str(transformer_torch_dtype).replace("torch.", ""),
    "generator_device": generator_device,
    "model_dir": model_dir,
    "weight_dir": weight_dir,
    "gpu": _gpu_info(),
    "load_to_gpu_threshold_gb": load_to_gpu_threshold_gb,
    "load_to_gpu": load_to_gpu,
    "steps": steps,
    "width": width,
    "height": height,
    "seed": seed,
}
print("==== 环境信息 ====")
print(json.dumps(env_info, ensure_ascii=False, indent=2))

timings = {}
print_gpu_memory_usage("开始")
t0 = time.perf_counter()

transformer_ckpt = f"{weight_dir}/nunchaku-qwen-image/svdq-{tag}_r32-qwen-image.safetensors"
base_model_path = f"{model_dir}/Qwen-Image"

# 1) Load transformer (quantized)
print_gpu_memory_usage("载入transformer前")
t = time.perf_counter()
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    transformer_ckpt,
    # 注意：如果你把 pipeline 设为 fp16，但 transformer 仍是 bf16，会触发 Half vs BF16 的 matmul 报错。
    # 因此 transformer dtype 用 VITOOM_TRANSFORMER_TORCH_DTYPE 单独控制，默认与 pipeline 一致。
    torch_dtype=transformer_torch_dtype,
    # 注意：你实测 pin_memory 可能会拖慢 UMA 场景，这里默认关；可用环境变量开启
    pin_memory=os.getenv("VITOOM_PIN_MEMORY", "").strip().lower() in ("1", "true", "yes"),
)

timings["load_transformer_s"] = time.perf_counter() - t
print_gpu_memory_usage("载入transformer后")

# 2) Load pipeline
print_gpu_memory_usage("载入pipeline前")
t = time.perf_counter()
pipe = QwenImagePipeline.from_pretrained(
    base_model_path,
    transformer=transformer,
    torch_dtype=pipe_torch_dtype,
)
timings["load_pipeline_s"] = time.perf_counter() - t
print_gpu_memory_usage("载入pipeline后")

# 3) Device / offload mode
t = time.perf_counter()
mode = "unknown"
if torch.cuda.is_available():
    if load_to_gpu:
        mode = "full_cuda"
        print("模式：full_cuda（整管道搬到GPU）")
        # 尽量确保 transformer 不做层级 offload（如果支持该 API）
        try:
            if hasattr(transformer, "set_offload"):
                transformer.set_offload(False)
        except Exception as e:
            print(f"提示：transformer.set_offload(False) 失败（可忽略）：{e}")
        # 仅在本项目环境存在时启用该加速（可选）
        try:
            from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
            pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name} 加速加载"))
        except Exception as e:
            print(f"跳过 pretouch_pipeline_cpu_tensors（不可用或失败）：{e}")
        pipe.to("cuda")
    else:
        mode = "sequential_cpu_offload"
        print("模式：sequential_cpu_offload（低显存：逐层CPU卸载）")
        transformer.set_offload(True)
        if hasattr(pipe, "_exclude_from_cpu_offload"):
            pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
timings["set_device_or_offload_s"] = time.perf_counter() - t
print_gpu_memory_usage("设备/卸载策略设置后")

# 4) Inference (cold, once)
print_gpu_memory_usage("推理前")
if torch.cuda.is_available():
    torch.cuda.synchronize()
t = time.perf_counter()
with torch.inference_mode():
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=4.0,
        guidance_scale=4.0,
        generator=torch.Generator(generator_device).manual_seed(seed),
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()
timings["inference_s"] = time.perf_counter() - t
print_gpu_memory_usage("推理后")

timings["total_s"] = time.perf_counter() - t0

result = {
    **env_info,
    "transformer_ckpt": transformer_ckpt,
    "base_model_path": base_model_path,
    "mode": mode,
    "timings": {k: round(v, 4) for k, v in timings.items()},
    "mem": _mem_snapshot(),
}

print("==== 计时结果 ====")
print(json.dumps(result["timings"], ensure_ascii=False, indent=2))
print("==== 单行JSON（用于两台机器对比/收集） ====")
print(json.dumps(result, ensure_ascii=False))


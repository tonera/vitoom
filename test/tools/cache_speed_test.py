import argparse
import time
import torch
from diffusers import ZImagePipeline,StableDiffusionXLPipeline
from inference.cache import apply_meancache_on_pipe
# python test.py --variant C --steps 10

def apply_cache(pipe, a):
    apply_meancache_on_pipe(
        pipe,
        # rel_l1_thresh=0.80,
        # skip_budget=0.50,
        # start_step=1,
        # # 对你这条 ZImage（L_K 常见 0.18~0.69）先放宽峰值抑制与累积误差上限，
        # # 先验证能触发 skip；后续再逐步收紧以保证画质。
        # peak_threshold=1.0,
        # max_accumulated_error=1.0,
        # gamma=1.0,
        # cache_device=a.cache_device,
        # enable_pssp=True,
        debug=True,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B","C"], required=True)  # A: nunchaku transformer; B: meancache
    p.add_argument("--model_dir", default="/home/tonera/aimodels/models/Beyond_Reality_Zimage_v2_svdq")
    p.add_argument("--svdq_ckpt", default="/home/tonera/aimodels/models/Beyond_Reality_Zimage_v2_svdq/svdq-int4_r32-Beyond_Reality_Zimage_v2_svdq.safetensors")
    p.add_argument("--prompt", default="a hot girl, sakimichan style")
    p.add_argument("--out", default="/home/tonera/website/output")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--h", type=int, default=1024)
    p.add_argument("--w", type=int, default=1024)
    p.add_argument("--cache_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--enable_cache", type=int, default=1)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    a = p.parse_args()

    print(f"model_dir = {a.model_dir}")

    dtype = torch.bfloat16
    if a.variant == "A":
        from nunchaku import NunchakuZImageTransformer2DModel
        from nunchaku.utils import is_turing
        dtype = torch.float16 if is_turing() else torch.bfloat16
        tr = NunchakuZImageTransformer2DModel.from_pretrained(a.svdq_ckpt, torch_dtype=dtype)
        pipe = ZImagePipeline.from_pretrained(a.model_dir, transformer=tr, torch_dtype=dtype, low_cpu_mem_usage=False)

    elif a.variant == "B":
        pipe = ZImagePipeline.from_pretrained(a.model_dir, torch_dtype=dtype, low_cpu_mem_usage=False)

    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            a.model_dir,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True,
        )

    pipe = pipe.to("cuda")
    if a.enable_cache == 1:
        apply_cache(pipe, a)

    t0 = time.perf_counter()
    out = pipe(prompt=a.prompt, height=a.h, width=a.w, num_inference_steps=a.steps, guidance_scale=a.guidance_scale, generator=torch.Generator().manual_seed(a.seed))
    t1 = time.perf_counter()
    print(f"[{a.variant}] infer_sec={(t1 - t0):.3f}")
    img = out.images[0]
    img.save(f"{a.out}/cache_test_{a.variant}.png")
    print(f"Saved to {a.out}/cache_test_{a.variant}.png")


if __name__ == "__main__":
    main()



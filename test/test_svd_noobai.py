import torch
import diffusers
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
import os
from inference.common.torch_transfer_utils import pretouch_pipeline_cpu_tensors
import time
start_time = time.time()
from inference.common.monitor import print_gpu_memory_usage
from typing import Any, Dict, List, Optional

device = "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "mps" if hasattr(torch,"mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print_gpu_memory_usage(f"from_pretrained device={device} ")
print(f"[info] torch={torch.__version__}")
try:
    print(f"[info] cuda_available={torch.cuda.is_available()} cuda_version={getattr(torch.version, 'cuda', None)}")
except Exception:
    pass

# ---- 可调开关：性能分析/加速策略 ----
# 更细的拆分分析：文本编码 / latent-only（跳过 VAE decode）/ 单独 decode
# 注意：开启会额外多跑 1~2 次推理用于对比，因此“总耗时(Time taken)”会显著变长。
DETAILED_BREAKDOWN = False
# 仅当 DETAILED_BREAKDOWN=True 时生效：是否额外跑 latent-only + vae decode 基准
RUN_EXTRA_BENCH_PASSES = False
# warmup 主要用于触发首轮缓存/初始化（不计入“单次出图”耗时）；关掉可减少等待
ENABLE_WARMUP = True
# 尝试开启 SDNQ 的量化 matmul（如果 SDNQ 支持且 triton 路径可用）
ENABLE_SDNQ_QUANT_MATMUL = False
#
# NOTE:
# 在某些 transformers 版本 + SDNQ 量化层组合下，from_pretrained() 如果检测到 missing/mismatched keys，
# 会触发 initialize_weights()，CLIP 的 _init_weights 会对 Linear/Embedding 做 normal_ 初始化。
# 但 SDNQ 量化层的权重可能是 uint8，因此会报：
#   RuntimeError: expected a floating-point or complex dtype, but got dtype=torch.uint8
#
# 这里做一个最小兼容补丁：CLIP 初始化遇到非浮点/复数权重时直接跳过初始化（量化权重不应该走 normal_）。
try:
    from transformers.models.clip import modeling_clip as _modeling_clip

    _orig_clip_init_weights = _modeling_clip.CLIPPreTrainedModel._init_weights

    def _sdnq_safe_clip_init_weights(self, module):
        # 更鲁棒：某些 SDNQ/量化模块触发的初始化路径不一定能靠 weight dtype 预判，
        # 直接兜底捕获 “uint8 不能 normal_” 之类错误并跳过。
        try:
            w = getattr(module, "weight", None)
            if w is not None and hasattr(w, "dtype"):
                if not (torch.is_floating_point(w) or torch.is_complex(w)):
                    return
            return _orig_clip_init_weights(self, module)
        except RuntimeError as e:
            msg = str(e)
            if "expected a floating-point or complex dtype" in msg:
                return
            raise

    _modeling_clip.CLIPPreTrainedModel._init_weights = _sdnq_safe_clip_init_weights
except Exception as _e:
    # 如果 transformers 结构变化或未安装 clip 模块，不阻塞运行（后续仍可能会报原始错误）
    print(f"[warn] Skip CLIP init patch: {_e}")

def _sync_if_needed() -> None:
    """不同 device 的同步：做性能计时时必须同步，避免测到异步队列提交时间。"""
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        # 同步失败不应阻塞业务，只是计时会不准
        pass


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.3f} s"


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    try:
        import numpy as np

        return float(np.percentile(np.asarray(values, dtype=float), q))
    except Exception:
        values_sorted = sorted(values)
        idx = int(round((q / 100.0) * (len(values_sorted) - 1)))
        idx = max(0, min(len(values_sorted) - 1, idx))
        return float(values_sorted[idx])


class _StageTimer:
    def __init__(self, name: str, do_sync: bool = True):
        self.name = name
        self.do_sync = do_sync
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        if self.do_sync:
            _sync_if_needed()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.do_sync:
            _sync_if_needed()
        self.elapsed = time.perf_counter() - self.start
        print(f"[perf] {self.name}: {_fmt_s(self.elapsed)}")
        return False


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "resources", "models")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


_repo_root = _find_repo_root(os.path.dirname(__file__))
model_dir = os.path.join(_repo_root, "resources", "models", "NoobAI-XL-Vpred-v1.0-SDNQ-uint4-svd-r128")
if not os.path.isdir(model_dir):
    raise FileNotFoundError(
        f"模型目录不存在：{model_dir}\n"
        f"请确认已把模型放到 vitoom/resources/models/ 下，或修改 test/test_svd_noobai.py 里的 model_dir。"
    )

print(f"[info] repo_root={_repo_root}")
print(f"[info] model_dir={model_dir}")

if device == "mps":
    torch_dtype = torch.float16
elif device == "cuda":
    torch_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
elif device == "xpu":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

with _StageTimer("from_pretrained()", do_sync=False):
    pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype)
    pretouch_pipeline_cpu_tensors(pipe, on_component=lambda name: print(f"{name} 加速加载"))

print_gpu_memory_usage("to device ")

with _StageTimer(f"pipe.to({device})", do_sync=True):
    pipe = pipe.to(device)

try:
    # 某些版本下关闭进度条能少一点 Python 端开销（尤其在回调统计时）
    pipe.set_progress_bar_config(disable=True)
except Exception:
    pass

if ENABLE_SDNQ_QUANT_MATMUL and device in ("cuda", "xpu"):
    try:
        # 参考 SDNQ 官方示例：use_torch_compile 作为 “是否可用 triton/compile 路径” 的判定
        # https://github.com/Disty0/sdnq
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        _triton_ok = triton_is_available() if callable(triton_is_available) else bool(triton_is_available)
        print(f"[perf] sdnq.triton_is_available={_triton_ok} (raw={triton_is_available})")

        if _triton_ok:
            with _StageTimer("apply_sdnq_options_to_model(use_quantized_matmul=True)", do_sync=True):
                if hasattr(pipe, "unet") and pipe.unet is not None:
                    pipe.unet = apply_sdnq_options_to_model(pipe.unet, use_quantized_matmul=True)
                if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
                if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                    pipe.text_encoder_2 = apply_sdnq_options_to_model(pipe.text_encoder_2, use_quantized_matmul=True)
        else:
            print("[perf] sdnq triton/compile 不可用，跳过 use_quantized_matmul")
    except Exception as e:
        print(f"[warn] 量化 matmul/compile 加速未启用：{e}")


prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell"
negative_prompt = "nsfw"
print_gpu_memory_usage("推理前")

gen = (torch.Generator(device=device).manual_seed(42) if device in ("cuda", "xpu") else torch.manual_seed(42))

# ---- 性能分析：warmup + step 级别耗时统计 ----
step_times: List[float] = []
_last_step_t: Dict[str, float] = {"t": 0.0}

def _on_step_end(pipe, step: int, timestep, callback_kwargs: Dict[str, Any]):
    # callback 在每一步结束时触发；同步后记录相邻 step 的时间差
    _sync_if_needed()
    t = time.perf_counter()
    if _last_step_t["t"] != 0.0:
        step_times.append(t - _last_step_t["t"])
    _last_step_t["t"] = t
    return callback_kwargs

def _decode_latents_to_image(latents):
    # 优先使用 pipeline 自带 decode（不同 diffusers 版本差异较大）
    if hasattr(pipe, "decode_latents"):
        return pipe.decode_latents(latents)
    # 兜底：手动 VAE decode
    scaling = None
    try:
        scaling = getattr(getattr(pipe, "vae", None), "config", None)
        scaling = getattr(scaling, "scaling_factor", None)
    except Exception:
        scaling = None
    if scaling is None:
        scaling = 0.13025  # SDXL 常见值，兜底
    latents = latents / scaling
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return image


def _try_encode_prompt() -> Optional[Dict[str, Any]]:
    """尝试单独测文本编码耗时，并返回可复用的 embeds（如果当前 diffusers 版本支持）。"""
    if not hasattr(pipe, "encode_prompt"):
        return None
    try:
        with _StageTimer("encode_prompt()", do_sync=True):
            res = pipe.encode_prompt(  # type: ignore[attr-defined]
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
            )
        # encode_prompt 在 SDXL 通常返回 (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        if isinstance(res, (tuple, list)) and len(res) == 4:
            return {
                "prompt_embeds": res[0],
                "negative_prompt_embeds": res[1],
                "pooled_prompt_embeds": res[2],
                "negative_pooled_prompt_embeds": res[3],
            }
    except Exception as e:
        print(f"[warn] encode_prompt() 拆分计时失败，跳过：{e}")
    return None


embeds_kwargs: Optional[Dict[str, Any]] = _try_encode_prompt() if DETAILED_BREAKDOWN else None

if ENABLE_WARMUP:
    print("[perf] warmup: 开始（用于触发首帧/缓存，结果不计入单次出图耗时）")
    with _StageTimer("warmup_generate(2 steps)", do_sync=True):
        _last_step_t["t"] = 0.0
        with torch.inference_mode():
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=832,
                height=1216,
                num_inference_steps=2,
                guidance_scale=3.5,
                generator=gen,
                callback_on_step_end=_on_step_end,
            ).images[0]

# 正式推理：清空 step 统计重新开始
step_times.clear()
_last_step_t["t"] = 0.0

def _run_generate(output_type: str = "pil", use_embeds: bool = False):
    kwargs: Dict[str, Any] = dict(
        width=832,
        height=1216,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=gen,
        callback_on_step_end=_on_step_end,
        output_type=output_type,
        return_dict=True,
    )
    if use_embeds and embeds_kwargs:
        kwargs.update(embeds_kwargs)
    else:
        kwargs.update(dict(prompt=prompt, negative_prompt=negative_prompt))
    return pipe(**kwargs)


with _StageTimer("generate(total)", do_sync=True):
    with torch.inference_mode():
        out = _run_generate(output_type="pil", use_embeds=bool(embeds_kwargs))
    image = out.images[0]

if DETAILED_BREAKDOWN and RUN_EXTRA_BENCH_PASSES:
    # 额外跑一遍 latent-only，粗略估计 VAE decode + 后处理开销
    step_times_latent: List[float] = []
    _last_step_t["t"] = 0.0

    def _on_step_end_latent(pipe, step: int, timestep, callback_kwargs: Dict[str, Any]):
        _sync_if_needed()
        t = time.perf_counter()
        if _last_step_t["t"] != 0.0:
            step_times_latent.append(t - _last_step_t["t"])
        _last_step_t["t"] = t
        return callback_kwargs

    with _StageTimer("generate(latent_only)", do_sync=True):
        with torch.inference_mode():
            out_latent = pipe(
                prompt_embeds=embeds_kwargs["prompt_embeds"] if embeds_kwargs else None,
                negative_prompt_embeds=embeds_kwargs["negative_prompt_embeds"] if embeds_kwargs else None,
                pooled_prompt_embeds=embeds_kwargs["pooled_prompt_embeds"] if embeds_kwargs else None,
                negative_pooled_prompt_embeds=embeds_kwargs["negative_pooled_prompt_embeds"] if embeds_kwargs else None,
                prompt=None if embeds_kwargs else prompt,
                negative_prompt=None if embeds_kwargs else negative_prompt,
                width=832,
                height=1216,
                num_inference_steps=28,
                guidance_scale=3.5,
                generator=gen,
                callback_on_step_end=_on_step_end_latent,
                output_type="latent",
                return_dict=True,
            )
        latents = out_latent.images

    with _StageTimer("vae_decode_only", do_sync=True):
        with torch.inference_mode():
            _ = _decode_latents_to_image(latents)

if step_times:
    avg = sum(step_times) / len(step_times)
    p50 = _percentile(step_times, 50)
    p90 = _percentile(step_times, 90)
    p99 = _percentile(step_times, 99)
    slow = max(step_times)
    fast = min(step_times)
    print("[perf] denoise step 统计（不含第 1 步起点）")
    print(f"  - steps_measured: {len(step_times)}")
    print(f"  - avg:  {_fmt_ms(avg)}")
    print(f"  - p50:  {_fmt_ms(p50) if p50 is not None else 'n/a'}")
    print(f"  - p90:  {_fmt_ms(p90) if p90 is not None else 'n/a'}")
    print(f"  - p99:  {_fmt_ms(p99) if p99 is not None else 'n/a'}")
    print(f"  - min:  {_fmt_ms(fast)}")
    print(f"  - max:  {_fmt_ms(slow)}")

# image.save("flux-2-dev-sdnq-uint4-svd-r32.png")

print_gpu_memory_usage("推理后")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken (includes setup + optional benches): {elapsed_time} seconds")
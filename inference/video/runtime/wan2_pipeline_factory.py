"""
Wan2 专用 pipeline 工厂与缓存（单进程内复用）。

说明：
- BaseInferrer.run_blocking 默认是单线程执行器，因此在该线程内访问缓存是安全的。
- 若未来引入多线程/多 worker，可在此增加锁或改为进程级缓存策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Any, Iterable

import glob

import torch
import json

from common.logger import get_logger
from common.runtime_cleanup import cleanup_after_oom as common_cleanup_after_oom, is_oom_error as common_is_oom_error
from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class Wan2PipeKey:
    name: str  # e.g. "t2v", "i2v", "ti2v", "control", "animate", "s2v", "inp", "ccv"
    model_root: str
    tokenizer_root: str
    audio_processor_root: str
    device: str
    torch_dtype: str  # "bf16" / "fp16" / "fp32"
    low_vram: bool  # whether ModelConfig uses vram_config (VRAM management/offload)
    vram_limit_gb: str  # stringified float (or empty) to avoid float hashing quirks


#
# NOTE:
# 旧实现曾在此维护全局 dict 缓存（_PIPE_CACHE）。
# 为对齐 image 侧的 PipelineCache(LRU=1 + TTL + 驱逐强释放)，视频侧缓存已上移到 inferrer/handlers，
# 本模块只负责：key 计算 + “创建新 pipeline”（无缓存）。
#


def _dtype_from_str(s: str) -> torch.dtype:
    s2 = (s or "").lower()
    if s2 in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s2 in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _device_default() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _is_oom(exc: BaseException) -> bool:
    """统一判断是否为 OOM（CUDA/MPS/部分 RuntimeError 文本）。"""
    return common_is_oom_error(exc)


def cleanup_after_oom() -> None:
    """
    OOM 后 best-effort 清理，避免同一次请求里连续 OOM。
    对外暴露，供 handler 在重试前调用。
    """
    common_cleanup_after_oom()


def is_oom(exc: BaseException) -> bool:
    """对外暴露，供 handler 判断是否为 OOM。"""
    return _is_oom(exc)


def _stable_sig(payload: Any) -> str:
    """将 payload 转为稳定字符串签名（用于日志）。"""
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(payload)


def _default_low_vram_config(*, device: str, torch_dtype: torch.dtype) -> dict:
    """
    Wan2 默认低显存 vram_config。
    设计目标：尽量把权重常驻 CPU，计算时再搬到 GPU（由 diffsynth VRAM management 调度）。
    """
    dev = str(device or "cpu")
    if dev.startswith("cuda"):
        return {
            "offload_device": "cpu",
            "offload_dtype": torch_dtype,
            "onload_device": "cpu",
            "onload_dtype": torch_dtype,
            # 低显存：preparing 放在 CPU，减少 preparing+computation 叠加导致的峰值显存
            "preparing_device": "cpu",
            "preparing_dtype": torch_dtype,
            "computation_device": dev,
            "computation_dtype": torch_dtype,
        }
    # CPU 推理：不做 offload（避免无意义的状态切换）
    return {
        "offload_device": dev,
        "offload_dtype": torch_dtype,
        "onload_device": dev,
        "onload_dtype": torch_dtype,
        "preparing_device": dev,
        "preparing_dtype": torch_dtype,
        "computation_device": dev,
        "computation_dtype": torch_dtype,
    }


def _auto_vram_limit_gb(device: str) -> Optional[float]:
    """尽量给 diffsynth vram_limit 一个合理默认值（GiB），仅在 CUDA 场景有效。"""
    dev = str(device or "")
    if not dev.startswith("cuda"):
        return None
    try:
        if not torch.cuda.is_available():
            return None
        # mem_get_info 返回 (free, total) bytes
        total = float(torch.cuda.mem_get_info(dev)[1]) / (1024**3)
        return max(0.0, total - 0.5)
    except Exception:
        return None


def _resolve_existing_dir(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Local path not found. Tried: {[str(p) for p in candidates]}")


def _resolve_model_root(models_base_dir: str, model_ref: str) -> Path:
    """
    model_ref:
    - 允许绝对路径
    - 或相对 models_base_dir 的一级子目录名（禁止包含路径分隔符）
    """
    p = Path(model_ref)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        return p
    # 严格：仅允许“目录名”，不允许传入带斜杠的相对路径，避免兜底/兼容导致路径不可控
    if "/" in model_ref or "\\" in model_ref:
        raise ValueError(f"load_name must be a directory name under models_dir (no slashes): {model_ref}")

    base = Path(models_base_dir).resolve()
    root = (base / model_ref).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model dir not found: {root}")
    return root


def _glob_one_or_many(root: Path, pattern: str) -> List[str]:
    matches = sorted(glob.glob(str(root / pattern)))
    if not matches:
        raise FileNotFoundError(f"Missing model files: pattern={pattern}, root={root}")
    return matches


def _glob_first_match(root: Path, patterns: List[str]) -> List[str]:
    """
    依次尝试多个 glob pattern，返回第一个命中的 matches。
    用于兼容不同打包方式（例如 FP8 模型可能没有 Wan2.2_VAE.* / models_t5_umt5-xxl-enc-bf16.* 这类文件名）。
    """
    tried: List[str] = []
    for pat in patterns:
        tried.append(pat)
        matches = sorted(glob.glob(str(root / pat)))
        if matches:
            return matches
    raise FileNotFoundError(f"Missing model files: patterns={tried}, root={root}")

def _glob_optional_first_match(root: Path, patterns: List[str]) -> Optional[List[str]]:
    """可选匹配：依次尝试多个 glob pattern，命中则返回 matches，否则返回 None。"""
    for pat in patterns:
        matches = sorted(glob.glob(str(root / pat)))
        if matches:
            return matches
    return None


def _glob_first_match_in_roots(roots: Iterable[Path], patterns: List[str]) -> List[str]:
    """
    In order, try each root with patterns and return the first matched list.
    Used for shared components (umt5/vae/clip) with fallback roots.
    """
    tried_roots: List[str] = []
    for r in roots:
        rr = Path(r).expanduser().resolve()
        tried_roots.append(str(rr))
        if not rr.exists():
            continue
        try:
            return _glob_first_match(rr, patterns)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Missing model files: patterns={patterns}, roots={tried_roots}")


def _glob_optional_first_match_in_roots(roots: Iterable[Path], patterns: List[str]) -> Optional[List[str]]:
    for r in roots:
        rr = Path(r).expanduser().resolve()
        if not rr.exists():
            continue
        matches = _glob_optional_first_match(rr, patterns)
        if matches:
            return matches
    return None


def _is_fp8_supported() -> bool:
    """
    Best-effort runtime check for FP8 availability (no env vars, tolerate older torch).
    """
    try:
        if not torch.cuda.is_available():
            return False
        if hasattr(torch.cuda, "is_fp8_supported"):
            try:
                return bool(torch.cuda.is_fp8_supported())  # type: ignore[attr-defined]
            except Exception:
                pass
        if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")):
            return False
        major, _minor = torch.cuda.get_device_capability()
        return int(major) >= 9
    except Exception:
        return False


def _t5_patterns_for_system(*, allow_fp8: bool = True) -> List[str]:
    """
    Prefer quantized umt5 checkpoints to reduce VRAM:
    - if fp8 supported: fp8.pth > int8.pth > bf16.pth > other .pth > safetensors
    - else:            int8.pth > bf16.pth > other .pth > safetensors

    When allow_fp8=False (low_vram / VRAM offload), skip fp8: Float8Tensor weights are
    incompatible with diffsynth AutoWrappedLinear.cast_to(empty_like) during CPU offload.
    """
    pats: List[str] = []
    if allow_fp8 and _is_fp8_supported():
        pats.append("models_t5_umt5-xxl-enc-fp8.pth")
    pats.extend(
        [
            "models_t5_umt5-xxl-enc-int8.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_t5_umt5-xxl-enc-*.pth",
            "*umt5*enc*.pth",
            "*umt5*.pth",
            "*t5*enc*.pth",
            "*t5*.pth",
            # fallback (if a package provides only safetensors)
            "*umt5*.safetensors",
            "*umt5*.*",
        ]
    )
    return pats


def _default_model_ref_for_name(name: str) -> str:
    """
    当调用方未提供 model_ref_override 时，给每个 pipeline name 一个默认模型目录名。
    注意：这些默认值不影响“通用构造器”能力；只是为了让系统在未传 load_name 时可运行。
    """
    defaults = {
        "t2v": "Wan2.2-T2V-A14B",
        "i2v": "Wan2.2-I2V-A14B",
        "ti2v": "Wan2.2-TI2V-5B",
        "control": "Wan2.2-Fun-A14B-Control",
        "animate": "Wan2.2-Animate-14B",
        "s2v": "Wan2.2-S2V-14B",
        "inp": "Wan2.2-Fun-A14B-InP",
        "ccv": "Wan2.2-Fun-A14B-Control-Camera",
    }
    if name not in defaults:
        raise ValueError(f"Unknown wan2 pipeline name: {name}")
    return defaults[name]


def _resolve_tokenizer_root(models_base_dir: str, *, model_root: Optional[Path] = None, weights_base_dir: Optional[str] = None) -> Path:
    """
    tokenizer 目录的布局在不同模型包里差异很大：
    - 我们的离线本地推理器约束：所有组件都应打包在同一个 model_root 目录内
      （即 {models_dir}/{load_name}/...）
    但为了减少重复（umt5/vae/tokenizer），我们允许从共享目录 `WanVideo` 兜底：
    - {models_dir}/{load_name}
    - {models_dir}/WanVideo
    - {weights_dir}/WanVideo
    """
    if model_root is None:
        raise FileNotFoundError("tokenizer_root requires model_root (expected under {models_dir}/{load_name})")

    mr = Path(model_root).resolve()
    shared_roots: List[Path] = [mr]
    shared_roots.append((Path(models_base_dir).expanduser().resolve() / "WanVideo"))
    if weights_base_dir:
        shared_roots.append((Path(weights_base_dir).expanduser().resolve() / "WanVideo"))

    tried: List[str] = []
    for r in shared_roots:
        tried.append(str(r))
        candidates = [r / "google/umt5-xxl", r / "google"]
        for c in candidates:
            if c.exists():
                return c
    raise FileNotFoundError(f"Local path not found. Tried: {tried}")


def build_wan2_model_configs(
    name: str,
    *,
    models_base_dir: str,
    weights_base_dir: Optional[str] = None,
    model_ref_override: Optional[str] = None,
    vram_config: Optional[dict] = None,
) -> Tuple[list[ModelConfig], ModelConfig, Optional[ModelConfig]]:
    """
    返回 (model_configs, tokenizer_config, audio_processor_config)
    """
    vram_kwargs = vram_config or {}

    # 解析模型根目录（默认值仅用于“未传 load_name 也能跑”）
    model_ref = model_ref_override or _default_model_ref_for_name(name)
    root = _resolve_model_root(models_base_dir, model_ref)
    tokenizer_root = _resolve_tokenizer_root(models_base_dir, model_root=root, weights_base_dir=weights_base_dir)
    tokenizer_config = ModelConfig(path=str(tokenizer_root))

    shared_roots: List[Path] = [root, Path(models_base_dir).expanduser().resolve() / "WanVideo"]
    if weights_base_dir:
        shared_roots.append(Path(weights_base_dir).expanduser().resolve() / "WanVideo")

    # 通用规则：按“存在性/命名规律”扫描组件（模型包假设：自包含）
    # - 扩散：若存在 high/low 子目录则双扩散，否则单扩散
    use_double_dit = (root / "high_noise_model").exists() and (root / "low_noise_model").exists()
    diffusion_high = None
    diffusion_low = None
    if use_double_dit:
        diffusion_high = _glob_one_or_many(root, "high_noise_model/diffusion_pytorch_model*.safetensors")
        diffusion_low = _glob_one_or_many(root, "low_noise_model/diffusion_pytorch_model*.safetensors")
    else:
        diffusion_high = _glob_one_or_many(root, "diffusion_pytorch_model*.safetensors")

    # - T5：优先量化版本（fp8/int8），并按系统能力选择 fp8；low_vram 时跳过 fp8（见 _t5_patterns_for_system）
    allow_fp8_t5 = vram_config is None
    if not allow_fp8_t5:
        logger.info(
            "[Wan2] low_vram enabled: prefer int8/bf16 umt5 over fp8 (Float8Tensor incompatible with VRAM offload)"
        )
    t5_files = _glob_first_match_in_roots(shared_roots, _t5_patterns_for_system(allow_fp8=allow_fp8_t5))

    # - VAE：优先明确版本名，其次宽松匹配（“包含 VAE/vae”）
    vae_files = _glob_first_match_in_roots(
        shared_roots,
        [
            "Wan2.1_VAE.*",
            "Wan2.2_VAE.*",
            "*VAE*.*",
            "vae*.safetensors",
            "*vae*.safetensors",
            "*vae*.*",
        ],
    )

    # - CLIP：可选（Wan2.1 I2V/Fun 等需要；Wan2.2 多数不需要；存在就加载，不存在就跳过）
    clip_files = _glob_optional_first_match_in_roots(
        shared_roots,
        [
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.*",
            "*clip*vit*huge*14*.*",
        ],
    )

    # - S2V：音频组件（可选，但 name==s2v 时应视为必需）
    audio_processor_config: Optional[ModelConfig] = None
    wav2vec2_model_files = _glob_optional_first_match(
        root,
        [
            "wav2vec2-large-xlsr-53-english/model.*",
            "wav2vec2*/model.*",
        ],
    )
    wav2vec2_dir = None
    if (root / "wav2vec2-large-xlsr-53-english").exists():
        wav2vec2_dir = root / "wav2vec2-large-xlsr-53-english"
    else:
        # 兼容：不同打包可能把 wav2vec2 目录名略微变化
        candidates = sorted([p for p in root.glob("wav2vec2*") if p.is_dir()])
        wav2vec2_dir = candidates[0] if candidates else None

    if name == "s2v":
        if not wav2vec2_model_files:
            raise FileNotFoundError(f"S2V audio encoder not found under: {root}")
        if not (wav2vec2_dir and wav2vec2_dir.exists()):
            raise FileNotFoundError(f"S2V audio processor dir not found under: {root}")
        audio_processor_config = ModelConfig(path=str(wav2vec2_dir))

    model_configs: list[ModelConfig] = []

    # 组件顺序的关键点：若是双扩散，必须先 high 再 low，保证 dit/dit2 对应
    if diffusion_high is not None:
        model_configs.append(ModelConfig(path=diffusion_high, **vram_kwargs))
    if diffusion_low is not None:
        model_configs.append(ModelConfig(path=diffusion_low, **vram_kwargs))

    # S2V 音频 encoder 权重需要被 auto_load_model 识别
    if name == "s2v" and wav2vec2_model_files is not None:
        model_configs.append(ModelConfig(path=wav2vec2_model_files, **vram_kwargs))

    # 主干组件
    model_configs.append(ModelConfig(path=t5_files, **vram_kwargs))
    model_configs.append(ModelConfig(path=vae_files, **vram_kwargs))

    # 可选组件：只要存在就加载（避免 Wan2.1 缺失 CLIP 导致 I2V/Fun 等失败）
    # 注：t2v/s2v 通常不需要 clip，但加载了也不会被使用；为了省内存，t2v/s2v 默认不加载。
    if clip_files is not None and name in ("i2v", "control", "animate", "inp", "ccv", "ti2v"):
        model_configs.append(ModelConfig(path=clip_files, **vram_kwargs))

    return model_configs, tokenizer_config, audio_processor_config


def compute_wan2_pipe_key(
    *,
    name: str,
    models_base_dir: str,
    weights_base_dir: Optional[str] = None,
    model_ref_override: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: str = "bf16",
    vram_limit: Optional[float] = None,
    low_vram: bool = False,
) -> Wan2PipeKey:
    dev = device or _device_default()
    effective_model_ref = model_ref_override or _default_model_ref_for_name(name)
    model_root_path = _resolve_model_root(models_base_dir, effective_model_ref)
    tokenizer_root = str(_resolve_tokenizer_root(models_base_dir, model_root=model_root_path, weights_base_dir=weights_base_dir))
    model_root = str(model_root_path)
    audio_root = ""
    if name == "s2v":
        # 缓存 key 里带上 audio_processor_root，避免未来扩展成“不同 wav2vec2 目录”时误复用
        ap = model_root_path / "wav2vec2-large-xlsr-53-english"
        audio_root = str(ap) if ap.exists() else ""
    # 仅当启用 low_vram 时，给 diffsynth 一个更合理的 vram_limit 默认值
    vram_limit2 = vram_limit
    if low_vram and vram_limit2 is None:
        vram_limit2 = _auto_vram_limit_gb(str(dev))
    vram_limit_sig = f"{float(vram_limit2):.3f}" if isinstance(vram_limit2, (int, float)) else ""
    key = Wan2PipeKey(
        name=name,
        model_root=model_root,
        tokenizer_root=tokenizer_root,
        audio_processor_root=audio_root,
        device=str(dev),
        torch_dtype=torch_dtype,
        low_vram=bool(low_vram),
        vram_limit_gb=vram_limit_sig,
    )
    return key


def create_wan2_pipe(
    *,
    name: str,
    models_base_dir: str,
    weights_base_dir: Optional[str] = None,
    model_ref_override: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: str = "bf16",
    vram_limit: Optional[float] = None,
    low_vram: bool = False,
) -> WanVideoPipeline:
    """
    创建一个新的 Wan2 pipeline（无缓存）。

    缓存/TTL/驱逐释放由上层（VideoInferrer + PipelineCache）负责。
    """
    dev = device or _device_default()
    # key 的计算/日志会用到 resolved model_ref（默认值仅用于“未传 load_name 也能跑”）
    effective_model_ref = model_ref_override or _default_model_ref_for_name(name)
    _ = compute_wan2_pipe_key(
        name=name,
        models_base_dir=models_base_dir,
        weights_base_dir=weights_base_dir,
        model_ref_override=effective_model_ref,
        device=str(dev),
        torch_dtype=torch_dtype,
        vram_limit=vram_limit,
        low_vram=low_vram,
    )

    dtype = _dtype_from_str(torch_dtype)
    vram_cfg = _default_low_vram_config(device=str(dev), torch_dtype=dtype) if low_vram else None
    vram_limit2 = vram_limit
    if low_vram and vram_limit2 is None:
        vram_limit2 = _auto_vram_limit_gb(str(dev))
    vram_limit_sig = f"{float(vram_limit2):.3f}" if isinstance(vram_limit2, (int, float)) else ""
    model_configs, tokenizer_config, audio_processor_config = build_wan2_model_configs(
        name,
        models_base_dir=models_base_dir,
        weights_base_dir=weights_base_dir,
        model_ref_override=effective_model_ref,
        vram_config=vram_cfg,
    )

    logger.info(
        f"Loading Wan2 pipeline (offline): name={name}, model_ref={model_ref_override}, "
        f"models_base_dir={models_base_dir}, device={dev}, dtype={torch_dtype}, "
        f"low_vram={bool(low_vram)} vram_limit_gb={vram_limit_sig} vram_config={_stable_sig(vram_cfg) if vram_cfg else ''}"
    )
    try:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=dtype,
            device=dev,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            vram_limit=vram_limit2,
        )
    except Exception as e:
        if _is_oom(e):
            logger.warning(f"Wan2 pipeline loading OOM: low_vram={bool(low_vram)} err={e}")
            cleanup_after_oom()
        raise
    return pipe


# Backward-compat: old callers may still import get_or_create_wan2_pipe.
# NOTE: 不再在此函数内做全局缓存；如需缓存请使用 VideoInferrer 的 PipelineCache。
def get_or_create_wan2_pipe(**kwargs) -> WanVideoPipeline:  # type: ignore[override]
    return create_wan2_pipe(**kwargs)


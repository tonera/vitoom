from __future__ import annotations

import os
from dataclasses import dataclass
import inspect
import time
from typing import Optional, Tuple

import torch

from .logging_utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen3LocalPaths:
    """
    - 传目录：目录里应包含 config/tokenizer/权重等（HF 标准结构）
    - 传 safetensors：需要你额外指定 config_dir/tokenizer_dir（本地），用于构建模型与 tokenizer
    """

    model_or_weights_path: str
    config_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None


def _pick_base_model_for_hidden_states(m):
    # sd-scripts 使用 AutoModelForCausalLM(...).model
    # 这里做个兼容：优先取 `.model`，否则返回自身（要求 forward 输出含 last_hidden_state）
    return getattr(m, "model", m)


def _strip_known_prefixes(sd: dict) -> dict:
    """
    Normalize common HF checkpoint prefixes so we can load into the base model.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
        elif k.startswith("base_model.model."):
            out[k[len("base_model.model.") :]] = v
        else:
            out[k] = v
    return out


def _collect_module_state_keys(m: torch.nn.Module) -> set[str]:
    keys: set[str] = set()
    for n, _ in m.named_parameters(recurse=True):
        keys.add(n)
    for n, _ in m.named_buffers(recurse=True):
        keys.add(n)
    return keys


def _transform_state_dict_for_target(sd: dict, mode: str) -> dict:
    if mode == "as_is":
        return sd
    if mode == "strip_model_prefix":
        return {(_k[len("model.") :] if _k.startswith("model.") else _k): v for _k, v in sd.items()}
    if mode == "strip_known_prefixes":
        return _strip_known_prefixes(sd)
    raise ValueError(f"Unknown transform mode: {mode}")


def _choose_best_loading_plan(
    *,
    raw_sd: dict,
    wrapper: torch.nn.Module,
    base: Optional[torch.nn.Module],
) -> tuple[torch.nn.Module, str, int]:
    """
    Pick (target_module, transform_mode, match_count) to maximize key name matches.
    Avoids the common pitfall: stripping "model." but loading into wrapper, or vice versa.
    """
    candidates: list[tuple[str, torch.nn.Module, str]] = []
    candidates.append(("wrapper", wrapper, "as_is"))
    candidates.append(("wrapper", wrapper, "strip_model_prefix"))
    candidates.append(("wrapper", wrapper, "strip_known_prefixes"))
    if base is not None and base is not wrapper:
        candidates.append(("base", base, "as_is"))
        candidates.append(("base", base, "strip_model_prefix"))
        candidates.append(("base", base, "strip_known_prefixes"))

    best = None
    best_match = -1
    for name, target, mode in candidates:
        keys = _collect_module_state_keys(target)
        sd_t = _transform_state_dict_for_target(raw_sd, mode)
        match = sum(1 for k in sd_t.keys() if k in keys)
        if match > best_match:
            best_match = match
            best = (target, mode, match)

    assert best is not None
    return best  # (target_module, transform_mode, match_count)


def load_qwen3_text_encoder(
    paths: Qwen3LocalPaths,
    *,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    loading_device: Optional[str] = None,
    pretouch_cpu_before_to_cuda: bool = False,
) -> Tuple[torch.nn.Module, "object"]:
    """
    返回 (text_encoder_model, tokenizer)
    其中 text_encoder_model.forward(...) 需能返回 `.last_hidden_state`
    """
    import transformers
    from transformers import AutoTokenizer

    qwen3_path = paths.model_or_weights_path
    logger.info(f"Loading Qwen3 from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(
            qwen3_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        # 使用 CausalLM 再取 .model，兼容 sd-scripts 逻辑
        m = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model = _pick_base_model_for_hidden_states(m)
    else:
        if paths.config_dir is None or paths.tokenizer_dir is None:
            raise ValueError(
                "当 qwen3_path 是 .safetensors 单文件时，必须提供 Qwen3LocalPaths(config_dir=..., tokenizer_dir=...)"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            paths.tokenizer_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        # 优先使用 transformers 内置 Qwen3 类；否则走 AutoConfig + from_config
        if hasattr(transformers, "Qwen3Config") and hasattr(transformers, "Qwen3ForCausalLM"):
            cfg = transformers.Qwen3Config.from_pretrained(paths.config_dir, local_files_only=local_files_only)
            m = transformers.Qwen3ForCausalLM(cfg)
        else:
            cfg = transformers.AutoConfig.from_pretrained(
                paths.config_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            m = transformers.AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)

        # load weights
        from safetensors.torch import load_file

        # loading_device 用于控制 safetensors 读入的位置（CPU 或直接 CUDA），以减少 H2D 时间
        if loading_device is None:
            sd_device = "cpu"
        else:
            sd_device = str(loading_device)

        t_load0 = time.perf_counter()
        sd = load_file(qwen3_path, device=sd_device)
        t_load1 = time.perf_counter()
        logger.info(f"Qwen3 safetensors load_file({sd_device}) time: {t_load1 - t_load0:.2f}s")

        base = _pick_base_model_for_hidden_states(m)
        # auto-pick best loading target + transform
        target, mode, matched = _choose_best_loading_plan(raw_sd=sd, wrapper=m, base=base)
        sd_t = _transform_state_dict_for_target(sd, mode)
        load_sig = inspect.signature(target.load_state_dict)
        t_sd0 = time.perf_counter()
        if "assign" in load_sig.parameters:
            info = target.load_state_dict(sd_t, strict=False, assign=True)
        else:
            info = target.load_state_dict(sd_t, strict=False)
        t_sd1 = time.perf_counter()
        logger.info(f"Qwen3 load_state_dict time: {t_sd1 - t_sd0:.2f}s")

        miss = len(info.missing_keys)
        unexp = len(info.unexpected_keys)
        logger.info(
            f"Loaded Qwen3 weights (strict=False) target={'base' if target is base else 'wrapper'} "
            f"mode={mode} matched_keys={matched}/{len(sd)} missing={miss} unexpected={unexp}"
        )
        if miss and miss <= 5:
            logger.info(f"Qwen3 missing keys: {info.missing_keys}")
        if unexp and unexp <= 5:
            logger.info(f"Qwen3 unexpected keys: {info.unexpected_keys}")
        if matched < max(1, int(0.8 * len(sd))):
            logger.warning(
                "Qwen3 权重 key 匹配率偏低：这可能导致提示词理解异常或效果不稳定。"
                "建议优先使用本地 HF 完整目录加载（model_or_weights_path 指向目录），"
                "或确认 config_dir/tokenizer_dir 与该 .safetensors 对应。"
            )

        # always return base model for embeddings
        model = base

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.requires_grad_(False)
    # 如果模型还在 CPU 且要迁移到 CUDA，可选先做 pretouch，避免 mmap 缺页风暴
    if pretouch_cpu_before_to_cuda and model.device.type == "cpu" and str(device).startswith("cuda"):
        from .torch_transfer_utils import pretouch_module_cpu_tensors

        t_mat0 = time.perf_counter()
        pretouch_module_cpu_tensors(model)
        t_mat1 = time.perf_counter()
        logger.info(f"Qwen3 pretouch CPU tensors: {t_mat1 - t_mat0:.2f}s")

    t_move0 = time.perf_counter()
    model = model.to(device=device, dtype=dtype)
    t_move1 = time.perf_counter()
    logger.info(f"Qwen3 model.to({device}) time: {t_move1 - t_move0:.2f}s")
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def load_t5_tokenizer(
    t5_tokenizer_dir: str,
    *,
    local_files_only: bool = True,
) -> "object":
    from transformers import T5TokenizerFast

    # 要求用户提供本地 tokenizer 目录（更可控/可复制）
    return T5TokenizerFast.from_pretrained(t5_tokenizer_dir, local_files_only=local_files_only)


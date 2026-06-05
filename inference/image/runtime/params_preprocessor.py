from __future__ import annotations

import re
from typing import Any, Optional

from common.Constant import JT_ED, JT_POSE, JT_RBG, JT_SR, JT_FS
from common.family_utils import to_model_family
from common.image_utils import load_image, constrain_size
from image.runtime.lora_manager import build_lora_list, append_trigger_words_to_prompt
from image.runtime.prompt_utils import sanitize_prompt
from schemas import InferenceRequestParams


SDXL_PONY_PROMPT_PREFIXES = ["score_9", "score_8_up", "score_7_up"]
SDXL_DEFAULT_PROMPT_PREFIXES = ["masterpiece", "best quality"]
EDITOR_MULTI_IMAGE_FAMILIES = {"flux2_klein", "qwen.edit"}
EDITOR_SINGLE_IMAGE_FAMILIES = {"flux_kontext"}
EDITOR_SPECIAL_MODEL_NAMES = {"flux.1-depth-dev", "flux.1-canny-dev"}


def _normalize_prompt_for_contains(s: str) -> str:
    return " ".join((s or "").lower().replace("_", " ").replace("-", " ").split())


def _prepend_prompt_tokens(prompt: str, tokens: list[str]) -> str:
    p = str(prompt or "").strip()
    if not tokens:
        return p
    p_norm = _normalize_prompt_for_contains(p)
    to_add: list[str] = []
    for t in tokens:
        t0 = str(t or "").strip()
        if not t0:
            continue
        t_norm = _normalize_prompt_for_contains(t0)
        if t_norm and t_norm in p_norm:
            continue
        to_add.append(t0)
    if not to_add:
        return p
    prefix = ", ".join(to_add)
    return f"{prefix}, {p}" if p else prefix


async def preprocess_inference_params(
    params: InferenceRequestParams,
    *,
    detector: Any,
    inference_config: Any,
    logger: Any,
    run_blocking: Optional[Any] = None,
) -> InferenceRequestParams:
    """
    推理前参数预处理（扁平入口，集中管理，避免调用长链）：
    - MK 图生图 url 探测：不可加载则回退文生图（清空 url），避免选错 img2img pipeline
    - family 归一化：必要时触发 detector 自动侦测
    - RBG 强制 png
    - SDXL prompt 自动加前缀（幂等）
    - LoRA：解析并缓存到 params.parsed_loras（sanitize 前），可选 trigger_word 拼接到 prompt，然后 sanitize
    - 宽高约束：总像素超过 1280×1280 则等比缩小，宽高对齐 16 倍数
    """

    # 1) url 探测回退（必须最早做：影响 is_img2img 的 pipeline 选择）
    if params.job_type not in (JT_ED, JT_POSE) and getattr(params, "url", None):
        if run_blocking:
            probe = await run_blocking(lambda: load_image(params.url))
        else:
            probe = load_image(params.url)
        if probe is None:
            logger.info("url image not loadable; fallback to text2img (clearing params.url)")
            params.url = None

    # 2) family 归一化（必要时自动侦测）
    required_mv = params.job_type not in (JT_RBG, JT_SR, JT_FS)
    raw_class = str(getattr(params, "family", "") or "").strip()
    if raw_class:
        fam = to_model_family(raw_class)
        if not fam:
            raise ValueError(f"Unsupported family={raw_class} (cannot map to canonical family)")
        params.family = fam
    else:
        raw_mv = str(getattr(params, "family", "") or "").strip()
        if raw_mv:
            params.family = raw_mv
            fam = to_model_family(raw_mv)
            if not fam:
                raise ValueError(f"Unsupported family={raw_mv} (cannot map to canonical family)")
            params.family = fam
        else:
            if required_mv:
                load_name = str(getattr(params, "load_name", "") or "").strip()
                if not load_name:
                    raise ValueError("family missing and load_name missing; cannot auto-detect model type")
                _ = detector.get_pipeline(params)
                detected = getattr(detector, "family", None)
                fam = to_model_family(detected)
                if not fam:
                    raise ValueError("family missing and detector failed to detect model type")
                params.family = fam

    # 3) job_type 规则
    if params.job_type == JT_RBG:
        params.file_type = "png"

    if params.job_type in (JT_ED, JT_POSE):
        tpl_list = list(getattr(params, "tpl_list", None) or [])
        if not tpl_list:
            raise ValueError(f"{params.job_type} requires non-empty tpl_list")

        load_name = str(getattr(params, "load_name", "") or "").strip().lower()
        if load_name in EDITOR_SPECIAL_MODEL_NAMES:
            pass
        elif mv := str(getattr(params, "family", "") or "").strip().lower():
            if mv not in EDITOR_SINGLE_IMAGE_FAMILIES and mv not in EDITOR_MULTI_IMAGE_FAMILIES:
                raise ValueError(
                    f"{params.job_type} only supports flux_kontext / flux2_klein / qwen.edit "
                    "or load_name=FLUX.1-Depth-dev / FLUX.1-Canny-dev"
                )
            if mv in EDITOR_MULTI_IMAGE_FAMILIES and len(tpl_list) > 9:
                raise ValueError(f"{params.job_type} supports at most 9 input images for family={mv}")
        else:
            raise ValueError(f"{params.job_type} requires a supported family or load_name")

    # 4) SDXL prompt 前缀（保持与旧实现一致：按 load_name/family 判 pony）
    mv = (getattr(params, "family", "") or "").strip().lower()
    if mv == "sdxl":
        mc = (getattr(params, "family", "") or "").strip().lower()
        mn = (getattr(params, "load_name", "") or "").strip().lower()
        is_pony = mc == "pony" or (mn and re.search(r"(?:^|[^a-z0-9])pony(?:$|[^a-z0-9])", mn))
        params.prompt = _prepend_prompt_tokens(
            params.prompt or "",
            SDXL_PONY_PROMPT_PREFIXES if is_pony else SDXL_DEFAULT_PROMPT_PREFIXES,
        )

    # 5) LoRA + prompt sanitize（关键：先解析并缓存，再 sanitize（会移除 <lora...>））
    lora_list = build_lora_list(params.prompt or "", getattr(params, "loras", None))
    params.parsed_loras = lora_list
    if lora_list:
        params.prompt = append_trigger_words_to_prompt(params.prompt or "", lora_list)
    params.prompt = sanitize_prompt(params.prompt or "")

    # 6) 尺寸约束：总像素超过 1280×1280 则等比缩小，宽高对齐 16 倍数
    try:
        w = int(getattr(params, "width", 0) or 0)
        h = int(getattr(params, "height", 0) or 0)
        if w > 0 and h > 0:
            nw, nh = constrain_size(w, h)
            if (nw, nh) != (w, h):
                logger.info(f"constrain_size: ({w}x{h}) -> ({nw}x{nh})")
                params.width = nw
                params.height = nh
    except Exception:
        pass

    return params


"""
Wan2 (diffsynth) LoRA manager for video inference.

Why a separate manager?
- Image side uses diffusers adapters / nunchaku compose logic.
- WanVideoPipeline uses diffsynth BasePipeline.load_lora(module, ...).

Design constraints:
- Per-request LoRA must NOT permanently mutate cached base weights.
  diffsynth BasePipeline.load_lora() is only reversible when running in "hotload" mode
  (VRAM management enabled + AutoWrappedLinear), which we get in our low_vram pipeline.
- Therefore: we require `pipe.vram_management_enabled == True` for LoRA at inference time.
  In handlers we automatically enable low_vram when loras are specified.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional


def build_lora_list(prompt: str, loras_param: Any) -> list[dict]:
    """
    Merge two LoRA sources:
    1) prompt tags: <lora:name:0.8>
    2) loras_param: JSON string / list / dict (preserves extra fields like target)

    Return unified list: [{name, weights, source, trigger_word?, target?/targets?}, ...]
    - loras_param overrides prompt tag when names collide
    - max 5
    """
    merged: dict[str, dict] = {}

    for item in _parse_loras_from_prompt(prompt or ""):
        key = _normalize_lora_key(item.get("name", ""))
        if key:
            merged[key] = item

    for item in _parse_loras_from_params(loras_param):
        key = _normalize_lora_key(item.get("name", ""))
        if key:
            merged[key] = item

    out = list(merged.values())
    return out[:5] if len(out) > 5 else out


def append_trigger_words_to_prompt(prompt: str, lora_list: list[dict]) -> str:
    base = (prompt or "").strip()
    if not lora_list:
        return prompt

    seen: set[str] = set()
    words: list[str] = []
    for item in lora_list:
        if not isinstance(item, dict):
            continue
        tw_raw = item.get("trigger_word") or item.get("triggerWord") or ""
        if not tw_raw:
            continue
        for part in str(tw_raw).split(","):
            w = part.strip()
            if not w:
                continue
            k = w.lower()
            if k in seen:
                continue
            seen.add(k)
            words.append(w)

    if not words:
        return prompt
    suffix = ",".join(words)
    if not base:
        return suffix
    if base.endswith(","):
        return f"{base}{suffix}"
    return f"{base},{suffix}"


def apply_wan2_loras(
    *,
    pipe: Any,
    lora_list: list[dict],
    loras_dir: str,
    logger: Any,
) -> None:
    """
    Apply LoRAs to WanVideoPipeline via diffsynth BasePipeline.load_lora.
    Requires low_vram pipeline (vram_management_enabled) so LoRA can be cleared per request.
    """
    if not lora_list:
        return

    if not bool(getattr(pipe, "vram_management_enabled", False)):
        raise RuntimeError("Wan2 LoRA requires low_vram pipeline (vram_management_enabled). Please enable force_offload/low_vram.")

    # Clear previous hotloaded loras (per-request isolation)
    try:
        if hasattr(pipe, "clear_lora"):
            pipe.clear_lora(verbose=0)
    except Exception:
        pass

    loras_dir_abs = os.path.abspath(loras_dir) if loras_dir else ""
    if loras_dir_abs and (not os.path.isdir(loras_dir_abs)):
        logger.warning(f"[Wan2][LoRA] loras_dir not found: {loras_dir_abs}")

    targets_default = _default_targets_for_pipe(pipe)
    for item in lora_list:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        w = item.get("weights", item.get("weight", item.get("value", 1.0)))
        try:
            alpha = float(w)
        except Exception:
            alpha = 1.0

        path = _resolve_lora_path(loras_dir_abs, name)
        if not path or not os.path.exists(path):
            logger.warning(f"[Wan2][LoRA] file not found: name={name} resolved={path}")
            continue

        targets = _resolve_targets(item, targets_default)
        if not targets:
            logger.warning(f"[Wan2][LoRA] no valid target modules for lora={name}")
            continue

        for t in targets:
            m = getattr(pipe, t, None)
            if m is None:
                logger.warning(f"[Wan2][LoRA] target module missing: {t} (skip) lora={name}")
                continue
            try:
                pipe.load_lora(m, lora_config=path, alpha=alpha, hotload=True, verbose=0)
                logger.info(f"[Wan2][LoRA] loaded: {name} -> {t} alpha={alpha}")
            except Exception as e:
                logger.warning(f"[Wan2][LoRA] load failed: lora={name} target={t} err={e}")


def clear_wan2_loras(*, pipe: Any, logger: Any) -> None:
    """Best-effort clear hotloaded LoRA (no-op if none)."""
    try:
        if hasattr(pipe, "clear_lora"):
            pipe.clear_lora(verbose=0)
    except Exception as e:
        try:
            logger.debug(f"[Wan2][LoRA] clear_lora failed (ignored): {e}")
        except Exception:
            pass


def _normalize_lora_key(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    base = n.replace("\\", "/").split("/")[-1]
    if base.lower().endswith(".safetensors"):
        base = base[:-11]
    if base.lower().endswith(".ckpt"):
        base = base[:-5]
    return base.strip().replace(".", "_")


def _parse_loras_from_prompt(prompt: str) -> list[dict]:
    pattern = re.compile(r"<lora(.*?)>", flags=re.IGNORECASE)
    tags = pattern.findall(prompt or "")
    out: list[dict] = []
    for lora_str in tags:
        item = {"name": "", "weights": 0.9, "source": "prompt"}
        parts = lora_str.split(":")
        if len(parts) == 1:
            item["name"] = parts[0].strip()
        elif len(parts) == 2:
            item["name"] = parts[1].strip()
        else:
            item["name"] = (parts[1] if len(parts) > 1 else "").strip()
            w_raw = (parts[2] if len(parts) > 2 else "").strip()
            try:
                item["weights"] = float(w_raw)
            except Exception:
                item["weights"] = 0.9
        if item["name"]:
            out.append(item)
    return out


def _parse_loras_from_params(loras: Any) -> list[dict]:
    if loras is None:
        return []

    payload: Any = loras
    arr: Optional[Any] = None
    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            return []
        try:
            arr = json.loads(s)
        except Exception:
            return []
    elif isinstance(payload, list):
        arr = payload
    elif isinstance(payload, dict):
        if "name" in payload:
            arr = [payload]
        else:
            return []
    else:
        return []

    if not isinstance(arr, list):
        return []

    out: list[dict] = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        name = str(x.get("name", "")).strip()
        if not name:
            continue
        v = x.get("value", x.get("weight", 0.8))
        try:
            w = float(v)
        except Exception:
            w = 0.8
        item = dict(x)  # preserve extra keys like target/targets
        item["name"] = name
        item["weights"] = w
        item["source"] = "params"
        out.append(item)
    return out


def _resolve_lora_path(loras_dir_abs: str, name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    # absolute path takes priority
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw
    # allow relative path if exists
    if os.path.exists(raw):
        return os.path.abspath(raw)
    base = raw.replace("\\", "/").split("/")[-1]
    if base.lower().endswith((".safetensors", ".ckpt")):
        fname = base
    else:
        # default to safetensors (training/validate_lora uses safetensors)
        fname = f"{base}.safetensors"
    return os.path.join(loras_dir_abs, fname) if loras_dir_abs else fname


def _default_targets_for_pipe(pipe: Any) -> list[str]:
    # dit always exists; dit2 optional
    targets = []
    if getattr(pipe, "dit", None) is not None:
        targets.append("dit")
    if getattr(pipe, "dit2", None) is not None:
        targets.append("dit2")
    return targets


def _resolve_targets(item: dict, default_targets: list[str]) -> list[str]:
    # explicit override
    tgt = item.get("target")
    tgts = item.get("targets")
    if isinstance(tgts, list) and tgts:
        return [str(x).strip() for x in tgts if str(x).strip()]
    if isinstance(tgt, str) and tgt.strip():
        parts = [p.strip() for p in tgt.replace(";", ",").split(",") if p.strip()]
        return parts

    # heuristic: allow naming convention high/low noise
    name = str(item.get("name", "")).lower()
    if ("low_noise" in name or "dit2" in name) and "dit2" in default_targets:
        return ["dit2"]
    if ("high_noise" in name or "dit1" in name or "dit" in name) and "dit" in default_targets and "dit2" not in default_targets:
        return ["dit"]
    return default_targets


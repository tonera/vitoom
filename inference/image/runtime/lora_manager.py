"""
LoRA 解析/加载/卸载（从 inferrer.py 抽离）

目标：
- inferrer.py 只做流程编排，一眼能看清。
- 兼容两种来源：
  1) prompt 内 <lora:lora_name:0.8>
  2) request_params.loras（推荐为 JSON 字符串：'[{"name":"xxx","weight":0.8,"trigger_word":"..."}]'）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional


def _ensure_transformers_pytorch_gelu_tanh(*, logger: Any) -> None:
    """
    兼容某些环境中 transformers 版本较旧，缺少 transformers.activations.PytorchGELUTanh，
    但 diffusers / transformers 某些组件会尝试 `from transformers.activations import PytorchGELUTanh`
    导致 LoRA 加载链路直接失败。
    这里做一个最小运行时补丁：若缺失则注入等价实现，避免因版本差异阻塞推理。
    """
    try:
        import transformers.activations as _ta  # type: ignore
    except Exception:
        return

    if hasattr(_ta, "PytorchGELUTanh"):
        return

    try:
        import torch.nn as _nn  # type: ignore
        import torch.nn.functional as _F  # type: ignore
    except Exception:
        return

    class PytorchGELUTanh(_nn.Module):  # type: ignore
        def forward(self, input):  # noqa: A002
            try:
                return _F.gelu(input, approximate="tanh")
            except TypeError:
                # 兜底：旧 torch 可能不支持 approximate 参数
                return _F.gelu(input)

    try:
        setattr(_ta, "PytorchGELUTanh", PytorchGELUTanh)
        # best-effort：有些环境依赖 __all__
        if hasattr(_ta, "__all__") and isinstance(getattr(_ta, "__all__"), list):
            if "PytorchGELUTanh" not in _ta.__all__:
                _ta.__all__.append("PytorchGELUTanh")
        try:
            logger.warning("transformers.activations.PytorchGELUTanh 缺失：已注入兼容实现（用于 diffusers LoRA 加载）")
        except Exception:
            pass
    except Exception:
        return


def build_lora_list(prompt: str, loras_param: Any) -> list[dict]:
    """
    合并两种 LoRA 来源：
    1) prompt 标签：<lora:lora_name:0.8>（支持多个）
    2) loras_param：推荐为 JSON 字符串 '[{"name":"xxx","weight":0.8,"trigger_word":"..."}]'（也兼容 list；value/weight 均可）

    返回统一结构：[{name, weights, adapter_name, source, trigger_word?}, ...]
    - loras_param 优先级高于 prompt（同名覆盖）
    - 兜底最多 5 个
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
    """
    从 lora_list 中提取 trigger_word，用逗号追加到 prompt：
    - prompt: "画一只小猫"
    - trigger_word: "cover hand"
    => "画一只小猫,cover hand"
    """
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
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(w)

    if not words:
        return prompt

    suffix = ",".join(words)
    if not base:
        return suffix
    if base.endswith(","):
        return f"{base}{suffix}"
    return f"{base},{suffix}"


def load_loras_into_pipe(pipe: Any, family: str, loras_dir: str, lora_list: list[dict], *, logger: Any) -> None:
    """
    加载 LoRA：
    - 若 pipe.transformer 存在且是 nunchaku（类名/模块名前缀匹配），使用 nunchaku 的 compose_lora -> transformer.update_lora_params
    - 其他所有情况，统一按 diffusers adapters（load_lora_weights + set_adapters）处理
    """
    loras_dir_abs = os.path.abspath(loras_dir)

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and _is_nunchaku_transformer(transformer):
        # nunchaku  transformer只有 NunchakuFluxTransformer2dModel支持lora
        # try:
        #     from nunchaku import NunchakuFluxTransformer2dModel # type: ignore
        # except Exception:
        #     NunchakuFluxTransformer2dModel = None  # type: ignore
        # if not isinstance(transformer, NunchakuFluxTransformer2dModel):
        #     logger.warning("Only NunchakuFluxTransformer2dModel is supported for LoRA")
        #     return

        # nunchaku: compose_lora -> transformer.update_lora_params
        try:
            from nunchaku.lora.common.compose import compose_lora  # type: ignore

            tuples: list[tuple[str, float]] = []
            for item in lora_list:
                path = _resolve_lora_path(loras_dir_abs, str(item.get("name", "")))
                if path and os.path.exists(path):
                    tuples.append((path, float(item.get("weights", 0.8))))
                else:
                    logger.warning(f"LoRA file not found: {path}")

            if tuples:
                logger.info(f"Loading nunchaku LoRA: {tuples}")
                composed = compose_lora(tuples)
                transformer.update_lora_params(composed)
                try:
                    setattr(pipe, "_vitoom_evict_after_use", True)
                    setattr(pipe, "_vitoom_evict_reason", "nunchaku_lora")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to load nunchaku LoRA: {e}", exc_info=True)
        return

    # diffusers adapters
    try:
        try:
            unload_loras_from_pipe(pipe, family, logger=logger)
        except Exception:
            logger.debug("cleanup: reset diffusers LoRA state before load failed (ignored)", exc_info=True)
        _ensure_transformers_pytorch_gelu_tanh(logger=logger)
        adapters = []
        weights = []
        for item in lora_list:
            weight_name = _resolve_lora_weight_name(str(item.get("name", "")))
            if not weight_name:
                continue
            adapter_name_raw = str(item.get("adapter_name") or _normalize_lora_key(str(item.get("name", ""))) or "lora")
            adapter_name = _sanitize_adapter_name(adapter_name_raw)
            try:
                pipe.load_lora_weights(loras_dir_abs, weight_name=weight_name, adapter_name=adapter_name)
                adapters.append(adapter_name)
                weights.append(float(item.get("weights", 0.8)))
                logger.info(f"Loaded diffusers LoRA: {weight_name} adapter={adapter_name} w={weights[-1]}")
            except Exception as e:
                logger.warning(f"Failed to load diffusers LoRA {weight_name}: {e}")
        if adapters:
            pipe.set_adapters(adapters, adapter_weights=weights)
            try:
                active = pipe.get_list_adapters() if hasattr(pipe, "get_list_adapters") else None
                logger.info(f"LoRA adapters activated: {adapters}, weights={weights}, active={active}")
            except Exception:
                logger.info(f"LoRA adapters activated: {adapters}, weights={weights}")
    except Exception as e:
        logger.error(f"Failed to load diffusers LoRA: {e}", exc_info=True)


def _is_nunchaku_transformer(obj: Any) -> bool:
    """
    尽量宽松地识别 nunchaku transformer：
    - 类名以 Nunchaku 开头（不区分大小写）
    - 或模块名以 nunchaku 开头
    - 确认目前只有flux nunchaku 支持lora
    """
    if obj is None:
        return False
    cls = getattr(obj, "__class__", None)
    name = (getattr(cls, "__name__", "") or "").lower()
    mod = (getattr(cls, "__module__", "") or "").lower()
    return name.startswith("nunchaku") or mod.startswith("nunchaku")
    


def unload_loras_from_pipe(pipe: Any, family: str, *, logger: Any) -> None:
    """尽力卸载 LoRA（参考旧 tonera/LoadLora.py），避免长时间运行时累积状态。"""
    mv = (family or "").lower()
    transformer = getattr(pipe, "transformer", None)

    # nunchaku LoRA 直接改写 transformer 内部量化低秩权重，不能依赖 diffusers adapters 卸载。
    # reset_lora() 会恢复到 _init_lora_state() 快照，避免 pipeline cache 复用时串请求状态。
    try:
        if transformer is not None and _is_nunchaku_transformer(transformer) and hasattr(transformer, "reset_lora"):
            transformer.reset_lora()
            return
    except Exception:
        logger.debug("reset nunchaku LoRA failed (ignored)", exc_info=True)

    if mv.startswith("flux"):
        try:
            if hasattr(pipe, "unload_lora_weights"):
                try:
                    pipe.unload_lora_weights(reset_to_overwritten_params=True)
                except TypeError:
                    pipe.unload_lora_weights()
            if hasattr(pipe, "unfuse_lora"):
                pipe.unfuse_lora()
        except Exception:
            logger.debug("cleanup: unload flux LoRA failed (ignored)", exc_info=True)
        return

    try:
        if hasattr(pipe, "get_list_adapters"):
            adapters = pipe.get_list_adapters()
            if adapters:
                logger.debug(f"Active adapters before unload: {adapters}")
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        if hasattr(pipe, "unfuse_lora"):
            pipe.unfuse_lora()
    except Exception:
        pass


def _normalize_lora_key(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    base = n.replace("\\", "/").split("/")[-1]
    if base.lower().endswith(".safetensors"):
        base = base[:-11]
    # PyTorch module name cannot contain '.' (used as hierarchy separator in state_dict keys)
    base = base.strip().replace(".", "_")
    return base


def _sanitize_adapter_name(name: str) -> str:
    """
    PyTorch nn.Module.add_module(name, ...) forbids '.' in module names.
    Keep adapter_name stable and safe.
    """
    raw = (name or "").strip()
    if not raw:
        return "lora"
    s = raw.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "lora"


def _parse_loras_from_prompt(prompt: str) -> list[dict]:
    """参考 tonera/DataFormat.py::getLoraList 解析 <lora...> 标签。"""
    pattern = re.compile(r"<lora(.*?)>", flags=re.IGNORECASE)
    tags = pattern.findall(prompt or "")
    out: list[dict] = []
    for lora_str in tags:
        item = {"name": "", "weights": 0.9, "adapter_name": "", "source": "prompt"}
        parts = lora_str.split(":")
        if len(parts) == 1:
            item["name"] = parts[0].strip()
        elif len(parts) == 2:
            item["name"] = parts[1].strip()
        else:
            item["name"] = (parts[1] if len(parts) > 1 else "").strip()
            w_raw = (parts[2] if len(parts) > 2 else "").strip()
            if re.match(r"^\d+(\.\d+)?$", w_raw or ""):
                item["weights"] = float(w_raw)
            else:
                item["weights"] = 0.9

        adapter_name = _normalize_lora_key(item["name"])
        item["adapter_name"] = _sanitize_adapter_name(adapter_name or item["name"].strip().split(".")[0])
        if item["name"]:
            out.append(item)
    return out


def _parse_loras_from_params(loras: Any) -> list[dict]:
    """解析 loras_param，支持 dict / JSON string / list。"""
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
        # allow single object payload: {"name": "...", "weight": 0.8, "trigger_word": "..."}
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
        adapter_name = _normalize_lora_key(name)
        out.append(
            {
                "name": name,
                "weights": w,
                "adapter_name": _sanitize_adapter_name(adapter_name or name.split(".")[0]),
                "source": "params",
                "trigger_word": str(x.get("trigger_word") or x.get("triggerWord") or "").strip(),
            }
        )
    return out


def _resolve_lora_weight_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    base = raw.replace("\\", "/").split("/")[-1]
    return base if base.lower().endswith(".safetensors") else f"{base}.safetensors"


def _resolve_lora_path(loras_dir_abs: str, name: str) -> str:
    weight_name = _resolve_lora_weight_name(name)
    if not weight_name:
        return ""
    return os.path.join(loras_dir_abs, weight_name)



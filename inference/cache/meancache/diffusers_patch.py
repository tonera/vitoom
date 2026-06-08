"""
Diffusers-style "one-liner" integration for MeanCache.

Goal: match TeaCache usage ergonomics:

    pipe = FluxPipeline.from_pretrained(...).to("cuda")
    apply_meancache_on_pipe(pipe, rel_l1_thresh=0.30, skip_budget=0.30, ...)
    image = pipe(prompt, num_inference_steps=50).images[0]
"""

from __future__ import annotations

import functools
import inspect
from types import MethodType
from typing import Any, Callable, Dict, Optional, Tuple

import torch  # type: ignore

from .config import MeanCacheConfig
from .engine import MeanCacheEngine


def _extract_kw_or_pos(name: str, args: tuple, kwargs: dict, pos_idx: int) -> Any:
    if name in kwargs:
        return kwargs.get(name)
    if len(args) > pos_idx:
        return args[pos_idx]
    return None


def _set_kw_or_pos(name: str, args: tuple, kwargs: dict, pos_idx: int, value: Any) -> Tuple[tuple, dict]:
    if name in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs[name] = value
        return args, new_kwargs
    if len(args) > pos_idx:
        new_args = list(args)
        new_args[pos_idx] = value
        return tuple(new_args), kwargs
    # fallback: set as kwarg
    new_kwargs = dict(kwargs)
    new_kwargs[name] = value
    return args, new_kwargs


def _extract_sample_and_packer(out: Any) -> Tuple[Optional[torch.Tensor], Callable[[torch.Tensor], Any]]:
    """
    Return (sample_tensor, pack_fn).
    pack_fn(sample) must reconstruct an output compatible with `out`.
    """

    def _identity(x: torch.Tensor) -> Any:
        return x

    def _find_first_tensor(obj: Any, path: list) -> Optional[Tuple[torch.Tensor, list]]:
        if isinstance(obj, torch.Tensor):
            return obj, path
        if isinstance(obj, (list, tuple)):
            for i, e in enumerate(obj):
                r = _find_first_tensor(e, path + [i])
                if r is not None:
                    return r
        return None

    def _set_by_path(obj: Any, path: list, new_tensor: torch.Tensor) -> Any:
        if len(path) == 0:
            return new_tensor
        idx = path[0]
        rest = path[1:]
        if isinstance(obj, list):
            vv = list(obj)
            vv[idx] = _set_by_path(vv[idx], rest, new_tensor)
            return vv
        if isinstance(obj, tuple):
            vv = list(obj)
            vv[idx] = _set_by_path(vv[idx], rest, new_tensor)
            return tuple(vv)
        # Unexpected type along the path: fall back
        return new_tensor

    if isinstance(out, torch.Tensor):
        return out, _identity

    # diffusers modeling outputs typically have `.sample`
    if hasattr(out, "sample") and isinstance(getattr(out, "sample"), torch.Tensor):
        template = out

        def _pack(x: torch.Tensor) -> Any:
            try:
                return type(template)(sample=x)
            except Exception:
                # best-effort fallback: return tensor
                return x

        return getattr(out, "sample"), _pack

    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        template = out

        def _pack(x: torch.Tensor) -> Any:
            return (x,) + tuple(template[1:])

        return out[0], _pack

    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        template = out

        def _pack(x: torch.Tensor) -> Any:
            vv = list(template)
            vv[0] = x
            return vv

        return out[0], _pack

    # Generic nested (tuple/list) case: find first tensor anywhere inside.
    nested = _find_first_tensor(out, [])
    if nested is not None:
        sample, path = nested

        def _pack(x: torch.Tensor) -> Any:
            return _set_by_path(out, path, x)

        return sample, _pack

    return None, _identity


def apply_meancache_on_pipe(
    pipe: Any,
    *,
    rel_l1_thresh: float = 0.80,
    skip_budget: float = 0.40,
    start_step: int = 2,
    end_step: int = -1,
    cache_device: str = "cpu",
    enable_pssp: bool = True,
    peak_threshold: float = 1.0,
    gamma: float = 1.0,
    max_accumulated_error: float = 1.0,
    max_cache_span: int = 3,
    assume_cfg_batch: bool = False,
    debug: bool = False,
    preset_name: str = "Custom",
    target: str = "auto",  # auto|transformer|unet
) -> Any:
    """
    对 Diffusers pipeline 做“实例级”patch，以一行方式启用 MeanCache（类似 TeaCache 的体验）。

    行为：
    - 会用 per-instance subclass 的方式覆盖 `pipe.__call__`（避免污染全局 class），并在每次推理前 `reset` 状态。
    - 会 wrap 目标模块（默认优先 `pipe.transformer`，否则 `pipe.unet`）的 `forward`，在部分 step 直接 skip 掉真实 forward，
      用缓存的 velocity + JVP 修正来预测当前步输出，从而减少 compute 次数。

    你关心的几个核心参数（调参方向很重要）：
    - **rel_l1_thresh**：skip 判定用的稳定性阈值（默认 0.80）。
      - 含义：主要用于对 online L_K / rel-L1 的“门槛”判断（值越大越容易 skip，速度更快但更可能伤画质）。
      - 经验：如果长期 `0 skipped`，可以逐步提高到 0.5/0.7/0.8 观察。
    - **skip_budget**：最大允许跳过的比例（0.0~0.5）。
      - 含义：只决定“最多能跳多少”，不保证一定跳到这个比例。
      - 越大越激进（例如 0.3 → 0.5）。
    - **peak_threshold**：PSSP 的“峰值抑制”硬阈值（默认 1.0）。
      - 含义：当某一步的 online 误差（velocity_similarity / accumulated_error）大于该阈值时，即使 schedule 允许也会强制 compute。
      - 越大越容易 skip（更快但更冒险）。
    - **max_accumulated_error**：累计误差上限（默认 1.0；上游实现默认更保守）。
      - 含义：当累计误差过大时会禁止继续 skip（避免漂移累积）。
      - 对某些 pipeline（比如 online L_K 偏大）需要调大（例如 1.0~2.0）才能真正触发 skip。
    - **gamma**：PSSP 的惩罚指数（默认 1.0）。
      - 含义：>1 会更强烈地抑制“峰值误差很大的 skip”（更稳但更慢）；越接近 1 越激进（更快但更可能出现质量波动）。
    """

    cfg = MeanCacheConfig(
        rel_l1_thresh=rel_l1_thresh,
        skip_budget=skip_budget,
        start_step=start_step,
        end_step=end_step,
        cache_device=cache_device,
        enable_pssp=enable_pssp,
        peak_threshold=peak_threshold,
        gamma=gamma,
        max_accumulated_error=max_accumulated_error,
        assume_cfg_batch=assume_cfg_batch,
        max_cache_span=max_cache_span,
        debug=debug,
        preset_name=preset_name,
    )

    # Reuse existing engine if already patched.
    engine: MeanCacheEngine
    if getattr(pipe, "_meancache_engine", None) is None:
        engine = MeanCacheEngine(cfg)
        pipe._meancache_engine = engine
    else:
        engine = pipe._meancache_engine
        engine.config = cfg  # type: ignore[assignment]
        engine.start_run(reset=True)

    module = _resolve_target_module(pipe, target=target)
    if module is None:
        raise ValueError("apply_meancache_on_pipe: cannot find target module (transformer/unet) on pipeline")

    _apply_meancache_on_module(module, engine)
    # Keep references for per-run diagnostics
    try:
        pipe._meancache_target_module = module
    except Exception:
        pass

    if debug and not getattr(pipe, "_meancache_patch_reported", False):
        has_tr = hasattr(pipe, "transformer")
        has_unet = hasattr(pipe, "unet")
        x_name = getattr(module, "_meancache_x_param_name", None)
        x_pos = getattr(module, "_meancache_x_param_pos", None)
        t_name = getattr(module, "_meancache_timestep_param_name", None)
        t_pos = getattr(module, "_meancache_timestep_param_pos", None)
        print(
            f"[MeanCache] patch pipe={type(pipe).__name__} has_transformer={has_tr} has_unet={has_unet} "
            f"target={target} module={type(module).__name__} x=({x_name}@{x_pos}) timestep=({t_name}@{t_pos})"
        )
        pipe._meancache_patch_reported = True

    # Patch pipeline __call__ (instance-isolated, like `fbcache_sdxl.apply_cache_on_pipe`)
    if not getattr(pipe, "_meancache_call_isolated", False):
        base_cls = pipe.__class__
        original_call = base_cls.__call__

        patched_cls = type(f"{base_cls.__name__}MeanCachePatched_{id(pipe)}", (base_cls,), {})
        try:
            root_base_cls = getattr(pipe, "_pipeline_base_class", None) or base_cls
            root_base_name = getattr(pipe, "_pipeline_base_class_name", None) or getattr(root_base_cls, "__name__", None)
            pipe._pipeline_base_class_name = root_base_name
            pipe._pipeline_base_class = root_base_cls
            pipe._meancache_pipeline_base_class_name = getattr(base_cls, "__name__", None)
            pipe._meancache_pipeline_base_class = base_cls
        except Exception:
            pass

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            # best-effort: infer num_inference_steps
            num_steps = kwargs.get("num_inference_steps", None)
            if num_steps is None and len(args) >= 2 and isinstance(args[1], int):
                # common signature: (prompt, num_inference_steps=..., ...)
                num_steps = args[1]

            # best-effort: get sigmas if scheduler already prepared (some pipelines set before loop)
            sigmas = None
            try:
                sched = getattr(self, "scheduler", None)
                sigmas = getattr(sched, "sigmas", None)
                if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
                    sigmas = sigmas.detach().clone()
                else:
                    sigmas = None
            except Exception:
                sigmas = None

            self._meancache_engine.start_run(total_steps=num_steps, sample_sigmas=sigmas, reset=True)

            out = original_call(self, *args, **kwargs)

            # Print simple summary (match upstream style).
            try:
                rep = self._meancache_engine.state.get(0)
                total = int(rep.get("step_index", 0)) if rep else 0
                skip_count = int(self._meancache_engine.state.get_skip_count(0)) if rep else 0
                compute_count = max(0, total - skip_count)
                skip_pct = (skip_count / total * 100.0) if total > 0 else 0.0
                speedup = (total / compute_count) if compute_count > 0 else 1.0
                if debug:
                    print(
                        f"[MeanCache] Sampling complete ({preset_name}): {total} steps, "
                        f"{skip_count} skipped, {compute_count} computed "
                        f"({skip_pct:.1f}% skip rate, ~{speedup:.2f}x speedup)"
                    )
                if debug and total == 0:
                    # Differentiate between "forward never called" vs "called but parse failed"
                    fwd_calls = None
                    try:
                        mod = getattr(self, "_meancache_target_module", None)
                        if mod is not None:
                            fwd_calls = getattr(mod, "_meancache_forward_calls", None)
                    except Exception:
                        fwd_calls = None

                    if isinstance(fwd_calls, int) and fwd_calls > 0:
                        print(
                            "[MeanCache] WARNING: 0 steps recorded but target forward WAS called. "
                            "This means x/timestep parsing failed in the wrapper. "
                            "Please paste one forward debug line above."
                        )
                    else:
                        print(
                            "[MeanCache] WARNING: 0 steps recorded and target forward was NOT called. "
                            "This means the pipeline is not using the patched module. "
                            "Try `apply_meancache_on_pipe(..., target='transformer')` or target='unet'."
                        )
            except Exception:
                pass
            return out

        patched_cls.__call__ = new_call
        pipe.__class__ = patched_cls
        pipe._meancache_call_isolated = True

    # Enable flag on module
    try:
        setattr(module, "enable_meancache", True)
    except Exception:
        pass

    return pipe


def _resolve_target_module(pipe: Any, *, target: str) -> Optional[Any]:
    if target == "transformer":
        return getattr(pipe, "transformer", None)
    if target == "unet":
        return getattr(pipe, "unet", None)
    # auto
    tr = getattr(pipe, "transformer", None)
    if tr is not None:
        return tr
    return getattr(pipe, "unet", None)


def _apply_meancache_on_module(module: Any, engine: MeanCacheEngine) -> Any:
    def _is_meancache_wrapped_forward(fn: Any) -> bool:
        try:
            f = getattr(fn, "__func__", fn)
            return bool(getattr(f, "_is_meancache_wrapper", False))
        except Exception:
            return False

    # Idempotent/self-healing: cached pipelines may restore/replace `forward` (e.g. via offload hooks).
    # If we already patched but the current `forward` is not our wrapper, re-apply the wrapper.
    if getattr(module, "_meancache_is_patched", False) and _is_meancache_wrapped_forward(getattr(module, "forward", None)):
        return module

    # Prefer the saved original forward (if present and not already wrapped) to avoid stacking wrappers.
    saved_original = getattr(module, "_meancache_original_forward", None)
    if callable(saved_original) and not _is_meancache_wrapped_forward(saved_original):
        original_forward = saved_original
    else:
        original_forward = module.forward
        module._meancache_original_forward = original_forward

    module._meancache_engine = engine
    module._meancache_last_template_by_pred: Dict[int, Any] = {}
    module._meancache_last_packer_by_pred: Dict[int, Callable[[torch.Tensor], Any]] = {}
    module._meancache_forward_calls = 0

    # Resolve parameter positions from signature (robust across UNet/Transformer variants)
    try:
        sig = inspect.signature(original_forward)
        params = [p.name for p in sig.parameters.values() if p.name != "self"]
    except Exception:
        params = []

    def _pos_of(name: str) -> Optional[int]:
        try:
            return params.index(name)
        except ValueError:
            return None

    x_param_name = None
    x_param_pos = None
    for cand in ("hidden_states", "sample", "input"):
        pos = _pos_of(cand)
        if pos is not None:
            x_param_name = cand
            x_param_pos = pos
            break
    if x_param_pos is None:
        x_param_name = "sample"
        x_param_pos = 0

    t_param_name = None
    t_param_pos = None
    for cand in ("timestep", "timesteps", "sigma"):
        pos = _pos_of(cand)
        if pos is not None:
            t_param_name = cand
            t_param_pos = pos
            break
    if t_param_pos is None:
        # fallback: common UNet case
        t_param_name = "timestep"
        t_param_pos = 1

    module._meancache_x_param_name = x_param_name
    module._meancache_x_param_pos = x_param_pos
    module._meancache_timestep_param_name = t_param_name
    module._meancache_timestep_param_pos = t_param_pos

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        try:
            self._meancache_forward_calls += 1
        except Exception:
            pass
        if not bool(getattr(self, "enable_meancache", True)):
            return original_forward(*args, **kwargs)

        # Some pipelines call forward with explicit self:
        #   module.forward(module, sample, timestep, ...)
        # In that case args[0] is `self` again. We must ignore it for parsing,
        # but preserve it when calling the original forward to keep behavior.
        explicit_self = bool(len(args) > 0 and args[0] is self)
        parse_args = args[1:] if explicit_self else args
        offset = 1 if explicit_self else 0

        # Try to locate x and sigma/timestep
        x_raw = _extract_kw_or_pos(x_param_name, parse_args, kwargs, x_param_pos)
        sigma = _extract_kw_or_pos(t_param_name, parse_args, kwargs, t_param_pos)

        # Z-Image transformer may pass `sample` as list/tuple (container).
        # We extract the first Tensor inside it for MeanCache, and remember its location
        # to put it back when we call the original forward.
        x_tensor: Optional[torch.Tensor] = x_raw if isinstance(x_raw, torch.Tensor) else None
        x_kind: Optional[str] = "direct" if isinstance(x_raw, torch.Tensor) else None
        x_elem_idx: Optional[int] = None
        if x_tensor is None and isinstance(x_raw, (list, tuple)):
            for i, e in enumerate(x_raw):
                if isinstance(e, torch.Tensor):
                    x_tensor = e
                    x_elem_idx = i
                    x_kind = "list" if isinstance(x_raw, list) else "tuple"
                    break

        # First-call debug to help integration across model variants
        try:
            if engine.config.debug and getattr(self, "_meancache_forward_calls", 0) <= 2:
                print(
                    f"[MeanCache][forward] module={type(self).__name__} "
                    f"explicit_self={explicit_self} args_len={len(args)} "
                    f"arg0_type={type(args[0]).__name__ if len(args) > 0 else None} "
                    f"x_raw_type={type(x_raw).__name__} "
                    f"x_ok={isinstance(x_tensor, torch.Tensor)} x_kind={x_kind} x_elem_idx={x_elem_idx} "
                    f"sigma_ok={sigma is not None} "
                    f"kw_keys={list(kwargs.keys())[:12]}"
                )
                if isinstance(x_raw, (list, tuple)):
                    et = [type(e).__name__ for e in list(x_raw)[:8]]
                    print(f"[MeanCache][forward] sample({type(x_raw).__name__}) len={len(x_raw)} head_types={et}")
                if isinstance(x_tensor, torch.Tensor):
                    print(
                        f"[MeanCache][forward] x_tensor shape={tuple(x_tensor.shape)} dtype={x_tensor.dtype} device={x_tensor.device}"
                    )
        except Exception:
            pass

        if not isinstance(x_tensor, torch.Tensor) or sigma is None:
            return original_forward(*args, **kwargs)

        # We'll call engine with a velocity_fn that calls original_forward and extracts a tensor sample.
        out_box: Dict[str, Any] = {"out": None, "pred": None}

        def velocity_fn(x_new: torch.Tensor, sigma_new, **kw) -> torch.Tensor:
            # Override x/sigma in the captured args/kwargs (best-effort)
            local_args, local_kwargs = args, dict(kwargs)
            # Put tensor back into original container shape if needed
            if x_kind == "direct" or x_elem_idx is None:
                x_replaced = x_new
            elif x_kind == "list":
                vv = list(x_raw)
                vv[x_elem_idx] = x_new
                x_replaced = vv
            elif x_kind == "tuple":
                vv = list(x_raw)
                vv[x_elem_idx] = x_new
                x_replaced = tuple(vv)
            else:
                x_replaced = x_new

            local_args, local_kwargs = _set_kw_or_pos(
                x_param_name, local_args, local_kwargs, x_param_pos + offset, x_replaced
            )
            local_args, local_kwargs = _set_kw_or_pos(
                t_param_name, local_args, local_kwargs, t_param_pos + offset, sigma_new
            )

            # QwenImage transformer (diffusers/nunchaku) requires either `max_txt_seq_len`
            # or `txt_seq_lens` to be provided. Some call-sites omit them; infer best-effort
            # from encoder hidden states / attention mask to avoid hard failure when fast_mode enables MeanCache.
            try:
                parse_local_args = local_args[1:] if explicit_self else local_args

                max_pos = _pos_of("max_txt_seq_len")
                lens_pos = _pos_of("txt_seq_lens")
                max_v = (
                    _extract_kw_or_pos("max_txt_seq_len", parse_local_args, local_kwargs, max_pos)
                    if ("max_txt_seq_len" in params)
                    else None
                )
                lens_v = (
                    _extract_kw_or_pos("txt_seq_lens", parse_local_args, local_kwargs, lens_pos)
                    if ("txt_seq_lens" in params)
                    else None
                )
                if max_v is None and lens_v is None and (("max_txt_seq_len" in params) or ("txt_seq_lens" in params)):
                    txt_src = None
                    for nm in (
                        "encoder_hidden_states",
                        "prompt_embeds",
                        "encoder_hidden_states_1",
                        "encoder_hidden_states_2",
                        "text_embeds",
                    ):
                        pos = _pos_of(nm)
                        if pos is None and nm not in local_kwargs:
                            continue
                        v = _extract_kw_or_pos(nm, parse_local_args, local_kwargs, pos if pos is not None else 10**9)
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            txt_src = v
                            break
                        if isinstance(v, (list, tuple)):
                            for e in v:
                                if isinstance(e, torch.Tensor) and e.ndim >= 2:
                                    txt_src = e
                                    break
                        if txt_src is not None:
                            break

                    attn_mask = None
                    if txt_src is None:
                        for nm in ("encoder_attention_mask", "attention_mask"):
                            pos = _pos_of(nm)
                            if pos is None and nm not in local_kwargs:
                                continue
                            v = _extract_kw_or_pos(nm, parse_local_args, local_kwargs, pos if pos is not None else 10**9)
                            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                                attn_mask = v
                                break

                    if txt_src is not None:
                        bsz = int(txt_src.shape[0])
                        seq = int(txt_src.shape[1])
                        if "max_txt_seq_len" in params:
                            local_args, local_kwargs = _set_kw_or_pos(
                                "max_txt_seq_len",
                                local_args,
                                local_kwargs,
                                (max_pos if max_pos is not None else 10**9) + offset,
                                seq,
                            )
                        elif "txt_seq_lens" in params:
                            lens = torch.full((bsz,), seq, dtype=torch.int64, device=txt_src.device)
                            local_args, local_kwargs = _set_kw_or_pos(
                                "txt_seq_lens",
                                local_args,
                                local_kwargs,
                                (lens_pos if lens_pos is not None else 10**9) + offset,
                                lens,
                            )
                    elif attn_mask is not None:
                        # Infer from mask (fallback): use max length and per-sample lengths if needed.
                        seq = int(attn_mask.shape[1])
                        if "max_txt_seq_len" in params:
                            local_args, local_kwargs = _set_kw_or_pos(
                                "max_txt_seq_len",
                                local_args,
                                local_kwargs,
                                (max_pos if max_pos is not None else 10**9) + offset,
                                seq,
                            )
                        elif "txt_seq_lens" in params:
                            try:
                                lens = attn_mask.to(torch.int64).sum(dim=-1)
                            except Exception:
                                lens = torch.full((int(attn_mask.shape[0]),), seq, dtype=torch.int64, device=attn_mask.device)
                            local_args, local_kwargs = _set_kw_or_pos(
                                "txt_seq_lens",
                                local_args,
                                local_kwargs,
                                (lens_pos if lens_pos is not None else 10**9) + offset,
                                lens,
                            )
            except Exception:
                pass

            out = original_forward(*local_args, **local_kwargs)
            sample, packer = _extract_sample_and_packer(out)
            out_box["out"] = out
            out_box["packer"] = packer
            if engine.config.debug and getattr(self, "_meancache_forward_calls", 0) <= 2:
                try:
                    ot = type(out).__name__
                    st = type(sample).__name__ if sample is not None else None
                    print(f"[MeanCache][forward] out_type={ot} extracted_sample_type={st}")
                except Exception:
                    pass
            if sample is None:
                # give up caching for unknown output
                raise RuntimeError("MeanCache: cannot extract tensor sample from module output")
            return sample

        try:
            sample_tensor = engine.forward(velocity_fn, x_tensor, sigma)
        except Exception:
            # fall back to original compute if anything goes wrong
            return original_forward(*args, **kwargs)

        # If computed, return the original output we already got.
        if out_box.get("out") is not None:
            out = out_box["out"]
            # Store template for future skip packing (split/batch handled by engine internally via sigma)
            pred_id = int(getattr(engine, "_last_pred_id", 0))
            self._meancache_last_template_by_pred[pred_id] = out
            self._meancache_last_packer_by_pred[pred_id] = out_box.get("packer")  # type: ignore[assignment]
            return out

        # If skipped: pack tensor output using last template; if missing, compute.
        pred_id = int(getattr(engine, "_last_pred_id", 0))
        template = self._meancache_last_template_by_pred.get(pred_id)
        packer = self._meancache_last_packer_by_pred.get(pred_id)
        if template is None or packer is None:
            return original_forward(*args, **kwargs)
        return packer(sample_tensor)

    # Mark wrapper for robust detection across cached pipeline reuse.
    try:
        setattr(new_forward, "_is_meancache_wrapper", True)
    except Exception:
        pass

    module.forward = MethodType(new_forward, module)
    module._meancache_is_patched = True
    return module


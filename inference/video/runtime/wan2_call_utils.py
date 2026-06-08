"""
Wan2 推理调用小工具：
- 根据 pipe.__call__ 的签名过滤 kwargs，避免不同版本/不同 pipeline 参数差异导致 TypeError。
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Iterable, Optional

from common.logger import print_info
from common.task_cancel import TaskCancelledError


def filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    如果 fn 支持 **kwargs，则原样返回；
    否则仅保留签名中存在的参数名。
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    # 若存在 VAR_KEYWORD(**kwargs)，无需过滤
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return dict(kwargs)

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _build_cancelable_progress_bar(
    *,
    base_progress_bar_cmd: Optional[Callable[[Iterable[Any]], Iterable[Any]]],
    task_id: str,
    stage: str,
    is_task_cancelled: Callable[[], bool],
) -> Callable[[Iterable[Any]], Iterable[Any]]:
    def _progress_bar_cmd(iterable: Iterable[Any]) -> Iterable[Any]:
        wrapped = iterable
        if callable(base_progress_bar_cmd):
            try:
                wrapped = base_progress_bar_cmd(iterable)
            except Exception:
                wrapped = iterable

        try:
            for step_index, item in enumerate(wrapped, start=1):
                if is_task_cancelled():
                    raise TaskCancelledError(task_id, f"{stage} step={step_index}")
                yield item
        finally:
            close = getattr(wrapped, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return _progress_bar_cmd


def call_pipe(
    pipe: Any,
    *,
    task_id: Optional[str] = None,
    stage: str = "wan2 inference",
    is_task_cancelled: Optional[Callable[[], bool]] = None,
    **kwargs: Any,
) -> Any:
    """
    调用 pipe(...)，并按 pipe.__call__ 的签名过滤 kwargs。
    """
    if task_id and callable(is_task_cancelled) and is_task_cancelled():
        raise TaskCancelledError(task_id, f"{stage} before pipe call")

    call_kwargs = dict(kwargs)
    if task_id and callable(is_task_cancelled):
        call_kwargs["progress_bar_cmd"] = _build_cancelable_progress_bar(
            base_progress_bar_cmd=call_kwargs.get("progress_bar_cmd"),
            task_id=task_id,
            stage=stage,
            is_task_cancelled=is_task_cancelled,
        )

    fn = getattr(pipe, "__call__", pipe)
    filtered = filter_kwargs_for_callable(fn, call_kwargs)

    # 在真正触发推理前打印一次最终参数（debug 级别；用于定位不同 pipeline 参数差异）
    try:
        pipe_name = getattr(getattr(pipe, "__class__", None), "__name__", None) or type(pipe).__name__
    except Exception:
        pipe_name = "pipe"
    try:
        log_kwargs = dict(filtered)
        for k in ("prompt", "negative_prompt"):
            v = log_kwargs.get(k)
            if isinstance(v, str) and len(v) > 200:
                log_kwargs[k] = v[:200] + "...(truncated)"
        print_info({"pipe": pipe_name, "kwargs": log_kwargs}, prefix="推理参数")
    except Exception:
        # best-effort：日志失败不影响推理
        pass

    result = pipe(**filtered)
    if task_id and callable(is_task_cancelled) and is_task_cancelled():
        raise TaskCancelledError(task_id, f"{stage} after pipe call")
    return result


# ==========================
# fast_mode -> TeaCache
# ==========================

# 注意：diffsynth 的 `WanVideoPipeline` 内置 TeaCache，但其 `model_id` 目前只支持部分 Wan2.1 模型。
# 为避免“fast_mode 打开但模型不支持”导致推理直接失败，这里做 best-effort：
# - 能推断出受支持的 model_id：注入 tea_cache 参数以启用
# - 推断不出/不支持：返回 {}，不启用 TeaCache（安全降级）

_WAN_TEACACHE_SUPPORTED_MODEL_IDS = {
    "Wan2.1-T2V-1.3B",
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B-480P",
    "Wan2.1-I2V-14B-720P",
}

_WAN_TEACACHE_DEFAULT_REL_L1_THRESH = 0.10


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _guess_wan_teacache_model_id(
    *,
    load_name: str,
    pipe_name: str,
    height: Optional[int],
    width: Optional[int],
) -> Optional[str]:
    """
    从本地模型目录名/路径（load_name）与 pipe_name 推断 TeaCache 的 model_id。
    仅覆盖当前 diffsynth/wan_video.py 内置 coefficients 的 Wan2.1 模型。
    """
    s = str(load_name or "")
    s_low = s.lower()
    k = _norm_key(s)

    # 显式包含分辨率标记时优先
    if "wan21i2v14b720p" in k or "720p" in s_low:
        if "wan21i2v14b" in k:
            return "Wan2.1-I2V-14B-720P"
    if "wan21i2v14b480p" in k or "480p" in s_low:
        if "wan21i2v14b" in k:
            return "Wan2.1-I2V-14B-480P"

    # 常见目录名：Wan2.1-T2V-14B / Wan2.1-T2V-1.3B / Wan2.1-I2V-14B
    if "wan21t2v13b" in k:
        return "Wan2.1-T2V-1.3B"
    if "wan21t2v14b" in k:
        return "Wan2.1-T2V-14B"
    if "wan21i2v14b" in k:
        # 若未写明 480/720，则按请求分辨率做一个 best-effort 选择
        try:
            h = int(height) if height is not None else None
        except Exception:
            h = None
        try:
            w = int(width) if width is not None else None
        except Exception:
            w = None
        if (h is not None and h >= 720) or (w is not None and w >= 1280):
            return "Wan2.1-I2V-14B-720P"
        return "Wan2.1-I2V-14B-480P"

    # 若模型目录名里不包含 Wan2.1 信息，就不要“拍脑袋启用”，避免不匹配系数导致质量问题
    _ = pipe_name  # 保留参数位，未来可按 pipe_name 扩展推断规则
    return None


def build_wan2_teacache_kwargs(
    req: Any,
    *,
    pipe_name: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    fast_mode 时，尽量为 Wan2 视频 pipeline 注入 TeaCache 参数（安全降级）。

    规则：
    - fast_mode=False：返回 {}
    - 仅从 req.load_name + 分辨率 best-effort 推断（仅 Wan2.1 支持）
      （按约定：不要从 req.model_cfg/model_config 读取任何 teacache 配置，避免多入口语义分裂）
    """
    if not bool(getattr(req, "fast_mode", False)):
        return {}

    # 阈值：固定默认值（若要关闭请用 fast_mode=false）
    thresh_f = float(_WAN_TEACACHE_DEFAULT_REL_L1_THRESH)

    # model_id：仅从 load_name 推断
    mn = str(getattr(req, "load_name", "") or "")
    model_id = _guess_wan_teacache_model_id(
        load_name=mn,
        pipe_name=str(pipe_name or ""),
        height=height if height is not None else getattr(req, "height", None),
        width=width if width is not None else getattr(req, "width", None),
    )

    def _warn(msg: str) -> None:
        try:
            w = getattr(logger, "warning", None)
            if callable(w):
                w(msg)
        except Exception:
            pass

    if not model_id:
        _warn(
            f"[Wan2][TeaCache] fast_mode=True 但未启用 TeaCache：无法从 load_name 推断受支持的 model_id。"
            f" pipe={pipe_name} load_name={mn!r} size={height}x{width} supported={sorted(_WAN_TEACACHE_SUPPORTED_MODEL_IDS)}"
        )
        return {}

    if str(model_id) not in _WAN_TEACACHE_SUPPORTED_MODEL_IDS:
        _warn(
            f"[Wan2][TeaCache] fast_mode=True 但未启用 TeaCache：推断的 model_id 不受支持。"
            f" pipe={pipe_name} load_name={mn!r} inferred_model_id={model_id!r} supported={sorted(_WAN_TEACACHE_SUPPORTED_MODEL_IDS)}"
        )
        return {}

    return {
        "tea_cache_l1_thresh": thresh_f,
        "tea_cache_model_id": str(model_id),
    }


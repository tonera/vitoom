"""
Framework-agnostic MeanCache engine.

It wraps a velocity/model function and decides whether to:
- compute a real forward pass, or
- skip and reuse cached velocity with JVP correction.

Algorithm is adapted from `facok/comfyui-meancache-z`:
`https://github.com/facok/comfyui-meancache-z/tree/master`
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

try:
    # 运行期常见：把 `inference/` 目录加入 PYTHONPATH，此时顶层包是 common/image/video/...
    from common.logger import get_logger  # type: ignore
except Exception:  # pragma: no cover
    # 单测/部分脚本可能以项目根为 PYTHONPATH，此时可通过 `inference.common.*` 访问
    from inference.common.logger import get_logger  # type: ignore

from .config import MeanCacheConfig
from .state import MeanCacheState
from .trajectory_scheduler import TrajectoryScheduler
from .velocity_cache import (
    compute_jvp_approximation,
    compute_online_L_K,
    compute_velocity_similarity,
    should_skip_step,
)

logger = get_logger(__name__)


def _to_sigma_scalar(sigma: Union[float, int, torch.Tensor]) -> float:
    if isinstance(sigma, (float, int)):
        return float(sigma)
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() == 0:
            return 1.0
        return float(sigma.flatten()[0].detach().cpu().item())
    return float(sigma)  # type: ignore[arg-type]


class MeanCacheEngine:
    """
    Main entry point.

    Usage:
        engine = MeanCacheEngine(config, sample_sigmas=sigmas)
        v_fn = engine.wrap(velocity_fn)
        v = v_fn(x, sigma, **kwargs)
    """

    def __init__(
        self,
        config: MeanCacheConfig = MeanCacheConfig(),
        *,
        total_steps: Optional[int] = None,
        sample_sigmas: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.cache_device = config.resolve_cache_device()
        self.state = MeanCacheState(cache_device=self.cache_device, max_cache_span=config.max_cache_span)

        self.last_sigma: Optional[float] = None
        # Some schedulers use increasing sigma; don't assume monotonic decreasing.
        # Learn the within-run direction from the first non-trivial delta.
        self._sigma_direction: Optional[int] = None  # +1 increasing, -1 decreasing
        self.split_cfg_unified_error: Optional[float] = None
        self._summary_printed: bool = False

        self.sample_sigmas: Optional[torch.Tensor] = sample_sigmas
        if total_steps is None:
            total_steps = (len(sample_sigmas) - 1) if sample_sigmas is not None else None
        self.total_steps: Optional[int] = total_steps

        self.scheduler: Optional[TrajectoryScheduler] = None
        if config.enable_pssp and total_steps is not None:
            self.scheduler = TrajectoryScheduler(
                total_steps=total_steps,
                skip_budget=config.skip_budget,
                peak_threshold=config.peak_threshold,
                gamma=config.gamma,
                max_accumulated_error=config.max_accumulated_error,
            )
            if sample_sigmas is not None:
                self.scheduler.adjust_for_sigmas(sample_sigmas)

        # Debug/introspection: record last call meta
        self._last_pred_id: int = 0
        self._last_batch_size: int = 1

    def reset(self) -> None:
        self.state.clear_all()
        self.last_sigma = None
        self._sigma_direction = None
        self.split_cfg_unified_error = None
        self._summary_printed = False

    def start_run(
        self,
        *,
        total_steps: Optional[int] = None,
        sample_sigmas: Optional[torch.Tensor] = None,
        reset: bool = True,
    ) -> None:
        """
        Prepare engine for a new sampling run.

        This is useful when integrating with Diffusers pipelines where
        `num_inference_steps` is known at `pipeline.__call__` time.
        """
        if reset:
            self.reset()
        if total_steps is not None:
            self.total_steps = int(total_steps)
        if sample_sigmas is not None:
            self.sample_sigmas = sample_sigmas

        # (Re)create scheduler per run so it matches the current step count/sigmas.
        self.scheduler = None
        if self.config.enable_pssp and self.total_steps is not None:
            self.scheduler = TrajectoryScheduler(
                total_steps=self.total_steps,
                skip_budget=self.config.skip_budget,
                peak_threshold=self.config.peak_threshold,
                gamma=self.config.gamma,
                max_accumulated_error=self.config.max_accumulated_error,
            )
            if self.sample_sigmas is not None:
                self.scheduler.adjust_for_sigmas(self.sample_sigmas)

    def stats(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "report": self.state.get_report(),
            "total_skips": self.state.get_total_skip_count(),
        }

    def _debug(self, msg: str) -> None:
        if self.config.debug:
            logger.info("[MeanCache] %s", msg)

    def _detect_new_run(self, current_sigma: float) -> None:
        """
        Best-effort detection of a new sampling run.

        We do NOT assume sigma direction. Instead, we learn the direction from
        early steps. A direction flip with a significant jump is treated as a new run.
        """
        if self.last_sigma is None:
            return

        delta = float(current_sigma) - float(self.last_sigma)
        if abs(delta) < 1e-8:
            return

        sign = 1 if delta > 0 else -1
        if self._sigma_direction is None:
            self._sigma_direction = sign
            return

        # Direction flip + noticeable jump => new run
        if sign != self._sigma_direction and abs(delta) > 0.1:
            self._debug(
                f"new run detected (sigma direction flip {self.last_sigma:.4f} -> {current_sigma:.4f}), reset"
            )
            self.reset()

    def wrap(self, velocity_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        """
        Return a wrapped callable with MeanCache enabled.

        Expected callable signature (flexible):
            velocity_fn(x: Tensor, sigma: Tensor|float, **kwargs) -> Tensor
        """

        def wrapped(x: torch.Tensor, sigma: Union[float, int, torch.Tensor], **kwargs) -> torch.Tensor:
            return self.forward(velocity_fn, x, sigma, **kwargs)

        return wrapped

    def forward(
        self,
        velocity_fn: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: Union[float, int, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        current_sigma = _to_sigma_scalar(sigma)
        self._detect_new_run(current_sigma)

        batch_size = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.ndim > 0 else 1
        self._last_batch_size = batch_size

        # split-CFG auto-detect: batch_size=1, called twice with same sigma.
        if batch_size == 1 and self.last_sigma is not None and abs(current_sigma - self.last_sigma) < 1e-6:
            pred_id = int(kwargs.pop("pred_id", 1))
            split_pred_id = pred_id
        else:
            pred_id = int(kwargs.pop("pred_id", 0))
            split_pred_id = pred_id
        self._last_pred_id = split_pred_id

        # update last_sigma after detecting split cfg
        self.last_sigma = current_sigma

        # Determine whether to treat the batch dimension as CFG branches.
        # Default: treat batched inputs as a single prediction (more robust).
        use_cfg_batch = bool(self.config.assume_cfg_batch and batch_size == 2)

        # step index resolution (match upstream off-by-one avoidance)
        if batch_size == 1:
            pred_state_target = self.state.get_or_create(split_pred_id, self.cache_device)
            step_index = int(pred_state_target.get("step_index", 0))
        elif use_cfg_batch:
            pred_state_ref = self.state.get_or_create(0, self.cache_device)
            step_index = int(pred_state_ref.get("step_index", 0))
        else:
            # Treat real batch as a single prediction state (pred_id=0).
            pred_state_ref = self.state.get_or_create(0, self.cache_device)
            step_index = int(pred_state_ref.get("step_index", 0))

        effective_end = self.config.end_step if self.config.end_step >= 0 else 999999
        in_active_range = self.config.start_step <= step_index < effective_end

        # unified error for split-CFG: ensure cond/uncond make same decision
        unified_error: Optional[float] = None
        if batch_size == 1:
            if split_pred_id == 0:
                err_0 = float(self.state.get_or_create(0, self.cache_device).get("accumulated_error", 1.0))
                err_1 = float(self.state.get_or_create(1, self.cache_device).get("accumulated_error", 1.0))
                self.split_cfg_unified_error = max(err_0, err_1)
            unified_error = self.split_cfg_unified_error

        self._debug(
            f"step={step_index} sigma={current_sigma:.4f} in_range={in_active_range} "
            f"batch={batch_size} pred_id={split_pred_id}"
        )

        if batch_size == 1:
            return self._process_single(
                velocity_fn=velocity_fn,
                x=x,
                sigma=sigma,
                pred_id=split_pred_id,
                step_index=step_index,
                current_sigma=current_sigma,
                in_active_range=in_active_range,
                unified_accumulated_error=unified_error,
                **kwargs,
            )

        if use_cfg_batch:
            # batched CFG: treat batch dimension as independent cond/uncond branches.
            return self._process_batch(
                velocity_fn=velocity_fn,
                x=x,
                sigma=sigma,
                batch_size=batch_size,
                step_index=step_index,
                current_sigma=current_sigma,
                in_active_range=in_active_range,
                **kwargs,
            )

        # real batch: treat as single prediction (cache a full-batch tensor)
        return self._process_single(
            velocity_fn=velocity_fn,
            x=x,
            sigma=sigma,
            pred_id=0,
            step_index=step_index,
            current_sigma=current_sigma,
            in_active_range=in_active_range,
            unified_accumulated_error=None,
            **kwargs,
        )

    def _process_single(
        self,
        *,
        velocity_fn: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: Union[float, int, torch.Tensor],
        pred_id: int,
        step_index: int,
        current_sigma: float,
        in_active_range: bool,
        unified_accumulated_error: Optional[float],
        **kwargs,
    ) -> torch.Tensor:
        pred_state = self.state.get_or_create(pred_id, self.cache_device)

        # decide skip
        should_skip = False
        if unified_accumulated_error is not None:
            accumulated_error = float(unified_accumulated_error)
        else:
            accumulated_error = float(pred_state.get("accumulated_error", 1.0))

        if in_active_range and pred_state.get("v_cache") is not None:
            if self.scheduler is not None:
                should_skip, accumulated_error = self.scheduler.get_skip_decision(
                    step_index=step_index,
                    velocity_similarity=accumulated_error,
                    accumulated_error=accumulated_error,
                )
            else:
                should_skip, accumulated_error = should_skip_step(
                    stability_deviation=accumulated_error,
                    threshold=self.config.rel_l1_thresh,
                    accumulated_error=accumulated_error,
                    max_accumulated=self.config.max_accumulated_error,
                )
            self.state.update(pred_id, accumulated_error=accumulated_error)
            self._debug(f"pred_id={pred_id}: decide should_skip={should_skip} metric={accumulated_error:.4f}")

        if should_skip and pred_state.get("v_cache") is not None:
            v_cache = pred_state["v_cache"].to(x.device)
            jvp_cache = pred_state.get("jvp_cache")
            sigma_cache = float(pred_state.get("sigma_cache", current_sigma))

            if jvp_cache is not None:
                # Direction-agnostic: predict along dt = current - cached (can be +/-).
                dt = current_sigma - sigma_cache
                output = v_cache + dt * jvp_cache.to(x.device)
                self._debug(f"pred_id={pred_id}: SKIP+JVP dt={dt:.4f} dev={accumulated_error:.4f}")
            else:
                output = v_cache
                self._debug(f"pred_id={pred_id}: SKIP (no JVP) dev={accumulated_error:.4f}")

            self.state.record_skip(pred_id, step_index)
        else:
            # compute
            output = velocity_fn(x, sigma, **kwargs)
            v_current = output
            self._debug(f"pred_id={pred_id}: COMPUTE (in_range={in_active_range}, has_cache={pred_state.get('v_cache') is not None})")

            self.state.update_history(pred_id, v_current, current_sigma)

            # prefer largest available JVP_K
            jvp = None
            for k in range(self.config.max_cache_span, 0, -1):
                jvp_k = self.state.get_jvp_k(pred_id, k)
                if jvp_k is not None:
                    jvp = jvp_k
                    break

            if jvp is None:
                v_prev = pred_state.get("v_prev")
                t_prev = pred_state.get("t_prev")
                if v_prev is not None and t_prev is not None:
                    dt_prev = current_sigma - float(t_prev)
                    if abs(dt_prev) > 1e-8:
                        jvp = compute_jvp_approximation(v_current, v_prev, dt_prev)

            v_cache_old = pred_state.get("v_cache")
            jvp_cache_old = pred_state.get("jvp_cache")
            sigma_cache_old = pred_state.get("sigma_cache")

            if v_cache_old is not None and jvp_cache_old is not None and sigma_cache_old is not None:
                dt_elapsed = current_sigma - float(sigma_cache_old)
                deviation = compute_online_L_K(v_current, v_cache_old, jvp_cache_old, dt_elapsed)
                self._debug(f"pred_id={pred_id}: L_K={deviation:.4f} thresh={self.config.rel_l1_thresh}")
            elif v_cache_old is not None:
                deviation = compute_velocity_similarity(v_current, v_cache_old)
                self._debug(f"pred_id={pred_id}: rel_L1={deviation:.4f} (no JVP yet)")
            else:
                deviation = 1.0
                self._debug(f"pred_id={pred_id}: first compute deviation={deviation:.4f}")

            self.state.update(pred_id, accumulated_error=deviation)
            self.state.update(pred_id, v_cache=v_current.to(self.cache_device))
            self.state.update(pred_id, sigma_cache=current_sigma)
            if jvp is not None:
                self.state.update(pred_id, jvp_cache=jvp.to(self.cache_device))

            self.state.update(pred_id, v_prev=v_current.to(self.cache_device), t_prev=current_sigma)

        self.state.increment_step(pred_id)
        return output

    def _process_batch(
        self,
        *,
        velocity_fn: Callable[..., torch.Tensor],
        x: torch.Tensor,
        sigma: Union[float, int, torch.Tensor],
        batch_size: int,
        step_index: int,
        current_sigma: float,
        in_active_range: bool,
        **kwargs,
    ) -> torch.Tensor:
        can_skip_all = True
        for pred_id in range(batch_size):
            pred_state = self.state.get_or_create(pred_id, self.cache_device)
            if (not in_active_range) or pred_state.get("v_cache") is None:
                can_skip_all = False
                break
            accumulated_error = float(pred_state.get("accumulated_error", 1.0))
            if self.scheduler is not None:
                should_skip, _ = self.scheduler.get_skip_decision(
                    step_index=step_index,
                    velocity_similarity=accumulated_error,
                    accumulated_error=accumulated_error,
                )
            else:
                should_skip, _ = should_skip_step(
                    stability_deviation=accumulated_error,
                    threshold=self.config.rel_l1_thresh,
                    accumulated_error=accumulated_error,
                )
            if not should_skip:
                can_skip_all = False
                break

        if can_skip_all:
            outputs = []
            for pred_id in range(batch_size):
                pred_state = self.state.get(pred_id)
                v_cache = pred_state["v_cache"].to(x.device)
                jvp_cache = pred_state.get("jvp_cache")
                sigma_cache = float(pred_state.get("sigma_cache", current_sigma))
                if jvp_cache is not None:
                    dt = current_sigma - sigma_cache
                    outputs.append(v_cache + dt * jvp_cache.to(x.device))
                else:
                    outputs.append(v_cache)
                self.state.record_skip(pred_id, step_index)
                self.state.increment_step(pred_id)
            return torch.cat(outputs, dim=0)

        output = velocity_fn(x, sigma, **kwargs)

        for pred_id in range(batch_size):
            pred_state = self.state.get_or_create(pred_id, self.cache_device)
            v_current = output[pred_id : pred_id + 1]

            self.state.update_history(pred_id, v_current, current_sigma)

            jvp = None
            for k in range(self.config.max_cache_span, 0, -1):
                jvp_k = self.state.get_jvp_k(pred_id, k)
                if jvp_k is not None:
                    jvp = jvp_k
                    break

            if jvp is None:
                v_prev = pred_state.get("v_prev")
                t_prev = pred_state.get("t_prev")
                if v_prev is not None and t_prev is not None:
                    dt_prev = current_sigma - float(t_prev)
                    if abs(dt_prev) > 1e-8:
                        jvp = compute_jvp_approximation(v_current, v_prev, dt_prev)

            v_cache_old = pred_state.get("v_cache")
            jvp_cache_old = pred_state.get("jvp_cache")
            sigma_cache_old = pred_state.get("sigma_cache")
            if v_cache_old is not None and jvp_cache_old is not None and sigma_cache_old is not None:
                dt_elapsed = current_sigma - float(sigma_cache_old)
                deviation = compute_online_L_K(v_current, v_cache_old, jvp_cache_old, dt_elapsed)
            elif v_cache_old is not None:
                deviation = compute_velocity_similarity(v_current, v_cache_old)
            else:
                deviation = 1.0

            self.state.update(pred_id, accumulated_error=deviation)
            self.state.update(pred_id, v_cache=v_current.to(self.cache_device))
            self.state.update(pred_id, sigma_cache=current_sigma)
            if jvp is not None:
                self.state.update(pred_id, jvp_cache=jvp.to(self.cache_device))
            self.state.update(pred_id, v_prev=v_current.to(self.cache_device), t_prev=current_sigma)
            self.state.increment_step(pred_id)

        return output


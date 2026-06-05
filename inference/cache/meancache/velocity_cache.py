"""
MeanCache velocity caching helpers.

Extracted and adapted from `facok/comfyui-meancache-z` (MIT):
`https://github.com/facok/comfyui-meancache-z/tree/master`

Core paper: UnicomAI MeanCache
`https://unicomai.github.io/MeanCache/`
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def compute_jvp_approximation(
    v_current: torch.Tensor,
    v_prev: torch.Tensor,
    dt_prev: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    JVP_{r→t} ≈ (v_t - v_r) / (t - r)
    """
    if abs(dt_prev) < eps:
        return torch.zeros_like(v_current)
    return (v_current - v_prev.to(v_current.device)) / dt_prev


def compute_jvp_k(
    v_history: list,
    t_history: list,
    k: int,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """
    JVP_K = (v_now - v_{now-K}) / (t_{now-K} - t_now)
    """
    if len(v_history) < k + 1 or len(t_history) < k + 1:
        return None

    v_now = v_history[-1]
    v_k_ago = v_history[-(k + 1)]
    t_now = t_history[-1]
    t_k_ago = t_history[-(k + 1)]

    # Direction-agnostic: use dt = t_now - t_k_ago (can be +/-).
    dt = t_now - t_k_ago
    if abs(dt) < eps:
        return None

    return (v_now - v_k_ago.to(v_now.device)) / dt


def compute_average_velocity(v_current: torch.Tensor, jvp: torch.Tensor, dt: float) -> torch.Tensor:
    """
    MeanCache paper uses: û(z_t, t, s) = v(z_t, t) + (s - t) · JVP
    """
    return v_current + dt * jvp


def compute_stability_deviation(
    v_true_avg: torch.Tensor,
    v_instant: torch.Tensor,
    jvp_k: torch.Tensor,
    dt: float,
) -> float:
    """
    L_K(t,s) = mean(| (u(z_t,t,s) - v(z_t,t)) - (s-t)·JVP_K |)
    """
    true_diff = v_true_avg - v_instant
    estimated_diff = dt * jvp_k.to(v_true_avg.device)
    return torch.abs(true_diff - estimated_diff).mean().item()


def compute_online_L_K(
    v_new: torch.Tensor,
    v_cached: torch.Tensor,
    jvp_cached: torch.Tensor,
    dt_elapsed: float,
    eps: float = 1e-8,
) -> float:
    """
    Online approximation:
      L_K ≈ mean(|v_new - (v_cached + dt * JVP_cached)|) / mean(|v_new|)
    """
    predicted_v = v_cached.to(v_new.device) + dt_elapsed * jvp_cached.to(v_new.device)
    prediction_error = torch.abs(v_new - predicted_v).mean()
    normalizer = torch.abs(v_new).mean() + eps
    return (prediction_error / normalizer).item()


def compute_velocity_similarity(
    v_current: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    metric: str = "l1_relative",
) -> float:
    if v_cache is None:
        return float("inf")

    v_cache = v_cache.to(v_current.device)

    if metric == "l1_relative":
        l1_distance = torch.abs(v_current - v_cache).mean()
        norm = torch.abs(v_cache).mean() + 1e-8
        return (l1_distance / norm).item()

    if metric == "cosine":
        v_curr_flat = v_current.flatten()
        v_cache_flat = v_cache.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            v_curr_flat.unsqueeze(0),
            v_cache_flat.unsqueeze(0),
        )
        return 1.0 - cos_sim.item()

    if metric == "l2":
        return torch.norm(v_current - v_cache, p=2).item()

    raise ValueError(f"Unknown metric: {metric}")


def should_skip_step(
    stability_deviation: float,
    threshold: float,
    accumulated_error: float,
    max_accumulated: float = 0.5,
) -> Tuple[bool, float]:
    """
    Adaptive thresholding with peak suppression.
    """
    accumulation_factor = 1.0 - (accumulated_error / max_accumulated)
    effective_threshold = threshold * max(0.1, accumulation_factor)

    if stability_deviation < effective_threshold:
        new_accumulated = accumulated_error + stability_deviation
        if new_accumulated < max_accumulated:
            return True, new_accumulated

    return False, 0.0


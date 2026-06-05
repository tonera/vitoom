"""
MeanCache 通用组件单测（纯 CPU）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# torch 是运行时必需依赖；在没有 torch 的测试环境里直接跳过本文件所有测试
torch = pytest.importorskip("torch")

from inference.cache.meancache.velocity_cache import (
    compute_jvp_approximation,
    compute_online_L_K,
)
from inference.cache.meancache import MeanCacheConfig, MeanCacheEngine


def test_compute_jvp_approximation():
    v_current = torch.tensor([2.0, 4.0])
    v_prev = torch.tensor([0.0, 2.0])
    jvp = compute_jvp_approximation(v_current, v_prev, dt_prev=2.0)
    assert torch.allclose(jvp, torch.tensor([1.0, 1.0]))


def test_compute_online_L_K_zero_when_perfect_prediction():
    v_cached = torch.tensor([1.0, 2.0])
    jvp = torch.zeros_like(v_cached)
    v_new = v_cached.clone()
    lk = compute_online_L_K(v_new=v_new, v_cached=v_cached, jvp_cached=jvp, dt_elapsed=1.0)
    assert lk == pytest.approx(0.0, abs=1e-8)


def test_meancache_engine_skips_after_warmup_constant_velocity_single():
    calls = {"n": 0}

    def velocity_fn(x: torch.Tensor, sigma, **kwargs) -> torch.Tensor:
        calls["n"] += 1
        return torch.ones_like(x)

    cfg = MeanCacheConfig(rel_l1_thresh=0.30, skip_budget=0.50, enable_pssp=False, cache_device="cpu")
    engine = MeanCacheEngine(cfg)
    wrapped = engine.wrap(velocity_fn)

    x = torch.zeros(1, 8)
    sigmas = [1.0, 0.9, 0.8, 0.7, 0.6]
    outs = [wrapped(x, s) for s in sigmas]

    assert all(o.shape == x.shape for o in outs)
    # 第 1/2 步需要 compute 来建立 cache + 误差度量，之后应基本全部 skip
    assert calls["n"] <= 3


def test_meancache_engine_skips_after_warmup_constant_velocity_batch_cfg():
    calls = {"n": 0}

    def velocity_fn(x: torch.Tensor, sigma, **kwargs) -> torch.Tensor:
        calls["n"] += 1
        return torch.ones_like(x)

    cfg = MeanCacheConfig(rel_l1_thresh=0.30, skip_budget=0.50, enable_pssp=False, cache_device="cpu")
    engine = MeanCacheEngine(cfg)
    wrapped = engine.wrap(velocity_fn)

    x = torch.zeros(2, 8)  # batched CFG: [cond, uncond]
    sigmas = [1.0, 0.9, 0.8, 0.7, 0.6]
    outs = [wrapped(x, s) for s in sigmas]

    assert all(o.shape == x.shape for o in outs)
    assert calls["n"] <= 3


"""
MeanCache state management.

Adapted from `facok/comfyui-meancache-z`:
`https://github.com/facok/comfyui-meancache-z/tree/master`
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

DEFAULT_MAX_CACHE_SPAN = 3


class MeanCacheState:
    """
    Tracks per-prediction (cond/uncond) cache state.

    This is intentionally model/framework agnostic: it stores tensors and
    step counters, and the engine decides how to interpret CFG modes.
    """

    def __init__(self, cache_device: Any = "cpu", max_cache_span: int = DEFAULT_MAX_CACHE_SPAN):
        self.cache_device = cache_device
        self.max_cache_span = max_cache_span
        self.states: Dict[int, Dict[str, Any]] = {}
        self._next_pred_id: int = 0

    def new_prediction(self, cache_device: Optional[Any] = None) -> int:
        if cache_device is not None:
            self.cache_device = cache_device

        pred_id = self._next_pred_id
        self._next_pred_id += 1

        self.states[pred_id] = {
            "v_prev": None,
            "v_cache": None,
            "t_prev": None,
            "dt_prev": None,
            "jvp_cache": None,
            "sigma_cache": None,
            "v_history": [],
            "t_history": [],
            "accumulated_distance": 0.0,
            "accumulated_error": 0.0,
            "skipped_steps": [],
            "step_index": 0,
            "trajectory_budget": None,
            "scheduled_skip_mask": None,
        }
        return pred_id

    def update(self, pred_id: int, **kwargs) -> None:
        if pred_id not in self.states:
            return
        for key, value in kwargs.items():
            if key in self.states[pred_id]:
                self.states[pred_id][key] = value

    def get(self, pred_id: int) -> Dict[str, Any]:
        return self.states.get(pred_id, {})

    def get_or_create(self, pred_id: int, cache_device: Optional[Any] = None) -> Dict[str, Any]:
        if pred_id not in self.states:
            while self._next_pred_id <= pred_id:
                self.new_prediction(cache_device)
        return self.states.get(pred_id, {})

    def update_history(self, pred_id: int, velocity: torch.Tensor, timestep: float) -> None:
        if pred_id not in self.states:
            return

        state = self.states[pred_id]
        v_cached = velocity.detach().clone().to(self.cache_device)
        state["v_history"].append(v_cached)
        state["t_history"].append(timestep)

        max_len = self.max_cache_span + 1
        if len(state["v_history"]) > max_len:
            old_v = state["v_history"].pop(0)
            del old_v
            state["t_history"].pop(0)

    def get_jvp_k(self, pred_id: int, k: int, eps: float = 1e-8) -> Optional[torch.Tensor]:
        if pred_id not in self.states:
            return None

        state = self.states[pred_id]
        v_history = state["v_history"]
        t_history = state["t_history"]

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

    def clear_all(self) -> None:
        for state in self.states.values():
            if state.get("v_prev") is not None:
                del state["v_prev"]
            if state.get("v_cache") is not None:
                del state["v_cache"]
            if state.get("jvp_cache") is not None:
                del state["jvp_cache"]
            for v in state.get("v_history", []):
                del v
            state["v_history"] = []
            state["t_history"] = []

        self.states = {}
        self._next_pred_id = 0

    def increment_step(self, pred_id: int) -> None:
        if pred_id in self.states:
            self.states[pred_id]["step_index"] += 1

    def record_skip(self, pred_id: int, step_index: int) -> None:
        if pred_id in self.states:
            self.states[pred_id]["skipped_steps"].append(step_index)

    def get_skip_count(self, pred_id: int) -> int:
        if pred_id in self.states:
            return len(self.states[pred_id]["skipped_steps"])
        return 0

    def get_total_skip_count(self) -> int:
        return sum(len(st.get("skipped_steps", [])) for st in self.states.values())

    def get_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        state_names = {0: "conditional", 1: "unconditional"}
        for pred_id, st in self.states.items():
            name = state_names.get(pred_id, f"prediction_{pred_id}")
            report[name] = {
                "skipped_steps": st.get("skipped_steps", []),
                "skip_count": len(st.get("skipped_steps", [])),
                "total_steps": st.get("step_index", 0),
            }
        return report


"""
Trajectory Stability Scheduling for MeanCache.

Implements Peak-Suppressed Shortest Path (PSSP) scheduling via DP.
Adapted from `facok/comfyui-meancache-z`:
`https://github.com/facok/comfyui-meancache-z/tree/master`
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class TrajectoryScheduler:
    def __init__(
        self,
        total_steps: int,
        skip_budget: float = 0.3,
        gamma: float = 2.0,
        peak_threshold: float = 0.15,
        max_accumulated_error: float = 0.5,
        min_compute_steps: int = 4,
        critical_start_ratio: float = 0.20,
        critical_end_ratio: float = 0.80,
    ):
        self.total_steps = max(1, int(total_steps))
        self.skip_budget = max(0.0, min(0.5, float(skip_budget)))
        self.gamma = float(gamma)
        self.peak_threshold = float(peak_threshold)
        self.max_accumulated_error = float(max_accumulated_error)
        self.min_compute_steps = int(min_compute_steps)
        self.critical_start_ratio = float(critical_start_ratio)
        self.critical_end_ratio = float(critical_end_ratio)

        self.skip_mask: List[bool] = [False] * self.total_steps
        self.step_weights: List[float] = [1.0] * self.total_steps

        self._compute_heuristic_schedule()

    def _compute_heuristic_schedule(self) -> None:
        n = self.total_steps
        max_skips = max(0, int(n * self.skip_budget))

        critical_start = max(1, int(n * self.critical_start_ratio))
        critical_end = min(n - 1, int(n * self.critical_end_ratio))

        self.skip_mask = [False] * n
        self.step_weights = [1.0] * n

        if max_skips == 0 or n <= self.min_compute_steps:
            return

        skip_candidates = list(range(critical_start, critical_end))
        if len(skip_candidates) == 0:
            return

        min_spacing = max(2, len(skip_candidates) // (max_skips + 1))

        skips_assigned = 0
        for i in range(0, len(skip_candidates), min_spacing):
            if skips_assigned >= max_skips:
                break
            idx = skip_candidates[i]
            self.skip_mask[idx] = True
            skips_assigned += 1

        self._update_step_weights()

    def _update_step_weights(self) -> None:
        n = self.total_steps
        self.step_weights = [1.0] * n

        for i in range(n):
            if i < len(self.skip_mask) and self.skip_mask[i]:
                self.step_weights[i] = 0.5

            if i < n * self.critical_start_ratio:
                self.step_weights[i] *= 1.5

            if i >= n * self.critical_end_ratio:
                self.step_weights[i] *= 1.3

    def compute_optimal_schedule_pssp(
        self,
        deviation_estimates: List[float],
        max_consecutive_skips: int = 3,
        protect_steps: Optional[List[int]] = None,
    ) -> List[bool]:
        n = self.total_steps
        max_skips = max(0, int(n * self.skip_budget))

        if protect_steps is not None:
            protected_set = set(protect_steps)
        else:
            protected_set = {0, n - 1}

        if max_skips == 0 or n <= self.min_compute_steps:
            self.skip_mask = [False] * n
            return self.skip_mask

        if len(deviation_estimates) != n:
            self._compute_heuristic_schedule()
            return self.skip_mask

        INF = float("inf")
        dp = [[INF] * (max_skips + 1) for _ in range(n + 1)]
        parent = [[None] * (max_skips + 1) for _ in range(n + 1)]
        consecutive = [[0] * (max_skips + 1) for _ in range(n + 1)]

        dp[0][0] = 0

        for i in range(1, n + 1):
            step_idx = i - 1
            is_protected = step_idx in protected_set

            for j in range(min(i, max_skips) + 1):
                # compute
                if dp[i - 1][j] < dp[i][j]:
                    dp[i][j] = dp[i - 1][j]
                    parent[i][j] = (i - 1, j, False)
                    consecutive[i][j] = 0

                # skip
                if j > 0 and not is_protected:
                    prev_consecutive = consecutive[i - 1][j - 1]
                    if prev_consecutive < max_consecutive_skips:
                        deviation = float(deviation_estimates[step_idx])
                        skip_cost = (deviation**self.gamma) if deviation > 0 else 0.0
                        new_cost = dp[i - 1][j - 1] + skip_cost
                        if new_cost < dp[i][j]:
                            dp[i][j] = new_cost
                            parent[i][j] = (i - 1, j - 1, True)
                            consecutive[i][j] = prev_consecutive + 1

        best_j = 0
        for j in range(max_skips, -1, -1):
            if dp[n][j] < INF:
                best_j = j
                break

        skip_mask = [False] * n
        current_i, current_j = n, best_j
        while current_i > 0 and parent[current_i][current_j] is not None:
            prev_i, prev_j, was_skip = parent[current_i][current_j]
            if was_skip:
                skip_mask[current_i - 1] = True
            current_i, current_j = prev_i, prev_j

        self.skip_mask = skip_mask
        self._update_step_weights()
        return skip_mask

    def get_skip_decision(
        self,
        step_index: int,
        velocity_similarity: float,
        accumulated_error: float,
    ) -> Tuple[bool, float]:
        if step_index < 0 or step_index >= len(self.skip_mask):
            max_accumulated = self.max_accumulated_error
            if velocity_similarity < self.peak_threshold * 0.5:
                if accumulated_error < max_accumulated * 0.6:
                    new_error = accumulated_error + velocity_similarity
                    return True, new_error
            return False, 0.0

        scheduled_skip = self.skip_mask[step_index]

        if scheduled_skip and velocity_similarity > self.peak_threshold:
            return False, 0.0

        max_accumulated = self.max_accumulated_error
        if scheduled_skip and accumulated_error > max_accumulated * 0.8:
            return False, 0.0

        if not scheduled_skip:
            if velocity_similarity < self.peak_threshold * 0.3:
                if accumulated_error < max_accumulated * 0.4:
                    new_error = accumulated_error + velocity_similarity
                    return True, new_error

        if scheduled_skip:
            new_error = accumulated_error + velocity_similarity
            return True, new_error

        return False, 0.0

    def get_timestep_weight(self, step_index: int) -> float:
        if 0 <= step_index < len(self.step_weights):
            return self.step_weights[step_index]
        return 1.0

    def get_schedule_summary(self) -> dict:
        skip_indices = [i for i, skip in enumerate(self.skip_mask) if skip]
        return {
            "total_steps": self.total_steps,
            "scheduled_skips": len(skip_indices),
            "skip_ratio": len(skip_indices) / max(1, self.total_steps),
            "skip_indices": skip_indices,
            "gamma": self.gamma,
            "critical_start": int(self.total_steps * self.critical_start_ratio),
            "critical_end": int(self.total_steps * self.critical_end_ratio),
        }

    def adjust_for_sigmas(self, sigmas: torch.Tensor) -> None:
        if sigmas is None or len(sigmas) <= 1:
            return

        n = len(sigmas) - 1
        if n != self.total_steps:
            self.total_steps = n
            self._compute_heuristic_schedule()
            return

        for i in range(n):
            s0 = float(sigmas[i].item())
            s1 = float(sigmas[i + 1].item())
            # Direction-agnostic ratio (>= 1.0), works for increasing or decreasing sigmas.
            denom = max(min(abs(s0), abs(s1)), 1e-8)
            sigma_ratio = max(abs(s0), abs(s1)) / denom
            if sigma_ratio > 2.0 and i < len(self.skip_mask):
                self.skip_mask[i] = False
                self.step_weights[i] *= 1.2


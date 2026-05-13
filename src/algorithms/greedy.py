"""Heuristic non-evolutionary baselines.

* GreedyTSP: visit-nearest-unvisited node tour, with the path discretized into
  N waypoints by linear interpolation. Approximates "fly toward the next
  highest-data node" without any global optimization.
* StraightLine: straight flight start -> end with no detour.

Both share the optimizer interface so they can be benchmarked side-by-side.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class GreedyConfig:
    pass


def _interp_path(points: torch.Tensor, N: int) -> torch.Tensor:
    """Resample a polyline to N equally-spaced waypoints (R^2)."""
    seg = points[1:] - points[:-1]
    seg_len = seg.norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=points.device), seg_len.cumsum(0)])
    total = cum[-1].clamp_min(1e-9)
    s = torch.linspace(0.0, total.item(), N + 2, device=points.device)
    out = torch.empty(N + 2, 2, device=points.device, dtype=points.dtype)
    j = 0
    for i in range(N + 2):
        while j < seg.shape[0] - 1 and cum[j + 1] < s[i]:
            j += 1
        t = (s[i] - cum[j]) / seg_len[j].clamp_min(1e-9)
        out[i] = points[j] + t * seg[j]
    return out


class _NonEvoOptimizer:
    name = "Greedy"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 env, device: str = "cpu", seed: int = 0,
                 cfg: GreedyConfig | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.env = env
        self.device = torch.device(device)
        self.cfg = cfg or GreedyConfig()

    def _flatten(self, traj_full: torch.Tensor) -> torch.Tensor:
        # drop start (0) and end (-1) since env adds them back
        return traj_full[1:-1].reshape(-1)

    def run(self) -> OptimizerResult:  # subclasses implement build
        raise NotImplementedError


class GreedyTSP(_NonEvoOptimizer):
    name = "Greedy"

    def run(self) -> OptimizerResult:
        t0 = perf_counter()
        env = self.env
        nodes = env.nodes
        data = env.data_required
        # weight by data so high-data nodes get visited earlier
        score = data / data.max().clamp_min(1e-9)
        visited = torch.zeros(env.sc.K, dtype=torch.bool, device=env.device)
        path = [env.start]
        cur = env.start
        for _ in range(env.sc.K):
            d = (nodes - cur).norm(dim=-1)
            # cost = dist / score (favor high data, nearby)
            cost = d / (score + 0.1)
            cost[visited] = float("inf")
            i = int(torch.argmin(cost))
            visited[i] = True
            cur = nodes[i]
            path.append(cur)
        path.append(env.end)
        pts = torch.stack(path, dim=0)
        traj_full = _interp_path(pts, env.sc.N)
        x = self._flatten(traj_full).unsqueeze(0)
        c = self.fn(x).item()
        return OptimizerResult(
            best_x=x.squeeze(0).detach().cpu(),
            best_cost=c,
            history=[c],
            wall_time_s=perf_counter() - t0,
            iterations=1,
        )


class StraightLine(_NonEvoOptimizer):
    name = "Straight"

    def run(self) -> OptimizerResult:
        t0 = perf_counter()
        env = self.env
        pts = torch.stack([env.start, env.end], dim=0)
        traj_full = _interp_path(pts, env.sc.N)
        x = self._flatten(traj_full).unsqueeze(0)
        c = self.fn(x).item()
        return OptimizerResult(
            best_x=x.squeeze(0).detach().cpu(),
            best_cost=c,
            history=[c],
            wall_time_s=perf_counter() - t0,
            iterations=1,
        )

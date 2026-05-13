"""Grey Wolf Optimizer (GWO).

Reference: Mirjalili, Mirjalili, Lewis. "Grey wolf optimizer."
Advances in Engineering Software, 2014.

At each iteration the swarm is ranked by fitness and the top three
wolves (alpha, beta, delta) drive the rest of the pack (omega) via
three candidate updates whose mean defines the new position:

  D_k = | C_k * X_k_top  -  X_i |             for k in {alpha, beta, delta}
  X_k = X_k_top  -  A_k * D_k
  X_i_new = (X_alpha + X_beta + X_delta) / 3

with A = 2*a*r1 - a, C = 2*r2, and  a  decreasing linearly from 2 to 0
across iterations. The decreasing  a  drives the swarm from
exploration (large |A|) to exploitation (small |A|).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class GWOConfig:
    pop_size: int = 60
    iterations: int = 300


class GWO:
    name = "GWO"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: GWOConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or GWOConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def run(self) -> OptimizerResult:
        cfg = self.cfg
        P, D = cfg.pop_size, self.dim
        rng = self.upper - self.lower
        if self.init_x is not None:
            X = self.init_x[:P].to(self.device).clamp(self.lower, self.upper)
        else:
            X = self.lower + self._rand(P, D) * rng

        cost = self.fn(X)
        # rank top three
        order = torch.argsort(cost)
        a_idx, b_idx, d_idx = order[0], order[1], order[2]
        gbest_c = cost[a_idx].item()
        gbest_x = X[a_idx].clone()

        history = [gbest_c]
        t0 = perf_counter()

        for t in range(cfg.iterations):
            # linearly decreasing exploration coefficient
            a = 2.0 - 2.0 * t / max(1, cfg.iterations - 1)

            X_a = X[a_idx].unsqueeze(0)
            X_b = X[b_idx].unsqueeze(0)
            X_d = X[d_idx].unsqueeze(0)

            updates = []
            for X_top in (X_a, X_b, X_d):
                A = 2.0 * a * self._rand(P, D) - a
                C = 2.0 * self._rand(P, D)
                Dvec = torch.abs(C * X_top - X)
                updates.append(X_top - A * Dvec)
            X_new = (updates[0] + updates[1] + updates[2]) / 3.0
            X_new = X_new.clamp(self.lower, self.upper)

            cost_new = self.fn(X_new)
            # greedy replacement to keep monotone-improving alpha (mild)
            better = cost_new < cost
            X = torch.where(better.unsqueeze(-1), X_new, X)
            cost = torch.where(better, cost_new, cost)

            order = torch.argsort(cost)
            a_idx, b_idx, d_idx = order[0], order[1], order[2]
            if cost[a_idx].item() < gbest_c:
                gbest_c = cost[a_idx].item()
                gbest_x = X[a_idx].clone()
            history.append(gbest_c)

        return OptimizerResult(
            best_x=gbest_x.detach().cpu(),
            best_cost=gbest_c,
            history=history,
            wall_time_s=perf_counter() - t0,
            iterations=cfg.iterations,
            extra={"pop_size": cfg.pop_size},
        )

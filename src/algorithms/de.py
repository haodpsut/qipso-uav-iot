"""Differential Evolution (rand/1/bin) baseline."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class DEConfig:
    pop_size: int = 60
    iterations: int = 300
    F: float = 0.5             # differential weight
    CR: float = 0.9            # crossover prob


class DE:
    name = "DE"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: DEConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or DEConfig()
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
            pop = self.init_x[:P].to(self.device).clamp(self.lower, self.upper)
        else:
            pop = self.lower + self._rand(P, D) * rng
        cost = self.fn(pop)

        g_idx = torch.argmin(cost)
        gbest_x = pop[g_idx].clone()
        gbest_c = cost[g_idx].item()
        history = [gbest_c]
        t0 = perf_counter()

        for it in range(cfg.iterations):
            # sample 3 distinct indices per target
            idx = torch.argsort(self._rand(P, P), dim=1)[:, :3]
            a, b, c = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]
            mutant = (a + cfg.F * (b - c)).clamp(self.lower, self.upper)
            cross_mask = self._rand(P, D) < cfg.CR
            # ensure at least one dim crosses
            j_rand = torch.randint(D, (P, 1), generator=self.gen, device=self.device)
            cross_mask.scatter_(1, j_rand, True)
            trial = torch.where(cross_mask, mutant, pop)
            trial_cost = self.fn(trial)
            better = trial_cost < cost
            pop = torch.where(better.unsqueeze(-1), trial, pop)
            cost = torch.where(better, trial_cost, cost)

            g_idx = torch.argmin(cost)
            if cost[g_idx].item() < gbest_c:
                gbest_c = cost[g_idx].item()
                gbest_x = pop[g_idx].clone()
            history.append(gbest_c)

        return OptimizerResult(
            best_x=gbest_x.detach().cpu(),
            best_cost=gbest_c,
            history=history,
            wall_time_s=perf_counter() - t0,
            iterations=cfg.iterations,
            extra={"pop_size": cfg.pop_size},
        )

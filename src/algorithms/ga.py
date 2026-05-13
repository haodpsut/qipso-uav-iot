"""Real-coded Genetic Algorithm baseline.

Tournament selection + BLX-alpha crossover + Gaussian mutation, elitism = 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class GAConfig:
    pop_size: int = 60
    iterations: int = 300
    tournament_k: int = 3
    crossover_p: float = 0.9
    mutation_p: float = 0.1
    mutation_sigma: float = 0.01    # fraction of search range
    blx_alpha: float = 0.5
    elite: int = 1


class GA:
    name = "GA"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: GAConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or GAConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randint(self, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(high, shape, generator=self.gen, device=self.device)

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
            # tournament selection -> 2 parents per offspring
            idx_a = self._randint(P, P, cfg.tournament_k)
            parents_a = pop[idx_a[torch.arange(P, device=self.device),
                                  cost[idx_a].argmin(dim=1)]]
            idx_b = self._randint(P, P, cfg.tournament_k)
            parents_b = pop[idx_b[torch.arange(P, device=self.device),
                                  cost[idx_b].argmin(dim=1)]]
            # BLX-alpha crossover
            do_cx = (self._rand(P, 1) < cfg.crossover_p)
            lo = torch.minimum(parents_a, parents_b)
            hi = torch.maximum(parents_a, parents_b)
            d = hi - lo
            child = lo - cfg.blx_alpha * d + self._rand(P, D) * (1 + 2 * cfg.blx_alpha) * d
            child = torch.where(do_cx, child, parents_a)
            # Gaussian mutation
            mut_mask = (self._rand(P, D) < cfg.mutation_p)
            noise = torch.randn(P, D, generator=self.gen, device=self.device) * cfg.mutation_sigma * rng
            child = torch.where(mut_mask, child + noise, child)
            child = child.clamp(self.lower, self.upper)

            child_cost = self.fn(child)
            # elitism: keep best `elite` parents
            if cfg.elite > 0:
                all_x = torch.cat([pop, child], dim=0)
                all_c = torch.cat([cost, child_cost], dim=0)
                top = torch.topk(all_c, P, largest=False).indices
                pop = all_x[top]
                cost = all_c[top]
            else:
                pop, cost = child, child_cost

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

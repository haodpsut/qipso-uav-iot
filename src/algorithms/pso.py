"""Standard Particle Swarm Optimization (Kennedy & Eberhart, 1995).

Inertia-weight variant of canonical PSO. Used as a baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class PSOConfig:
    swarm_size: int = 60
    iterations: int = 300
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    v_clip: float = 0.2     # fraction of search range


class PSO:
    name = "PSO"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: PSOConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or PSOConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def run(self) -> OptimizerResult:
        cfg = self.cfg
        S, D = cfg.swarm_size, self.dim
        rng = self.upper - self.lower
        if self.init_x is not None:
            x = self.init_x[:S].to(self.device).clamp(self.lower, self.upper)
        else:
            x = self.lower + self._rand(S, D) * rng
        v = (self._rand(S, D) - 0.5) * cfg.v_clip * rng

        cost = self.fn(x)
        pbest_x = x.clone()
        pbest_c = cost.clone()
        g_idx = torch.argmin(pbest_c)
        gbest_x = pbest_x[g_idx].clone()
        gbest_c = pbest_c[g_idx].item()

        history = [gbest_c]
        t0 = perf_counter()
        vclip = cfg.v_clip * rng

        for it in range(cfg.iterations):
            r1 = self._rand(S, D)
            r2 = self._rand(S, D)
            v = cfg.w * v + cfg.c1 * r1 * (pbest_x - x) + cfg.c2 * r2 * (gbest_x - x)
            v.clamp_(-vclip, vclip)
            x = (x + v).clamp(self.lower, self.upper)
            cost = self.fn(x)
            better = cost < pbest_c
            pbest_x = torch.where(better.unsqueeze(-1), x, pbest_x)
            pbest_c = torch.where(better, cost, pbest_c)
            g_idx = torch.argmin(pbest_c)
            if pbest_c[g_idx].item() < gbest_c:
                gbest_c = pbest_c[g_idx].item()
                gbest_x = pbest_x[g_idx].clone()
            history.append(gbest_c)

        return OptimizerResult(
            best_x=gbest_x.detach().cpu(),
            best_cost=gbest_c,
            history=history,
            wall_time_s=perf_counter() - t0,
            iterations=cfg.iterations,
            extra={"swarm_size": cfg.swarm_size},
        )

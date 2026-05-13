"""Hybrid Quantum-Inspired PSO + Differential Evolution (QIPSO-DE).

Motivation. On the multi-UAV problem the classical DE/rand/1/bin update
out-performs canonical PSO at large M because the random differential
vector (a-b) provides a wider, scale-adaptive search step than the
PSO inertia-weighted update. QIPSO, in turn, beats DE at M=2 because
its quantum mix biases the swarm toward the global best.

This module combines the two: each particle simultaneously maintains
(i) a Q-bit state (alpha, beta) and (ii) a classical position x; each
iteration applies a DE mutation/crossover step in classical space and
then a quantum rotation gate + decaying quantum mix on top. Selection
is greedy (DE-style) to preserve high-quality solutions.

Hyperparameters merge those of DE (F, CR) and QIPSO (theta_max,
q_mix0, q_mix_decay). Default values are inherited from the parent
algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class QIPSODEConfig:
    pop_size: int = 60
    iterations: int = 300
    F: float = 0.5                  # DE differential weight
    CR: float = 0.9                 # DE crossover probability
    theta_max: float = 0.05 * 3.14159265   # max quantum rotation angle
    q_mix0: float = 0.08            # initial quantum-mix weight
    q_mix_decay: float = 0.97       # geometric decay per iteration
    use_rotation: bool = True


class QIPSODE:
    name = "QIPSODE"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: QIPSODEConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or QIPSODEConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _measure(self, beta: torch.Tensor) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * beta.pow(2)

    def run(self) -> OptimizerResult:
        cfg = self.cfg
        P, D = cfg.pop_size, self.dim
        rng = self.upper - self.lower

        # ---- initialize population ----
        if self.init_x is not None:
            pop = self.init_x[:P].to(self.device).clamp(self.lower, self.upper)
        else:
            pop = self.lower + self._rand(P, D) * rng

        # Q-bits track the current population
        beta = torch.sqrt(torch.clamp((pop - self.lower) / rng, min=0.0, max=1.0))
        alpha = torch.sqrt(torch.clamp(1.0 - beta.pow(2), min=0.0))

        cost = self.fn(pop)
        g_idx = torch.argmin(cost)
        gbest_x = pop[g_idx].clone()
        gbest_c = cost[g_idx].item()

        history = [gbest_c]
        t0 = perf_counter()
        q_mix = cfg.q_mix0

        for it in range(cfg.iterations):
            # ---- DE/rand/1/bin mutation ----
            # sample 3 distinct indices per target
            idx = torch.argsort(self._rand(P, P), dim=1)[:, :3]
            a, b, c = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]
            mutant = (a + cfg.F * (b - c)).clamp(self.lower, self.upper)
            cross_mask = self._rand(P, D) < cfg.CR
            j_rand = torch.randint(D, (P, 1), generator=self.gen, device=self.device)
            cross_mask.scatter_(1, j_rand, True)
            trial = torch.where(cross_mask, mutant, pop)

            # ---- quantum rotation gate (toward gbest) ----
            if cfg.use_rotation:
                norm = (gbest_x - self.lower) / rng
                target_beta = torch.sqrt(torch.clamp(norm, min=0.0, max=1.0))
                direction = torch.sign(target_beta.unsqueeze(0) - beta)
                theta = direction * cfg.theta_max * self._rand(P, D)
                ca, sa = torch.cos(theta), torch.sin(theta)
                new_alpha = ca * alpha - sa * beta
                new_beta = sa * alpha + ca * beta
                alpha, beta = new_alpha, new_beta
                xq = self._measure(beta)
                # decaying quantum mix: nudge the DE trial vector toward Q
                trial = trial + q_mix * (xq - trial)
                trial.clamp_(self.lower, self.upper)
                q_mix = q_mix * cfg.q_mix_decay

            # ---- DE-style greedy selection ----
            trial_cost = self.fn(trial)
            better = trial_cost < cost
            pop = torch.where(better.unsqueeze(-1), trial, pop)
            cost = torch.where(better, trial_cost, cost)

            # Recompute beta from accepted positions so Q-bits track pop
            beta = torch.sqrt(torch.clamp((pop - self.lower) / rng, min=0.0, max=1.0))
            alpha = torch.sqrt(torch.clamp(1.0 - beta.pow(2), min=0.0))

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

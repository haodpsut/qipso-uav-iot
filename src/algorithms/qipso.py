"""Quantum-Inspired Particle Swarm Optimization (QIPSO).

Each decision component j of particle i is represented by a Q-bit
( alpha_{i,j}, beta_{i,j} ) with alpha^2 + beta^2 = 1. The continuous
position is obtained by measurement:

    x_{i,j} = lower_j + (upper_j - lower_j) * beta_{i,j}^2

That is, beta^2 plays the role of the probability of the "1" basis state,
mapped onto the search box for that dimension.

Evolution combines the classical PSO velocity update on the *position*
with a *quantum rotation gate* on the Q-bit angles so the swarm both
exploits (toward Pbest/Gbest, via velocity) and explores (via the
rotation gate, which can flip the bias).

This implementation is fully batched on (swarm_size x dim) tensors and
runs on GPU when the env tensors live on cuda.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class QIPSOConfig:
    swarm_size: int = 60
    iterations: int = 300
    w: float = 0.7                  # inertia
    c1: float = 1.5                 # cognitive
    c2: float = 1.5                 # social
    v_clip: float = 0.2             # fraction of search range
    theta_max: float = 0.05 * 3.14159265   # max rotation angle (rad)
    q_mix0: float = 0.08            # starting weight of quantum perturbation
    q_mix_decay: float = 0.97       # geometric decay per iteration
    use_rotation: bool = True       # ablation switch
    use_quantum_init: bool = True   # ablation switch (superposition init)


class QIPSO:
    name = "QIPSO"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: QIPSOConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or QIPSOConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _measure(self, beta: torch.Tensor) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * beta.pow(2)

    def run(self) -> OptimizerResult:
        cfg = self.cfg
        S, D = cfg.swarm_size, self.dim
        rng = self.upper - self.lower

        # ---- Q-bit initialization ----
        if cfg.use_quantum_init:
            theta0 = self._rand(S, D) * 2.0 * torch.pi
            alpha = torch.cos(theta0)
            beta = torch.sin(theta0)
        else:
            beta = self._rand(S, D)
            alpha = torch.sqrt(torch.clamp(1.0 - beta.pow(2), min=0.0))

        # ---- classical position / velocity ----
        if self.init_x is not None:
            x = self.init_x[:S].to(self.device).clamp(self.lower, self.upper)
            # set beta to match this position so the rotation gate has a sensible start
            beta = torch.sqrt(torch.clamp((x - self.lower) / rng, min=0.0, max=1.0))
            alpha = torch.sqrt(torch.clamp(1.0 - beta.pow(2), min=0.0))
        else:
            x = self._measure(beta)
        v = (self._rand(S, D) - 0.5) * cfg.v_clip * rng

        # initial evaluation
        cost = self.fn(x)
        pbest_x = x.clone()
        pbest_c = cost.clone()
        g_idx = torch.argmin(pbest_c)
        gbest_x = pbest_x[g_idx].clone()
        gbest_c = pbest_c[g_idx].item()

        history = [gbest_c]
        t0 = perf_counter()
        vclip = cfg.v_clip * rng
        q_mix = cfg.q_mix0

        for it in range(cfg.iterations):
            # ---- velocity / position update ----
            r1 = self._rand(S, D)
            r2 = self._rand(S, D)
            v = cfg.w * v + cfg.c1 * r1 * (pbest_x - x) + cfg.c2 * r2 * (gbest_x - x)
            v.clamp_(-vclip, vclip)
            x = (x + v).clamp(self.lower, self.upper)

            # ---- quantum rotation gate ----
            if cfg.use_rotation:
                # target Q-bit derived from current gbest direction
                norm = (gbest_x - self.lower) / rng                  # in [0,1]
                target_beta = torch.sqrt(torch.clamp(norm, min=0.0, max=1.0))
                # angle sign chosen to rotate beta toward target_beta
                direction = torch.sign(target_beta.unsqueeze(0) - beta)
                theta = direction * cfg.theta_max * self._rand(S, D)
                ca, sa = torch.cos(theta), torch.sin(theta)
                new_alpha = ca * alpha - sa * beta
                new_beta = sa * alpha + ca * beta
                alpha, beta = new_alpha, new_beta
                # Soft quantum perturbation: nudge x toward the measured
                # Q-bit position by a small (and decaying) fraction. This
                # adds biased exploration without wiping out PSO progress.
                xq = self._measure(beta)
                x = x + q_mix * (xq - x)
                q_mix = q_mix * cfg.q_mix_decay

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

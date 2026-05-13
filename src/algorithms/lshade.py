"""L-SHADE: SHADE with Linear population size reduction.

Reference: Tanabe & Fukunaga. "Improving the search performance of
SHADE using linear population size reduction."  IEEE CEC 2014. Winner
of the CEC 2014 competition on real-parameter single-objective opt.

Core ideas:
  * "current-to-pbest/1" mutation that pulls toward one of the top-100p%
    individuals (instead of always toward the global best).
  * Per-individual control parameters  F_i, CR_i  drawn from two
    Cauchy/Normal "memory" arrays  M_F, M_CR  that store the
    success-history of recent improvements.
  * An external archive  A  of recently-replaced parents that augments
    the donor pool for diversity.
  * Linear population-size reduction (LPSR) from  N_init  down to
    N_min  (typically 4) across the iteration budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .base import OptimizerResult


@dataclass
class LSHADEConfig:
    pop_size: int = 60              # N_init
    pop_min: int = 8
    iterations: int = 300
    memory_size: int = 5            # H
    p_best_frac: float = 0.11       # fraction of pbest pool
    archive_factor: float = 1.4     # |A| <= archive_factor * |P|


class LSHADE:
    name = "LSHADE"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: LSHADEConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or LSHADEConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randint(self, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(high, shape, generator=self.gen, device=self.device)

    @staticmethod
    def _lehmer_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        num = (w * x.pow(2)).sum()
        den = (w * x).sum().clamp_min(1e-12)
        return num / den

    def run(self) -> OptimizerResult:
        cfg = self.cfg
        D = self.dim
        rng = self.upper - self.lower
        N_init = cfg.pop_size

        if self.init_x is not None:
            pop = self.init_x[:N_init].to(self.device).clamp(self.lower, self.upper)
        else:
            pop = self.lower + self._rand(N_init, D) * rng
        cost = self.fn(pop)

        archive = torch.empty(0, D, device=self.device, dtype=pop.dtype)
        M_F = torch.full((cfg.memory_size,), 0.5, device=self.device)
        M_CR = torch.full((cfg.memory_size,), 0.5, device=self.device)
        mem_pos = 0

        gbest_c = cost.min().item()
        gbest_x = pop[cost.argmin()].clone()
        history = [gbest_c]
        t0 = perf_counter()

        for t in range(cfg.iterations):
            N = pop.shape[0]

            # --- sample F_i, CR_i from memory ---
            r_mem = self._randint(cfg.memory_size, N)
            mu_F  = M_F[r_mem]
            mu_CR = M_CR[r_mem]

            # CR_i ~ N(mu_CR, 0.1)
            CR = (mu_CR + 0.1 * torch.randn(N, generator=self.gen, device=self.device)).clamp(0.0, 1.0)
            # F_i ~ Cauchy(mu_F, 0.1); resample until F > 0; cap at 1
            u = self._rand(N) - 0.5
            F = (mu_F + 0.1 * torch.tan(torch.pi * u)).clamp(min=1e-6, max=1.0)

            # --- current-to-pbest/1 mutation ---
            p = max(2, int(cfg.p_best_frac * N))
            top_idx = torch.argsort(cost)[:p]
            pbest = pop[top_idx[self._randint(p, N)]]

            # r1 from population, r2 from pop ∪ archive (distinct from i and r1)
            r1 = torch.zeros(N, dtype=torch.long, device=self.device)
            for i in range(N):
                while True:
                    j = int(self._randint(N, 1).item())
                    if j != i:
                        r1[i] = j
                        break

            pool = torch.cat([pop, archive], dim=0) if archive.shape[0] else pop
            r2 = torch.zeros(N, dtype=torch.long, device=self.device)
            for i in range(N):
                while True:
                    j = int(self._randint(pool.shape[0], 1).item())
                    if j != i and j != int(r1[i].item()):
                        r2[i] = j
                        break

            x_r1 = pop[r1]
            x_r2 = pool[r2]
            mutant = pop + F.unsqueeze(1) * (pbest - pop) + F.unsqueeze(1) * (x_r1 - x_r2)
            mutant = mutant.clamp(self.lower, self.upper)

            # --- binomial crossover ---
            cross = self._rand(N, D) < CR.unsqueeze(1)
            j_rand = self._randint(D, N, 1)
            cross.scatter_(1, j_rand, True)
            trial = torch.where(cross, mutant, pop)

            trial_cost = self.fn(trial)
            better = trial_cost < cost
            # archive the replaced parents
            if better.any():
                replaced = pop[better].clone()
                archive = torch.cat([archive, replaced], dim=0)
                max_a = int(cfg.archive_factor * N)
                if archive.shape[0] > max_a:
                    # keep the most recent max_a entries
                    archive = archive[-max_a:]

            # adapt memory M_F, M_CR with weighted (Lehmer) mean
            if better.any():
                # weights proportional to |delta_cost|
                delta = (cost[better] - trial_cost[better]).clamp_min(1e-12)
                w = delta / delta.sum().clamp_min(1e-12)
                M_F[mem_pos]  = self._lehmer_mean(F[better],  w)
                M_CR[mem_pos] = self._lehmer_mean(CR[better], w)
                mem_pos = (mem_pos + 1) % cfg.memory_size

            pop = torch.where(better.unsqueeze(-1), trial, pop)
            cost = torch.where(better, trial_cost, cost)

            # --- linear population-size reduction ---
            N_new = int(round(N_init - (N_init - cfg.pop_min) * (t + 1) / cfg.iterations))
            if N_new < N:
                order = torch.argsort(cost)[:N_new]
                pop = pop[order]
                cost = cost[order]

            min_idx = int(cost.argmin().item())
            if cost[min_idx].item() < gbest_c:
                gbest_c = cost[min_idx].item()
                gbest_x = pop[min_idx].clone()
            history.append(gbest_c)

        return OptimizerResult(
            best_x=gbest_x.detach().cpu(),
            best_cost=gbest_c,
            history=history,
            wall_time_s=perf_counter() - t0,
            iterations=cfg.iterations,
            extra={"pop_init": N_init, "pop_final": int(pop.shape[0])},
        )

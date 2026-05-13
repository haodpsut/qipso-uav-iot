"""(mu/mu_w, lambda)-CMA-ES with rank-mu update.

Reference: Hansen & Ostermeier. "Completely derandomized
self-adaptation in evolution strategies."  Evolutionary Computation,
2001.  Tutorial: Hansen 2023 (arXiv:1604.00772).

State maintained between iterations:
  * mean vector  m  in R^D
  * step-size    sigma  > 0
  * covariance   C  in R^{D x D}
  * evolution paths  p_sigma, p_c

Each iteration samples  lambda  offspring  x_k ~ m + sigma * N(0, C),
keeps the top  mu  by fitness, updates the mean by a weighted average,
and adapts  sigma  (cumulative step-size adaptation) and  C
(rank-1 + rank-mu update).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import math
import torch

from .base import OptimizerResult


@dataclass
class CMAESConfig:
    pop_size: int = 60         # lambda (offspring per generation)
    iterations: int = 300
    sigma0_frac: float = 0.05  # initial step-size = sigma0_frac * range
    rank_one_only: bool = False


class CMAES:
    name = "CMAES"

    def __init__(self, fitness_fn, dim: int, lower: float, upper: float,
                 cfg: CMAESConfig | None = None, device: str = "cpu",
                 seed: int = 0, init_x: torch.Tensor | None = None):
        self.fn = fitness_fn
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.cfg = cfg or CMAESConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.init_x = init_x

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def run(self) -> OptimizerResult:
        cfg = self.cfg
        D = self.dim
        lam = cfg.pop_size
        rng = self.upper - self.lower
        dtype = torch.float64                # numerical stability for C

        # --- selection weights (Hansen 2016 tutorial defaults) ---
        mu = lam // 2
        weights_raw = torch.tensor(
            [math.log((lam + 1) / 2.0) - math.log(i + 1) for i in range(lam)],
            dtype=dtype, device=self.device)
        weights = weights_raw.clone()
        weights[mu:] = 0.0
        weights = weights / weights[:mu].sum()
        mueff = 1.0 / (weights[:mu].pow(2).sum().item())

        # --- learning rates (canonical) ---
        c_sigma = (mueff + 2.0) / (D + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (D + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mueff / D) / (D + 4.0 + 2.0 * mueff / D)
        c_1 = 2.0 / ((D + 1.3) ** 2 + mueff)
        c_mu = min(1.0 - c_1,
                   2.0 * (mueff - 2.0 + 1.0 / mueff) / ((D + 2.0) ** 2 + mueff))
        if cfg.rank_one_only:
            c_mu = 0.0
        chi_N = math.sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D * D))

        # --- initialize m, sigma, C ---
        if self.init_x is not None:
            m = self.init_x[:lam].to(self.device).to(dtype).mean(dim=0)
        else:
            m = torch.full((D,), 0.5 * (self.lower + self.upper),
                           dtype=dtype, device=self.device)
        sigma = cfg.sigma0_frac * rng
        C = torch.eye(D, dtype=dtype, device=self.device)
        p_sigma = torch.zeros(D, dtype=dtype, device=self.device)
        p_c = torch.zeros(D, dtype=dtype, device=self.device)

        gbest_c = float("inf")
        gbest_x = m.clone()
        history = []
        t0 = perf_counter()

        # eigendecomposition cache; re-evaluate every few generations
        eig_period = max(1, int(1 / (10 * D * (c_1 + c_mu))))

        def _eig(C):
            # symmetrize for stability
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = torch.linalg.eigh(C)
            eigvals = eigvals.clamp_min(1e-12)
            return eigvecs, eigvals

        eigvecs, eigvals = _eig(C)
        diag_D = torch.sqrt(eigvals)

        for t in range(cfg.iterations):
            # ---- sample offspring ----
            z = self._randn(lam, D).to(dtype)
            # y = B * diag(D) * z
            y = (eigvecs @ (diag_D.unsqueeze(-1) * z.T)).T   # (lam, D)
            x = m.unsqueeze(0) + sigma * y
            x_clamped = x.clamp(self.lower, self.upper)

            cost = self.fn(x_clamped.to(torch.float32)).to(dtype)
            order = torch.argsort(cost)[:mu]
            best_idx = int(order[0].item())
            if float(cost[best_idx].item()) < gbest_c:
                gbest_c = float(cost[best_idx].item())
                gbest_x = x_clamped[best_idx].clone()

            # weighted mean update (note: use the *unclamped* y so the
            # covariance update sees the true offspring direction)
            y_sel = y[order]                        # (mu, D)
            yw = (weights[:mu].unsqueeze(-1) * y_sel).sum(dim=0)  # (D,)
            m = m + sigma * yw
            m = m.clamp(self.lower, self.upper)

            # ---- cumulation for sigma (p_sigma) ----
            # C^{-1/2} * yw = B * diag(1/D) * B^T * yw
            inv_sqrt_C_yw = eigvecs @ ((eigvecs.T @ yw) / diag_D)
            p_sigma = (1 - c_sigma) * p_sigma + \
                      math.sqrt(c_sigma * (2 - c_sigma) * mueff) * inv_sqrt_C_yw
            sigma = sigma * math.exp((c_sigma / d_sigma) *
                                     (float(p_sigma.norm().item()) / chi_N - 1.0))
            # clip sigma to a sensible band
            sigma = max(min(sigma, rng), 1e-6 * rng)

            # ---- cumulation for C (p_c) ----
            hsig_lhs = float(p_sigma.norm().item()) / \
                       math.sqrt(1 - (1 - c_sigma) ** (2 * (t + 1)))
            hsig = 1.0 if hsig_lhs < (1.4 + 2.0 / (D + 1)) * chi_N else 0.0
            p_c = (1 - c_c) * p_c + \
                  hsig * math.sqrt(c_c * (2 - c_c) * mueff) * yw

            # rank-1 and rank-mu updates
            artmp = y_sel                            # (mu, D)
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * (torch.outer(p_c, p_c)
                          + (1 - hsig) * c_c * (2 - c_c) * C)
                 + c_mu * (weights[:mu].unsqueeze(-1).unsqueeze(-1)
                           * (artmp.unsqueeze(-1) * artmp.unsqueeze(1))).sum(dim=0))

            # re-eigendecompose periodically
            if (t + 1) % eig_period == 0:
                eigvecs, eigvals = _eig(C)
                diag_D = torch.sqrt(eigvals)

            history.append(gbest_c)

        return OptimizerResult(
            best_x=gbest_x.detach().to(torch.float32).cpu(),
            best_cost=gbest_c,
            history=history,
            wall_time_s=perf_counter() - t0,
            iterations=cfg.iterations,
            extra={"lambda": lam, "mu": mu, "sigma_final": float(sigma)},
        )

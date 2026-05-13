"""Common interfaces / dataclasses for optimizers."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch


@dataclass
class OptimizerResult:
    best_x: torch.Tensor                      # (dim,)
    best_cost: float
    history: list = field(default_factory=list)   # per-iteration best cost
    wall_time_s: float = 0.0
    iterations: int = 0
    extra: dict = field(default_factory=dict)


class _Timer:
    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = perf_counter() - self.t0

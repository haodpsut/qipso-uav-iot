"""Shared helpers for experiment runners.

* build_env(cfg)          -> UAVIoTEnv or MultiUAVEnv
* build_optimizer(name, env, seed) -> optimizer instance
* run_one(name, env, seed) -> dict row (csv-friendly)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# allow running scripts directly from experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import OPTIMIZERS
from src.algorithms.qipso import QIPSOConfig
from src.algorithms.qipso_de import QIPSODEConfig
from src.algorithms.pso import PSOConfig
from src.algorithms.ga import GAConfig
from src.algorithms.de import DEConfig
from src.env import (
    MultiUAVEnv,
    MultiUAVScenario,
    Scenario,
    UAVIoTEnv,
)


def device_auto() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_env(K: int = 20, N: int = 80, M: int = 1, area: float = 1000.0,
              H: float = 100.0, seed: int = 0,
              device: str | None = None) -> UAVIoTEnv | MultiUAVEnv:
    device = device or device_auto()
    sc = Scenario(area=area, K=K, H=H, N=N, M=M, seed=seed, device=device)
    if M <= 1:
        return UAVIoTEnv(sc)
    return MultiUAVEnv(MultiUAVScenario(base=sc, M=M))


def build_optimizer(name: str, env, dim: int, lower: float, upper: float,
                    iterations: int = 300, swarm_size: int = 60,
                    seed: int = 0, warm_start: bool = True, **overrides):
    name = name.lower()
    cls = OPTIMIZERS[name]
    device = str(env.device)
    init_x = env.warm_start(swarm_size, seed=seed) if warm_start else None
    if name == "qipso":
        cfg = QIPSOConfig(iterations=iterations, swarm_size=swarm_size, **overrides)
        return cls(env.fitness, dim, lower, upper, cfg=cfg, device=device,
                   seed=seed, init_x=init_x)
    if name == "qipsode":
        cfg = QIPSODEConfig(iterations=iterations, pop_size=swarm_size, **overrides)
        return cls(env.fitness, dim, lower, upper, cfg=cfg, device=device,
                   seed=seed, init_x=init_x)
    if name == "pso":
        cfg = PSOConfig(iterations=iterations, swarm_size=swarm_size, **overrides)
        return cls(env.fitness, dim, lower, upper, cfg=cfg, device=device,
                   seed=seed, init_x=init_x)
    if name == "ga":
        cfg = GAConfig(iterations=iterations, pop_size=swarm_size, **overrides)
        return cls(env.fitness, dim, lower, upper, cfg=cfg, device=device,
                   seed=seed, init_x=init_x)
    if name == "de":
        cfg = DEConfig(iterations=iterations, pop_size=swarm_size, **overrides)
        return cls(env.fitness, dim, lower, upper, cfg=cfg, device=device,
                   seed=seed, init_x=init_x)
    if name in ("greedy", "straight"):
        return cls(env.fitness, dim, lower, upper, env=env, device=device, seed=seed)
    raise ValueError(name)


def run_one(alg: str, env, iterations: int, swarm_size: int, seed: int,
            extra: dict | None = None) -> dict:
    dim = env.dim
    opt = build_optimizer(alg, env, dim, 0.0, env.sc.area,
                          iterations=iterations, swarm_size=swarm_size,
                          seed=seed, **(extra or {}))
    res = opt.run()
    x = res.best_x.to(env.device).unsqueeze(0)
    m = env.metrics(x)
    return {
        "alg": alg,
        "seed": seed,
        "best_cost": float(res.best_cost),
        "energy_j": float(m["energy_j"].item()),
        "completion": float(m["completion"].item()),
        "wall_time_s": float(res.wall_time_s),
        "iterations": int(res.iterations),
        "history": [float(h) for h in res.history],
        "best_x": res.best_x.tolist(),
    }

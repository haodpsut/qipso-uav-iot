"""Experiment 5 — QIPSO hyperparameter sensitivity.

Grid sweep over the maximum rotation angle (theta_max) and inertia weight
(w) of QIPSO; report final cost. Used for the sensitivity heatmap figure.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import build_env, device_auto
from src.algorithms.qipso import QIPSO, QIPSOConfig
from src.utils import save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--swarm", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    theta_grid = [0.01, 0.03, 0.05, 0.08, 0.12, 0.2]
    w_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for th in theta_grid:
        for w in w_grid:
            for s in range(args.seeds):
                env = build_env(K=args.K, N=args.N, seed=s,
                                device=args.device or device_auto())
                cfg = QIPSOConfig(iterations=args.iters, swarm_size=args.swarm,
                                  theta_max=th * math.pi, w=w)
                init_x = env.warm_start(args.swarm, seed=s)
                opt = QIPSO(env.fitness, env.dim, 0.0, env.sc.area,
                            cfg=cfg, device=str(env.device), seed=s, init_x=init_x)
                res = opt.run()
                rows.append({"theta_frac_pi": th, "w": w, "seed": s,
                             "best_cost": float(res.best_cost),
                             "wall_time_s": float(res.wall_time_s)})
                print(f"theta={th}pi w={w} seed={s} cost={res.best_cost:.2f}")

    save_csv(rows, Path(args.out) / "csv" / "sensitivity_metrics.csv")
    print(f"\nSaved {len(rows)} rows.")


if __name__ == "__main__":
    main()

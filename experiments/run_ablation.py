"""Experiment 4 — QIPSO ablation.

Isolate the effect of (1) quantum superposition initialization and
(2) the quantum rotation gate. Four variants on single-UAV K=20:

  * QIPSO-full        : both on   (proposed)
  * QIPSO-no_rotation : quantum init only
  * QIPSO-no_qinit    : rotation only (classical PSO init)
  * QIPSO-neither     : equivalent to vanilla PSO

Output: ablation_metrics.csv + per-seed convergence histories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import build_env, device_auto
from src.algorithms.qipso import QIPSO, QIPSOConfig
from src.utils import save_csv, save_json


VARIANTS = {
    "full":        dict(use_rotation=True,  use_quantum_init=True),
    "no_rotation": dict(use_rotation=False, use_quantum_init=True),
    "no_qinit":    dict(use_rotation=True,  use_quantum_init=False),
    "neither":     dict(use_rotation=False, use_quantum_init=False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--swarm", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    rows, histories = [], []
    for name, flags in VARIANTS.items():
        for s in range(args.seeds):
            env = build_env(K=args.K, N=args.N, seed=s,
                            device=args.device or device_auto())
            cfg = QIPSOConfig(iterations=args.iters, swarm_size=args.swarm, **flags)
            init_x = env.warm_start(args.swarm, seed=s)
            opt = QIPSO(env.fitness, env.dim, 0.0, env.sc.area,
                        cfg=cfg, device=str(env.device), seed=s, init_x=init_x)
            res = opt.run()
            x = res.best_x.to(env.device).unsqueeze(0)
            m = env.metrics(x)
            row = {"variant": name, "seed": s,
                   "best_cost": float(res.best_cost),
                   "energy_j": float(m["energy_j"].item()),
                   "completion": float(m["completion"].item()),
                   "wall_time_s": float(res.wall_time_s)}
            rows.append(row)
            histories.append({"variant": name, "seed": s,
                              "history": [float(h) for h in res.history]})
            print(f"{name:12s} seed={s:02d} cost={row['best_cost']:10.2f} "
                  f"E={row['energy_j']:.1f}")

    out = Path(args.out)
    save_csv(rows, out / "csv" / "ablation_metrics.csv")
    save_json(histories, out / "logs" / "ablation_histories.json")
    print(f"\nSaved {len(rows)} rows.")


if __name__ == "__main__":
    main()

"""Experiment 3 — Multi-UAV sweep.

Vary M (number of UAVs) in {1, 2, 3, 4} with K=60 nodes. Compare QIPSO vs
PSO/GA/DE on the joint trajectory problem; report total energy, completion,
and per-UAV breakdown. We also store one representative trajectory per
(alg, M) for figure plotting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import build_env, build_optimizer, device_auto
from src.utils import save_csv, save_json


ALGS = ["qipso", "pso", "ga", "de"]
M_LIST = [1, 2, 3, 4]


def run_one_multi(alg, env, iterations, swarm_size, seed):
    opt = build_optimizer(alg, env, env.dim, 0.0, env.sc.area,
                          iterations=iterations, swarm_size=swarm_size, seed=seed)
    res = opt.run()
    x = res.best_x.to(env.device).unsqueeze(0)
    m = env.metrics(x)
    return res, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--swarm", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--Ms", type=int, nargs="+", default=M_LIST)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    rows = []
    samples = []
    for M in args.Ms:
        for alg in ALGS:
            for s in range(args.seeds):
                env = build_env(K=args.K, N=args.N, M=M, seed=s,
                                device=args.device or device_auto())
                res, m = run_one_multi(alg, env, args.iters, args.swarm, s)
                row = {
                    "alg": alg, "M": M, "seed": s,
                    "best_cost": float(res.best_cost),
                    "energy_j": float(m["energy_j"].item()),
                    "completion": float(m["completion"].item()),
                    "wall_time_s": float(res.wall_time_s),
                }
                rows.append(row)
                print(f"M={M} {alg:6s} seed={s:02d} cost={row['best_cost']:10.2f} "
                      f"E={row['energy_j']:.1f} comp={row['completion']:.3f}")
                if s == 0:                       # store a representative trajectory
                    samples.append({
                        "alg": alg, "M": M, "seed": s,
                        "trajectory": m["trajectory"].squeeze(0).cpu().tolist(),
                        "nodes": env.nodes.cpu().tolist(),
                        "labels": env.labels.cpu().tolist() if hasattr(env, "labels")
                                  else [0] * env.sc.K,
                        "data_required": env.data_required.cpu().tolist(),
                    })

    save_csv(rows, out / "csv" / "multi_uav_metrics.csv")
    save_json(samples, out / "logs" / "multi_uav_trajectories.json")
    print(f"\nSaved {len(rows)} rows.")


if __name__ == "__main__":
    main()

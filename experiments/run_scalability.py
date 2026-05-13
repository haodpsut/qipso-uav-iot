"""Experiment 2 — Scalability vs number of IoT nodes K.

Sweep K in {10, 20, 50, 100, 150}, single-UAV, all algorithms (including
non-evolutionary baselines). Each (alg, K) pair is run with `--seeds`
independent seeds and we record final cost / energy / completion / runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import build_env, run_one, device_auto
from src.utils import save_csv


ALGS = ["qipso", "pso", "ga", "de", "greedy", "straight"]
K_LIST = [10, 20, 50, 100, 150]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--swarm", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--Ks", type=int, nargs="+", default=K_LIST)
    parser.add_argument("--algs", type=str, nargs="+", default=ALGS)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    rows = []
    for K in args.Ks:
        for alg in args.algs:
            for s in range(args.seeds):
                env = build_env(K=K, N=args.N, seed=s,
                                device=args.device or device_auto())
                r = run_one(alg, env, args.iters, args.swarm, s)
                r["K"] = K
                r.pop("history", None)
                r.pop("best_x", None)
                rows.append(r)
                print(f"K={K:3d} {alg:8s} seed={s:02d} "
                      f"cost={r['best_cost']:10.2f} t={r['wall_time_s']:.2f}s")

    out = Path(args.out)
    save_csv(rows, out / "csv" / "scalability_metrics.csv")
    print(f"\nSaved {len(rows)} rows.")


if __name__ == "__main__":
    main()

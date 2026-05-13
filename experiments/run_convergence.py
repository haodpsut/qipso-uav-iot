"""Experiment 1 — Convergence study.

Single-UAV, fixed K=20, fixed N=80. Run QIPSO / PSO / GA / DE for
`--iters` iterations across `--seeds` independent seeds. Save per-iteration
best-cost history (for the convergence plot) plus final metrics CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import build_env, run_one, device_auto
from src.utils import save_csv, save_json


ALGS = ["qipso", "qipsode", "pso", "ga", "de", "gwo", "lshade", "cmaes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--swarm", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    rows = []
    histories = []
    for alg in ALGS:
        for s in range(args.seeds):
            env = build_env(K=args.K, N=args.N, seed=s, device=args.device or device_auto())
            r = run_one(alg, env, args.iters, args.swarm, s)
            print(f"{alg:6s} seed={s:02d} cost={r['best_cost']:10.2f} "
                  f"energy={r['energy_j']:9.2f} comp={r['completion']:.3f} "
                  f"time={r['wall_time_s']:.2f}s")
            histories.append({"alg": alg, "seed": s, "history": r.pop("history"),
                              "best_x": r.pop("best_x")})
            rows.append(r)

    save_csv(rows, out / "csv" / "convergence_metrics.csv")
    save_json(histories, out / "logs" / "convergence_histories.json")
    print(f"\nSaved {len(rows)} rows to {out/'csv/convergence_metrics.csv'}")


if __name__ == "__main__":
    main()

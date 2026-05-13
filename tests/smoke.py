"""Quick smoke test runnable from repo root:

    python tests/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import build_env, build_optimizer


def main():
    for K in (10, 20, 50):
        print(f"--- single-UAV K={K} N=80 ---")
        env = build_env(K=K, N=80, seed=0, device="cpu")
        for alg in ("qipso", "pso", "ga", "de", "greedy", "straight"):
            opt = build_optimizer(alg, env, env.dim, 0.0, env.sc.area,
                                  iterations=300, swarm_size=50, seed=0)
            r = opt.run()
            x = r.best_x.to(env.device).unsqueeze(0)
            m = env.metrics(x)
            print(f"  {alg:9s}: cost={r.best_cost:11.1f}  "
                  f"E={m['energy_j'].item():9.1f}J  "
                  f"comp={m['completion'].item():.3f}  "
                  f"t={r.wall_time_s:.2f}s")
        print()

    print("--- multi-UAV M=3 K=30 N=80 ---")
    env = build_env(K=30, N=80, M=3, seed=0, device="cpu")
    for alg in ("qipso", "pso", "ga", "de"):
        opt = build_optimizer(alg, env, env.dim, 0.0, env.sc.area,
                              iterations=300, swarm_size=50, seed=0)
        r = opt.run()
        x = r.best_x.to(env.device).unsqueeze(0)
        m = env.metrics(x)
        print(f"  {alg:6s}: cost={r.best_cost:11.1f}  "
              f"E={m['energy_j'].item():9.1f}  "
              f"comp={m['completion'].item():.3f}")
    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()

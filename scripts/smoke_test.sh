#!/usr/bin/env bash
# 30-second smoke test to verify the install works.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
python - <<'PY'
import torch
from src.env import UAVIoTEnv, Scenario
from src.algorithms.qipso import QIPSO, QIPSOConfig
from src.algorithms.pso  import PSO,  PSOConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
sc = Scenario(K=10, N=40, seed=0, device=device)
env = UAVIoTEnv(sc)

for cls, name in [(QIPSO, "QIPSO"), (PSO, "PSO")]:
    opt = cls(env.fitness, env.dim, 0.0, sc.area, device=device, seed=0)
    if name == "QIPSO":
        opt.cfg = QIPSOConfig(iterations=30, swarm_size=20)
    else:
        opt.cfg = PSOConfig(iterations=30, swarm_size=20)
    r = opt.run()
    print(f"{name}: best cost = {r.best_cost:.1f}  wall = {r.wall_time_s:.2f}s")
print("smoke test OK")
PY

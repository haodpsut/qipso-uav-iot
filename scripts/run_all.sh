#!/usr/bin/env bash
# Full experiment pipeline on the RTX 4090 server.
# Usage:  bash scripts/run_all.sh                # full run (~hours)
#         bash scripts/run_all.sh --quick        # smoke test (~minutes)
set -e

QUICK=0
if [ "${1:-}" = "--quick" ]; then QUICK=1; fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

if [ "$QUICK" = "1" ]; then
  ITERS=40; SWARM=20; SEEDS=3; KS="10 20"; MS="1 2"
else
  ITERS=300; SWARM=60; SEEDS=30; KS="10 20 50 100 150"; MS="1 2 3 4"
fi

mkdir -p results/{csv,logs,figures}

echo "==[ 1/5 ] Convergence study"
python experiments/run_convergence.py \
  --iters "$ITERS" --swarm "$SWARM" --seeds "$SEEDS"

echo "==[ 2/5 ] Scalability sweep"
python experiments/run_scalability.py \
  --iters "$ITERS" --swarm "$SWARM" --seeds "$SEEDS" --Ks $KS

echo "==[ 3/5 ] Multi-UAV sweep"
python experiments/run_multi_uav.py \
  --iters "$ITERS" --swarm "$SWARM" --seeds "$SEEDS" --Ms $MS

echo "==[ 4/5 ] QIPSO ablation"
python experiments/run_ablation.py \
  --iters "$ITERS" --swarm "$SWARM" --seeds "$SEEDS"

echo "==[ 5/5 ] Sensitivity grid"
python experiments/run_sensitivity.py \
  --iters "$ITERS" --swarm "$SWARM" --seeds 5

echo "==[ figures ]"
python -m src.viz.plot_convergence
python -m src.viz.plot_results
python -m src.viz.plot_trajectory

echo "Done. See results/{csv,logs,figures}/"

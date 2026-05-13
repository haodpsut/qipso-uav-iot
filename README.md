# QIPSO for UAV-IoT Trajectory Optimization

Code companion for the paper *"Quantum-Inspired Particle Swarm
Optimization for Energy-Efficient Trajectory Control in UAV-assisted IoT
Networks"* (single- and multi-UAV).

## Setup (conda only, tested on RTX 4090)

```bash
conda env create -f environment.yml
conda activate qipso-uav
bash scripts/smoke_test.sh        # ~30 seconds; should print "smoke test OK"
```

If you need a different CUDA build of PyTorch, edit `environment.yml`
(swap `pytorch-cuda=12.1` for the version that matches the driver) before
running `conda env create`.

## Running experiments

### Full pipeline (~hours on a 4090)

```bash
bash scripts/run_all.sh
```

### Quick run (~minutes, for sanity-check / debugging)

```bash
bash scripts/run_all.sh --quick
```

### Individual experiments

```bash
python experiments/run_convergence.py --iters 300 --swarm 60 --seeds 30
python experiments/run_scalability.py --iters 300 --swarm 60 --seeds 30 \
       --Ks 10 20 50 100 150
python experiments/run_multi_uav.py   --iters 300 --swarm 60 --seeds 30 \
       --Ms 1 2 3 4
python experiments/run_ablation.py    --iters 300 --swarm 60 --seeds 20
python experiments/run_sensitivity.py --iters 200 --swarm 60 --seeds 5
```

Outputs land under `results/`:

```
results/
  csv/      *.csv                 per-run metrics (used by plotting)
  logs/     *.json                histories + sample trajectories
  figures/  *.pdf                 publication-ready PDFs
```

After experiments finish, regenerate every figure with:

```bash
python -m src.viz.plot_convergence
python -m src.viz.plot_results
python -m src.viz.plot_trajectory
```

## Layout

```
src/
  env/            scenario + propulsion + A2G channel + multi-UAV
  algorithms/     QIPSO, PSO, GA, DE, Greedy-TSP, StraightLine
  utils/          seeding + I/O helpers
  viz/            matplotlib plotting
experiments/      Convergence / Scalability / MultiUAV / Ablation / Sensitivity
scripts/          run_all.sh + smoke_test.sh
configs/          (reserved for YAML scenario overrides)
results/          CSV / JSON / PDFs produced by the runs
```

## Workflow with the paper repo

1. Author runs `bash scripts/run_all.sh` on the 4090.
2. Author commits `results/csv/`, `results/logs/`, `results/figures/`.
3. `git push origin main` — figures are then consumed by the LaTeX
   build of the paper.

## Reference

* Zeng, Xu, Zhang. "Energy Minimization for Wireless Communication With
  Rotary-Wing UAV." *IEEE Trans. Wireless Commun.*, 2019.
* Han, Kim. "Quantum-inspired evolutionary algorithm for a class of
  combinatorial optimization." *IEEE Trans. Evol. Comput.*, 2002.
* Al-Hourani, Kandeepan, Lardner. "Optimal LAP altitude for maximum
  coverage." *IEEE Wireless Commun. Lett.*, 2014.

# QIPSO-UAV-IoT

Open-source implementation, datasets, and evaluation pipeline for the
paper

> **Energy-Efficient Multi-UAV Trajectory Control for Mobile IoT Data
> Collection: A Quantum-Inspired Hybrid Metaheuristic and a Benchmark
> of Eight Algorithms**
>
> Phuc Hao Do, Tran Duc Le, Truong Duy Dinh, Van Dai Pham, 2026.

The repository contains **(i)** a fully-vectorized PyTorch simulator
of multi-UAV trajectory optimization with rotary-wing propulsion,
air-to-ground LoS/NLoS channel, service-radius-gated data collection,
and inter-UAV collision avoidance; **(ii)** reference implementations
of eight metaheuristics: the two proposed methods (QIPSO and Hybrid
QIPSO-DE) plus PSO, GA, DE, GWO, L-SHADE, CMA-ES, Greedy-TSP, and a
no-detour straight-line tour; **(iii)** the exact experiment scripts
that produce every figure and CSV file in the paper.

## Setup (conda-only, tested on RTX 4090)

```bash
git clone https://github.com/haodpsut/qipso-uav-iot.git
cd qipso-uav-iot

conda env create -f environment.yml
conda activate qipso-uav

bash scripts/smoke_test.sh        # ~30 seconds; should print "smoke test OK"
```

If you need a different CUDA build of PyTorch, edit
`environment.yml` (swap `pytorch-cuda=12.1` for the version that
matches the driver) before running `conda env create`.

A CPU-only fallback is also possible — the smoke test runs in about a
minute on a recent laptop CPU and is sufficient to verify code
correctness.

## Reproducing the paper

### Full pipeline (~hours on a 4090)

```bash
bash scripts/run_all.sh
```

This runs five experiments:

```
experiments/
  run_convergence.py     single-UAV K=20, all 8 algorithms (Fig. 6, Fig. 10)
  run_scalability.py     K in {10,20,50,100,150}            (Fig. 7)
  run_multi_uav.py       M in {1,2,3,4}                     (Fig. 8, Fig. 9, Fig. 11, Tab. III)
  run_ablation.py        QIPSO rotation-gate ablation       (Tab. IV, Fig. 13)
  run_sensitivity.py     (theta_max, w) grid                (Fig. 14)
```

Each experiment produces a CSV summary under `results/csv/`, a JSON
log under `results/logs/`, and a PDF figure under `results/figures/`.

### Quick run for sanity-check (~minutes)

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

### Re-generating the figures

After the experiments finish, the figures can be re-built from the
CSVs/JSONs at any time:

```bash
python -m src.viz.plot_convergence
python -m src.viz.plot_results        # scalability + boxplot + runtime + multi-UAV + ablation + sensitivity + Wilcoxon
python -m src.viz.plot_trajectory     # trajectory M=1..4
python -m src.viz.plot_insights       # Pareto frontier + budget-aware ranking
```

## Repository layout

```
src/
  env/            scenario + propulsion + A2G channel + multi-UAV
  algorithms/     QIPSO, QIPSO-DE, PSO, GA, DE, GWO, L-SHADE, CMA-ES,
                  Greedy-TSP, StraightLine
  utils/          seeding + I/O helpers
  viz/            matplotlib plotting + insight figures
experiments/      run_convergence / run_scalability / run_multi_uav
                  run_ablation / run_sensitivity
scripts/          run_all.sh + smoke_test.sh
configs/          (reserved for YAML scenario overrides)
results/          CSV / JSON / PDFs produced by the runs
tests/            smoke.py
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{do2026qipso,
  author  = {Do, Phuc Hao and Le, Tran Duc and Dinh, Truong Duy and Pham, Van Dai},
  title   = {Energy-Efficient Multi-UAV Trajectory Control for Mobile {IoT}
             Data Collection: A Quantum-Inspired Hybrid Metaheuristic and
             a Benchmark of Eight Algorithms},
  journal = {IEEE Transactions on Mobile Computing},
  year    = {2026},
  note    = {Under review.}
}
```

## References for the implemented baselines

* PSO: J. Kennedy and R. Eberhart, *Particle Swarm Optimization*, IEEE ICNN 1995.
* GA (real-coded BLX-$\alpha$): J. H. Holland, *Adaptation in Natural and Artificial Systems*, MIT Press 1992.
* DE/rand/1/bin: R. Storn and K. Price, *Differential evolution*, J. Global Optim. 1997.
* GWO: S. Mirjalili, S. M. Mirjalili, and A. Lewis, *Grey wolf optimizer*, Adv. Eng. Softw. 2014.
* L-SHADE: R. Tanabe and A. S. Fukunaga, *Improving the search performance of SHADE using linear population size reduction*, IEEE CEC 2014.
* CMA-ES: N. Hansen and A. Ostermeier, *Completely derandomized self-adaptation in evolution strategies*, Evol. Comput. 2001.
* Rotary-wing energy model: Y. Zeng, J. Xu, and R. Zhang, *Energy minimization for wireless communication with rotary-wing UAV*, IEEE TWC 2019.
* A2G channel model: A. Al-Hourani, S. Kandeepan, and S. Lardner, *Optimal LAP altitude for maximum coverage*, IEEE WCL 2014.

## License

Released under the MIT License -- see `LICENSE` for the full text.

## Contact

Truong Duy Dinh (corresponding author) -- `duydt@ptit.edu.vn`.

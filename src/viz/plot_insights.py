"""Insight-driven figures for the paper:

  * fig_pareto.pdf       runtime-vs-cost Pareto frontier (multi-UAV M=3)
  * fig_budget.pdf       cost as a function of the iteration budget;
                         exposes the slow-start of CMA-ES and the
                         fast convergence of L-SHADE.

Both figures are read from results/csv/multi_uav_metrics.csv and
results/logs/convergence_histories.json. They complement the main
performance/scalability/multi-UAV plots.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style, color_for
from .plot_results import _markers_for, _linestyle_for


def _read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def plot_pareto(csv_path: Path, out_path: Path, M: int = 3) -> None:
    """Runtime vs. cost scatter for the M-UAV scenario, mean over seeds."""
    apply_style()
    rows = _read_csv(csv_path)
    by = defaultdict(lambda: {"cost": [], "wall": []})
    for r in rows:
        if int(r["M"]) != M:
            continue
        by[r["alg"]]["cost"].append(float(r["best_cost"]))
        by[r["alg"]]["wall"].append(float(r["wall_time_s"]))
    algs = sorted(by.keys())
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for alg in algs:
        c = np.mean(by[alg]["cost"])
        cs = np.std(by[alg]["cost"])
        w = np.mean(by[alg]["wall"])
        ws = np.std(by[alg]["wall"])
        ax.errorbar(w, c, xerr=ws, yerr=cs,
                    fmt=_markers_for(alg), markersize=8,
                    color=color_for(alg), capsize=3, lw=0.9,
                    label=alg.upper())
        ax.annotate(alg.upper(),
                    xy=(w, c), xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=7.5, color=color_for(alg))
    ax.set_xlabel("Wall-clock time per run (s)")
    ax.set_ylabel("Final mission cost (J + penalties)")
    ax.set_yscale("log")
    ax.set_title(f"Runtime--cost trade-off, $M={M}$, $K=60$", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    # No legend (annotations serve)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def plot_budget(history_path: Path, out_path: Path,
                budgets=(20, 50, 100, 200, 300)) -> None:
    """For each algorithm and each iteration budget t in `budgets`, plot the
    mean best_cost-so-far. Reveals the budget-dependent ranking."""
    apply_style()
    data = json.loads(Path(history_path).read_text(encoding="utf-8"))
    by_alg = defaultdict(list)
    for r in data:
        by_alg[r["alg"]].append(r["history"])
    preferred = ["lshade", "cmaes", "qipsode", "de", "qipso",
                 "pso", "ga", "gwo"]
    algs_order = [a for a in preferred if a in by_alg] + \
                 [a for a in by_alg if a not in preferred]

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    x = np.arange(len(budgets))
    width = 0.10
    for i, alg in enumerate(algs_order):
        H = np.asarray(by_alg[alg], dtype=np.float64)
        means = []
        for t in budgets:
            t_idx = min(t, H.shape[1] - 1)
            means.append(H[:, t_idx].mean())
        ax.bar(x + (i - len(algs_order) / 2) * width,
               means, width,
               color=color_for(alg), label=alg.upper(),
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$t={b}$" for b in budgets])
    ax.set_yscale("log")
    ax.set_ylabel("Best cost so far (J + penalties)")
    ax.set_xlabel("Iteration budget")
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()
    root = Path(args.results)
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot_pareto(root / "csv" / "multi_uav_metrics.csv",
                figs / "fig_pareto.pdf", M=3)
    plot_budget(root / "logs" / "convergence_histories.json",
                figs / "fig_budget.pdf")


if __name__ == "__main__":
    main()

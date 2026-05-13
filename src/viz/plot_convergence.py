"""Plot mean +/- std convergence curve per algorithm.

Inputs:
  results/logs/convergence_histories.json
Output:
  results/figures/fig_convergence.pdf
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style, color_for
from .plot_results import _markers_for, _linestyle_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="results/logs/convergence_histories.json")
    parser.add_argument("--out", type=str, default="results/figures/fig_convergence.pdf")
    args = parser.parse_args()

    apply_style()
    rows = json.loads(Path(args.logs).read_text(encoding="utf-8"))
    by_alg: dict[str, list[list[float]]] = defaultdict(list)
    for r in rows:
        by_alg[r["alg"]].append(r["history"])

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    # plot in a fixed order so the legend is predictable
    preferred = ["qipso", "qipsode", "pso", "ga", "de",
                 "gwo", "lshade", "cmaes"]
    algs_order = [a for a in preferred if a in by_alg] + \
                 [a for a in by_alg if a not in preferred]

    for alg in algs_order:
        H = np.asarray(by_alg[alg], dtype=np.float64)
        mu = H.mean(axis=0)
        sd = H.std(axis=0)
        x = np.arange(mu.size)
        # sparsely-placed markers for legend-and-line clarity
        n_mark = 8
        mark_every = max(1, mu.size // n_mark)
        ax.plot(x, mu, label=alg.upper(),
                color=color_for(alg),
                linestyle=_linestyle_for(alg),
                marker=_markers_for(alg),
                markersize=4.5, markevery=mark_every,
                lw=1.3)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.13,
                        color=color_for(alg), lw=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost (J + penalties)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

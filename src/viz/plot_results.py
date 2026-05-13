"""Aggregate result plots: scalability curve, boxplot, runtime bar,
ablation bar, sensitivity heatmap, Wilcoxon p-value table.

Reads CSVs under results/csv/ and writes PDFs under results/figures/.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .style import apply_style, color_for, WONG_PALETTE


def _read(csv_path):
    with Path(csv_path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _markers_for(alg: str) -> str:
    return {
        "qipso":    "o",
        "qipsode":  "s",      # square — distinguishes from qipso
        "pso":      "^",
        "ga":       "v",
        "de":       "D",
        "gwo":      "<",
        "lshade":   ">",
        "cmaes":    "P",
        "greedy":   "X",
        "straight": "*",
    }.get(alg.lower(), "o")


def _linestyle_for(alg: str) -> str:
    return {
        "qipso":   "-",
        "qipsode": "-",
        "pso":     "--",
        "ga":      "--",
        "de":      "-.",
        "gwo":     ":",
        "lshade":  "-.",
        "cmaes":   ":",
        "greedy":  ":",
        "straight": ":",
    }.get(alg.lower(), "-")


def plot_scalability(csv_path: Path, out_dir: Path):
    """Two-panel scalability: (a) evolutionary algorithms + straight-line on
    a linear-ish scale where they are visible; (b) the divergent Greedy-TSP
    baseline on a separate axis. Plotting them on the same axes hides the
    inter-algorithm differences in the first group."""
    rows = _read(csv_path)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["alg"]][int(r["K"])].append(float(r["energy_j"]))

    # split into "compact" (close-to-optimal) and "divergent" (Greedy)
    divergent = {"greedy"}
    compact_algs = sorted([a for a in by if a not in divergent],
                          key=lambda a: a)
    div_algs = sorted([a for a in by if a in divergent])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.0, 2.6),
                                     gridspec_kw={"width_ratios": [3, 1]})

    for alg in compact_algs:
        Kd = by[alg]
        Ks = sorted(Kd.keys())
        mu = np.array([np.mean(Kd[k]) for k in Ks])
        sd = np.array([np.std(Kd[k]) for k in Ks])
        ax_a.errorbar(Ks, mu, yerr=sd, label=alg.upper(),
                      color=color_for(alg), marker=_markers_for(alg),
                      linestyle=_linestyle_for(alg),
                      markersize=4, capsize=2, lw=1.2)
    ax_a.set_xlabel("Number of IoT nodes $K$")
    ax_a.set_ylabel("Mission energy (J)")
    ax_a.legend(frameon=False, fontsize=7, ncol=2)

    for alg in div_algs:
        Kd = by[alg]
        Ks = sorted(Kd.keys())
        mu = np.array([np.mean(Kd[k]) for k in Ks])
        sd = np.array([np.std(Kd[k]) for k in Ks])
        ax_b.errorbar(Ks, mu, yerr=sd, label=alg.upper(),
                      color=color_for(alg), marker=_markers_for(alg),
                      markersize=4, capsize=2, lw=1.2)
    ax_b.set_xlabel("$K$")
    ax_b.set_ylabel("Mission energy (J)")
    ax_b.set_title("Greedy (divergent)", fontsize=8)
    ax_b.legend(frameon=False, fontsize=7)
    # use scientific notation on the divergent panel
    ax_b.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_b.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.tight_layout()
    path = out_dir / "fig_scalability.pdf"
    fig.savefig(path)
    print(f"Saved {path}")


def plot_boxplot(csv_path: Path, out_dir: Path):
    rows = _read(csv_path)
    by = defaultdict(list)
    for r in rows:
        by[r["alg"]].append(float(r["energy_j"]))
    algs = list(by.keys())
    data = [by[a] for a in algs]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=[a.upper() for a in algs], patch_artist=True)
    for patch, alg in zip(bp["boxes"], algs):
        patch.set_facecolor(color_for(alg))
        patch.set_alpha(0.6)
    ax.set_ylabel("Final energy (J)")
    path = out_dir / "fig_box_energy.pdf"
    fig.savefig(path)
    print(f"Saved {path}")


def plot_runtime(csv_path: Path, out_dir: Path):
    rows = _read(csv_path)
    by = defaultdict(list)
    for r in rows:
        by[r["alg"]].append(float(r["wall_time_s"]))
    algs = list(by.keys())
    mu = [np.mean(by[a]) for a in algs]
    sd = [np.std(by[a]) for a in algs]
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(algs)), mu, yerr=sd, capsize=3,
                  color=[color_for(a) for a in algs])
    ax.set_xticks(range(len(algs)))
    ax.set_xticklabels([a.upper() for a in algs])
    ax.set_ylabel("Wall time / run (s)")
    path = out_dir / "fig_runtime.pdf"
    fig.savefig(path)
    print(f"Saved {path}")


def plot_multi_uav(csv_path: Path, out_dir: Path):
    """Two-panel multi-UAV: (a) total cost (objective with penalties); (b)
    data completion ratio. Plotting cost reveals the QIPSO win at M=2 that
    is hidden by the energy-only view."""
    rows = _read(csv_path)
    by_cost = defaultdict(lambda: defaultdict(list))
    by_comp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cost[r["alg"]][int(r["M"])].append(float(r["best_cost"]))
        by_comp[r["alg"]][int(r["M"])].append(float(r["completion"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.8))

    algs = sorted(by_cost.keys())
    for alg in algs:
        Md = by_cost[alg]
        Ms = sorted(Md.keys())
        mu = np.array([np.mean(Md[m]) for m in Ms])
        sd = np.array([np.std(Md[m]) for m in Ms])
        ax1.errorbar(Ms, mu, yerr=sd, label=alg.upper(),
                     color=color_for(alg), marker=_markers_for(alg),
                     linestyle=_linestyle_for(alg),
                     markersize=4, capsize=2, lw=1.2)
    ax1.set_xlabel("Number of UAVs $M$")
    ax1.set_ylabel("Total mission cost (J + penalties)")
    ax1.set_yscale("log")
    ax1.set_xticks(sorted({m for d in by_cost.values() for m in d}))
    ax1.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax1.set_title("(a) cost", fontsize=9)

    for alg in algs:
        Md = by_comp[alg]
        Ms = sorted(Md.keys())
        mu = np.array([np.mean(Md[m]) for m in Ms])
        sd = np.array([np.std(Md[m]) for m in Ms])
        ax2.errorbar(Ms, mu, yerr=sd, label=alg.upper(),
                     color=color_for(alg), marker=_markers_for(alg),
                     linestyle=_linestyle_for(alg),
                     markersize=4, capsize=2, lw=1.2)
    ax2.set_xlabel("Number of UAVs $M$")
    ax2.set_ylabel("Data completion ratio")
    ax2.set_xticks(sorted({m for d in by_comp.values() for m in d}))
    ax2.set_title("(b) completion", fontsize=9)

    fig.tight_layout()
    path = out_dir / "fig_multi_uav.pdf"
    fig.savefig(path)
    print(f"Saved {path}")


def plot_ablation(csv_path: Path, out_dir: Path):
    rows = _read(csv_path)
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(float(r["best_cost"]))
    order = ["neither", "no_qinit", "no_rotation", "full"]
    labels = ["PSO\n(no Q-bit,\nno rotation)", "rotation\nonly",
              "Q-init\nonly", "QIPSO\n(both)"]
    mu = [np.mean(by[v]) for v in order]
    sd = [np.std(by[v]) for v in order]
    fig, ax = plt.subplots()
    bars = ax.bar(range(4), mu, yerr=sd, capsize=3,
                  color=[WONG_PALETTE[i] for i in range(4)])
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final cost (J)")
    path = out_dir / "fig_ablation.pdf"
    fig.savefig(path)
    print(f"Saved {path}")


def plot_sensitivity(csv_path: Path, out_dir: Path):
    rows = _read(csv_path)
    thetas = sorted({float(r["theta_frac_pi"]) for r in rows})
    ws = sorted({float(r["w"]) for r in rows})
    Z = np.zeros((len(thetas), len(ws)))
    cnt = np.zeros_like(Z)
    for r in rows:
        i = thetas.index(float(r["theta_frac_pi"]))
        j = ws.index(float(r["w"]))
        Z[i, j] += float(r["best_cost"])
        cnt[i, j] += 1
    Z = Z / cnt.clip(min=1)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([f"{w:g}" for w in ws])
    ax.set_yticks(range(len(thetas)))
    ax.set_yticklabels([f"{t:g}$\\pi$" for t in thetas])
    ax.set_xlabel(r"Inertia weight $\omega$")
    ax.set_ylabel(r"Max rotation angle $\theta_{\max}$")
    fig.colorbar(im, ax=ax, label="Final cost (J)")
    path = out_dir / "fig_sensitivity.pdf"
    fig.savefig(path)
    print(f"Saved {path}")


def wilcoxon_table(csv_path: Path, out_dir: Path):
    rows = _read(csv_path)
    by = defaultdict(list)
    for r in rows:
        by[r["alg"]].append((int(r["seed"]), float(r["energy_j"])))
    # only compare paired seeds
    algs = list(by.keys())
    table = {}
    if "qipso" not in algs:
        return
    base = dict(by["qipso"])
    for alg in algs:
        if alg == "qipso":
            continue
        other = dict(by[alg])
        seeds = sorted(set(base) & set(other))
        if len(seeds) < 5:
            continue
        a = [base[s] for s in seeds]
        b = [other[s] for s in seeds]
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="zsplit", alternative="less")
            table[alg] = {"stat": float(stat), "p": float(p), "n": len(seeds)}
        except ValueError as e:
            table[alg] = {"error": str(e)}
    out_csv = out_dir.parent / "csv" / "wilcoxon_vs_qipso.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alg", "stat", "p_value", "n_paired"])
        for alg, v in table.items():
            w.writerow([alg, v.get("stat", ""), v.get("p", ""), v.get("n", "")])
    print(f"Saved {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()
    apply_style()
    root = Path(args.results)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = root / "csv"

    targets = {
        "convergence_metrics.csv": [plot_boxplot, plot_runtime],
        "scalability_metrics.csv": [plot_scalability, wilcoxon_table],
        "multi_uav_metrics.csv":   [plot_multi_uav],
        "ablation_metrics.csv":    [plot_ablation],
        "sensitivity_metrics.csv": [plot_sensitivity],
    }
    for name, fns in targets.items():
        path = csv_dir / name
        if not path.exists():
            print(f"skip {name} (not found)")
            continue
        for fn in fns:
            fn(path, fig_dir)


if __name__ == "__main__":
    main()

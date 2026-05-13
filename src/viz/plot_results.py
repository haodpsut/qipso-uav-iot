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


def plot_scalability(csv_path: Path, out_dir: Path):
    rows = _read(csv_path)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["alg"]][int(r["K"])].append(float(r["energy_j"]))
    fig, ax = plt.subplots()
    for alg, Kd in by.items():
        Ks = sorted(Kd.keys())
        mu = [np.mean(Kd[k]) for k in Ks]
        sd = [np.std(Kd[k]) for k in Ks]
        ax.errorbar(Ks, mu, yerr=sd, label=alg.upper(),
                    color=color_for(alg), marker="o", capsize=2)
    ax.set_xlabel("Number of IoT nodes $K$")
    ax.set_ylabel("Mission energy (J)")
    ax.legend(frameon=False)
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
    rows = _read(csv_path)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["alg"]][int(r["M"])].append(float(r["energy_j"]))
    fig, ax = plt.subplots()
    for alg, Md in by.items():
        Ms = sorted(Md.keys())
        mu = [np.mean(Md[m]) for m in Ms]
        sd = [np.std(Md[m]) for m in Ms]
        ax.errorbar(Ms, mu, yerr=sd, label=alg.upper(),
                    color=color_for(alg), marker="o", capsize=2)
    ax.set_xlabel("Number of UAVs $M$")
    ax.set_ylabel("Total mission energy (J)")
    ax.legend(frameon=False)
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

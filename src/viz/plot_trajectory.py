"""Plot a 2D top-down trajectory: nodes (radius ~ data), labels by cluster
when multi-UAV, polyline path per UAV. Used for both single-UAV (fig_traj)
and multi-UAV (fig_traj_multi).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style, color_for, WONG_PALETTE


def _plot_one(ax, sample: dict, area: float = 1000.0):
    nodes = np.asarray(sample["nodes"])
    data = np.asarray(sample["data_required"])
    labels = np.asarray(sample.get("labels", [0] * len(nodes)))
    traj = np.asarray(sample["trajectory"])
    sizes = 20 + 80 * (data - data.min()) / (data.max() - data.min() + 1e-9)
    for cid in np.unique(labels):
        mask = labels == cid
        ax.scatter(nodes[mask, 0], nodes[mask, 1], s=sizes[mask],
                   color=WONG_PALETTE[int(cid) % len(WONG_PALETTE)],
                   edgecolor="black", linewidth=0.4, alpha=0.85,
                   label=f"Cluster {cid}" if labels.max() > 0 else "IoT nodes")
    if traj.ndim == 2:
        ax.plot(traj[:, 0], traj[:, 1], color=color_for(sample.get("alg", "qipso")),
                linewidth=1.8)
        ax.plot(traj[0, 0], traj[0, 1], "s", color="black", markersize=7, label="Start")
        ax.plot(traj[-1, 0], traj[-1, 1], "*", color="black", markersize=10, label="End")
    else:  # multi-UAV: (M, N+2, 2)
        for m in range(traj.shape[0]):
            ax.plot(traj[m, :, 0], traj[m, :, 1],
                    color=WONG_PALETTE[m % len(WONG_PALETTE)],
                    linewidth=1.8, label=f"UAV {m}")
    ax.set_xlim(0, area)
    ax.set_ylim(0, area)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str,
                        default="results/logs/multi_uav_trajectories.json")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    parser.add_argument("--area", type=float, default=1000.0)
    args = parser.parse_args()

    apply_style()
    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    by_M = {}
    for s in samples:
        by_M.setdefault(s["M"], []).append(s)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for M, group in by_M.items():
        ncol = len(group)
        fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.0), squeeze=False)
        for ax, sample in zip(axes[0], group):
            _plot_one(ax, sample, args.area)
            ax.set_title(sample["alg"].upper())
        if ncol > 1:
            axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
        path = out_dir / f"fig_traj_M{M}.pdf"
        fig.savefig(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

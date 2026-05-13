"""Publication-quality matplotlib defaults.

Vector PDF output, Times New Roman / serif fonts (closest to IEEE), gray
grids, color-blind-safe palette (Wong 2011).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

WONG_COLORS = {
    "qipso": "#0072B2",   # blue
    "pso":   "#D55E00",   # vermillion
    "ga":    "#009E73",   # green
    "de":    "#CC79A7",   # pink
    "greedy": "#E69F00",  # orange
    "straight": "#999999",
}
WONG_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                "#E69F00", "#56B4E9", "#F0E442", "#999999"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (4.0, 3.0),
    })


def color_for(alg: str) -> str:
    return WONG_COLORS.get(alg.lower(), "#000000")

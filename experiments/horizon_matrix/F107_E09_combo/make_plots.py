#!/usr/bin/env python3
"""E09 champion plots -> out/plots/*.png."""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

OUT = os.path.join(HERE, "out", "plots")
os.makedirs(OUT, exist_ok=True)

# Reference palette (dataviz skill), light mode.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.labelsize": 10,
})

daily = pd.read_csv(os.path.join(common.DATA, "daily.csv"),
                    index_col="date", parse_dates=True)
obs = daily["f107_obs"]


def load_pred(cell):
    return pd.read_csv(os.path.join(HERE, "..", cell, "out", "predictions.csv"),
                       parse_dates=["t_date", "target_date"])


def per_lead_rmse(cell):
    p = load_pred(cell)
    p["y"] = obs.reindex(p.target_date).to_numpy()
    return p.groupby("lead").apply(
        lambda g: float(np.sqrt(np.mean((g.pred_obs - g.y) ** 2))),
        include_groups=False,
    )

leads = np.arange(1, 31)

# ---- Plot 1: champion lineage, RMSE by lead ------------------------------
lineage = [
    ("E00 control", "F107_E00_control", ORANGE),
    ("E06 stack (old champion)", "F107_E06_stack", AQUA),
    ("E09 combo (champion)", "F107_E09_combo", BLUE),
]
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(leads, per_lead_rmse("F107_C00_persist"), color=MUTED, lw=1.4,
        ls=(0, (4, 3)), label="persistence baseline")
for name, cell, color in lineage:
    ax.plot(leads, per_lead_rmse(cell), color=color,
            lw=2.2 if cell.endswith("combo") else 1.6, label=name)
ax.set_xlim(1, 30); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("forecast lead (days)")
ax.set_ylabel("test RMSE (sfu), observed F10.7")
ax.set_title("Champion lineage — test 2024-01 → 2026-05 (cycle-25 max)")
ax.legend(loc="lower right", fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rmse_lineage.png"))

# ---- Plot 2: blend weights by lead ---------------------------------------
wlog = open(os.path.join(HERE, "out", "blend_weights.log")).read()
W = np.array([
    [float(x) for x in m.group(1).split(",")]
    for m in re.finditer(r"= \[([^\]]+)\]", wlog)
])
series = [
    ("LGBM 1996-span", BLUE), ("LGBM deep-span 1947", ORANGE),
    ("27-d recurrence", AQUA), ("81-d mean", YELLOW), ("persistence", MAGENTA),
]
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for i, (name, color) in enumerate(series):
    ax.plot(leads, W[:, i], color=color, lw=1.8, label=name)
# direct-label the four biggest enders; legend carries all five
ends = W[-1]
for i in np.argsort(ends)[-4:]:
    ax.annotate(series[i][0], xy=(30, ends[i]), xytext=(30.4, ends[i]),
                fontsize=8, color=INK2, va="center")
ax.set_xlim(1, 33.5); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_ylim(bottom=0)
ax.set_xlabel("forecast lead (days)")
ax.set_ylabel("OOF-fitted blend weight (nonneg)")
ax.set_title("Who the blend trusts, by lead — E09's per-lead weights")
ax.legend(loc="upper right", fontsize=8.5, labelcolor=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "blend_weights.png"))

# ---- Plot 3: what the forecast looks like (leads 3 / 14 / 27) ------------
p = load_pred("F107_E09_combo")
fig, axes = plt.subplots(3, 1, figsize=(9, 7), dpi=150, sharex=True,
                         gridspec_kw=dict(hspace=0.15))
for ax, lead in zip(axes, (3, 14, 27)):
    g = p[p.lead == lead].set_index("target_date").sort_index()
    ax.plot(obs.loc["2024-01":"2026-06"], color=INK2, lw=0.9,
            label="observed" if lead == 3 else None)
    ax.plot(g["pred_obs"], color=BLUE, lw=1.1,
            label=f"E09 forecast" if lead == 3 else None)
    ax.text(0.008, 0.92, f"lead {lead} d", transform=ax.transAxes,
            fontsize=9.5, color=INK, va="top", fontweight="bold")
    ax.set_ylabel("F10.7 (sfu)")
axes[0].set_title("E09 forecasts vs observed F10.7 over the test years")
axes[0].legend(loc="upper right", fontsize=9, labelcolor=INK2)
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "forecast_vs_truth.png"))

print("wrote", OUT)

#!/usr/bin/env python3
"""Diagnostic plots for the E10 far-side verdict -> out/plots/*.png."""

import os
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
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
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

truth = common.load_daily()["f107_adj"]  # only for index; truth col below
daily = pd.read_csv(os.path.join(common.DATA, "daily.csv"),
                    index_col="date", parse_dates=True)
obs = daily["f107_obs"]


def per_lead_rmse(cell):
    p = pd.read_csv(
        os.path.join(HERE, "..", cell, "out", "predictions.csv"),
        parse_dates=["t_date", "target_date"],
    )
    p["y"] = obs.reindex(p.target_date).to_numpy()
    return p.groupby("lead").apply(
        lambda g: float(np.sqrt(np.mean((g.pred_obs - g.y) ** 2))),
        include_groups=False,
    )

cells = {
    "E10 far-side": ("F107_E10_farside", BLUE),
    "E00 control": ("F107_E00_control", ORANGE),
    "E06 stack (champion)": ("F107_E06_stack", AQUA),
}
rmse = {k: per_lead_rmse(c) for k, (c, _) in cells.items()}
rmse["persistence"] = per_lead_rmse("F107_C00_persist")
leads = np.arange(1, 31)

# ---- Plot 1: RMSE by lead -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(leads, rmse["persistence"], color=MUTED, lw=1.4, ls=(0, (4, 3)),
        label="persistence baseline")
for name, (_, color) in cells.items():
    ax.plot(leads, rmse[name], color=color, lw=1.8, label=name)
ax.annotate("far-side features close most of\nthe control's gap to the stack",
            xy=(11, rmse["E10 far-side"][11]), xytext=(13.5, 24.5),
            fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlim(1, 30); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("forecast lead (days)")
ax.set_ylabel("test RMSE (sfu), observed F10.7")
ax.set_title("F10.7 forecast error by lead — test 2024-01 → 2026-05 (cycle-25 max)")
ax.legend(loc="lower right", fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rmse_by_lead.png"))

# ---- Plot 2: where the gain lives ----------------------------------------
delta = rmse["E00 control"] - rmse["E10 far-side"]  # >0 = E10 better
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
colors = [BLUE if d >= 0 else RED for d in delta]
ax.bar(leads, delta, width=0.72, color=colors, zorder=3)
ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
ax.axvspan(0.5, 14.5, color=BLUE, alpha=0.05, zorder=1)
ax.text(7.5, ax.get_ylim()[1] * 0.93, "L1–14: the return window\n(far side rotates on in ≤ ~13.6 d)",
        ha="center", va="top", fontsize=8.5, color=INK2)
ax.set_xlim(0.3, 30.7); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("forecast lead (days)")
ax.set_ylabel("RMSE improvement vs control (sfu)")
ax.set_title("Where the far-side features pay: E00 control − E10, by lead")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gain_by_lead.png"))

# ---- Plot 3: the far-side signal vs F10.7, 2010 -> now --------------------
fs = pd.read_csv(os.path.join(common.DATA, "farside_daily.csv"),
                 index_col="date", parse_dates=True)
lo = "2010-01-01"
fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), dpi=150, sharex=True,
                         gridspec_kw=dict(hspace=0.12))
a = axes[0]
a.plot(obs.loc[lo:], color=BLUE, lw=0.4, alpha=0.35)
a.plot(obs.loc[lo:].rolling(81, center=True).mean(), color=BLUE, lw=1.8)
a.set_ylim(0, 340)  # a handful of flare-day spikes (to ~900) run off-panel
a.set_ylabel("F10.7 observed (sfu)")
a.set_title("Earth-side activity …and what the far side saw (81-day smooth over daily)")
b = axes[1]
sig = fs["fs_flux_wsum"].loc[lo:]
sig = sig.clip(upper=sig.quantile(0.995))  # display clip: residual GONG glitch days
b.plot(sig, color=ORANGE, lw=0.4, alpha=0.35)
b.plot(sig.rolling(81, center=True, min_periods=41).mean(), color=ORANGE, lw=1.8)
b.set_ylabel("far-side flux, prob-weighted\n(clipped at p99.5)")
b.set_xlabel("")
for a_ in axes:
    a_.margins(x=0.01)
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "farside_vs_f107.png"))

print("wrote", OUT)

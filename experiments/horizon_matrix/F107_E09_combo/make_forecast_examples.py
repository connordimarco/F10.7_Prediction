#!/usr/bin/env python3
"""Three example 30-day E09 forecasts vs observed F10.7 (test years, so the
forecasts are fully out of sample) -> out/plots/forecast_examples.png."""

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

GOLD = "#eda100"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.labelsize": 9,
})

obs = pd.read_csv(os.path.join(common.DATA, "daily.csv"),
                  index_col="date", parse_dates=True)["f107_obs"]
# champion point forecast (E15) + calibrated conformal band (E14)
p = pd.read_csv(os.path.join(HERE, "..", "F107_E15_combo_tuned", "out", "predictions.csv"),
                parse_dates=["t_date", "target_date"])
rec = pd.read_csv(os.path.join(HERE, "..", "F107_C01_recur27", "out", "predictions.csv"),
                  parse_dates=["t_date", "target_date"])
qb = pd.read_csv(os.path.join(HERE, "..", "F107_E14_calbounds", "out", "bounds.csv"),
                 parse_dates=["t_date", "target_date"])
origins = pd.DatetimeIndex(sorted(p.t_date.unique()))

rng = np.random.default_rng(7)
picks = []
while len(picks) < 3:  # seeded-random origins, kept >=120 d apart
    cand = origins[rng.integers(len(origins))]
    if all(abs((cand - q).days) >= 120 for q in picks):
        picks.append(cand)
picks.sort()

fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), dpi=150,
                         gridspec_kw=dict(hspace=0.42))
GREEN = "#008300"
for ax, t0 in zip(axes, picks):
    g = p[p.t_date == t0].sort_values("lead")
    r = rec[rec.t_date == t0].sort_values("lead")
    ctx = obs.loc[t0 - pd.Timedelta(days=60): t0]
    tru = obs.loc[t0: t0 + pd.Timedelta(days=31)]
    y = obs.reindex(g.target_date).to_numpy()
    e_g = float(np.sqrt(np.nanmean((g.pred_obs.to_numpy() - y) ** 2)))
    e_r = float(np.sqrt(np.nanmean((r.pred_obs.to_numpy() - y) ** 2)))
    q = qb[qb.t_date == t0].sort_values("lead")
    ax.fill_between(q.target_date, q.p10_obs, q.p90_obs, color=GOLD,
                    alpha=0.18, linewidth=0, label="10–90% band")
    ax.plot(ctx.index, ctx, color=INK, lw=1.4)
    ax.plot(tru.index, tru, color=INK, lw=1.4, label="observation")
    ax.plot(r.target_date, r.pred_obs, color=GREEN, lw=1.3, alpha=0.9,
            label="27-day repeat")
    ax.plot(g.target_date, g.pred_obs, color=GOLD, lw=2.0,
            label="30-day forecast")
    ax.axvline(t0, color=AXIS, lw=0.9, ls=(0, (3, 3)))
    ymax = max(ctx.max(), tru.max(), g.pred_obs.max(), r.pred_obs.max())
    ax.text(t0 - pd.Timedelta(days=1), ymax, "forecast issued",
            ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_title(f"{t0:%B %d, %Y}", loc="left")
    ax.text(0.99, 1.02, f"30-day RMSE — forecast {e_g:.0f}, repeat {e_r:.0f} sfu",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=INK2)
    ax.set_ylabel("F10.7 (sfu)")
axes[0].legend(fontsize=8.5, labelcolor=INK2, loc="upper left")
fig.tight_layout()
out = os.path.join(HERE, "out", "plots", "forecast_examples.png")
fig.savefig(out)
print("wrote", out, "| origins:", ", ".join(str(t.date()) for t in picks))

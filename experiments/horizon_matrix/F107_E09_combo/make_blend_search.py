#!/usr/bin/env python3
"""How the blend weights are actually found (lead 14, real OOF data).

Uses the cached out-of-fold predictions (out/oof_lead14.json, built by
make_blend_explainer.py) -> out/plots/blend_search.png:
  left   the raw material: five honest predictions vs what happened
  right  error of a two-component mix as the weight slides
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

LEAD = 14
c = json.load(open(os.path.join(HERE, "out", f"oof_lead{LEAD}.json")))
orig = pd.DatetimeIndex(c["origins"])
y_adj = np.array(c["y_adj"])
Z = np.array(c["Z"])  # cols: lgbm96, lgbm47, recur, mean81, persist

DEEPGOLD, GOLD, GREEN, PINK, GRAY = "#c98500", "#eda100", "#008300", "#e87ba4", "#898781"
BLUE = "#2a78d6"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NAMES = ["ML — all data", "ML — long record", "27-d recurrence", "81-d mean", "persistence"]
COLORS = [GOLD, DEEPGOLD, GREEN, PINK, GRAY]

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

tdates = orig + pd.Timedelta(days=LEAD)
y_obs = np.asarray(common.adj_to_obs(y_adj, tdates))
comp_obs = np.stack([np.asarray(common.adj_to_obs(Z[:, i], tdates)) for i in range(5)])
rmse = lambda p: float(np.sqrt(np.mean((p - y_obs) ** 2)))

fit = LinearRegression(positive=True).fit(Z, y_adj)
w, b = fit.coef_, fit.intercept_

fig = plt.figure(figsize=(10.5, 4.4), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.24)
axA, axB = [fig.add_subplot(gs[0, i]) for i in range(2)]

# --- A: the raw material ---------------------------------------------------
win = (orig >= "2014-01-01") & (orig <= "2014-12-31")
t = tdates[win]
for i in (4, 2, 3, 1, 0):  # draw weakest first so the MLs sit on top
    axA.plot(t, comp_obs[i][win], color=COLORS[i], lw=0.9, alpha=0.85,
             label=NAMES[i])
axA.plot(t, y_obs[win], color=INK, lw=1.6, label="observation")
axA.set_title("Observed vs modeled F10.7", loc="left")
axA.set_ylabel(f"F10.7 at lead {LEAD} (sfu) — 2014 shown")
handles, labels = axA.get_legend_handles_labels()
axA.legend(handles[::-1], labels[::-1], fontsize=7, labelcolor=INK2,
           loc="upper right", ncols=2)
axA.tick_params(axis="x", labelrotation=25)

# --- B: RMSE over two of the five weights (others at fitted values) --------
g1 = np.linspace(0, 0.8, 81)   # weight on ML — all data
g2 = np.linspace(0, 0.8, 81)   # weight on 81-d mean
G1, G2 = np.meshgrid(g1, g2)
rest = Z[:, 1] * w[1] + Z[:, 2] * w[2] + Z[:, 4] * w[4] + b
dist2 = np.asarray(common.sun_earth_distance_au(tdates)) ** 2
E = np.empty_like(G1)
for j, w2 in enumerate(g2):
    preds = (Z[:, 0][None, :] * g1[:, None] + Z[:, 3][None, :] * w2 + rest[None, :])
    po = preds / dist2[None, :]
    E[j] = np.sqrt(np.mean((po - y_obs[None, :]) ** 2, axis=1))
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "seqblue", ["#0d366b", "#2a78d6", "#9ec5f4", "#cde2fb"])
vmax = 32.0  # focus the ramp on the valley; everything worse saturates
cf = axB.contourf(G1, G2, np.minimum(E, vmax),
                  levels=np.linspace(E.min(), vmax, 25), cmap=cmap, extend="max")
axB.contour(G1, G2, E, levels=np.arange(23.5, vmax, 1.0), colors=SURFACE,
            linewidths=0.6)
axB.plot([w[0]], [w[3]], marker="*", ms=14, color="#eda100",
         markeredgecolor=INK, markeredgewidth=0.7)
axB.annotate(f"fitted weights\n({w[0]:.2f}, {w[3]:.2f})", xy=(w[0], w[3]),
             xytext=(0.06, 0.62), fontsize=8, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
axB.grid(False)
axB.set_xlabel("weight on ML — all data")
axB.set_ylabel("weight on 81-day mean")
axB.set_title("RMSE over two of the five weights", loc="left")
cb = fig.colorbar(cf, ax=axB, shrink=0.85, pad=0.02)
cb.set_label("RMSE of the blend (sfu)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)

fig.tight_layout()
out = os.path.join(HERE, "out", "plots", "blend_search.png")
fig.savefig(out)
print("wrote", out)
print(f"NNLS weights {np.round(w, 2)}")

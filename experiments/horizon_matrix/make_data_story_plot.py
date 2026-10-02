#!/usr/bin/env python3
"""The data story: what each added data family buys, same model throughout.

Three identical LightGBM setups differing ONLY in inputs:
  F107_E12_f107ssn   f10.7 + sunspot number (the originally requested inputs)
  F107_E00_control   + sunspot group data (SRS active regions)
  F107_E10_farside   + far-side activity (GONG/HMI helioseismic catalogs)

-> plots/data_story.png (RMSE by lead + skill-vs-persistence by lead)
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shared"))
import common

OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
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

obs = pd.read_csv(os.path.join(common.DATA, "daily.csv"),
                  index_col="date", parse_dates=True)["f107_obs"]


def per_lead_rmse(cell):
    p = pd.read_csv(os.path.join(HERE, cell, "out", "predictions.csv"),
                    parse_dates=["t_date", "target_date"])
    p["y"] = obs.reindex(p.target_date).to_numpy()
    return p.groupby("lead").apply(
        lambda g: float(np.sqrt(np.mean((g.pred_obs - g.y) ** 2))),
        include_groups=False,
    )

series = [
    ("F10.7 + sunspot number", "F107_E12_f107ssn", BLUE, 1.7),
    ("+ sunspot group data", "F107_E00_control", ORANGE, 1.7),
    ("+ far-side activity", "F107_E10_farside", AQUA, 1.7),
    ("All Data + Blending", "F107_E15_combo_tuned", YELLOW, 2.4),
]
persist = per_lead_rmse("F107_C00_persist")
leads = np.arange(1, 31)

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(8, 10.8), dpi=150, sharex=True,
    gridspec_kw=dict(hspace=0.14, height_ratios=[1, 1, 0.75]),
)
for name, cell, color, lw in series:
    r = per_lead_rmse(cell)
    ax1.plot(leads, r, color=color, lw=lw, label=name)
    ax2.plot(leads, 1 - r / persist, color=color, lw=lw, label=name)
ax1.plot(leads, persist, color=MUTED, lw=1.3, ls=(0, (4, 3)),
         label="persistence (no model)")
ax1.set_ylabel("test RMSE (sfu)")
ax1.set_title("F10.7 Prediction")
ax1.legend(loc="lower right", fontsize=9, labelcolor=INK2)
ax2.axhline(0, color=AXIS, lw=0.9)
ax2.set_ylabel("skill vs persistence")
ax2.set_xlim(1, 30)
ax2.text(1.3, 0.005, "persistence = 0", ha="left", va="bottom",
         fontsize=8, color=MUTED)

# ---- panel 3: inside the winner — normalized blend weight share ----------
GREEN, GRAY = "#008300", "#898781"
wlog = open(os.path.join(HERE, "F107_E09_combo", "out",
                         "blend_weights.log")).read()
W = np.array([[float(x) for x in m.group(1).split(",")]
              for m in re.finditer(r"= \[([^\]]+)\]", wlog)])
bands = [  # the two LightGBMs as two steps of one gold family
    ("ML — long F10.7 +\nsunspot record", W[:, 1], "#c98500"),
    ("ML — all data\nsources", W[:, 0], YELLOW),
    ("81-day mean", W[:, 3], "#e87ba4"),
    ("27-day recurrence", W[:, 2], GREEN),
    ("persistence", W[:, 4], GRAY),
]
share = np.stack([b[1] for b in bands])
share = share / share.sum(axis=0)
ax3.stackplot(leads, share, colors=[b[2] for b in bands],
              edgecolor=SURFACE, linewidth=1.5)
ax3.grid(False)
ax3.set_ylim(0, 1)
ax3.set_ylabel("share of blend weight")
ax3.set_xlabel("forecast lead (days)")
ax3.set_xticks([1, 5, 10, 15, 20, 25, 30])
cum = np.cumsum(share, axis=0)


def band_mid(i, lead):
    j = lead - 1
    lo = cum[i - 1][j] if i else 0.0
    return (lo + cum[i][j]) / 2


labels = [  # (band, lead to center the label on, text color)
    (0, 4, SURFACE), (1, 13, SURFACE), (2, 20, SURFACE),
    (3, 13, SURFACE), (4, 24, INK),
]
for i, lead, color in labels:
    ax3.text(lead, band_mid(i, lead), bands[i][0], ha="center", va="center",
             fontsize=8.5, color=color,
             fontweight="bold" if i <= 1 else "normal")
fig.align_ylabels([ax1, ax2, ax3])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "data_story.png"))
print("wrote", os.path.join(OUT, "data_story.png"))

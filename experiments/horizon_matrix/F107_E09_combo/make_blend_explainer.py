#!/usr/bin/env python3
"""Four-panel walkthrough of how the blend weights are found.

Recomputes the real out-of-fold predictions for one lead (14 d) with the
exact E09 fold/seed setup, caches them (out/oof_lead14.json), and renders
out/plots/blend_explainer.png.
"""

import json
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

LEAD = 14
CACHE = os.path.join(HERE, "out", f"oof_lead{LEAD}.json")
OUT = os.path.join(HERE, "out", "plots")
os.makedirs(OUT, exist_ok=True)

DEEPGOLD, GOLD, GREEN, PINK, GRAY = "#c98500", "#eda100", "#008300", "#e87ba4", "#898781"
BLUE = "#2a78d6"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COMPONENTS = [  # (label, color) in E09 weight order [lgbm96, lgbm47, recur, mean81, persist]
    ("ML — all data sources", GOLD),
    ("ML — long F10.7 + sunspot record", DEEPGOLD),
    ("27-day recurrence", GREEN),
    ("81-day mean", PINK),
    ("persistence", GRAY),
]

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


def compute_oof():
    import lightgbm as lgb
    PARAMS = dict(objective="regression", n_estimators=3000, learning_rate=0.03,
                  num_leaves=63, min_child_samples=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    df = common.load_daily()
    adj = df["f107_adj"]
    f107a = adj.rolling(81, min_periods=60).mean()
    X96, y96, orig96, _ = common.build_samples(df, "train", extra=common.FS_COLS)
    X47, y47, orig47, _ = common.build_samples(
        df, "train47", required=["f107_adj", "ssn"], extra=common.FS_COLS)
    Xva, yva, _, _ = common.build_samples(df, "val", extra=common.FS_COLS)
    folds = np.array_split(np.arange(len(orig96)), 5)
    in47 = orig47.get_indexer(orig96)
    hc = LEAD - 1

    def fit(X, y):
        m = lgb.LGBMRegressor(random_state=11, **PARAMS)
        m.fit(X, y[:, hc], eval_set=[(Xva, yva[:, hc])], eval_metric="rmse",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        return m

    oof96, oof47 = np.empty(len(orig96)), np.empty(len(orig96))
    for k in folds:
        m96, m47mask = np.ones(len(orig96), bool), np.ones(len(orig47), bool)
        m96[k] = False
        m = fit(X96[m96], y96[m96])
        oof96[k] = m.predict(X96[k], num_iteration=m.best_iteration_)
        m47mask[in47[k]] = False
        m = fit(X47[m47mask], y47[m47mask])
        oof47[k] = m.predict(X47[in47[k]], num_iteration=m.best_iteration_)
        print(f"fold done ({len(k)} origins)", flush=True)

    lag = 27 if LEAD <= 27 else 54
    rec = np.array([adj.loc[t + np.timedelta64(LEAD - lag, "D")] for t in orig96])
    base = np.stack([rec, f107a.loc[orig96].to_numpy(), adj.loc[orig96].to_numpy()], axis=1)
    json.dump(dict(
        origins=[str(t.date()) for t in orig96],
        y_adj=y96[:, hc].tolist(),
        Z=np.column_stack([oof96, oof47, base]).tolist(),
        fold_ends=[str(orig96[k[-1]].date()) for k in folds],
        fold_starts=[str(orig96[k[0]].date()) for k in folds],
    ), open(CACHE, "w"))


if not os.path.exists(CACHE):
    compute_oof()
c = json.load(open(CACHE))
orig = pd.DatetimeIndex(c["origins"])
y_adj = np.array(c["y_adj"])
Z = np.array(c["Z"])  # cols: lgbm96, lgbm47, recur, mean81, persist

from sklearn.linear_model import LinearRegression
blend = LinearRegression(positive=True).fit(Z, y_adj)
w = blend.coef_

tdates = orig + pd.Timedelta(days=LEAD)
y_obs = np.asarray(common.adj_to_obs(y_adj, tdates))
comp_obs = np.stack([common.adj_to_obs(Z[:, i], tdates) for i in range(5)])
blend_obs = np.asarray(common.adj_to_obs(blend.predict(Z), tdates))
rmse = lambda p: float(np.sqrt(np.mean((p - y_obs) ** 2)))
comp_rmse = [rmse(comp_obs[i]) for i in range(5)]
blend_rmse = rmse(blend_obs)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), dpi=150)
(axA, axB), (axC, axD) = axes

# --- A: chronological folds ------------------------------------------------
starts = pd.DatetimeIndex(c["fold_starts"])
ends = pd.DatetimeIndex(c["fold_ends"])
for row in range(5):
    for k in range(5):
        x0, x1 = starts[k], ends[k]
        held = k == row
        axA.barh(4 - row, (x1 - x0).days, left=x0, height=0.62,
                 color=BLUE if held else GRID,
                 edgecolor=SURFACE, linewidth=1.2)
axA.set_yticks(range(5))
axA.set_yticklabels([f"pass {5 - r}" for r in range(5)], fontsize=8.5)
axA.grid(False)
axA.set_title("1 · Train on gray years, predict the blue block", loc="left")
axA.set_xlabel("every training day gets a prediction from a model that never saw it",
               fontsize=8.5)
axA.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2022-01-01"))

# --- B: honest error of each component ------------------------------------
order = np.argsort(comp_rmse)[::-1]
names = [COMPONENTS[i][0] for i in order] + ["weighted blend"]
vals = [comp_rmse[i] for i in order] + [blend_rmse]
cols = [COMPONENTS[i][1] for i in order] + [BLUE]
ypos = np.arange(len(vals))
axB.barh(ypos, vals, color=cols, height=0.62)
axB.set_yticks(ypos)
axB.set_yticklabels(names, fontsize=8.5)
for y, v in zip(ypos, vals):
    axB.text(v - 0.3, y, f"{v:.1f}", ha="right", va="center",
             fontsize=8, color=SURFACE, fontweight="bold")
axB.set_xlabel(f"out-of-fold RMSE at lead {LEAD} (sfu), 1996–2021")
axB.set_title("2 · Each component alone vs the blend", loc="left")
axB.grid(axis="y", visible=False)

# --- C: the fitted weights -------------------------------------------------
axC.barh(np.arange(5), [w[i] for i in order[::-1]][::-1], height=0.62,
         color=[COMPONENTS[i][1] for i in order])
axC.set_yticks(np.arange(5))
axC.set_yticklabels([COMPONENTS[i][0] for i in order], fontsize=8.5)
for y, i in enumerate(order):
    axC.text(max(w[i], 0.004) + 0.008, y, f"{w[i]:.2f}", va="center",
             fontsize=8, color=INK2)
axC.set_xlabel(f"fitted weight at lead {LEAD} (nonnegative least squares)")
axC.set_title("3 · Weights that minimize the blended error", loc="left")
axC.grid(axis="y", visible=False)

# --- D: repeat for all 30 leads -------------------------------------------
wlog = open(os.path.join(HERE, "out", "blend_weights.log")).read()
W = np.array([[float(x) for x in m.group(1).split(",")]
              for m in re.finditer(r"= \[([^\]]+)\]", wlog)])
leads = np.arange(1, 31)
for i, (name, color) in enumerate(COMPONENTS):
    axD.plot(leads, W[:, i], color=color, lw=1.7, label=name)
axD.axvline(LEAD, color=AXIS, lw=0.9, ls=(0, (3, 3)))
axD.text(LEAD + 0.4, 0.02, f"lead {LEAD} (panels 2–3)", fontsize=7.5,
         color=MUTED, va="bottom")
axD.set_xlim(1, 30)
axD.set_xticks([1, 5, 10, 15, 20, 25, 30])
axD.set_xlabel("forecast lead (days) — colors as in panels 2–3")
axD.set_ylabel("fitted weight")
axD.set_title("4 · The fit repeated for all 30 leads", loc="left")

fig.tight_layout(w_pad=2.5, h_pad=2.2)
fig.savefig(os.path.join(OUT, "blend_explainer.png"))
print("wrote", os.path.join(OUT, "blend_explainer.png"))
print(f"weights lead {LEAD}: {np.round(w, 2)}  blend {blend_rmse:.2f} vs best single {min(comp_rmse):.2f}")

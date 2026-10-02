#!/usr/bin/env python3
"""RMSE vs days ahead: our forecast vs SWPC's operational 27-day outlook on
the matched set (origin = issue date - 1, same target days).

Usage: plot_swpc_by_lead.py <model_dir> [--ens <ensemble_cell>]
Reads data/swpc_prf/swpc_27day.csv (from swpc_27day.py). Writes
plots/swpc_vs_ours_by_lead.png and prints the per-lead table.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import common

cell = sys.argv[1]
ens = sys.argv[sys.argv.index("--ens") + 1] if "--ens" in sys.argv else None

df = common.load_daily()
orig = common.origins("test", df)
truth = df["f107_obs"]
f107a = truth.rolling(81, min_periods=60).mean()
sw = pd.read_csv(os.path.join(common.DATA, "swpc_prf", "swpc_27day.csv"), parse_dates=["issue_date", "target_date"])
sw["t_date"] = sw.issue_date - pd.Timedelta(days=1)
sw["lead"] = (sw.target_date - sw.t_date).dt.days
sw = sw[sw.t_date.isin(orig)].copy()
sw["y"] = truth.reindex(sw.target_date).to_numpy()
sw = sw.dropna(subset=["y"]).sort_values(["t_date", "lead"])
sw["persist"] = truth.reindex(sw.t_date).to_numpy()
sw["mean81"] = f107a.reindex(sw.t_date).to_numpy()
lag = np.where(sw.lead <= 27, 27, 54)
sw["recur27"] = truth.reindex(sw.t_date + pd.to_timedelta(sw.lead - lag, unit="D")).to_numpy()


def cell_pred(c):
    p = pd.read_csv(os.path.join(c, "out", "predictions.csv"), parse_dates=["t_date", "target_date"])
    return p.set_index(["t_date", "target_date"]).pred_obs.reindex(pd.MultiIndex.from_frame(sw[["t_date", "target_date"]])).to_numpy()


sw["ours"] = cell_pred(cell)
series = {"ours": "This forecast", "f107_swpc": "SWPC 27-day outlook", "persist": "Persistence",
          "recur27": "27-day recurrence", "mean81": "81-day mean"}
if ens:
    e = pd.read_csv(os.path.join(ens, "out", "ensemble.csv"), parse_dates=["t_date", "target_date"])
    e = e.set_index(["t_date", "target_date"]).reindex(pd.MultiIndex.from_frame(sw[["t_date", "target_date"]]))
    sw["q10"], sw["q90"] = e.q10.to_numpy(), e.q90.to_numpy()

hi = sw.y >= 150
leads = np.arange(1, 28)


def rmse_by_lead(col, mask=None):
    m = np.ones(len(sw), bool) if mask is None else mask.to_numpy()
    return np.array([np.sqrt(np.mean((sw[col] - sw.y)[m & (sw.lead == h).to_numpy()] ** 2)) for h in leads])


R = {k: rmse_by_lead(k) for k in series}
Rhi = {k: rmse_by_lead(k, hi) for k in ("ours", "f107_swpc")}
tab = pd.DataFrame({"SWPC": R["f107_swpc"], "ours": R["ours"], "persist": R["persist"], "recur27": R["recur27"],
                    "mean81": R["mean81"], "ours/SWPC": R["ours"] / R["f107_swpc"],
                    "SWPC hi-act": Rhi["f107_swpc"], "ours hi-act": Rhi["ours"]}, index=pd.Index(leads, name="lead"))
pd.set_option("display.width", 200)
print(f"matched set: {sw.t_date.nunique()} outlook issues, {len(sw)} target days, {int(hi.sum())} with truth >= 150")
print(tab.round(2).to_string())

# --- figure -------------------------------------------------------------
BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titlecolor": INK, "axes.labelsize": 9,
})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160, gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
refs = [("persist", (0, (1, 2)), "Persistence"), ("recur27", (0, (4, 2)), "27-day recurrence"), ("mean81", "solid", "81-day mean")]
for k, ls, lab in refs:
    ax.plot(leads, R[k], color=MUTED, lw=1.2, ls=ls, label=lab)
ax.plot(leads, R["f107_swpc"], color=ORANGE, lw=2, label="SWPC 27-day outlook")
ax.plot(leads, R["ours"], color=BLUE, lw=2, label="This forecast")
ax.text(27.4, R["f107_swpc"][-1], "SWPC", color=INK2, fontsize=8.5, va="center")
ax.text(27.4, R["ours"][-1], "ours", color=INK2, fontsize=8.5, va="center")
ax.set_xlim(0.5, 30); ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
ax.set_xlabel("days ahead"); ax.set_ylabel("RMSE (sfu), observed F10.7")
ax.set_title(f"RMSE vs lead — {sw.t_date.nunique()} matched SWPC issues, {sw.t_date.min():%b %Y}–{sw.t_date.max():%b %Y}", loc="left")
ax.legend(fontsize=8, labelcolor=INK2, loc="lower right", ncol=1)
ax2.axhline(0, color=AXIS, lw=0.8)
ax2.plot(leads, 100 * (1 - R["ours"] / R["f107_swpc"]), color=BLUE, lw=2, label="all days")
ax2.plot(leads, 100 * (1 - Rhi["ours"] / Rhi["f107_swpc"]), color=BLUE, lw=1.6, ls=(0, (4, 2)), label="days with F10.7 ≥ 150")
ax2.set_xlim(0.5, 28); ax2.set_xticks([1, 5, 10, 15, 20, 25, 27])
ax2.set_xlabel("days ahead"); ax2.set_ylabel("RMSE reduction vs SWPC (%)")
ax2.set_title("Improvement over the SWPC outlook", loc="left")
ax2.legend(fontsize=8, labelcolor=INK2, loc="upper right")
fig.text(0.01, 0.01, "Both forecasts know data through the day before the SWPC issue date; SWPC's day-1 value is its issue-day entry.",
         fontsize=7.5, color=MUTED)
out = os.path.join(HERE, "swpc_vs_ours_by_lead.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", os.path.normpath(out))

#!/usr/bin/env python3
"""E51: geometry-informed linear model. Per lead, ordinary least squares on
[flux(t), env81(t), recur27(t+h), S_0(t), S_h(t)] where S_h is the E50 disk
forward series (Earth-side + far-side regions rotated h days ahead).
Five coefficients + intercept per lead, fitted on train origins. Answers:
how much of the flux does the explicit geometry explain with (almost) no
learner? Requires F107_E50_diskfwd/out/S.parquet."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

FLUX = "f107_adj_rob"
H = common.HORIZON

df = common.load_daily()
flux, env = df[FLUX], common.envelope(df, FLUX)
S = pd.read_parquet(os.path.join(HERE, "..", "F107_E50_diskfwd", "out", "S.parquet"))
otr, ote = common.origins("train", df), common.origins("test", df)
tgt = flux.to_numpy()
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)


def feats(orig, h):
    lag = 27 if h <= 27 else 54
    rec = np.array([flux.loc[t + np.timedelta64(h - lag, "D")] for t in orig])
    return np.column_stack([flux.loc[orig], env.loc[orig], rec, S.loc[orig, "S0"], S.loc[orig, f"S{h}"]])


pred = np.empty((len(ote), H))
for h in range(1, H + 1):
    ytr = tgt[df.index.get_indexer(otr) + h]
    Xtr, Xte = feats(otr, h), feats(ote, h)
    m = LinearRegression().fit(Xtr, ytr)
    pred[:, h - 1] = m.predict(Xte)
    pd.DataFrame({"t_date": ote.date, "pred_adj": np.round(pred[:, h - 1], 4)}).to_csv(
        os.path.join(HERE, "out", f"lead_{h:02d}.csv"), index=False)
    pd.DataFrame({"t_date": otr.date, "oof_adj": np.round(m.predict(Xtr), 4), "y_adj": np.round(ytr, 4)}).to_csv(
        os.path.join(HERE, "out", f"oof_{h:02d}.csv"), index=False)
    if h in (1, 7, 14, 27):
        print(f"lead {h:2d}: coef[flux env recur S0 Sh] = " + " ".join(f"{c:.3f}" for c in m.coef_) + f"  b={m.intercept_:.1f}")
common.write_predictions(HERE, ote, pred)

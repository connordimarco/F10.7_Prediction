#!/usr/bin/env python3
"""Calibrate the E24 uncertainty band -> models/band.json.

Residuals are log(observed / forecast) per lead over every origin from
2024-01-01 (the untouched test era, cycle-25 maximum) whose 30-day truth is
complete. Their quantiles (q05..q95) are stored per lead and applied
multiplicatively at predict time. Note: in-sample on this era, no model
selection happened here. Re-run after each data refresh to extend the record.
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

import e24


def main():
    df = e24.load_table()
    models, meta = e24.load_models()
    orig = e24.valid_origins(df, "2024-01-01", "2099-12-31", need_truth=True)
    X = e24.feature_rows(df, orig)
    env = e24.envelope(df).loc[orig].to_numpy()
    P = e24.to_obs(e24.predict_adj(models, X, env), orig)
    obs = df["f107_obs"].to_numpy()
    pos = df.index.get_indexer(orig)
    Y = np.stack([obs[p + 1 : p + 1 + e24.HORIZON] for p in pos])
    R = np.log(Y / P)
    band = {str(h + 1): {f"q{q:02d}": round(float(np.percentile(R[:, h], q)), 5) for q in e24.QS} for h in range(e24.HORIZON)}
    rmse = float(np.sqrt(np.mean((Y - P) ** 2)))
    out = dict(kind="log-ratio quantiles of observed/forecast, per lead", n_origins=int(len(orig)),
               origins=[str(orig[0].date()), str(orig[-1].date())], pooled_rmse_sfu=round(rmse, 2),
               calibrated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), band=band)
    json.dump(out, open(os.path.join(e24.MODELS, "band.json"), "w"), indent=1)
    w = [np.exp(band[str(h)]["q90"]) - np.exp(band[str(h)]["q10"]) for h in (1, 7, 14, 30)]
    print(f"band from {len(orig)} origins {orig[0].date()}..{orig[-1].date()}; pooled RMSE {rmse:.2f} sfu; "
          f"10-90 band width as a fraction of the forecast at leads 1/7/14/30: {np.round(w, 2).tolist()}")


if __name__ == "__main__":
    main()

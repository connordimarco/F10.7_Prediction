#!/usr/bin/env python3
"""Website views -> deploy/web/ (published by deploy/f107_daily.sh).

  forecast.json   observed daily F10.7 (last 183 days, measured days only)
                  + the newest issued forecast (point forecast and q05..q95)
  hindcast.json   the forecast the frozen model gives from every origin since
                  2024-01-01 (data it was never trained or tuned on), computed
                  from today's data archive, + the per-lead band and the
                  observed series, so the page can draw any start date
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "realtime"))
import e24  # noqa: E402

OUT = os.path.join(ROOT, "deploy", "web")
DAYS = 183  # trailing ~6 months; the page pans through it
HINDCAST_START = "2024-01-01"


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(OUT, name))


def observed(since):
    d = pd.read_csv(os.path.join(ROOT, "data", "daily.csv"), parse_dates=["date"])
    d = d[(d.f107_filled == 0) & d.f107_obs.notna() & (d.date >= since)]
    return {"t": d.date.dt.strftime("%Y-%m-%d").tolist(), "f107": [round(float(v), 1) for v in d.f107_obs]}


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(OUT, exist_ok=True)

    fc = pd.read_csv(os.path.join(ROOT, "realtime", "forecasts", "latest.csv"))
    since = (pd.Timestamp(fc.origin[0]) - pd.Timedelta(days=DAYS - 1)).strftime("%Y-%m-%d")
    write("forecast.json", {
        "generated_utc": now,
        "origin": str(fc.origin[0]),
        "issued_at": str(fc.issued_at[0]),
        "model": str(fc.model[0]),
        "observed": observed(since),
        "forecast": {"t": fc.target_date.astype(str).tolist(),
                     **{k: fc[k].round(1).tolist()
                        for k in ["f107", "q05", "q10", "q25", "q50", "q75", "q90", "q95"]}},
    })

    df = e24.load_table()
    models, meta = e24.load_models()
    orig = e24.valid_origins(df, HINDCAST_START, "2099-12-31", need_truth=False)
    X = e24.feature_rows(df, orig)
    env = e24.envelope(df).loc[orig].to_numpy()
    P = e24.to_obs(e24.predict_adj(models, X, env), orig)
    band = json.load(open(os.path.join(e24.MODELS, "band.json")))["band"]
    write("hindcast.json", {
        "generated_utc": now,
        "model": meta["model"],
        "origins": [t.strftime("%Y-%m-%d") for t in orig],
        "f107": np.round(P, 1).tolist(),
        "band": [band[str(h)] for h in range(1, e24.HORIZON + 1)],
        "observed": observed((orig[0] - pd.Timedelta(days=60)).strftime("%Y-%m-%d")),
    })
    print(f"web views: forecast origin {fc.origin[0]}; hindcast {len(orig)} origins "
          f"{orig[0].date()}..{orig[-1].date()}")


if __name__ == "__main__":
    main()

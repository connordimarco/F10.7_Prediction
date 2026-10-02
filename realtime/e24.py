"""E24 as a frozen realtime model: shared pieces for train / calibrate / predict.

E24 (experiments/horizon_matrix/F107_E24_ratio_deep) is one LightGBM per lead
(1..30 days) with cfg-44 params, trained on origins 1947-2021 (val 2022-2023
for early stopping), features = 60-day windows of 26 daily series
(flare-robust adjusted F10.7, SSN, 10 SRS active-region aggregates, 14 far-side
aggregates; the AR/far-side families are optional-NaN), and target
    y(t+h) = f107_adj_rob(t+h) / env81(t),   env81 = trailing 81-day mean,
multiplied back by env81(t) and converted to observed flux at predict time.
Everything here reuses the matrix's shared code so the live feature row is
built exactly as the training rows were.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "horizon_matrix", "shared"))
import common  # noqa: E402
from single_model import CFG44  # noqa: E402

MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "out")
FORECASTS = os.path.join(HERE, "forecasts")
FLUX = "f107_adj_rob"
REQUIRED = ["f107_adj", "ssn"]  # windows that must be gap-free (train47 rule)
COLS = [FLUX if c == "f107_adj" else c for c in common.FEATURE_COLS] + list(common.FS_COLS)
NAMES = [f"{c}_lag{lag}" for c in COLS for lag in range(common.HIST - 1, -1, -1)]
HIST, HORIZON = common.HIST, common.HORIZON
QS = [5, 10, 25, 50, 75, 90, 95]


def load_table():
    return common.load_daily()


def envelope(df):
    return common.envelope(df, FLUX)


def valid_origins(df, lo, hi, need_truth):
    """Origins in [lo, hi] whose 60-day REQUIRED windows are gap-free (and,
    if need_truth, whose 30-day target window is complete)."""
    x_ok = df[REQUIRED].notna().all(axis=1).rolling(HIST).sum() == HIST
    ok = x_ok & envelope(df).notna()
    if need_truth:
        ok &= df["f107_adj"].notna().rolling(HORIZON).sum().shift(-HORIZON) == HORIZON
    return df.index[(df.index >= lo) & (df.index <= hi) & ok]


def feature_rows(df, orig):
    vals = df[COLS].to_numpy()
    pos = df.index.get_indexer(orig)
    assert (pos >= HIST - 1).all()
    return np.stack([vals[p - HIST + 1 : p + 1].T.ravel() for p in pos])


def targets(df, orig):
    tgt = df[FLUX].to_numpy()
    pos = df.index.get_indexer(orig)
    return np.stack([tgt[p + 1 : p + 1 + HORIZON] for p in pos])


def load_models():
    import joblib

    meta = json.load(open(os.path.join(MODELS, "meta.json")))
    assert meta["feature_names"] == NAMES, "feature layout changed since training"
    return [joblib.load(os.path.join(MODELS, f"lead_{h:02d}.joblib")) for h in range(1, HORIZON + 1)], meta


def predict_adj(models, X, env):
    """-> (n, 30) adjusted-flux predictions for feature rows X with envelope env (n,)."""
    P = np.stack([m.predict(X, num_iteration=m.best_iteration_) for m in models], axis=1)
    return P * np.asarray(env)[:, None]


def to_obs(P_adj, orig):
    """Adjusted -> observed per target date; returns (n, 30)."""
    out = np.empty_like(P_adj)
    for i, t in enumerate(orig):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=HORIZON)
        out[i] = common.adj_to_obs(P_adj[i], tdates)
    return out


def long_frame(orig, P_obs, col="pred_obs"):
    rows = []
    for i, t in enumerate(orig):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=HORIZON)
        for h in range(HORIZON):
            rows.append((t.date(), h + 1, tdates[h].date(), round(float(P_obs[i, h]), 2)))
    return pd.DataFrame(rows, columns=["t_date", "lead", "target_date", col])

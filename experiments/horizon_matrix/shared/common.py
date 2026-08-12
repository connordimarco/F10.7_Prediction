"""Shared data loading, windowing, and conventions for the horizon matrix.

Conventions (decided 2026-08-12):
  - Models work in ADJUSTED (1-AU) F10.7 space — the deterministic Earth–Sun
    distance annual cycle (~±3.3%) is removed before learning and restored
    analytically at predict time. Scoring is on OBSERVED F10.7, the
    operational quantity.
  - Input window HIST=60 days (t-59..t), forecast horizon HORIZON=30 days
    (t+1..t+30).
  - Splits by forecast origin date t:
      train 1996-01-01..2021-12-31 · val 2022-01-01..2023-12-31 (early
      stopping / tuning only) · test 2024-01-01.. (untouched by tuning).
    Test covers the cycle-25 maximum — deliberately hard.
  - The canonical origin set per split is test_origins()/etc: origins whose
    full input window AND full target window are NaN-free. Every cell
    predicts exactly the canonical test set so n is identical across cells.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data"))

HIST = 60
HORIZON = 30
FEATURE_COLS = [
    "f107_adj",
    "ssn",
    "ar_area",
    "ar_area_proj",
    "ar_area_east",
    "ar_area_west",
    "ar_area_max",
    "ar_nreg",
    "ar_nspots",
    "ar_ncomplex",
    "ar_nplage",
    "ar_nreturn",
]
SW_COLS = [
    "sw_Bx_med",
    "sw_By_med",
    "sw_Bz_med",
    "sw_Ux_med",
    "sw_rho_med",
    "sw_T_med",
    "sw_Bz_min",
    "sw_Ux_min",
]
SPLITS = {
    "train": ("1996-01-01", "2021-12-31"),
    "train47": ("1947-01-01", "2021-12-31"),  # full F10.7 record; ar_* NaN pre-1996
    "train05": ("2005-01-01", "2021-12-31"),  # MIDL-constrained span
    "val": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2099-12-31"),
}


def load_daily():
    """Base daily table joined with solar wind (sw_* columns NaN pre-2005)."""
    df = pd.read_csv(os.path.join(DATA, "daily.csv"), index_col="date", parse_dates=True)
    assert df.index.freq or (df.index[1:] - df.index[:-1]).max() == pd.Timedelta(days=1)
    sw_path = os.path.join(DATA, "sw_daily.csv")
    if os.path.exists(sw_path):
        sw = pd.read_csv(sw_path, index_col="date", parse_dates=True)
        df = df.join(sw[SW_COLS])
    return df


def sun_earth_distance_au(dates):
    """Earth–Sun distance (AU), ~0.01% accurate — plenty for the ±3.3% cycle."""
    n = (pd.DatetimeIndex(dates) - pd.Timestamp("2000-01-01 12:00")).total_seconds() / 86400.0
    g = np.deg2rad(357.529 + 0.98560028 * n)
    return 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)


def adj_to_obs(f_adj, dates):
    return np.asarray(f_adj) / sun_earth_distance_au(dates) ** 2


def _valid_origins(df, split, cols):
    lo, hi = SPLITS[split]
    x_ok = df[cols].notna().all(axis=1).rolling(HIST).sum() == HIST
    y_ok = (
        df["f107_adj"].notna().rolling(HORIZON).sum().shift(-HORIZON) == HORIZON
    )
    ok = x_ok & y_ok
    return df.index[(df.index >= lo) & (df.index <= hi) & ok]


def origins(split, df=None, sw=False, required=None):
    """Valid forecast origins. The TEST set is canonical: it always requires
    valid solar-wind windows too (when sw_daily.csv exists), so every cell —
    with or without SW features — predicts the identical origin set.

    sw: False = SW not required; True = require valid SW windows;
    "optional" = SW columns go into X but NaN windows are allowed
    (LightGBM handles missing natively) — origins as if sw=False.
    required: override the columns whose windows must be NaN-free (train/val
    only — e.g. ["f107_adj", "ssn"] for the 1947 span where ar_* is NaN
    pre-1996). The test split ignores it: the canonical set stands."""
    df = load_daily() if df is None else df
    cols = list(FEATURE_COLS if (required is None or split == "test") else required)
    if (split == "test" and SW_COLS[0] in df.columns) or sw is True:
        cols += SW_COLS
    return _valid_origins(df, split, cols)


def build_samples(df, split, sw=False, required=None):
    """-> X (n, HIST*ncols), y_adj (n, HORIZON), origin dates, feature names.

    X columns are (FEATURE_COLS [+ SW_COLS]) x lags 59..0 (oldest first);
    y is t+1..t+30. `required` relaxes which columns gate origin validity
    (see origins); X always carries the full column set.
    """
    cols = FEATURE_COLS + SW_COLS if sw else FEATURE_COLS
    orig = origins(split, df, sw=sw, required=required)
    pos = df.index.get_indexer(orig)
    vals = df[cols].to_numpy()
    tgt = df["f107_adj"].to_numpy()
    X = np.stack([vals[p - HIST + 1 : p + 1].T.ravel() for p in pos])
    y = np.stack([tgt[p + 1 : p + 1 + HORIZON] for p in pos])
    names = [f"{c}_lag{lag}" for c in cols for lag in range(HIST - 1, -1, -1)]
    return X, y, orig, names


def write_predictions(cell_dir, orig, pred_adj):
    """pred_adj (n, HORIZON) in adjusted space -> out/predictions.csv in
    OBSERVED space (the cell contract: t_date, lead, target_date, pred_obs)."""
    rows = []
    for i, t in enumerate(orig):
        tdates = pd.date_range(t + pd.Timedelta(days=1), periods=HORIZON)
        obs = adj_to_obs(pred_adj[i], tdates)
        for h in range(HORIZON):
            rows.append((t.date(), h + 1, tdates[h].date(), round(float(obs[h]), 2)))
    out = pd.DataFrame(rows, columns=["t_date", "lead", "target_date", "pred_obs"])
    os.makedirs(os.path.join(cell_dir, "out"), exist_ok=True)
    path = os.path.join(cell_dir, "out", "predictions.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}: {len(orig)} origins x {HORIZON} leads")

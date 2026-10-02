#!/usr/bin/env python3
"""E50: disk forward model — explicit solar-rotation geometry, no learner.

Every SRS region on the origin day (heliographic lat/lon, area) and every
GONG far-side detection (Carrington lon/lat, effective area x probability,
2010->) is rotated forward day by day at the latitude-dependent synodic
rate, decayed with an e-folding time tau, and weighted by a
center-to-limb function g(mu) = a + (1-a) mu^p (mu = cos(lat) cos(lon),
zero behind the limb). The disk-integrated proxy
    S_h(t) = sum_earth A^q e^{-h/tau} g(mu_h) + s_fs sum_farside A_fs^q e^{-h/tau} g(mu_h)
gives the *change* in region-driven flux from the origin, and the forecast
is persistence plus that change:
    pred(t+h) = flux_rob(t) + k_h [S_h(t) - S_0(t)].
(p, a, q, tau, s_fs) come from a small grid and k_h from least squares,
both fitted on the 1996-2021 train origins (35 numbers in total; the test
years are untouched). Outputs the standard predictions.csv, the
lead/oof files for the stack (train rows are in-sample for the 35
parameters), and out/S.parquet (S_h for every day) for feature use.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

FLUX = "f107_adj_rob"
H = common.HORIZON
LEADS = np.arange(H + 1)  # 0..30, S_0 is the origin-day geometry
OMEGA0, OMEGA2 = 13.39, 2.8  # synodic deg/day: omega = 13.39 - 2.8 sin^2(lat)
GRID = dict(p=[0.5, 1.0, 1.5, 2.0], a=[0.0, 0.15, 0.3], q=[0.5, 0.75, 1.0],
            tau=[10.0, 20.0, 40.0, 1e9], s_fs=[0.0, 1.0, 3.0, 10.0])


def load_srs():
    srs = pd.read_csv(os.path.join(common.DATA, "srs_parsed.csv"), parse_dates=["valid_date"])
    one = srs[srs.section == "I"].dropna(subset=["lat", "lon"]).copy()
    one["area"] = one["area"].fillna(0).clip(lower=5.0)  # tiny regions still radiate
    # central-meridian Carrington longitude from the regions themselves:
    # Carrington longitude increases westward, SRS lon is east-positive
    one["L0"] = (one["carrington_lo"] + one["lon"]) % 360
    have = set(pd.read_csv(os.path.join(common.DATA, "daily.csv"), index_col="date", parse_dates=True)
               .query("srs_present == 1").index)
    return one, have


def l0_series(one, idx):
    """Central-meridian Carrington longitude per day: median from SRS regions,
    gaps filled by rotating the nearest known value at 13.199 deg/day."""
    ang = one.groupby("valid_date")["L0"].apply(
        lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(), np.cos(np.radians(s)).mean())) % 360)
    known = ang.reindex(idx)
    # unwrap onto a straight line, fill by the mean rotation rate
    t = np.arange(len(idx))
    k = known.notna().to_numpy()
    base = (known[k].to_numpy() + 13.199 * t[k]) % 360  # should be ~constant
    c = np.degrees(np.arctan2(np.sin(np.radians(base)).mean(), np.cos(np.radians(base)).mean()))
    filled = (c - 13.199 * t) % 360
    out = known.to_numpy().copy()
    out[~k] = filled[~k]
    return pd.Series(out, index=idx)


def load_farside():
    rows = []
    for p in sorted(glob.glob(os.path.join(common.DATA, "farside", "f6x", "mrf6x*.txt"))):
        name = os.path.basename(p)
        day = pd.Timestamp("20" + name[5:11])
        hh = name[12:16]
        try:
            with open(p) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            continue
        for ln in lines[1:]:
            q = ln.split()
            if len(q) < 9:
                continue
            try:
                rows.append((day, hh, float(q[0]), float(q[1]), float(q[4]) / 100.0, min(float(q[5]), 5e3)))
            except ValueError:
                continue
    fs = pd.DataFrame(rows, columns=["day", "hh", "clon", "lat", "prob", "area"])
    latest = fs.groupby("day")["hh"].transform("max")
    return fs[fs.hh == latest].drop(columns="hh")


def region_arrays(days, one, have, fs, L0):
    """Padded per-day arrays -> (A, lat, lon, is_fs) each (n_days, max_reg)."""
    by_day = {d: g for d, g in one.groupby("valid_date")}
    fs_by_day = {d: g for d, g in fs.groupby("day")} if len(fs) else {}
    srs_days = np.array(sorted(by_day))
    recs = []
    for d in days:
        # Earth side: latest SRS day <= d, longitudes advanced by the gap
        i = np.searchsorted(srs_days, np.datetime64(d), side="right") - 1
        A, la, lo, isf = [], [], [], []
        if i >= 0 and (d - pd.Timestamp(srs_days[i])).days <= 5:
            g = by_day[pd.Timestamp(srs_days[i])]
            gap = (d - pd.Timestamp(srs_days[i])).days
            for r in g.itertuples():
                om = OMEGA0 - OMEGA2 * np.sin(np.radians(r.lat)) ** 2
                A.append(r.area); la.append(r.lat); lo.append(r.lon - om * gap); isf.append(0.0)
        if d in fs_by_day:
            for r in fs_by_day[d].itertuples():
                lon_e = (r.clon - L0.loc[d] + 180.0) % 360 - 180.0  # east-positive CMD longitude
                A.append(r.area * r.prob); la.append(r.lat); lo.append(lon_e); isf.append(1.0)
        recs.append((A, la, lo, isf))
    m = max(1, max(len(r[0]) for r in recs))
    out = [np.zeros((len(days), m)) for _ in range(4)]
    for j, (A, la, lo, isf) in enumerate(recs):
        n = len(A)
        if n:
            out[0][j, :n], out[1][j, :n], out[2][j, :n], out[3][j, :n] = A, la, lo, isf
    return out


def main():
    df = common.load_daily()
    flux = df[FLUX]
    one, have = load_srs()
    days = pd.DatetimeIndex(df.index[df.index >= "1996-01-01"])
    L0 = l0_series(one, days)
    fs = load_farside()
    A, lat, lon, isf = region_arrays(days, one, have, fs, L0)
    print(f"{len(days)} days, max {A.shape[1]} regions/day; far-side rows {len(fs)}; "
          f"L0 check: SRS-derived spread within day (deg, median) = "
          f"{one.groupby('valid_date')['L0'].apply(lambda s: (s.max() - s.min()) if len(s) > 1 else np.nan).median():.1f}")
    # geometry for all leads, independent of the fitted parameters
    om = OMEGA0 - OMEGA2 * np.sin(np.radians(lat)) ** 2                  # (n, m)
    lon_h = lon[:, :, None] - om[:, :, None] * LEADS[None, None, :]      # (n, m, 31)
    mu = np.cos(np.radians(lat))[:, :, None] * np.cos(np.radians(lon_h))
    mu = np.where(np.abs(((lon_h + 180) % 360) - 180) < 90, np.clip(mu, 0, None), 0.0)
    present = A > 0

    def S_of(p, a, q, tau, s_fs):
        g = a + (1 - a) * mu**p
        g = np.where(mu > 0, g, 0.0)
        w = (A**q) * present * np.where(isf > 0, s_fs, 1.0)                 # (n, m)
        dec = np.exp(-LEADS / tau)[None, None, :]
        return np.einsum("nm,nmh->nh", w, g * dec)                          # (n, 31)

    # fit on train origins: pred = flux(t) + k_h (S_h - S_0)
    otr = common.origins("train", df)
    ote = common.origins("test", df)
    pos_tr = days.get_indexer(otr); pos_te = days.get_indexer(ote)
    assert (pos_tr >= 0).all() and (pos_te >= 0).all()
    tgt = flux.to_numpy()
    dpos = df.index.get_indexer(otr)
    Ytr = np.stack([tgt[dpos + h] for h in range(1, H + 1)], axis=1) - flux.loc[otr].to_numpy()[:, None]
    best = None
    for p in GRID["p"]:
        for a in GRID["a"]:
            for q in GRID["q"]:
                for tau in GRID["tau"]:
                    for s_fs in GRID["s_fs"]:
                        S = S_of(p, a, q, tau, s_fs)
                        D = (S[:, 1:] - S[:, :1])[pos_tr]                   # (n_tr, 30)
                        k = (D * Ytr).sum(0) / np.maximum((D * D).sum(0), 1e-9)
                        rmse = np.sqrt(np.mean((D * k - Ytr) ** 2))
                        if best is None or rmse < best[0]:
                            best = (rmse, dict(p=p, a=a, q=q, tau=tau, s_fs=s_fs), k, S)
    rmse, par, k, S = best
    print(f"best train RMSE (anomaly, adj) {rmse:.2f} at {par}; k_h at h=1,7,14,27: "
          f"{k[0]:.3f} {k[6]:.3f} {k[13]:.3f} {k[26]:.3f}")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    pd.DataFrame(S, index=days, columns=[f"S{h}" for h in LEADS]).to_parquet(os.path.join(HERE, "out", "S.parquet"))
    D_te = (S[:, 1:] - S[:, :1])[pos_te]
    pred_te = flux.loc[ote].to_numpy()[:, None] + D_te * k
    D_tr = (S[:, 1:] - S[:, :1])[pos_tr]
    pred_tr = flux.loc[otr].to_numpy()[:, None] + D_tr * k
    for h in range(1, H + 1):
        pd.DataFrame({"t_date": ote.date, "pred_adj": np.round(pred_te[:, h - 1], 4)}).to_csv(
            os.path.join(HERE, "out", f"lead_{h:02d}.csv"), index=False)
        pd.DataFrame({"t_date": otr.date, "oof_adj": np.round(pred_tr[:, h - 1], 4),
                      "y_adj": np.round(Ytr[:, h - 1] + flux.loc[otr].to_numpy(), 4)}).to_csv(
            os.path.join(HERE, "out", f"oof_{h:02d}.csv"), index=False)
    common.write_predictions(HERE, ote, pred_te)


if __name__ == "__main__":
    main()

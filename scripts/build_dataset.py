#!/usr/bin/env python3
"""Build the unified daily modeling table -> data/daily.csv.

One row per calendar day, 1996-01-01 -> last day with F10.7:
  f107_obs, f107_adj   Penticton via LISIRD (0.0 = missing -> NaN). Multiple
                       obs/day post-1990: the one nearest 20:00 UT (local noon)
                       is taken, matching the canonical daily value (the one
                       SWPC/CelesTrak publish, flare spikes included).
  f107_obs_rob,        Flare-robust variants for TRAINING: on days whose
  f107_adj_rob         readings disagree by >10% (max/min of the 17/20/23 UT
                       values) the reading nearest the previous day's robust
                       value is used (causal; flares go up, glitches go
                       down); any remaining one-day spike >1.3x / dip <0.7x
                       both neighbours is replaced by the neighbour mean.
  f107_flare           1 on days where the robust value differs from the
                       canonical one; f107_nread = readings that day.
  ssn                  SILSO daily total ISN v2 (-1 -> NaN).
  ar_*                 per-day aggregates of the SRS active-region table
                       (srs_parsed.csv). Days whose SRS file exists but lists
                       no regions are genuine zeros; days with no SRS file are
                       forward-filled and flagged srs_present=0.

Gaps <= 3 days in f107/ssn are linearly interpolated (flagged f107_filled /
ssn_filled); longer gaps stay NaN and the window builder drops those samples.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
START = "1947-02-14"  # first LISIRD F10.7 day; ar_* stay NaN before 1996


def load_f107():
    df = pd.read_csv(os.path.join(ROOT, "f107_penticton_lisird.csv"))
    df.columns = ["jd", "obs", "adj"]
    # pandas' julian origin is JD 0 = noon, so the JD converts directly to UT
    # (2026-09-02 fix: a former "+0.5" here dated every reading one day late).
    df["t"] = pd.to_datetime(df["jd"], unit="D", origin="julian")
    df["date"] = df["t"].dt.normalize()
    df["hour"] = df["t"].dt.hour + df["t"].dt.minute / 60.0
    df = df.replace({"obs": {0.0: np.nan}, "adj": {0.0: np.nan}}).dropna(subset=["obs"])
    # canonical daily value: the observation closest to 20:00 UT (local noon)
    df["dist"] = (df["hour"] - 20.0).abs()
    noon = df.sort_values(["date", "dist"]).groupby("date").first()[["obs", "adj"]]
    g = df.groupby("date")
    out = noon.rename(columns={"obs": "f107_obs", "adj": "f107_adj"})
    out["f107_nread"] = g.size()
    spread = g["obs"].max() / g["obs"].min()
    # flare/glitch day: the day's readings disagree by >10%. Flares push a
    # reading UP, instrument glitches push one DOWN, so neither min nor
    # median is safe; take the reading nearest the previous day's robust
    # value (causal, so the realtime path can apply the same rule).
    rob_obs = out["f107_obs"].copy()
    rob_adj = out["f107_adj"].copy()
    prev = np.nan
    for day in out.index:
        if spread.loc[day] > 1.10 and np.isfinite(prev):
            r = df[df["date"] == day]
            pick = r.iloc[(r["obs"] - prev).abs().argmin()]
            rob_obs.loc[day], rob_adj.loc[day] = pick["obs"], pick["adj"]
        prev = rob_obs.loc[day]
    out["f107_obs_rob"], out["f107_adj_rob"] = rob_obs, rob_adj
    # residual one-day spikes (>1.3x both neighbours) and dips (<0.7x both),
    # mostly the single-reading era: replace by the neighbour mean
    r = out["f107_obs_rob"]
    prev_, next_ = r.shift(1), r.shift(-1)
    bad = ((r > 1.3 * prev_) & (r > 1.3 * next_)) | ((r < 0.7 * prev_) & (r < 0.7 * next_))
    ratio = out["f107_adj_rob"] / out["f107_obs_rob"]
    out.loc[bad, "f107_obs_rob"] = ((prev_ + next_) / 2)[bad]
    out.loc[bad, "f107_adj_rob"] = (out["f107_obs_rob"] * ratio)[bad]
    out["f107_flare"] = (out["f107_obs_rob"] != out["f107_obs"]).astype(int)
    return out[["f107_obs", "f107_adj", "f107_obs_rob", "f107_adj_rob", "f107_flare", "f107_nread"]]


def load_ssn():
    df = pd.read_csv(
        os.path.join(ROOT, "ssn_daily_silso.csv"),
        sep=";",
        header=None,
        usecols=[0, 1, 2, 4],
        names=["y", "m", "d", "ssn"],
    )
    df["date"] = pd.to_datetime(df[["y", "m", "d"]].rename(columns={"y": "year", "m": "month", "d": "day"}))
    df.loc[df["ssn"] < 0, "ssn"] = np.nan
    return df.set_index("date")[["ssn"]]


def srs_days_present():
    """Valid-dates covered by an SRS file (file date minus one day)."""
    days = set()
    for p in glob.glob(os.path.join(ROOT, "srs", "**", "[12]*SRS.txt"), recursive=True):
        if os.path.getsize(p) == 0:
            continue
        m = re.match(r"(\d{8})SRS\.txt", os.path.basename(p))
        days.add(pd.Timestamp(m.group(1)) - pd.Timedelta(days=1))
    return days


def load_regions():
    df = pd.read_csv(os.path.join(ROOT, "srs_parsed.csv"), parse_dates=["valid_date"])
    one = df[df.section == "I"].copy()
    one["area"] = one["area"].fillna(0)
    rad = np.pi / 180.0
    # projected area: cos(lat)*cos(lon), clamped — disk-center contribution proxy
    one["proj"] = one.area * np.maximum(np.cos(one.lat * rad) * np.cos(one.lon * rad), 0.0)
    one["complex"] = one.mag_type.str.contains("GAMMA|DELTA", na=False)
    g = one.groupby("valid_date")
    out = pd.DataFrame(
        {
            "ar_area": g.area.sum(),
            "ar_area_proj": g.proj.sum(),
            "ar_area_east": one[one.lon >= 30].groupby("valid_date").area.sum(),
            "ar_area_west": one[one.lon <= -30].groupby("valid_date").area.sum(),
            "ar_area_max": g.area.max(),
            "ar_nreg": g.size(),
            "ar_nspots": g.num_spots.sum(),
            "ar_ncomplex": g["complex"].sum(),
        }
    )
    ia = df[df.section == "IA"].groupby("valid_date").size().rename("ar_nplage")
    ii = df[df.section == "II"].groupby("valid_date").size().rename("ar_nreturn")
    return pd.concat([out, ia, ii], axis=1, sort=True)


def main():
    f107, ssn, ar = load_f107(), load_ssn(), load_regions()
    idx = pd.date_range(START, f107.index.max(), freq="D")
    df = pd.DataFrame(index=idx).join(f107).join(ssn).join(ar)

    for col in ("f107_obs", "f107_adj", "ssn"):
        flag = col.split("_")[0] + "_filled"
        filled = df[col].interpolate(limit=3, limit_area="inside")
        df[flag] = (df[col].isna() & filled.notna()).astype(int)
        df[col] = filled
    for col in ("f107_obs_rob", "f107_adj_rob"):
        df[col] = df[col].interpolate(limit=3, limit_area="inside")
    df["f107_flare"] = df["f107_flare"].fillna(0).astype(int)
    df["f107_nread"] = df["f107_nread"].fillna(0).astype(int)

    ar_cols = [c for c in df.columns if c.startswith("ar_")]
    present = srs_days_present()
    df["srs_present"] = [int(d in present) for d in df.index]
    # file present + no rows -> genuine zeros; file missing -> forward-fill
    df.loc[df.srs_present == 1, ar_cols] = df.loc[df.srs_present == 1, ar_cols].fillna(0)
    df[ar_cols] = df[ar_cols].ffill()

    df.index.name = "date"
    out = os.path.join(ROOT, "daily.csv")
    df.to_csv(out, float_format="%.4f")

    print(f"{out}: {len(df)} days {df.index[0].date()} -> {df.index[-1].date()}")
    for col in ("f107_obs", "f107_adj", "ssn"):
        print(f"  {col}: {df[col].isna().sum()} NaN, {df[col.split('_')[0] + '_filled'].sum()} interpolated")
    print(f"  srs missing days (ffilled): {(df.srs_present == 0).sum()}")
    print(f"  flare-robust: {int(df.f107_flare.sum())} days differ from the canonical noon value")
    jan = df[df.index.month == 1]
    print(f"  sanity: Jan mean obs/adj = {(jan.f107_obs / jan.f107_adj).mean():.4f} (expect ~1.03, perihelion)")
    print(f"  sanity: Jul mean obs/adj = {(df[df.index.month == 7].f107_obs / df[df.index.month == 7].f107_adj).mean():.4f} (expect ~0.97)")


if __name__ == "__main__":
    main()

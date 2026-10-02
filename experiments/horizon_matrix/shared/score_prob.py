#!/usr/bin/env python3
"""Probabilistic scorecard (added 2026-09-02; separate from the fixed
deterministic scorer, which stays untouched).

Usage: score_prob.py <cell_dir>

Reads <cell>/out/ensemble.csv — one row per (t_date, lead, target_date) on
the canonical test origins with quantile columns q05..q95 (any subset of
qNN) and optionally `mean` and member columns m000..mNNN — and scores
OBSERVED F10.7. Writes <cell>/out/scorecard_prob.json.

Metrics:
  crps        exact from members if present, else the quantile-weighted
              approximation 2/K * sum_k pinball_tau_k(y - q_k)
  crps_skill  1 - crps / MAE of persistence (a point forecast's CRPS is its
              absolute error, so this is "value added over persistence in
              CRPS units")
  cov80/cov50 empirical coverage of [q10,q90] / [q25,q75]; width80 sharpness
  by year, by lead band, by truth bin (>=150, >=200), plus
  spike_capture: among target days whose truth exceeds the origin-day flux
              by >= 30 / 50 sfu, the share with truth <= q90 (the band
              reaches the spike) and the share with truth <= q95.
  det         RMSE/MAE of q50 and of `mean` (if given) — for the leaderboard
              cross-check against the deterministic scorer.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def crps_members(M, y):
    """M (n, m) members, y (n,) -> per-row CRPS (exact, O(m log m))."""
    m = M.shape[1]
    Ms = np.sort(M, axis=1)
    term1 = np.mean(np.abs(Ms - y[:, None]), axis=1)
    # E|X - X'| = 2/m^2 * sum_i (2i - m - 1) x_(i)   (sorted ascending, i 1-based)
    i = np.arange(1, m + 1)
    term2 = (2.0 / m**2) * np.sum((2 * i - m - 1) * Ms, axis=1)
    return term1 - 0.5 * term2


def crps_quantiles(Q, taus, y):
    """Q (n, K) at levels taus -> per-row approximate CRPS."""
    d = y[:, None] - Q
    pin = np.where(d >= 0, taus * d, (taus - 1) * d)
    return 2.0 * pin.mean(axis=1)


def block(g, qcols, mcols, taus):
    y = g.y.to_numpy()
    if mcols:
        crps = crps_members(g[mcols].to_numpy(), y)
    else:
        crps = crps_quantiles(g[qcols].to_numpy(), taus, y)
    pers_mae = np.mean(np.abs(g.persist - y))
    out = {
        "n": int(len(g)),
        "crps": float(crps.mean()),
        "crps_skill_vs_persist": float(1 - crps.mean() / pers_mae),
        "cov80": float(((y >= g.q10) & (y <= g.q90)).mean()),
        "cov50": float(((y >= g.q25) & (y <= g.q75)).mean()),
        "width80": float((g.q90 - g.q10).mean()),
        "above_q90": float((y > g.q90).mean()),
        "below_q10": float((y < g.q10).mean()),
        "rmse_q50": float(np.sqrt(np.mean((g.q50 - y) ** 2))),
        "mae_q50": float(np.mean(np.abs(g.q50 - y))),
    }
    if "mean" in g:
        out["rmse_mean"] = float(np.sqrt(np.mean((g["mean"] - y) ** 2)))
    return out


def main(cell_dir):
    cell_dir = cell_dir.rstrip("/")
    df = common.load_daily()
    orig = common.origins("test", df)
    truth = df["f107_obs"]
    e = pd.read_csv(os.path.join(cell_dir, "out", "ensemble.csv"),
                    parse_dates=["t_date", "target_date"])
    e = e[e.t_date.isin(orig)]
    if len(e) != len(orig) * common.HORIZON:
        sys.exit(f"FAIL: {len(e)} rows on canonical test origins, expected {len(orig) * common.HORIZON}")
    qcols = sorted(c for c in e.columns if re.fullmatch(r"q\d\d", c))
    mcols = sorted(c for c in e.columns if re.fullmatch(r"m\d+", c))
    for need in ("q10", "q25", "q50", "q75", "q90"):
        assert need in qcols, f"ensemble.csv needs column {need}"
    taus = np.array([int(c[1:]) / 100 for c in qcols])
    e["y"] = truth.reindex(e.target_date).to_numpy()
    e["persist"] = truth.reindex(e.t_date).to_numpy()
    e = e.dropna(subset=["y"])
    e["rise"] = e.y - e.persist

    card = {
        "cell": os.path.basename(cell_dir),
        "data_version": common.DATA_VERSION,
        "n_origins": int(len(orig)),
        "quantiles": qcols,
        "members": len(mcols),
        "pooled": block(e, qcols, mcols, taus),
        "bands": {
            "L1_7": block(e[e.lead <= 7], qcols, mcols, taus),
            "L8_14": block(e[(e.lead >= 8) & (e.lead <= 14)], qcols, mcols, taus),
            "L15_30": block(e[e.lead >= 15], qcols, mcols, taus),
        },
        "by_year": {int(yr): block(g, qcols, mcols, taus) for yr, g in e.groupby(e.target_date.dt.year)},
        "truth_ge150": block(e[e.y >= 150], qcols, mcols, taus),
        "truth_ge200": block(e[e.y >= 200], qcols, mcols, taus),
        "spike_capture": {},
    }
    for thr in (30, 50):
        s = e[e.rise >= thr]
        card["spike_capture"][f"rise_ge{thr}"] = {
            "n": int(len(s)),
            "within_q90": float((s.y <= s.q90).mean()) if len(s) else None,
            "within_q95": float((s.y <= s.q95).mean()) if len(s) and "q95" in s else None,
            "q50_reaches_half": float((s.q50 - s.persist >= 0.5 * s.rise).mean()) if len(s) else None,
        }
    with open(os.path.join(cell_dir, "out", "scorecard_prob.json"), "w") as f:
        json.dump(card, f, indent=1)
    p = card["pooled"]
    print(f"{card['cell']}: CRPS {p['crps']:.2f} (skill vs persist {p['crps_skill_vs_persist']:+.3f})  "
          f"cov80 {p['cov80']:.3f} cov50 {p['cov50']:.3f} width80 {p['width80']:.1f}  "
          f"RMSE(q50) {p['rmse_q50']:.2f}" + (f" RMSE(mean) {p['rmse_mean']:.2f}" if "rmse_mean" in p else ""))
    for k, v in card["bands"].items():
        print(f"  {k:6} CRPS {v['crps']:6.2f}  cov80 {v['cov80']:.3f}  width80 {v['width80']:5.1f}")
    for yr, v in card["by_year"].items():
        print(f"  {yr}   CRPS {v['crps']:6.2f}  cov80 {v['cov80']:.3f}  width80 {v['width80']:5.1f}")
    for k, v in (("truth_ge150", card["truth_ge150"]), ("truth_ge200", card["truth_ge200"])):
        print(f"  {k}: CRPS {v['crps']:.2f} cov80 {v['cov80']:.3f} above_q90 {v['above_q90']:.3f}")
    for k, v in card["spike_capture"].items():
        print(f"  spike {k}: n={v['n']} within_q90 {v['within_q90']}  q50_reaches_half {v['q50_reaches_half']}")


if __name__ == "__main__":
    main(sys.argv[1])

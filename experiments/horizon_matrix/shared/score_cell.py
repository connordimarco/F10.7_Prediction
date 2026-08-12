#!/usr/bin/env python3
"""Fixed scorecard — NEVER edited per cell (LAUREN rule).

Usage: score_cell.py <cell_dir>

Reads <cell>/out/predictions.csv, scores OBSERVED F10.7 on the canonical test
origins, writes <cell>/out/scorecard.json.

Reference for skill: raw persistence, pred_obs(t+h) = f107_obs(t). A cell
named *persist* scoring skill != 0 means the harness is broken.

Guards (screening): pooled r and RMSE — a candidate must not lose to the best
baseline cell on pooled RMSE while claiming a win on any target.
Targets: skill vs persistence per lead band (1-7, 8-14, 15-30), RMSE in the
high-activity bin (true F10.7 >= 150).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def metrics(y, p):
    err = p - y
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan"),
    }


def main(cell_dir):
    cell_dir = cell_dir.rstrip("/")
    df = common.load_daily()
    orig = common.origins("test", df)
    truth = df["f107_obs"]

    pred = pd.read_csv(
        os.path.join(cell_dir, "out", "predictions.csv"),
        parse_dates=["t_date", "target_date"],
    )
    pred = pred[pred.t_date.isin(orig)]
    n_expect = len(orig) * common.HORIZON
    if len(pred) != n_expect:
        sys.exit(f"FAIL: {len(pred)} rows on canonical test origins, expected {n_expect}")

    pred["y"] = truth.reindex(pred.target_date).to_numpy()
    pred["persist"] = truth.reindex(pred.t_date).to_numpy()
    pred = pred.dropna(subset=["y"])

    per_lead = []
    for h, g in pred.groupby("lead"):
        m = metrics(g.y.to_numpy(), g.pred_obs.to_numpy())
        m["lead"] = int(h)
        m["skill_vs_persist"] = float(
            1.0 - m["rmse"] / np.sqrt(np.mean((g.persist - g.y) ** 2))
        )
        per_lead.append(m)

    def band(lo, hi):
        g = pred[(pred.lead >= lo) & (pred.lead <= hi)]
        m = metrics(g.y.to_numpy(), g.pred_obs.to_numpy())
        m["skill_vs_persist"] = float(
            1.0 - m["rmse"] / np.sqrt(np.mean((g.persist - g.y) ** 2))
        )
        return m

    hi = pred[pred.y >= 150]
    card = {
        "cell": os.path.basename(cell_dir),
        "n_origins": int(len(orig)),
        "test_span": [str(orig[0].date()), str(orig[-1].date())],
        "pooled": band(1, common.HORIZON),
        "bands": {"L1_7": band(1, 7), "L8_14": band(8, 14), "L15_30": band(15, 30)},
        "high_activity_rmse": float(
            np.sqrt(np.mean((hi.pred_obs - hi.y) ** 2))
        ) if len(hi) else None,
        "per_lead": per_lead,
    }
    path = os.path.join(cell_dir, "out", "scorecard.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=1)
    p = card["pooled"]
    print(
        f"{card['cell']}: pooled RMSE {p['rmse']:.2f} MAE {p['mae']:.2f} r {p['r']:.3f} "
        f"skill {p['skill_vs_persist']:+.3f}  (n={card['n_origins']} origins)"
    )
    for k, v in card["bands"].items():
        print(f"  {k:6} RMSE {v['rmse']:6.2f}  skill {v['skill_vs_persist']:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])

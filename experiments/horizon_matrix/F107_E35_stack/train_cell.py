#!/usr/bin/env python3
"""E35: level-2 stack over finished cells' per-lead OOF records.

Each source cell leaves out/oof_XX.csv (train-origin OOF prediction + truth,
adjusted space) and out/lead_XX.csv (test prediction). Per lead, a nonneg
linear blend (+ intercept) of the sources is fitted on the OOF rows and
applied to the test rows; the stack's own OOF blend is written in the same
format so E31-style ensembles can sit on top of it. Sources whose files are
missing are skipped (logged). Mild optimism caveat: a source's OOF blend
weights were fitted on these same rows, but with 5-7 nonneg coefficients
per lead that is negligible next to the 9,347-row fit.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import common

SOURCES = os.environ.get("STACK_SOURCES", "F107_E30_combo_v2 F107_E33_combo_v2_raw F107_E34_combo_gated F107_E40_nn F107_E41_mlp").split()
FLUX = "f107_adj_rob"


def main():
    df = common.load_daily()
    _, _, o96, _ = common.build_samples(df, "train", extra=common.FS_COLS, flux=FLUX)
    _, _, ote, _ = common.build_samples(df, "test", extra=common.FS_COLS, flux=FLUX)
    srcs = [s for s in SOURCES if os.path.exists(os.path.join(HERE, "..", s, "out", "oof_30.csv"))
            and os.path.exists(os.path.join(HERE, "..", s, "out", "lead_30.csv"))]
    print("sources:", srcs, "| missing:", [s for s in SOURCES if s not in srcs])
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    pred = np.empty((len(ote), common.HORIZON))
    for h in range(1, common.HORIZON + 1):
        Ztr, Zte, y = [], [], None
        for s in srcs:
            o = pd.read_csv(os.path.join(HERE, "..", s, "out", f"oof_{h:02d}.csv"), parse_dates=["t_date"])
            t = pd.read_csv(os.path.join(HERE, "..", s, "out", f"lead_{h:02d}.csv"), parse_dates=["t_date"])
            assert (pd.DatetimeIndex(o.t_date) == o96).all() and (pd.DatetimeIndex(t.t_date) == ote).all(), s
            Ztr.append(o.oof_adj.to_numpy()); Zte.append(t.pred_adj.to_numpy()); y = o.y_adj.to_numpy()
        Ztr, Zte = np.column_stack(Ztr), np.column_stack(Zte)
        blend = LinearRegression(positive=True).fit(Ztr, y)
        pred[:, h - 1] = blend.predict(Zte)
        pd.DataFrame({"t_date": o96.date, "oof_adj": np.round(blend.predict(Ztr), 4), "y_adj": np.round(y, 4)}).to_csv(
            os.path.join(HERE, "out", f"oof_{h:02d}.csv"), index=False)
        pd.DataFrame({"t_date": ote.date, "pred_adj": np.round(pred[:, h - 1], 4)}).to_csv(
            os.path.join(HERE, "out", f"lead_{h:02d}.csv"), index=False)
        print(f"lead {h:2d}: w=[" + " ".join(f"{w:.2f}" for w in blend.coef_) + f"] b={blend.intercept_:.1f}", flush=True)
    common.write_predictions(HERE, ote, pred)


if __name__ == "__main__":
    main()

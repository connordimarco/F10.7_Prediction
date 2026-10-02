#!/usr/bin/env python3
"""Daily plage/EUV proxy table -> data/proxy_daily.csv (optional-NaN feature
family; never gates origin validity).

  mgii   Bremen composite Mg II core-to-wing index via LISIRD
         (`bremen_composite_mgii`, daily 1978-11-07 -> ~yesterday). Gaps of
         <= 3 days linearly interpolated, longer gaps NaN.
Fetch: curl -o data/mgii_composite_lisird.csv
       https://lasp.colorado.edu/lisird/latis/dap/bremen_composite_mgii.csv
"""

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def main():
    m = pd.read_csv(os.path.join(ROOT, "mgii_composite_lisird.csv"))
    m.columns = ["date", "mgii", "unc", "source"]
    m["date"] = pd.to_datetime(m["date"], format="%Y %m %d")
    m = m.set_index("date").sort_index()
    m = m[m.mgii > 0]
    idx = pd.date_range("1947-02-14", max(m.index.max(), pd.Timestamp.today().normalize()), freq="D")
    out = pd.DataFrame(index=idx)
    out["mgii"] = m["mgii"].reindex(idx).interpolate(limit=3, limit_area="inside")
    out.index.name = "date"
    out.to_csv(os.path.join(ROOT, "proxy_daily.csv"), float_format="%.5f")
    have = out.mgii.notna()
    print(f"proxy_daily.csv: mgii {have.sum()} days {out.index[have][0].date()} -> {out.index[have][-1].date()}, "
          f"{(~have[out.index >= '1978-11-07']).sum()} NaN days since start")


if __name__ == "__main__":
    main()

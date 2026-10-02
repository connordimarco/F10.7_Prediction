# F10.7_Prediction

Forecast daily F10.7 solar radio flux 1–30 days ahead, for satellite drag.

One LightGBM per lead day on the last 60 days of F10.7, sunspot number, SWPC
active-region summaries and GONG/JSOC far-side detections, predicting the
ratio of future flux to the current 81-day mean. Trained 1947–2021, early-
stopped on 2022–23, scored on 2024 onward (cycle-25 maximum): pooled RMSE
29.1 sfu; against SWPC's operational 27-day outlook on 125 matched issues,
28.6 vs 32.1 sfu (−11%), 32.2 vs 37.3 on days ≥150 sfu (−14%).

- `scripts/` — rebuild `data/` from the raw archives (`fetch_raw.sh` →
  `parse_srs.py` → `build_dataset.py`; `fetch_farside.py` →
  `build_farside_daily.py`). Endpoints and archives: `data/SOURCES.md`.
- `experiments/horizon_matrix/` — the model (`F107_E24_ratio_deep`), the
  shared loader/trainer/scorer it uses, and the SWPC benchmark
  (`benchmarks/swpc_27day.py`, `plots/swpc_vs_ours_by_lead.png`).
- `realtime/` — the frozen model run daily: saved boosters, calibrated band,
  data refresh, predictor, verifier; issued forecasts archived in
  `realtime/forecasts/`. See `realtime/README.md`.

Env: `python3 -m venv env && env/bin/pip install -r requirements.txt`
(LightGBM needs Homebrew `libomp`).
